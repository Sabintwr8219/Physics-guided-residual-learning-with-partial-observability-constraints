"""
Generate ground-truth queue join and queue length files.

This script uses the filtered northbound trajectory file to identify queue-join
events using standstill and creep-based rules. It saves the GT queue-join file,
GT queue-length file, and a plot-ready cumulative GT curve file.

Outputs:
    data/processed_data/gt_queue_join_runXXX.csv
    data/processed_data/gt_queue_length_runXXX_0p1s.csv
    data/processed_data/gt_curve_plot_ready_allruns.csv
"""

from __future__ import annotations

import gc
import math

import numpy as np
import pandas as pd

from config import (
    RUN_IDS,
    TRAJ_NB_FILTERED_CSV,
    GT_JOIN_PATTERN,
    GT_QUEUE_LENGTH_PATTERN,
    GT_CURVE_PLOT_READY_CSV,
    CHUNKSIZE_TRAJ,
    M_TO_FT,
    CORRIDOR_MIN_FT,
    CORRIDOR_MAX_FT,
    V_STOP_FPS,
    STOP_PERSIST_SEC,
    CREEP_DROP_FPS,
    CREEP_LOOKBACK_SEC,
    CREEP_PERSIST_SEC,
    NEIGHBOR_TIME_TOL_SEC,
    NEIGHBOR_MAX_GAP_FT,
    NEIGHBOR_LOW_SPEED_FPS,
    NEIGHBOR_DROP_FPS,
    BACKTRACK_MAX_SEC,
    ONSET_DROP_EPS_FPS,
    JOIN_PERSIST_SEC,
    GROW_TOL_FT,
    DEFAULT_L_EFF_FT_PER_VEH,
    ensure_project_directories,
)


# ============================================================
# Required columns
# ============================================================

TRAJ_USECOLS = [
    "run_id",
    "veh_uid",
    "vehID",
    "Total_Sim_Time_Sec",
    "s_rel_stop_m",
    "Speed_fps",
]

TRAJ_DTYPES = {
    "run_id": "int16",
    "veh_uid": "string",
    "vehID": "int32",
    "Total_Sim_Time_Sec": "float32",
    "s_rel_stop_m": "float32",
    "Speed_fps": "float32",
}


# ============================================================
# Utility functions
# ============================================================

def robust_dt(times: np.ndarray, default: float = 0.1) -> float:
    """Estimate the representative time step from trajectory timestamps."""
    if len(times) < 3:
        return default

    diffs = np.diff(np.sort(times))
    diffs = diffs[(diffs > 1e-9) & (diffs < 10)]

    if len(diffs) == 0:
        return default

    return float(np.median(diffs))


def nearest_index_at_time(times: np.ndarray, target_time: float):
    """
    Return the index of the timestamp closest to target_time.

    Returns None if target_time is outside the available time range.
    """
    if len(times) == 0:
        return None

    if target_time < times[0] or target_time > times[-1]:
        return None

    idx = np.searchsorted(times, target_time, side="left")

    if idx == 0:
        return 0

    if idx >= len(times):
        return len(times) - 1

    left = idx - 1
    right = idx

    if abs(times[right] - target_time) < abs(times[left] - target_time):
        return right

    return left


def compute_stopbar_cross_time(group: pd.DataFrame) -> float:
    """
    Compute first stopbar crossing time for one vehicle.

    If the vehicle never reaches s_rel_stop_ft >= 0, the last observed time is used.
    """
    g = group.sort_values("Total_Sim_Time_Sec")

    t = g["Total_Sim_Time_Sec"].to_numpy(dtype=float)
    y = g["s_rel_stop_ft"].to_numpy(dtype=float)

    crossing_idx = np.where(y >= 0)[0]

    if len(crossing_idx) > 0:
        return float(t[crossing_idx[0]])

    return float(t[-1])


def map_times_to_index(times: np.ndarray, event_times: np.ndarray) -> np.ndarray:
    """
    Map event times to nearest indices on the available simulation time grid.
    """
    if len(event_times) == 0:
        return np.array([], dtype=int)

    idx = np.searchsorted(times, event_times, side="left")
    idx[idx >= len(times)] = len(times) - 1

    mask = (idx > 0) & (
        np.abs(times[idx] - event_times) > np.abs(times[idx - 1] - event_times)
    )
    idx[mask] -= 1

    return idx


def build_hybrid_queue_length(
    times: np.ndarray,
    tail_len_raw: np.ndarray,
    dep_counts: np.ndarray,
    l_eff_ft: float,
    grow_tol_ft: float,
) -> np.ndarray:
    """
    Build a smooth GT queue-length profile.

    Growth is taken from the observed tail location. Dissipation is handled using
    downstream departures multiplied by an effective vehicle length.
    """
    n = len(times)

    if n == 0:
        return np.array([], dtype=float)

    q = np.zeros(n, dtype=float)
    q[0] = max(0.0, float(tail_len_raw[0]))

    for i in range(1, n):
        previous_q = q[i - 1]
        tail_now = max(0.0, float(tail_len_raw[i]))
        departure_reduction = float(dep_counts[i]) * float(l_eff_ft)

        if tail_now > previous_q + grow_tol_ft:
            q[i] = tail_now
        else:
            q[i] = max(0.0, previous_q - departure_reduction)
            if tail_now > q[i]:
                q[i] = tail_now

    q[q < 0] = 0.0
    return q


# ============================================================
# Vehicle-level queue-join logic
# ============================================================

def backtrack_to_slowdown_onset(
    times: np.ndarray,
    speeds: np.ndarray,
    in_corridor: np.ndarray,
    confirm_idx: int | None,
    backtrack_max_sec: float,
    onset_drop_eps_fps: float,
):
    """
    Backtrack from the confirmation point to the slowdown onset point.
    """
    if confirm_idx is None:
        return None

    t_confirm = float(times[confirm_idx])
    left_time = t_confirm - backtrack_max_sec

    candidate_idx = np.where(
        (times >= left_time) &
        (times <= t_confirm) &
        in_corridor
    )[0]

    if len(candidate_idx) == 0:
        return confirm_idx

    local_speeds = speeds[candidate_idx]
    peak_speed = np.max(local_speeds)

    peak_candidates = candidate_idx[np.where(np.isclose(local_speeds, peak_speed))[0]]

    if len(peak_candidates) == 0:
        return confirm_idx

    peak_idx = int(peak_candidates[-1])

    for j in range(peak_idx, confirm_idx + 1):
        if not in_corridor[j]:
            continue

        if speeds[peak_idx] - speeds[j] >= onset_drop_eps_fps:
            return j

    return peak_idx


def build_vehicle_debug_flags(group: pd.DataFrame) -> pd.DataFrame:
    """
    Compute standstill and creep candidate flags for one vehicle trajectory.
    """
    g = group.sort_values("Total_Sim_Time_Sec").copy()

    t = g["Total_Sim_Time_Sec"].to_numpy(dtype=float)
    v = g["Speed_fps"].to_numpy(dtype=float)
    s = g["s_rel_stop_ft"].to_numpy(dtype=float)

    n = len(g)
    dt_local = robust_dt(t, default=0.1)

    in_corridor = (
        (s >= CORRIDOR_MIN_FT - 1e-9) &
        (s <= CORRIDOR_MAX_FT + 1e-9)
    )

    standstill_confirm = np.zeros(n, dtype=bool)
    creep_candidate = np.zeros(n, dtype=bool)
    baseline_speed_arr = np.full(n, np.nan)

    for i in range(n):
        if not in_corridor[i]:
            continue

        t0 = float(t[i])

        # --------------------
        # Standstill confirmation
        # --------------------
        t_end_stop = t0 + STOP_PERSIST_SEC
        j_stop = np.searchsorted(t, t_end_stop, side="right") - 1

        if j_stop >= i and t[j_stop] >= t_end_stop - 0.5 * dt_local:
            if np.all(in_corridor[i:j_stop + 1]):
                if np.nanmax(v[i:j_stop + 1]) <= V_STOP_FPS + 1e-9:
                    standstill_confirm[i] = True

        # --------------------
        # Creep candidate
        # --------------------
        previous_time = t0 - CREEP_LOOKBACK_SEC
        previous_idx = nearest_index_at_time(t, previous_time)

        if previous_idx is None:
            continue

        if not in_corridor[previous_idx]:
            continue

        baseline_speed = float(v[previous_idx])
        baseline_speed_arr[i] = baseline_speed

        speed_drop = baseline_speed - float(v[i])

        if speed_drop < CREEP_DROP_FPS:
            continue

        t_end_creep = t0 + CREEP_PERSIST_SEC
        j_creep = np.searchsorted(t, t_end_creep, side="right") - 1

        if j_creep >= i and t[j_creep] >= t_end_creep - 0.5 * dt_local:
            if np.all(in_corridor[i:j_creep + 1]):
                future_speeds = v[i:j_creep + 1]

                reduced_state_ok = (
                    np.nanmin(future_speeds) <= baseline_speed - CREEP_DROP_FPS + 1.0
                    and np.nanmedian(future_speeds) <= baseline_speed - 4.0
                )

                if reduced_state_ok:
                    creep_candidate[i] = True

    g["in_corridor_flag"] = in_corridor
    g["standstill_confirm_flag"] = standstill_confirm
    g["creep_candidate_flag"] = creep_candidate
    g["baseline_speed_1s_ago"] = baseline_speed_arr

    return g


def vehicle_has_similar_constrained_condition(
    vehicle_debug: pd.DataFrame,
    target_time: float,
) -> bool:
    """
    Check whether a neighboring vehicle shows a similar constrained state near target_time.
    """
    t = vehicle_debug["Total_Sim_Time_Sec"].to_numpy(dtype=float)
    v = vehicle_debug["Speed_fps"].to_numpy(dtype=float)
    in_corridor = vehicle_debug["in_corridor_flag"].to_numpy(dtype=bool)

    idx = np.where(np.abs(t - target_time) <= NEIGHBOR_TIME_TOL_SEC)[0]

    if len(idx) == 0:
        return False

    if "standstill_confirm_flag" in vehicle_debug.columns:
        if np.any(vehicle_debug["standstill_confirm_flag"].to_numpy(dtype=bool)[idx]):
            return True

    for i in idx:
        if not in_corridor[i]:
            continue

        v_now = float(v[i])

        if v_now <= NEIGHBOR_LOW_SPEED_FPS:
            return True

        previous_idx = nearest_index_at_time(t, float(t[i]) - 1.0)

        if previous_idx is None:
            continue

        if not in_corridor[previous_idx]:
            continue

        if float(v[previous_idx]) - v_now >= NEIGHBOR_DROP_FPS:
            return True

    return False


def get_neighbor_rows_at_time(
    frame: pd.DataFrame,
    veh_uid: str,
    s_current: float,
) -> tuple[pd.Series | None, pd.Series | None]:
    """
    Find nearest ahead and behind vehicles within the neighbor gap threshold.
    """
    others = frame[frame["veh_uid"] != veh_uid].copy()

    if others.empty:
        return None, None

    ahead = others[others["s_rel_stop_ft"] > s_current].copy()
    behind = others[others["s_rel_stop_ft"] < s_current].copy()

    ahead_row = None
    behind_row = None

    if not ahead.empty:
        ahead["gap_ft"] = ahead["s_rel_stop_ft"] - s_current
        ahead = ahead[ahead["gap_ft"] <= NEIGHBOR_MAX_GAP_FT + 1e-9]

        if not ahead.empty:
            ahead_row = ahead.sort_values("gap_ft", ascending=True).iloc[0]

    if not behind.empty:
        behind["gap_ft"] = s_current - behind["s_rel_stop_ft"]
        behind = behind[behind["gap_ft"] <= NEIGHBOR_MAX_GAP_FT + 1e-9]

        if not behind.empty:
            behind_row = behind.sort_values("gap_ft", ascending=True).iloc[0]

    return ahead_row, behind_row


# ============================================================
# Data loading
# ============================================================

def load_run_trajectory(run_id: int) -> pd.DataFrame:
    """
    Load one simulation run from the filtered trajectory file using chunked reading.
    """
    chunks = []

    for chunk in pd.read_csv(
        TRAJ_NB_FILTERED_CSV,
        usecols=lambda col: col in TRAJ_USECOLS or col == "s_rel_stop_ft",
        dtype={k: v for k, v in TRAJ_DTYPES.items() if k != "s_rel_stop_m"},
        chunksize=CHUNKSIZE_TRAJ,
        low_memory=False,
    ):
        chunk["run_id"] = pd.to_numeric(chunk["run_id"], errors="coerce")
        chunk = chunk.dropna(subset=["run_id"]).copy()
        chunk["run_id"] = chunk["run_id"].astype(int)

        sub = chunk[chunk["run_id"] == int(run_id)].copy()

        if not sub.empty:
            chunks.append(sub)

    if not chunks:
        return pd.DataFrame(columns=TRAJ_USECOLS + ["s_rel_stop_ft"])

    df_run = pd.concat(chunks, ignore_index=True)

    df_run["veh_uid"] = df_run["veh_uid"].astype(str).str.strip()
    df_run["vehID"] = pd.to_numeric(df_run["vehID"], errors="coerce").astype(int)
    df_run["Total_Sim_Time_Sec"] = pd.to_numeric(df_run["Total_Sim_Time_Sec"], errors="coerce")
    df_run["Speed_fps"] = pd.to_numeric(df_run["Speed_fps"], errors="coerce")
    df_run["s_rel_stop_m"] = pd.to_numeric(df_run["s_rel_stop_m"], errors="coerce")

    if "s_rel_stop_ft" not in df_run.columns:
        df_run["s_rel_stop_ft"] = df_run["s_rel_stop_m"] * M_TO_FT
    else:
        df_run["s_rel_stop_ft"] = pd.to_numeric(df_run["s_rel_stop_ft"], errors="coerce")

    df_run = df_run.dropna(
        subset=[
            "veh_uid",
            "vehID",
            "Total_Sim_Time_Sec",
            "Speed_fps",
            "s_rel_stop_ft",
        ]
    ).copy()

    return df_run


# ============================================================
# Main GT processing
# ============================================================

def identify_queue_join_events_for_run(run_df: pd.DataFrame, run_id: int) -> pd.DataFrame:
    """
    Identify queue join events for all vehicles in one run.
    """
    debug_store: dict[str, pd.DataFrame] = {}

    for veh_uid, group in run_df.groupby("veh_uid", sort=False):
        debug_store[str(veh_uid)] = build_vehicle_debug_flags(group)

    time_groups = {
        float(time_value): group.copy()
        for time_value, group in run_df.groupby("Total_Sim_Time_Sec", sort=True)
    }
    time_keys = np.array(list(time_groups.keys()), dtype=float)

    join_records = []

    for veh_uid, g_debug in debug_store.items():
        g_debug = g_debug.sort_values("Total_Sim_Time_Sec").copy()

        t = g_debug["Total_Sim_Time_Sec"].to_numpy(dtype=float)
        v = g_debug["Speed_fps"].to_numpy(dtype=float)
        in_corridor = g_debug["in_corridor_flag"].to_numpy(dtype=bool)
        standstill_confirm = g_debug["standstill_confirm_flag"].to_numpy(dtype=bool)
        creep_candidate = g_debug["creep_candidate_flag"].to_numpy(dtype=bool)

        creep_confirm = np.zeros(len(g_debug), dtype=bool)
        support_side = np.array([None] * len(g_debug), dtype=object)

        queue_confirm = standstill_confirm.copy()
        rule_family = np.array([None] * len(g_debug), dtype=object)
        rule_family[standstill_confirm] = "standstill"

        candidate_indices = np.where(creep_candidate)[0]

        for i in candidate_indices:
            ti = float(t[i])

            k = nearest_index_at_time(time_keys, ti)
            if k is None:
                continue

            frame_time = float(time_keys[k])
            frame = time_groups[frame_time]

            row_now = frame[frame["veh_uid"] == veh_uid]

            if row_now.empty:
                continue

            s_now = float(row_now["s_rel_stop_ft"].iloc[0])

            ahead_row, behind_row = get_neighbor_rows_at_time(
                frame=frame,
                veh_uid=veh_uid,
                s_current=s_now,
            )

            ahead_ok = False
            behind_ok = False

            if ahead_row is not None:
                ahead_uid = str(ahead_row["veh_uid"])
                ahead_ok = vehicle_has_similar_constrained_condition(
                    debug_store[ahead_uid],
                    target_time=ti,
                )

            if behind_row is not None:
                behind_uid = str(behind_row["veh_uid"])
                behind_ok = vehicle_has_similar_constrained_condition(
                    debug_store[behind_uid],
                    target_time=ti,
                )

            if ahead_ok or behind_ok:
                creep_confirm[i] = True
                queue_confirm[i] = True
                rule_family[i] = "creep"

                if ahead_ok and behind_ok:
                    support_side[i] = "ahead+behind"
                elif ahead_ok:
                    support_side[i] = "ahead"
                else:
                    support_side[i] = "behind"

        dt_local = robust_dt(t, default=0.1)
        min_join_samples = max(1, int(math.ceil(JOIN_PERSIST_SEC / dt_local)))

        hit_idx = np.where(queue_confirm)[0]

        first_confirm_idx = None
        first_rule = None
        first_support = None

        if len(hit_idx) > 0:
            run_length = 0

            for j in range(len(hit_idx)):
                if j == 0 or hit_idx[j] == hit_idx[j - 1] + 1:
                    run_length += 1
                else:
                    run_length = 1

                if run_length >= min_join_samples:
                    first_confirm_idx = int(hit_idx[j - run_length + 1])
                    first_rule = rule_family[first_confirm_idx]
                    first_support = support_side[first_confirm_idx]
                    break

        if first_confirm_idx is not None:
            onset_idx = backtrack_to_slowdown_onset(
                times=t,
                speeds=v,
                in_corridor=in_corridor,
                confirm_idx=first_confirm_idx,
                backtrack_max_sec=BACKTRACK_MAX_SEC,
                onset_drop_eps_fps=ONSET_DROP_EPS_FPS,
            )

            t_join_onset = float(t[onset_idx]) if onset_idx is not None else np.nan
            t_join_confirm = float(t[first_confirm_idx])

            if np.isfinite(t_join_onset):
                t_queue_join = float(t_join_onset)
            else:
                t_queue_join = float(t_join_confirm)

            joined_queue = True
            join_rule_family = str(first_rule)
            is_standstill_join = bool(first_rule == "standstill")
            is_creep_join = bool(first_rule == "creep")
            creep_support_side = first_support
        else:
            t_join_onset = np.nan
            t_join_confirm = np.nan
            t_queue_join = np.nan

            joined_queue = False
            join_rule_family = None
            is_standstill_join = False
            is_creep_join = False
            creep_support_side = None

        t_stopbar = compute_stopbar_cross_time(g_debug)

        join_records.append(
            {
                "run_id": int(run_id),
                "veh_uid": str(veh_uid),
                "vehID": int(g_debug["vehID"].iloc[0]),
                "joined_queue": joined_queue,
                "join_rule_family": join_rule_family,
                "is_standstill_join": is_standstill_join,
                "is_creep_join": is_creep_join,
                "creep_support_side": creep_support_side,
                "t_queue_join_sec": t_queue_join,
                "t_join_onset_sec": t_join_onset,
                "t_join_confirm_sec": t_join_confirm,
                "t_stopbar_cross_sec": t_stopbar,
            }
        )

    gt = pd.DataFrame(join_records)

    # Non-queued vehicles use stopbar crossing as their operational event time.
    gt["t_queue_join_sec"] = gt["t_queue_join_sec"].fillna(gt["t_stopbar_cross_sec"])

    gt = gt[
        [
            "run_id",
            "veh_uid",
            "vehID",
            "joined_queue",
            "join_rule_family",
            "is_standstill_join",
            "is_creep_join",
            "creep_support_side",
            "t_queue_join_sec",
            "t_join_onset_sec",
            "t_join_confirm_sec",
            "t_stopbar_cross_sec",
        ]
    ].sort_values(["t_queue_join_sec", "vehID"]).reset_index(drop=True)

    gt["N_gt"] = np.arange(1, len(gt) + 1, dtype=int)

    return gt


def build_queue_length_for_run(run_df: pd.DataFrame, gt: pd.DataFrame) -> pd.DataFrame:
    """
    Build a GT queue-length profile from queue-join and departure events.
    """
    times = np.sort(run_df["Total_Sim_Time_Sec"].unique())

    joined_gt = gt[gt["joined_queue"] == True][
        ["veh_uid", "t_queue_join_sec", "t_stopbar_cross_sec"]
    ].copy()
    joined_gt["veh_uid"] = joined_gt["veh_uid"].astype(str)

    join_times = joined_gt["t_queue_join_sec"].to_numpy(dtype=float)
    departure_times = joined_gt["t_stopbar_cross_sec"].to_numpy(dtype=float)

    arr_idx = map_times_to_index(times, join_times)
    dep_idx = map_times_to_index(times, departure_times)

    arr_counts = np.zeros(len(times), dtype=int)
    dep_counts = np.zeros(len(times), dtype=int)

    for i in arr_idx:
        arr_counts[i] += 1

    for i in dep_idx:
        dep_counts[i] += 1

    queue_vehicle_count = np.zeros(len(times), dtype=float)

    for i in range(1, len(times)):
        queue_vehicle_count[i] = max(
            0.0,
            queue_vehicle_count[i - 1] + arr_counts[i] - dep_counts[i],
        )

    tail_len_raw = np.zeros(len(times), dtype=float)

    time_group = run_df.groupby("Total_Sim_Time_Sec", sort=True)

    for time_idx, current_time in enumerate(times):
        frame = time_group.get_group(current_time).copy()
        frame["veh_uid"] = frame["veh_uid"].astype(str)

        frame = frame.merge(joined_gt, on="veh_uid", how="inner")

        frame = frame[
            (frame["t_queue_join_sec"] <= current_time) &
            (current_time <= frame["t_stopbar_cross_sec"]) &
            (frame["s_rel_stop_ft"] < 0)
        ].copy()

        if frame.empty:
            continue

        back_s = float(frame["s_rel_stop_ft"].min())
        tail_len_raw[time_idx] = max(0.0, abs(back_s))

    ratio_samples = []

    for i in np.where(queue_vehicle_count > 0)[0]:
        if queue_vehicle_count[i] > 0 and tail_len_raw[i] > 0:
            ratio = tail_len_raw[i] / queue_vehicle_count[i]
            if 5.0 <= ratio <= 80.0:
                ratio_samples.append(float(ratio))

    if len(ratio_samples) == 0:
        l_eff_est = DEFAULT_L_EFF_FT_PER_VEH
    else:
        l_eff_est = float(np.median(np.asarray(ratio_samples, dtype=float)))

    queue_length = build_hybrid_queue_length(
        times=times,
        tail_len_raw=tail_len_raw,
        dep_counts=dep_counts,
        l_eff_ft=l_eff_est,
        grow_tol_ft=GROW_TOL_FT,
    )

    q_df = pd.DataFrame(
        {
            "time_sec": times,
            "queue_length_ft": queue_length,
            "queue_length_ft_raw": tail_len_raw,
            "Qveh": queue_vehicle_count,
            "arrivals_count": arr_counts,
            "departures_count": dep_counts,
            "L_eff_ft_per_veh": np.full(len(times), float(l_eff_est)),
        }
    )

    return q_df


def process_all_runs() -> None:
    """
    Generate GT files for all configured simulation runs.
    """
    ensure_project_directories()

    plot_ready_parts = []

    for run_id in RUN_IDS:
        print("=" * 80)
        print(f"Processing GT for run {run_id:03d}")
        print("=" * 80)

        run_df = load_run_trajectory(run_id)

        if run_df.empty:
            print(f"[Run {run_id:03d}] No trajectory rows found. Skipping.")
            continue

        print(f"[Run {run_id:03d}] Rows     : {len(run_df):,}")
        print(f"[Run {run_id:03d}] Vehicles : {run_df['veh_uid'].nunique():,}")

        gt = identify_queue_join_events_for_run(run_df, run_id)
        q_df = build_queue_length_for_run(run_df, gt)

        gt_out = GT_JOIN_PATTERN.as_posix().format(run_id=run_id)
        q_out = GT_QUEUE_LENGTH_PATTERN.as_posix().format(run_id=run_id)

        pd.DataFrame(gt).to_csv(gt_out, index=False)
        pd.DataFrame(q_df).to_csv(q_out, index=False)

        plot_ready = gt[
            [
                "run_id",
                "veh_uid",
                "vehID",
                "N_gt",
                "t_queue_join_sec",
                "joined_queue",
                "join_rule_family",
            ]
        ].copy()

        plot_ready_parts.append(plot_ready)

        print(
            f"[Run {run_id:03d}] Saved GT join and queue length | "
            f"joined={int(gt['joined_queue'].sum()):,}, "
            f"standstill={int(gt['is_standstill_join'].sum()):,}, "
            f"creep={int(gt['is_creep_join'].sum()):,}"
        )

        del run_df, gt, q_df
        gc.collect()

    if not plot_ready_parts:
        raise ValueError("No GT plot-ready rows were created.")

    plot_ready_all = pd.concat(plot_ready_parts, ignore_index=True)
    GT_CURVE_PLOT_READY_CSV.parent.mkdir(parents=True, exist_ok=True)
    plot_ready_all.to_csv(GT_CURVE_PLOT_READY_CSV, index=False)

    print("\nGT processing complete.")
    print(f"Plot-ready GT curve file saved to: {GT_CURVE_PLOT_READY_CSV}")


# ============================================================
# Main entry point
# ============================================================

def main() -> None:
    """Run GT queue generation for all configured runs."""
    process_all_runs()


if __name__ == "__main__":
    main()