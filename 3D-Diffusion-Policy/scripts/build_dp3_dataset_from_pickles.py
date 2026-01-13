#!/usr/bin/env python3
"""Convert pickled episodes into a DP3-style zarr dataset.

Expected pickle contents (per episode):
- observations: list/tuple of per-step dicts or a dict of arrays. Each step must include
  robot_state fields and point_cloud. Optional color_image1/color_image2.
- actions: array shaped (T, Da) or (T-1, Da). If length is T-1 we repeat the last action.
- rewards: optional array shaped (T,) or (T-1,).
- success/task/action_type: optional per-episode metadata.

The generated zarr contains data/state, data/action, data/point_cloud and any available
color images/rewards, plus meta/episode_ends and optional meta fields for success,
task, and action_type.
"""

import argparse
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Iterable, List, Sequence, Tuple
from datetime import datetime

import numpy as np
import zarr

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diffusion_policy_3d.common.replay_buffer import ReplayBuffer

# Order matters; this defines how the low-dimensional state vector is constructed.
STATE_FIELDS: List[Tuple[str, ...]] = [
    ("robot_state", "joint_positions"),
    ("robot_state", "joint_velocities"),
    ("robot_state", "joint_torques"),
    ("robot_state", "ee_pos"),
    ("robot_state", "ee_quat"),
    ("robot_state", "ee_pos_vel"),
    ("robot_state", "ee_ori_vel"),
    ("robot_state", "gripper_width"),
]


def infer_length(obs_block: Any) -> int:
    """Infer sequence length from observations."""
    if isinstance(obs_block, (list, tuple)):
        return len(obs_block)
    if isinstance(obs_block, dict):
        for value in obs_block.values():
            if isinstance(value, (list, tuple, np.ndarray)):
                return len(value)
            if isinstance(value, dict):
                nested = infer_length(value)
                if nested is not None:
                    return nested
    raise ValueError("Unable to infer observation length")


def resolve_path(obj: Any, path: Sequence[str]) -> Any:
    cur = obj
    for key in path:
        cur = cur[key]
    return cur


def field_exists(obs_block: Any, path: Sequence[str]) -> bool:
    try:
        if isinstance(obs_block, (list, tuple)):
            _ = resolve_path(obs_block[0], path)
        else:
            _ = resolve_path(obs_block, path)
        return True
    except Exception:
        return False


def stack_field(obs_block: Any, path: Sequence[str], length: int) -> np.ndarray:
    if isinstance(obs_block, (list, tuple)):
        arrays = []
        for step in obs_block:
            arr = np.asarray(resolve_path(step, path))
            arrays.append(arr)
        
        # Check if this is point_cloud field and handle shape inconsistencies
        if len(path) == 1 and path[0] == "point_cloud":
            # Find the maximum number of points to pad to
            max_points = max(arr.shape[0] for arr in arrays)
            padded_arrays = []
            
            for arr in arrays:
                if arr.shape[0] < max_points:
                    # Pad with zeros to match max_points
                    pad_shape = (max_points - arr.shape[0],) + arr.shape[1:]
                    padding = np.zeros(pad_shape, dtype=arr.dtype)
                    arr_padded = np.concatenate([arr, padding], axis=0)
                    padded_arrays.append(arr_padded)
                else:
                    padded_arrays.append(arr)
            arrays = padded_arrays
        
        arr = np.stack(arrays, axis=0)
    elif isinstance(obs_block, dict):
        arr = np.asarray(resolve_path(obs_block, path))
    else:
        raise TypeError(f"Unsupported observation container type: {type(obs_block)}")

    if arr.shape[0] != length:
        raise ValueError(f"Field {'/'.join(path)} has length {arr.shape[0]} but expected {length}")
    return arr


def pad_to_obs_length(arr: Any, obs_len: int, name: str) -> np.ndarray:
    if arr is None:
        return None
    arr = np.asarray(arr)
    if arr.shape[0] == obs_len:
        return arr
    if arr.shape[0] == obs_len - 1:
        pad = arr[-1:]
        return np.concatenate([arr, pad], axis=0)
    raise ValueError(f"{name} length {arr.shape[0]} is incompatible with observations {obs_len}")


def build_state(obs_block: Any, obs_len: int) -> np.ndarray:
    parts = []
    for path in STATE_FIELDS:
        arr = stack_field(obs_block, path, obs_len)
        parts.append(arr.reshape(obs_len, -1))
    return np.concatenate(parts, axis=1)


def load_episode(pkl_path: Path) -> Tuple[dict, dict]:
    with open(pkl_path, "rb") as f:
        raw = pickle.load(f)

    if "observations" not in raw:
        raise KeyError(f"observations missing in {pkl_path}")
    if "actions" not in raw:
        raise KeyError(f"actions missing in {pkl_path}")

    obs_block = raw["observations"]
    obs_len = infer_length(obs_block)

    state = build_state(obs_block, obs_len).astype(np.float32)
    point_cloud = stack_field(obs_block, ("point_cloud",), obs_len).astype(np.float32)
    actions = pad_to_obs_length(raw["actions"], obs_len, "actions").astype(np.float32)

    rewards = None
    if "rewards" in raw and raw["rewards"] is not None:
        rewards = pad_to_obs_length(raw["rewards"], obs_len, "rewards")
        rewards = rewards.astype(np.float32)

    # Skip color images to save space and processing time
    # color_image1 = None
    # color_image2 = None
    # if field_exists(obs_block, ("color_image1",)):
    #     color_image1 = stack_field(obs_block, ("color_image1",), obs_len)
    # if field_exists(obs_block, ("color_image2",)):
    #     color_image2 = stack_field(obs_block, ("color_image2",), obs_len)

    episode = {
        "state": state,
        "action": actions,
        "point_cloud": point_cloud,
    }
    if rewards is not None:
        episode["reward"] = rewards
    # Skip color images
    # if color_image1 is not None:
    #     episode["color_image1"] = color_image1
    # if color_image2 is not None:
    #     episode["color_image2"] = color_image2

    meta = {
        "success": raw.get("success"),
        "task": raw.get("task"),
        "action_type": raw.get("action_type"),
        "obs_len": obs_len,
        "state_dim": state.shape[1],
        "action_dim": actions.shape[1],
    }
    return episode, meta


def collect_pickles(source_dir: Path, data_date_since: str = None) -> List[Path]:
    exts = {".pkl", ".pickle"}
    paths_with_dates: List[Tuple[Path, datetime]] = []
    
    # Parse cut-off date if provided
    since_date = None
    if data_date_since:
        try:
            since_date = datetime.fromisoformat(data_date_since)
        except ValueError:
            # Fallback for simple date format if fromisoformat is too strict in some python versions
            # or if input is slightly off standard.
            # But fromisoformat usually handles "YYYY-MM-DD" correctly.
            # Let's try appending time if it looks like a date only
            if len(data_date_since) == 10: # YYYY-MM-DD
                 try:
                     since_date = datetime.fromisoformat(data_date_since + "T00:00:00")
                 except ValueError:
                     pass
            
            if since_date is None:
                print(f"Warning: Could not parse data_date_since '{data_date_since}', ignoring filter.")

    for path in source_dir.rglob("*"):
        if path.suffix.lower() in exts:
            # Try to extract date from filename
            # Filename format: 2026-01-10T23:51:56.685824.pkl
            try:
                dt = datetime.fromisoformat(path.stem)
                paths_with_dates.append((path, dt))
            except ValueError:
                # If filename is not a timestamp, we might still want to include it if no filter is set
                # But if we want to sort by date, better to use mtime as fallback or put at end?
                # For now, let's use file mtime as fallback for sorting
                mtime = datetime.fromtimestamp(path.stat().st_mtime)
                paths_with_dates.append((path, mtime))

    # Filter by date if requested
    if since_date:
        paths_with_dates = [
            (p, d) for p, d in paths_with_dates 
            if d >= since_date
        ]

    # Sort by date descending (newest first)
    paths_with_dates.sort(key=lambda x: x[1], reverse=True)
    
    return [p for p, d in paths_with_dates]


def main(args: argparse.Namespace) -> None:
    source_dir = Path(args.source_dir).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    pickle_paths = collect_pickles(source_dir, args.data_date_since)
    if not pickle_paths:
        raise FileNotFoundError(f"No pickle files found under {source_dir}")

    # Limit number of rollouts if specified
    if args.max_rollouts > 0:
        pickle_paths = pickle_paths[:args.max_rollouts]
        print(f"Processing first {len(pickle_paths)} rollouts (sorted newest first, limited by max_rollouts={args.max_rollouts})")

    store = zarr.DirectoryStore(str(output_path))
    buffer = ReplayBuffer.create_empty_zarr(storage=store)

    meta_success: List[bool] = []
    meta_task: List[str] = []
    meta_action_type: List[str] = []

    for idx, pkl_path in enumerate(pickle_paths):
        episode, meta = load_episode(pkl_path)
        buffer.add_episode(episode)

        if meta["success"] is not None:
            meta_success.append(bool(meta["success"]))
        if meta["task"] is not None:
            meta_task.append(str(meta["task"]))
        if meta["action_type"] is not None:
            meta_action_type.append(str(meta["action_type"]))

        print(
            f"[{idx+1}/{len(pickle_paths)}] {pkl_path.name}: obs={meta['obs_len']} "
            f"state_dim={meta['state_dim']} action_dim={meta['action_dim']}"
        )

    meta_payload = {}
    if meta_success:
        meta_payload["success"] = np.array(meta_success, dtype=np.bool_)
    if meta_task:
        max_len = max(len(x) for x in meta_task)
        meta_payload["task"] = np.array(meta_task, dtype=f"<U{max_len}")
    if meta_action_type:
        max_len = max(len(x) for x in meta_action_type)
        meta_payload["action_type"] = np.array(meta_action_type, dtype=f"<U{max_len}")
    if meta_payload:
        buffer.update_meta(meta_payload)

    print(f"Wrote dataset with {buffer.n_episodes} episodes to {output_path}")


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a DP3 zarr dataset from pickles")
    parser.add_argument("--source-dir", type=str, required=True, help="Directory containing episode pickles")
    parser.add_argument("--output", type=str, required=True, help="Output zarr directory")
    parser.add_argument("--max-rollouts", type=int, default=50, help="Maximum number of rollouts to process (default: 50, 0 for unlimited)")
    parser.add_argument("--data-date-since", type=str, default=None, help="Filter files modified/created since this date (ISO format), e.g. 2026-01-10T12:00:00")
    return parser.parse_args(list(argv))


if __name__ == "__main__":
    main(parse_args(sys.argv[1:]))
