from __future__ import annotations

from collections import defaultdict


PROFILE_TAGS = [
    "SETUP",
    "ISSUE_TMA",
    "ISSUE_MMA",
    "WAIT_TMA",
    "WAIT_MMA",
    "WAIT_MAINLOOP",
    "WAIT_EPILOGUE",
    "EPILOGUE",
]

PROFILE_TAG_CATEGORIES = {
    "SETUP": "setup",
    "ISSUE_TMA": "issue",
    "ISSUE_MMA": "issue",
    "WAIT_TMA": "wait",
    "WAIT_MMA": "wait",
    "WAIT_MAINLOOP": "wait",
    "WAIT_EPILOGUE": "wait",
    "EPILOGUE": "epilogue",
}


def _unpack_field(value: int, shift: int, bits: int) -> int:
    mask = (1 << bits) - 1
    field = (value >> shift) & mask
    return -1 if field == mask else field


def decode_profile_record(record: list[int] | tuple[int, int, int, int]) -> dict[str, int | str]:
    word0, word1, start, duration = record
    tag = _unpack_field(word0, 32, 8)
    tag_name = PROFILE_TAGS[tag] if 0 <= tag < len(PROFILE_TAGS) else f"TAG_{tag}"
    warp_id = _unpack_field(word0, 16, 8)
    role = warp_role(warp_id)
    return {
        "sm_id": _unpack_field(word0, 0, 16),
        "warp_id": warp_id,
        "cta_rank": _unpack_field(word0, 24, 8),
        "tag": tag,
        "tag_name": tag_name,
        "stage": _unpack_field(word0, 40, 8),
        "phase": _unpack_field(word0, 48, 8),
        "bid": _unpack_field(word1, 0, 16),
        "bid_m": _unpack_field(word1, 16, 16),
        "bid_n": _unpack_field(word1, 32, 16),
        "iter_k": _unpack_field(word1, 48, 16),
        "start": start,
        "duration": duration,
        "role": role,
    }


def warp_role(warp_id: int) -> str:
    if warp_id == 0:
        return "TMA"
    if warp_id == 1:
        return "MMA"
    if warp_id >= 2:
        return f"EPI{warp_id - 2}"
    return "UNKNOWN"


def _track_name(decoded: dict[str, int | str], row_id: int) -> str:
    role = decoded["role"]
    warp_id = decoded["warp_id"]
    cta_rank = decoded["cta_rank"]
    parts = [f"row={row_id}", f"role={role}"]
    if warp_id >= 0:
        parts.append(f"warp={warp_id}")
    if cta_rank >= 0:
        parts.append(f"rank={cta_rank}")
    return " | ".join(parts)


def build_trace_events(profile_data: list[list[int]]) -> tuple[list[dict], list[dict]]:
    decoded_events: list[dict] = []
    track_names: dict[tuple[int, int], str] = {}
    process_names: dict[int, str] = {}

    for row_id, data in enumerate(profile_data):
        count = data[0]
        for index in range(count):
            decoded = decode_profile_record(data[1 + index * 4 : 1 + (index + 1) * 4])
            sm_id = decoded["sm_id"]
            tid = row_id
            decoded["tid"] = tid
            decoded_events.append(decoded)
            track_names.setdefault((sm_id, tid), _track_name(decoded, row_id))
            process_names.setdefault(sm_id, f"SM {sm_id}")

    if not decoded_events:
        return [], []

    offset = min(evt["start"] for evt in decoded_events)
    trace_events = []
    for sm_id, process_name in sorted(process_names.items()):
        trace_events.append(
            {
                "name": "process_name",
                "ph": "M",
                "pid": sm_id,
                "tid": 0,
                "args": {"name": process_name},
            }
        )

    for (sm_id, tid), track_name in sorted(track_names.items()):
        trace_events.append(
            {
                "name": "thread_name",
                "ph": "M",
                "pid": sm_id,
                "tid": tid,
                "args": {"name": track_name},
            }
        )

    for decoded in decoded_events:
        args = {
            "role": decoded["role"],
            "warp_id": decoded["warp_id"],
            "cta_rank": decoded["cta_rank"],
            "block_idx": decoded["bid"],
            "tile_m": decoded["bid_m"],
            "tile_n": decoded["bid_n"],
            "iter_k": decoded["iter_k"],
            "stage": decoded["stage"],
            "phase": decoded["phase"],
        }
        trace_events.append(
            {
                "name": decoded["tag_name"],
                "cat": PROFILE_TAG_CATEGORIES.get(decoded["tag_name"], "other"),
                "ph": "X",
                "ts": decoded["start"] - offset,
                "dur": decoded["duration"],
                "pid": decoded["sm_id"],
                "tid": decoded["tid"],
                "args": args,
            }
        )

    return trace_events, decoded_events


def summarize_profile(decoded_events: list[dict], top_n: int = 12) -> dict:
    role_total = defaultdict(int)
    role_wait = defaultdict(int)
    tag_total = defaultdict(int)
    top_waits = []

    for event in decoded_events:
        role = event["role"]
        tag_name = event["tag_name"]
        duration = event["duration"]
        role_total[role] += duration
        tag_total[tag_name] += duration
        if tag_name.startswith("WAIT_"):
            role_wait[role] += duration
            top_waits.append(event)

    top_waits.sort(key=lambda event: event["duration"], reverse=True)

    return {
        "event_count": len(decoded_events),
        "tag_total": dict(sorted(tag_total.items())),
        "role_total": dict(sorted(role_total.items())),
        "role_wait": dict(sorted(role_wait.items())),
        "top_waits": top_waits[:top_n],
    }


def format_profile_summary(summary: dict) -> str:
    lines = [f"[PROFILE] decoded {summary['event_count']} events"]

    if summary["tag_total"]:
        tag_parts = [f"{name}={cycles}" for name, cycles in summary["tag_total"].items()]
        lines.append("[PROFILE] tag cycles: " + ", ".join(tag_parts))

    if summary["role_total"]:
        role_parts = []
        for role, total_cycles in summary["role_total"].items():
            wait_cycles = summary["role_wait"].get(role, 0)
            idle_pct = 100.0 * wait_cycles / total_cycles if total_cycles else 0.0
            role_parts.append(f"{role}: idle={idle_pct:.1f}% ({wait_cycles}/{total_cycles})")
        lines.append("[PROFILE] role idle: " + ", ".join(role_parts))

    if summary["top_waits"]:
        lines.append("[PROFILE] top bubbles:")
        for event in summary["top_waits"]:
            lines.append(
                "  "
                f"{event['tag_name']} dur={event['duration']} role={event['role']} "
                f"block={event['bid']} tile=({event['bid_m']},{event['bid_n']}) "
                f"iter_k={event['iter_k']} stage={event['stage']} phase={event['phase']}"
            )

    return "\n".join(lines)