#!/usr/bin/env python3
"""Convert Prosperity replay JSON into easier-to-read files.

Prosperity replay files often store large logs as semicolon-delimited strings
inside JSON fields such as `activitiesLog` and `graphLog`. This script expands
those embedded tables into normal CSV files and also produces a short Markdown
summary so a run is easier to inspect.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent

# Edit these defaults if you want to run the script without command-line arguments.
DEFAULT_INPUT_FILE = SCRIPT_DIR / "24345" / "24345.json"
DEFAULT_OUTPUT_DIR: Path | None = None
ACTION_PATTERN = re.compile(r"^(BUY|SELL)\s+(\d+)x\s+(-?\d+(?:\.\d+)?)$")


def parse_args() -> argparse.Namespace:
    """Read command-line overrides for the input file and output directory."""
    parser = argparse.ArgumentParser(
        description=(
            "Expand a Prosperity replay JSON file into readable CSV tables "
            "and a Markdown summary."
        )
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        help=(
            "Path to the replay JSON file. If omitted, the script uses "
            "DEFAULT_INPUT_FILE."
        ),
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        help="Directory for converted files. Defaults to <input_stem>_readable",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    """Load the replay payload from disk."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def is_embedded_table(value: Any) -> bool:
    """Detect Prosperity log fields stored as multiline `;`-delimited text."""
    if not isinstance(value, str):
        return False
    lines = [line for line in value.splitlines() if line.strip()]
    return len(lines) >= 2 and ";" in lines[0]


def parse_embedded_table(value: str) -> list[dict[str, str]]:
    """Parse a Prosperity log string into row dictionaries."""
    lines = [line for line in value.splitlines() if line.strip()]
    reader = csv.DictReader(lines, delimiter=";")
    return [dict(row) for row in reader]


def find_log_payload(
    input_path: Path, payload: dict[str, Any]
) -> tuple[dict[str, Any] | None, Path | None]:
    """Locate the replay log payload used to extract strategy BUY/SELL actions."""
    if isinstance(payload.get("logs"), list):
        return payload, input_path

    candidate = input_path.with_suffix(".log")
    if candidate.exists():
        return load_json(candidate), candidate

    return None, None


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    """Write a list of dictionaries to CSV while preserving column discovery order."""
    if not rows:
        return

    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def maybe_number(value: Any) -> float | int | None:
    """Best-effort conversion of replay fields into numeric values."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return value

    text = str(value).strip()
    if not text:
        return None

    try:
        if any(ch in text for ch in ".eE"):
            return float(text)
        return int(text)
    except ValueError:
        return None


def fmt_number(value: float | int | None, digits: int = 2) -> str:
    """Format numbers for human-readable Markdown output."""
    if value is None:
        return "-"
    if isinstance(value, int):
        return f"{value:,}"
    return f"{value:,.{digits}f}"


def fmt_side(row: dict[str, str], side: str) -> str:
    """Render up to three bid or ask levels from one activity row."""
    levels = []
    for level in range(1, 4):
        price = row.get(f"{side}_price_{level}", "")
        volume = row.get(f"{side}_volume_{level}", "")
        if price:
            levels.append(f"{price} x {volume}")
    return ", ".join(levels) if levels else "-"


def normalize_price(value: float | int | None) -> float | int | None:
    """Drop trailing `.0` for prices that are effectively integers."""
    if value is None:
        return None
    if isinstance(value, int):
        return value
    return int(value) if value.is_integer() else value


def build_activity_price_index(
    rows: list[dict[str, str]],
) -> dict[int, list[tuple[str, set[float]]]]:
    """Index visible order-book prices by timestamp so actions can be labeled."""
    index: dict[int, list[tuple[str, set[float]]]] = defaultdict(list)
    for row in rows:
        timestamp = maybe_number(row.get("timestamp"))
        product = row.get("product", "")
        if not isinstance(timestamp, (int, float)) or not product:
            continue

        visible_prices: set[float] = set()
        for side in ("bid", "ask"):
            for level in range(1, 4):
                price = maybe_number(row.get(f"{side}_price_{level}"))
                if isinstance(price, (int, float)):
                    visible_prices.add(float(price))

        index[int(timestamp)].append((product, visible_prices))

    return index


def infer_product_for_action(
    timestamp: int,
    price: float | int | None,
    price_index: dict[int, list[tuple[str, set[float]]]],
) -> str:
    """Match an action price against the visible book for that timestamp."""
    if price is None:
        return ""

    candidates = [
        product
        for product, visible_prices in price_index.get(timestamp, [])
        if float(price) in visible_prices
    ]
    unique_candidates = sorted(set(candidates))
    if len(unique_candidates) == 1:
        return unique_candidates[0]
    return ""


def extract_user_actions(
    activity_rows: list[dict[str, str]],
    log_payload: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Parse strategy BUY/SELL lines from the replay log into tabular rows."""
    if not log_payload or not isinstance(log_payload.get("logs"), list):
        return []

    price_index = build_activity_price_index(activity_rows)
    actions: list[dict[str, Any]] = []

    for log_entry in log_payload["logs"]:
        timestamp = maybe_number(log_entry.get("timestamp"))
        if not isinstance(timestamp, (int, float)):
            continue

        for line in str(log_entry.get("lambdaLog", "")).splitlines():
            match = ACTION_PATTERN.match(line.strip())
            if not match:
                continue

            action, quantity_text, price_text = match.groups()
            price = maybe_number(price_text)
            normalized_timestamp = int(timestamp)
            product = infer_product_for_action(
                normalized_timestamp,
                price,
                price_index,
            )
            actions.append(
                {
                    "timestamp": normalized_timestamp,
                    "product": product,
                    "action": action,
                    "quantity": int(quantity_text),
                    "price": normalize_price(price),
                    "product_matched": "yes" if product else "no",
                    "raw_line": line.strip(),
                }
            )

    return actions


def summarize_positions(positions: list[dict[str, Any]]) -> list[str]:
    """Create a small Markdown table for the final portfolio positions."""
    if not positions:
        return ["No position data present."]

    lines = ["| Symbol | Quantity |", "| --- | ---: |"]
    for position in positions:
        lines.append(
            f"| {position.get('symbol', '-')} | {position.get('quantity', '-')} |"
        )
    return lines


def summarize_graph(rows: list[dict[str, str]]) -> list[str]:
    """Summarize portfolio PnL over time from the replay graph log."""
    if not rows:
        return ["No graph data present."]

    points: list[tuple[int, float]] = []
    for row in rows:
        timestamp = maybe_number(row.get("timestamp"))
        value = maybe_number(row.get("value"))
        if isinstance(timestamp, (int, float)) and isinstance(value, (int, float)):
            points.append((int(timestamp), float(value)))

    if not points:
        return ["Graph data could not be parsed."]

    min_point = min(points, key=lambda item: item[1])
    max_point = max(points, key=lambda item: item[1])
    # Sampling a few checkpoints keeps the summary compact while still showing
    # the rough shape of the run.
    sample_count = min(10, len(points))
    step = max(1, len(points) // sample_count)
    checkpoint_rows = points[::step][:sample_count]
    if checkpoint_rows[-1] != points[-1]:
        checkpoint_rows[-1] = points[-1]

    lines = [
        f"- Start: `{points[0][0]}` -> `{fmt_number(points[0][1])}`",
        f"- End: `{points[-1][0]}` -> `{fmt_number(points[-1][1])}`",
        f"- Lowest: `{min_point[0]}` -> `{fmt_number(min_point[1])}`",
        f"- Highest: `{max_point[0]}` -> `{fmt_number(max_point[1])}`",
        "",
        "| Timestamp | PnL |",
        "| ---: | ---: |",
    ]
    for timestamp, value in checkpoint_rows:
        lines.append(f"| {timestamp} | {fmt_number(value)} |")
    return lines


def summarize_activities(rows: list[dict[str, str]]) -> list[str]:
    """Group activity snapshots by product and extract simple market statistics."""
    if not rows:
        return ["No activity data present."]

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row.get("product", "UNKNOWN")].append(row)

    lines: list[str] = []
    for product in sorted(grouped):
        product_rows = grouped[product]
        # Mid prices and spreads are convenient first-pass diagnostics when you
        # want to understand whether a strategy was fighting or following the
        # market.
        mid_prices = [
            float(value)
            for value in (row.get("mid_price") for row in product_rows)
            if value not in (None, "")
        ]
        spreads = []
        for row in product_rows:
            bid = maybe_number(row.get("bid_price_1"))
            ask = maybe_number(row.get("ask_price_1"))
            if isinstance(bid, (int, float)) and isinstance(ask, (int, float)):
                spreads.append(float(ask) - float(bid))

        start_row = product_rows[0]
        end_row = product_rows[-1]
        final_pnl = maybe_number(end_row.get("profit_and_loss"))

        lines.extend(
            [
                f"### {product}",
                f"- Snapshots: `{len(product_rows):,}`",
                (
                    f"- Timestamp range: `{start_row.get('timestamp', '-')}` -> "
                    f"`{end_row.get('timestamp', '-')}`"
                ),
                (
                    f"- Mid price: start `{start_row.get('mid_price', '-')}`, "
                    f"end `{end_row.get('mid_price', '-')}`, "
                    f"min `{fmt_number(min(mid_prices) if mid_prices else None)}`, "
                    f"max `{fmt_number(max(mid_prices) if mid_prices else None)}`, "
                    f"avg `{fmt_number(mean(mid_prices) if mid_prices else None)}`"
                ),
                f"- Average spread: `{fmt_number(mean(spreads) if spreads else None)}`",
                f"- Final per-product PnL: `{fmt_number(final_pnl)}`",
                (
                    f"- Opening book: bids `{fmt_side(start_row, 'bid')}` | "
                    f"asks `{fmt_side(start_row, 'ask')}`"
                ),
                (
                    f"- Closing book: bids `{fmt_side(end_row, 'bid')}` | "
                    f"asks `{fmt_side(end_row, 'ask')}`"
                ),
                "",
            ]
        )

    return lines


def build_summary(
    source_path: Path,
    output_dir: Path,
    payload: dict[str, Any],
    extracted_tables: dict[str, list[dict[str, Any]]],
) -> str:
    """Assemble the human-readable Markdown report for one replay file."""
    activity_rows = extracted_tables.get("activitiesLog", [])
    graph_rows = extracted_tables.get("graphLog", [])
    positions = payload.get("positions", [])
    user_actions = extracted_tables.get("user_actions", [])

    lines = [
        "# Replay Summary",
        "",
        f"- Source: `{source_path}`",
        f"- Output directory: `{output_dir}`",
        f"- Round: `{payload.get('round', '-')}`",
        f"- Status: `{payload.get('status', '-')}`",
        f"- Reported profit: `{fmt_number(maybe_number(payload.get('profit')))}`",
        f"- Activity rows: `{len(activity_rows):,}`",
        f"- Graph points: `{len(graph_rows):,}`",
        f"- User actions: `{len(user_actions):,}`",
        "",
        "## Final Positions",
        "",
        *summarize_positions(positions if isinstance(positions, list) else []),
        "",
        "## Portfolio PnL",
        "",
        *summarize_graph(graph_rows),
        "",
        "## Products",
        "",
        *summarize_activities(activity_rows),
        "## Extracted Files",
        "",
    ]

    for name, rows in sorted(extracted_tables.items()):
        lines.append(f"- `{name}.csv` with `{len(rows):,}` rows")

    return "\n".join(lines) + "\n"


def main() -> int:
    """Convert one replay JSON into CSV tables plus a Markdown summary."""
    args = parse_args()
    input_candidate = args.input if args.input is not None else DEFAULT_INPUT_FILE
    input_path = input_candidate.expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Replay file not found: {input_path}")

    output_candidate = args.output_dir if args.output_dir is not None else DEFAULT_OUTPUT_DIR
    output_dir = (
        output_candidate.expanduser().resolve()
        if output_candidate is not None
        else input_path.parent / f"{input_path.stem}_readable"
    )

    payload = load_json(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    extracted_tables: dict[str, list[dict[str, Any]]] = {}

    for key, value in payload.items():
        # Replay files mix metadata with large embedded tables. Each supported
        # table gets expanded into its own CSV for easier filtering and analysis.
        if is_embedded_table(value):
            rows = parse_embedded_table(value)
            extracted_tables[key] = rows
            write_csv(rows, output_dir / f"{key}.csv")
        elif isinstance(value, list) and value and all(
            isinstance(item, dict) for item in value
        ):
            rows = [dict(item) for item in value]
            extracted_tables[key] = rows
            write_csv(rows, output_dir / f"{key}.csv")

    log_payload, _ = find_log_payload(input_path, payload)
    user_actions = extract_user_actions(
        extracted_tables.get("activitiesLog", []),
        log_payload,
    )
    if user_actions:
        extracted_tables["user_actions"] = user_actions
        write_csv(user_actions, output_dir / "user_actions.csv")

    summary = build_summary(input_path, output_dir, payload, extracted_tables)
    summary_path = output_dir / "summary.md"
    summary_path.write_text(summary, encoding="utf-8")

    pretty_json_path = output_dir / "metadata.json"
    pretty_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Wrote readable files to {output_dir}")
    print(f"- {summary_path.name}")
    for key in sorted(extracted_tables):
        print(f"- {key}.csv")
    print(f"- {pretty_json_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
