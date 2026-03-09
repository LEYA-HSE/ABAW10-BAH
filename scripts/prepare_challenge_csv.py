# coding: utf-8
from __future__ import annotations

import csv
from pathlib import Path


DEFAULT_INPUT_TXT = Path(
    r"E:\BAH\videos_normal\test-set-30-participants-unlabeled\data\split\test.txt"
)
DEFAULT_OUTPUT_CSV = Path(r"c:\Prgrm\ABAW26\data\challenge\test.csv")


def _parse_line(line: str) -> tuple[str, int, str]:
    raw = line.rstrip("\n")
    parts = raw.split(",", 2)
    if len(parts) < 2:
        raise ValueError(f"Bad line (expected at least 2 commas): {raw[:200]}")

    video_path = parts[0].strip()
    label = int(parts[1].strip())
    transcript = parts[2] if len(parts) == 3 else ""
    return video_path, label, transcript


def prepare_challenge_csv(
    input_txt: Path = DEFAULT_INPUT_TXT,
    output_csv: Path = DEFAULT_OUTPUT_CSV,
) -> Path:
    input_txt = Path(input_txt)
    output_csv = Path(output_csv)
    if not input_txt.exists():
        raise FileNotFoundError(f"Challenge split not found: {input_txt}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    rows: list[tuple[str, str, int, str, str]] = []
    with input_txt.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip():
                continue
            video_path, label, transcript = _parse_line(line)
            video_name = Path(video_path).stem
            rows.append((video_path, video_name, label, transcript, transcript))

    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["video_path", "video_name", "label", "transcript", "text"])
        writer.writerows(rows)

    return output_csv


if __name__ == "__main__":
    out = prepare_challenge_csv()
    print(f"Saved: {out}")
