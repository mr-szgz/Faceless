from pathlib import Path
import subprocess
from fractions import Fraction

from faceless.config import DEPENDENCIES_DIR
import os
env = {**os.environ, "PYTHONNOUSERSITE": "1", "PYTHONPATH": ""}

def run_ffprobe_dump(source_files: list[Path]):
    from tqdm import tqdm
    dependencies = Path(DEPENDENCIES_DIR).expanduser().resolve()
    source_path = source_files[0].parent if source_files else Path.cwd()
    output_dir = source_path / "ffprobes"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    progress = tqdm(total=len(source_files), desc="Probing media files", unit="file", mininterval=0.5, miniters=32)

    try:
        for item in source_files:
            output_path = output_dir / f"{item.stem}.txt"
            with output_path.open("wb") as handle:
                subprocess.check_call(
                    [
                        str(dependencies / "ffprobe.exe"),
                        "-v",
                        "error",
                        "-probesize",
                        "32k",
                        "-analyzeduration",
                        "0",
                        "-select_streams",
                        "v:0",
                        "-show_entries",
                        "stream=avg_frame_rate,r_frame_rate,nb_frames:format=duration",
                        "-of",
                        "default=noprint_wrappers=1:nokey=0",
                        str(item),
                    ],
                    env=env,
                    stdout=handle,
                )
            progress.update(1)
    finally:
        progress.close()
            
    return output_dir

def get_probe_for_source_file(source: Path):
    source = source.expanduser().resolve()
    return source / "ffprobes" / f"{source.stem}.txt"

def parse_probe_file(probe_file: Path) -> dict[str, str]:
    properties: dict[str, str] = {}

    with probe_file.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            key, separator, value = line.strip().partition("=")
            if not separator:
                continue

            properties[key] = value

    return properties

# MATHS

def get_first_valid_fraction(properties: dict[str, str], *keys: str) -> Fraction | None:
    for key in keys:
        value = properties.get(key)
        if value in (None, "", "N/A"):
            continue

        try:
            fraction = Fraction(value)
        except (ValueError, ZeroDivisionError):
            continue

        if fraction > 0:
            return fraction

    return None

# CALCULATIONS

def calc_median_fps(source: Path):
    source = source.expanduser().resolve()
    probes_dir = source / "ffprobes"
    output_path = source / "median_framerate.txt"
    frame_rates: list[Fraction] = []

    with os.scandir(probes_dir) as entries:
        for entry in entries:
            if not entry.is_file() or not entry.name.endswith(".txt"):
                continue

            properties = parse_probe_file(Path(entry.path))
            frame_rate = get_first_valid_fraction(properties, "r_frame_rate", "avg_frame_rate")
            if frame_rate is None:
                continue

            frame_rates.append(frame_rate)

    if not frame_rates:
        raise FileNotFoundError(f"No valid r_frame_rate entries found in {probes_dir}")

    frame_rates.sort()
    middle = len(frame_rates) // 2
    if len(frame_rates) % 2:
        median_rate = frame_rates[middle]
    else:
        median_rate = (frame_rates[middle - 1] + frame_rates[middle]) / 2

    output_path.write_text(
        f"r_frame_rate={float(median_rate)}\n",
        encoding="utf-8",
    )
    return output_path

def calc_median_frames(source: Path):
    source = source.expanduser().resolve()
    probes_dir = source / "ffprobes"
    output_path = source / "median_frames.txt"
    frame_counts: list[Fraction] = []

    with os.scandir(probes_dir) as entries:
        for entry in entries:
            if not entry.is_file() or not entry.name.endswith(".txt"):
                continue

            properties = parse_probe_file(Path(entry.path))
            frame_count = properties.get("nb_frames")
            if frame_count not in (None, "", "N/A"):
                try:
                    frame_counts.append(Fraction(frame_count))
                    continue
                except (ValueError, ZeroDivisionError):
                    pass

            try:
                duration = Fraction(properties["duration"])
                frame_rate = get_first_valid_fraction(properties, "avg_frame_rate", "r_frame_rate")
                if frame_rate is not None and duration > 0:
                    frame_counts.append(duration * frame_rate)
            except (KeyError, ValueError, ZeroDivisionError):
                pass

    if not frame_counts:
        raise FileNotFoundError(f"No valid frame count metadata found in {probes_dir}")

    frame_counts.sort()
    middle = len(frame_counts) // 2
    if len(frame_counts) % 2:
        median_frames = Fraction(frame_counts[middle], 1)
    else:
        median_frames = Fraction(frame_counts[middle - 1] + frame_counts[middle], 2)

    output_path.write_text(
        f"nb_read_frames={float(median_frames)}\n",
        encoding="utf-8",
    )
    return output_path
