"""Download UCI 235 atomically; aggregate complete observed hours without inventing labels."""
import argparse
import hashlib
import json
import shutil
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from config import DATA
from src.utils import write_json

URL = "https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip"
SOURCE = "https://doi.org/10.24432/C58K54"
SCHEMA = "complete-hours-v2"


def download_dataset(directory=DATA / "raw"):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "household_power_consumption.txt"
    if target.exists() and target.stat().st_size > 1_000_000:
        return target
    archive = directory / "uci235.zip"
    if not archive.exists():
        partial = archive.with_suffix(".partial")
        for attempt in range(3):
            try:
                request = urllib.request.Request(URL, headers={"User-Agent": "OptiEnergy-AI/1.0 educational research"})
                with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as out:
                    shutil.copyfileobj(response, out)
                with zipfile.ZipFile(partial) as z:
                    if z.testzip():
                        raise ValueError("UCI archive failed its CRC integrity check")
                partial.replace(archive)
                break
            except (OSError, ValueError, zipfile.BadZipFile):
                partial.unlink(missing_ok=True)
                if attempt == 2:
                    raise RuntimeError("Cannot download UCI data. Download the ZIP manually into data/raw/uci235.zip.")
                time.sleep(attempt + 1)
    with zipfile.ZipFile(archive) as z:
        # Read only the named member; never extract arbitrary archive paths.
        member = next((n for n in z.namelist() if Path(n).name == target.name), None)
        if member is None:
            raise ValueError("Dataset file is missing from archive")
        with z.open(member) as source, target.with_suffix(".partial").open("wb") as out:
            shutil.copyfileobj(source, out)
    target.with_suffix(".partial").replace(target)
    return target


def aggregate_minutes(frame):
    """An index at 14:00 represents energy observed in [13:00, 14:00)."""
    frame = frame.copy()
    timestamps = pd.to_datetime(frame["Date"].astype(str) + " " + frame["Time"].astype(str),
                                format="%d/%m/%Y %H:%M:%S", errors="coerce")
    power = pd.to_numeric(frame["Global_active_power"], errors="coerce")
    power = power.where(np.isfinite(power) & (power >= 0))
    series = pd.Series(power.to_numpy(), index=timestamps).loc[timestamps.notna().to_numpy()]
    duplicates = int(series.index.duplicated().sum())
    series = series[~series.index.duplicated(keep="first")].sort_index()
    series = series.asfreq("min")
    groups = series.resample("h", closed="left", label="right")
    counts = groups.count()
    energy = (groups.sum(min_count=60) / 60.0).where(counts == 60)
    hourly = pd.DataFrame({"energy_kwh": energy, "observed_minutes": counts})
    hourly.index.name = "origin"
    stats = {"source": SOURCE, "license": "CC BY 4.0; Hebrail & Berard",
             "raw_rows": len(frame), "duplicate_timestamps": duplicates,
             "missing_or_invalid_minutes": int(series.isna().sum()),
             "missing_minute_fraction": float(series.isna().mean()),
             "hours": len(hourly), "complete_hours": int(energy.notna().sum()),
             "incomplete_hours": int(energy.isna().sum()), "schema": SCHEMA,
             "timestamp_convention": "naive local clock; right endpoint of observed hourly interval"}
    return hourly, stats


def load_hourly(data_dir=DATA, force=False):
    data_dir = Path(data_dir)
    cache, metadata = data_dir / "processed/hourly.csv", data_dir / "processed/metadata.json"
    if cache.exists() and metadata.exists() and not force:
        stats = json.loads(metadata.read_text(encoding="utf-8"))
        if stats.get("schema") == SCHEMA:
            return pd.read_csv(cache, index_col=0, parse_dates=True), stats
    path = download_dataset(data_dir / "raw")
    frame = pd.read_csv(path, sep=";", na_values=["?", ""],
                        usecols=["Date", "Time", "Global_active_power"], low_memory=False)
    hourly, stats = aggregate_minutes(frame)
    stats["raw_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    cache.parent.mkdir(parents=True, exist_ok=True)
    hourly.to_csv(cache)
    write_json(metadata, stats)
    return hourly, stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    _, metadata = load_hourly(force=args.force)
    print(json.dumps(metadata, indent=2))
