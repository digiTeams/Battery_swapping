"""Shared result-file naming: {policy}_{stream}[_plus{n}]_{YYYY-MM-DD}_{type}.csv
and {policy}_train_{YYYY-MM-DD}.pt checkpoints; readers pick the newest family."""

import os
import re
import stat
import tempfile
import time
from datetime import date
from pathlib import Path

from config import data_params


def scenario_size(scenario):
    """Resolve an expansion selector (index or literal size) to station count."""
    sizes = data_params["scenarios"]
    return (
        sizes[scenario]
        if isinstance(scenario, int) and 0 <= scenario < len(sizes)
        else int(scenario)
    )


def scenario(scenario):
    """Return the "_plus{n}" filename tag, or "" for None."""
    if scenario is None:
        return ""
    return f"_plus{scenario_size(scenario)}"


def expansion_station_file(size):
    """Return the expansion station table name for the configured city."""
    return f"{data_params['city']}_stations_test_{int(size)}.csv"


def run_stem(policy, stream, scenario_id=None, day=None):
    """Return the dated output stem."""
    iso = (day or date.today()).isoformat()
    return f"{policy}_{stream}{scenario(scenario_id)}_{iso}"


def latest_file(directory, prefix, tail):
    """Return the newest dated file matching prefix + date + tail, or None."""
    pattern = re.compile(re.escape(prefix) + r"\d{4}-\d{2}-\d{2}" + re.escape(tail))
    matches = [
        path
        for path in Path(directory).glob(f"{prefix}*{tail}")
        if pattern.match(path.name)
    ]
    return sorted(matches)[-1] if matches else None


def atomic_write(path, write, retries=5, delay=0.5):
    """Write through a unique sibling temp file, then atomically replace.
    On Windows, retry briefly on locked destinations, then fall back to a
    direct write so records are never lost."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        write(temporary)
        for attempt in range(retries):
            try:
                if path.exists():
                    try:
                        path.chmod(path.stat().st_mode | stat.S_IWRITE)
                    except OSError:
                        pass
                temporary.replace(path)
                return
            except PermissionError:
                if attempt == retries - 1:
                    break
                time.sleep(delay)
        # Last resort: replace was blocked by an external handle; write the
        # content in place so the records are still recorded.
        try:
            write(path)
        except PermissionError as exc:
            raise PermissionError(
                f"Cannot write {path}: the file appears to be open or locked "
                "by another program (Excel, an editor, antivirus, or a second "
                "run of the same task). Close it and resume; the previous file "
                "remains intact."
            ) from exc
    finally:
        temporary.unlink(missing_ok=True)


def atomic_to_csv(frame, path, retries=5, delay=0.5):
    """Atomically write a DataFrame as CSV (no index column)."""
    atomic_write(
        Path(path),
        lambda target: frame.to_csv(target, index=False),
        retries,
        delay,
    )
