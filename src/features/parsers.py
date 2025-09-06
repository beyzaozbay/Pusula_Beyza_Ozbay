import re
from typing import Optional

_NUM_RE = re.compile(r"(\d+(?:[.,]\d+)?)")
_RANGE_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*[-–]\s*(\d+(?:[.,]\d+)?)", re.UNICODE)

def _to_float(num_str: str) -> Optional[float]:
    if num_str is None:
        return None
    # Convert Turkish/European decimal comma to dot
    num_str = num_str.replace(",", ".")
    try:
        return float(num_str)
    except Exception:
        return None

def extract_number_or_range(value: str) -> Optional[float]:
    """
    Extract a numeric value from text. If a range like '10-12' appears,
    return the average (11.0). If only a single number exists, return that.
    """
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    # Range first
    m = _RANGE_RE.search(s)
    if m:
        a = _to_float(m.group(1))
        b = _to_float(m.group(2))
        if a is not None and b is not None:
            return (a + b) / 2.0
    # Single number
    m = _NUM_RE.search(s)
    if m:
        return _to_float(m.group(1))
    return None

def parse_sessions(value: str) -> Optional[float]:
    """
    Parse session count from text like '10 seans', '8-10 seans', '12'.
    Non-negative. Returns float (can be cast to int later).
    """
    n = extract_number_or_range(value)
    if n is None:
        return None
    return n if n >= 0 else None

def parse_duration_minutes(value: str) -> Optional[float]:
    """
    Parse application duration from text. Heuristics:
    - 'saat' or 'hour' or 'h' => minutes = number * 60
    - 'dk' or 'dakika' or 'min' => minutes = number
    - 'sn' or 'saniye' or 'sec' => minutes = number / 60
    - 'gün' or 'day' => minutes = number * 24 * 60
    - otherwise: assume minutes
    """
    if value is None:
        return None
    s = str(value).lower().strip()
    if not s:
        return None
    n = extract_number_or_range(s)
    if n is None:
        return None

    # unit detection
    if any(u in s for u in ["saat", " hour", " hours", " hrs", " hr", " h "]):
        return n * 60.0
    if any(u in s for u in ["dk", "dakika", " min", "mins", "minute"]):
        return n
    if any(u in s for u in ["sn", "saniye", " sec", "second"]):
        return n / 60.0
    if any(u in s for u in ["gün", " day"]):
        return n * 24.0 * 60.0

    # default: treat as minutes
    return n

def to_int_safe(x: Optional[float]) -> Optional[int]:
    """Cast float to int when close to integer; otherwise round sensibly."""
    if x is None:
        return None
    try:
        if abs(x - round(x)) < 1e-9:
            return int(round(x))
        return int(round(x))
    except Exception:
        return None
