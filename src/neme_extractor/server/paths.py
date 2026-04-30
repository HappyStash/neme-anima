"""Path normalization for user-supplied paths.

The web UI runs in the user's browser, which on Windows hosts may submit
Windows-style paths (``C:\\Users\\me\\foo.mkv``) or ``file://`` URIs even
when the server is running inside WSL. We normalize them here so the rest
of the pipeline can treat them as ordinary POSIX paths.
"""

from __future__ import annotations

import functools
import re
import sys
from pathlib import Path
from urllib.parse import unquote

_WIN_DRIVE_RE = re.compile(r"^([A-Za-z]):[\\/](.*)$", re.DOTALL)
_FILE_URI_RE = re.compile(r"^file://(/[A-Za-z]:/.*|/.*)$")


@functools.lru_cache(maxsize=1)
def is_wsl() -> bool:
    if sys.platform != "linux":
        return False
    try:
        with open("/proc/version") as f:
            return "microsoft" in f.read().lower()
    except OSError:
        return False


def normalize_input_path(raw: str) -> Path:
    """Best-effort canonicalization of a user-supplied path string.

    Handles three forms beyond a plain POSIX path:
      * ``file:///...`` URIs (decoded, drive prefix stripped)
      * Windows drive paths (``C:\\...`` or ``C:/...``)
      * vfs:// fallback emitted by the drop-zone when the browser hid the path
    """
    s = raw.strip().strip('"').strip("'")
    if not s:
        raise ValueError("empty path")
    if s.startswith("vfs://"):
        raise ValueError(
            f"path was not exposed by the browser; paste the absolute path manually: {s}"
        )

    m = _FILE_URI_RE.match(s)
    if m:
        s = unquote(m.group(1))
        # /C:/foo → C:/foo
        if re.match(r"^/[A-Za-z]:", s):
            s = s[1:]

    m = _WIN_DRIVE_RE.match(s)
    if m:
        drive = m.group(1).lower()
        rest = m.group(2).replace("\\", "/")
        if is_wsl():
            return Path(f"/mnt/{drive}/{rest}").expanduser()
        # On native Windows / other platforms, hand back a Path the OS understands.
        return Path(f"{drive.upper()}:/{rest}").expanduser()

    return Path(s).expanduser()
