#!/usr/bin/env python3
"""
bootstrap.py

- Checks required packages (by importing).
- Installs missing packages using pip.
- Launches Streamlit app.py.

Usage:
  python bootstrap.py

Recommended:
  python -m venv .venv
  # activate venv
  python bootstrap.py
"""

from __future__ import annotations

import os
import sys
import subprocess
import importlib
from typing import List, Tuple


# Map pip package name -> import name
REQUIREMENTS: List[Tuple[str, str]] = [
    ("python-dotenv>=1.0.0", "dotenv"),
    ("pandas>=2.0.0", "pandas"),
    ("numpy>=1.24.0", "numpy"),
    ("google-api-python-client>=2.110.0", "googleapiclient"),
    ("google-auth>=2.25.0", "google.auth"),
    ("google-auth-httplib2>=0.1.1", "google_auth_httplib2"),
    ("google-auth-oauthlib>=1.2.0", "google_auth_oauthlib"),
    ("httplib2>=0.22.0", "httplib2"),
    ("isodate>=0.6.1", "isodate"),
    ("streamlit>=1.32.0", "streamlit"),
    ("plotly>=5.18.0", "plotly"),
    ("sqlalchemy>=2.0.0", "sqlalchemy"),
    # Optional but recommended
    ("pyarrow>=15.0.0", "pyarrow"),
    ("typing-extensions>=4.9.0", "typing_extensions"),
    ("watchdog>=3.0.0", "watchdog"),
]


def _print(msg: str) -> None:
    print(msg, flush=True)


def import_exists(import_name: str) -> bool:
    try:
        importlib.import_module(import_name)
        return True
    except Exception:
        return False


def pip_install(pkgs: List[str]) -> None:
    if not pkgs:
        return

    _print("\nInstalling missing packages:")
    for p in pkgs:
        _print(f"  - {p}")

    cmd = [sys.executable, "-m", "pip", "install", "--upgrade"] + pkgs
    _print("\nRunning: " + " ".join(cmd))
    subprocess.check_call(cmd)


def ensure_pip_ready() -> None:
    # Ensure pip exists and is recent enough
    subprocess.check_call([sys.executable, "-m", "pip", "--version"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])


def run_streamlit(app_path: str = "app.py") -> None:
    if not os.path.exists(app_path):
        raise FileNotFoundError(f"Cannot find {app_path}. Run bootstrap.py in the project folder.")
    cmd = [sys.executable, "-m", "streamlit", "run", app_path]
    _print("\nStarting Streamlit:")
    _print("  " + " ".join(cmd))
    subprocess.check_call(cmd)


def main() -> None:
    _print("Bootstrap: checking environment...\n")

    # Optional: warn if not in venv
    in_venv = (sys.prefix != sys.base_prefix)
    if not in_venv:
        _print("Warning: You are not in a virtual environment.")
        _print("Recommended:")
        _print("  python -m venv .venv")
        _print("  source .venv/bin/activate   (macOS/Linux)")
        _print("  .venv\\Scripts\\activate      (Windows)\n")

    ensure_pip_ready()

    missing = []
    for pip_spec, import_name in REQUIREMENTS:
        if import_exists(import_name):
            _print(f"[OK]   {import_name}")
        else:
            _print(f"[MISS] {import_name}  -> will install {pip_spec}")
            missing.append(pip_spec)

    if missing:
        pip_install(missing)
        _print("\nRe-checking imports after installation...\n")
        still_missing = []
        for pip_spec, import_name in REQUIREMENTS:
            if not import_exists(import_name):
                still_missing.append((pip_spec, import_name))

        if still_missing:
            _print("Some packages still failed to import after installation:")
            for pip_spec, import_name in still_missing:
                _print(f"  - {pip_spec} (import: {import_name})")
            _print("\nResolve the above manually, then re-run bootstrap.py.")
            sys.exit(1)

    _print("\nAll requirements satisfied.")
    run_streamlit("app.py")


if __name__ == "__main__":
    main()
