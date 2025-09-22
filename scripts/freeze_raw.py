#!/usr/bin/env python3
"""
Mark files in data/raw as read-only to 'freeze' the originals.

- On Unix (Linux/macOS): removes write bits.
- On Windows: sets FILE_ATTRIBUTE_READONLY.

This version works even if configs/data_paths.yaml is missing.
"""
import os, stat, platform
from pathlib import Path

def load_raw_dir():
    cfg_path = Path("configs/data_paths.yaml")
    if cfg_path.exists():
        try:
            import yaml
            cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
            raw_dir = Path(cfg.get("raw_dir", "data/raw"))
            return raw_dir
        except Exception as e:
            print(f"[warn] Could not read YAML ({cfg_path}): {e}. Falling back to data/raw.")
    return Path("data/raw")

def make_readonly(p: Path):
    try:
        if platform.system().lower().startswith("win"):
            import ctypes
            FILE_ATTRIBUTE_READONLY = 0x01
            ctypes.windll.kernel32.SetFileAttributesW(str(p), FILE_ATTRIBUTE_READONLY)
        else:
            mode = p.stat().st_mode
            p.chmod(mode & ~stat.S_IWUSR & ~stat.S_IWGRP & ~stat.S_IWOTH)
    except Exception as e:
        print(f"[warn] Failed to set read-only for {p}: {e}")

def main():
    raw_dir = load_raw_dir()
    if not raw_dir.exists():
        print(f"[error] Raw dir not found: {raw_dir} (create it or add configs/data_paths.yaml)")
        return
    cnt = 0
    for p in raw_dir.glob("*.csv"):
        make_readonly(p); cnt += 1
    print(f"Set read-only on {cnt} files in {raw_dir}")

if __name__ == "__main__":
    main()
