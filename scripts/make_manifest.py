#!/usr/bin/env python3
import os, csv, hashlib
from pathlib import Path

def sha256_of_file(p: Path, chunk=1024 * 1024):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()

def head_preview(path: Path, n=2):
    lines = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.strip():
                lines.append(line.strip())
            if len(lines) >= n:
                break
    return " | ".join(lines)

def autodetect_family(name: str):
    base = Path(name).stem.upper()
    if base.startswith("RC"):
        return "RC"
    if base.startswith("C"):
        return "C"
    if base.startswith("R"):
        return "R"
    return "UNKNOWN"

def load_paths():
    cfg_path = Path("configs/data_paths.yaml")
    if cfg_path.exists():
        try:
            import yaml
            cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
            raw_dir = Path(cfg.get("raw_dir", "data/raw"))
            out_csv = Path(cfg.get("manifest", "data/manifest.csv"))
            return raw_dir, out_csv
        except Exception as e:
            print(f"[warn] Could not read YAML: {e}. Falling back to defaults.")
    return Path("data/raw"), Path("data/manifest.csv")

def main():
    raw_dir, out_csv = load_paths()
    if not raw_dir.exists():
        print(f"[error] Raw dir not found: {raw_dir}")
        return

    rows = []
    for p in sorted(raw_dir.glob("*.csv")):
        st = p.stat()
        rows.append({
            "filename": p.name,
            "family": autodetect_family(p.name),
            "size_bytes": st.st_size,
            "sha256": sha256_of_file(p),
            "head_preview": head_preview(p, n=2),
        })

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["filename", "family", "size_bytes", "sha256", "head_preview"]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"Manifest written: {out_csv} (files: {len(rows)})")

if __name__ == "__main__":
    main()
