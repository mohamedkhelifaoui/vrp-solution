from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from pathlib import Path
import subprocess, sys, tempfile, shutil, pandas as pd

BASE = Path(__file__).resolve().parents[1]
OUT  = BASE / "data" / "solutions_champions" / "api"
OUT.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="ML Champions API")

def run_pipeline(csv_path: Path, outdir: Path):
    cmd = [
        sys.executable, str(BASE/"scripts"/"ml_predict_and_run.py"),
        "--csv", str(csv_path),
        "--K", "200", "--seed", "42",
        "--cv_global", "0.20", "--cv_link", "0.10",
        "--fallbacks", "Q120", "Gamma1-q1p645",
        "--outdir", str(outdir)
    ]
    subprocess.run(cmd, check=True)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / file.filename
        tmp.write_bytes(await file.read())
        run_pipeline(tmp, OUT)
        inst = tmp.stem
        summary  = OUT / f"{inst}__summary.csv"
        cand_eval= OUT / f"{inst}__candidates_eval.csv"
        plan     = next(OUT.glob(f"{inst}__champion_*.json"), None)
        resp = {"instance": inst, "plan_json": str(plan.relative_to(BASE)) if plan else None}
        if summary.exists():
            resp["summary"] = pd.read_csv(summary).to_dict(orient="records")[0]
        if cand_eval.exists():
            resp["candidates_eval"] = pd.read_csv(cand_eval).to_dict(orient="records")
        return JSONResponse(resp)
