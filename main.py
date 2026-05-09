"""
main.py  (generic)
==================
FastAPI backend. Receives user code strings from frontend,
parses/validates them, then runs the generic solver + optimiser pipeline.

Run:  uvicorn main:app --reload --port 8000
"""

from __future__ import annotations
import asyncio, copy, io, json, os, queue, threading, time, uuid, zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from fastapi import FastAPI, BackgroundTasks, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from user_code_runner import parse_user_code, validate_user_model
from solver            import build_road_signals, run_one_case, compute_cabin_rms, estimate_single_eval_time
from optimiser         import run_optimisation
from plotter           import generate_all_plots, generate_time_history

# ---------------------------------------------------------------------------
app = FastAPI(title="Ride Optimisation — Generic", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

UPLOAD_DIR  = Path("uploads");  UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR = Path("results");  RESULTS_DIR.mkdir(exist_ok=True)

_jobs: Dict[str, Dict]         = {}
_log_queues: Dict[str, queue.Queue] = {}


# ---------------------------------------------------------------------------
# Request schema
# ---------------------------------------------------------------------------
class RunRequest(BaseModel):
    # User code strings from editor panels
    ode_code:  str          # ODE panel  (CONFIG, BOUNDS, OUTPUTS, ode_rhs)
    geom_code: str = ""     # Geometry panel (constraints fn) — empty = disabled

    # Optimiser
    mode:      str = "nsga2"
    pop_size:  int = 10
    n_gen:     int = 5
    n_calls:   int = 50
    n_initial: int = 10

    # Simulation time
    T_END:    float = 466.945
    T_IGNORE: float = 0.5

    # CSV paths (key -> server path, returned by /upload-csv)
    csv_paths: Dict[str, str] = {}


# ---------------------------------------------------------------------------
# Job runner (background thread)
# ---------------------------------------------------------------------------
def _run_job(job_id: str, req: RunRequest):
    job  = _jobs[job_id]
    logq = _log_queues[job_id]

    def log(msg: str):
        logq.put(msg)
        job["logs"].append(msg)

    try:
        job["status"] = "running"
        log(f"Job {job_id} started | mode={req.mode}")

        # ---- 1. Parse user code ----
        log("Parsing user code …")
        try:
            model = parse_user_code(req.ode_code, req.geom_code or None)
        except Exception as e:
            log(f"[ERROR] Code parse failed: {e}")
            job["status"] = "failed"; job["error"] = str(e)
            logq.put("__DONE__"); return

        # ---- 2. Validate ----
        violations = validate_user_model(model)
        if violations:
            for v in violations:
                log(f"[CONFIG ERROR] {v}")
            job["status"] = "failed"
            job["error"]  = "; ".join(violations)
            logq.put("__DONE__"); return
        log(f"Model OK | n_states={model.n_states} | geom={model.geom_enabled} "
            f"| params={list(model.bounds.keys())}")

        # ---- 3. Inject CSV paths + time settings into cfg ----
        csv_key_map = {
            "fa_lh":  "axlefront_left_csv",
            "fa_rh":  "axlefront_right_csv",
            "ra1_lh": "axlerear1_left_csv",
            "ra1_rh": "axlerear1_right_csv",
            "ra2_lh": "axlerear2_left_csv",
            "ra2_rh": "axlerear2_right_csv",
        }
        for short, cfg_key in csv_key_map.items():
            if short in req.csv_paths:
                model.cfg[cfg_key] = req.csv_paths[short]
        model.cfg["T_END"]    = req.T_END
        model.cfg["T_IGNORE"] = req.T_IGNORE

        # ---- 4. Build road signals ----
        log("Loading road signals …")
        try:
            signals = build_road_signals(model.cfg)
            log(f"Road signals loaded: {list(signals.keys())}")
        except Exception as e:
            log(f"[ERROR] CSV load failed: {e}")
            job["status"] = "failed"; job["error"] = str(e)
            logq.put("__DONE__"); return

        # ---- 5. Time estimate ----
        log("Estimating simulation time …")
        try:
            single_t = estimate_single_eval_time(model, signals, t_sample=3.0)
            n_evals  = (req.pop_size * (req.n_gen + 1)
                        if req.mode == "nsga2" else req.n_calls)
            est_total = single_t * n_evals
            mins, secs = divmod(int(est_total), 60)
            log(f"Est. per eval={single_t:.1f}s | total evals={n_evals} "
                f"| Est. total time ≈ {mins}m {secs}s")
            job["est_seconds"] = est_total
            job["n_evals_total"] = n_evals
        except Exception as e:
            log(f"[WARN] Time estimation failed: {e}")
            job["est_seconds"] = None

        # ---- 6. Time array ----
        dt     = float(model.cfg.get("DT", 0.001))
        t_eval = np.arange(0.0, req.T_END + dt, dt)
        log(f"t_eval: {len(t_eval)} points | dt={dt} | T_END={req.T_END}")

        # ---- 7. Run optimisation ----
        opt_settings = dict(
            pop_size=req.pop_size, n_gen=req.n_gen,
            n_calls=req.n_calls,   n_initial=req.n_initial,
        )
        result = run_optimisation(
            model=model, signals=signals, t_eval=t_eval,
            mode=req.mode, opt_settings=opt_settings, log_cb=log,
        )

        # ---- 8. Save outputs ----
        out_dir = str(RESULTS_DIR / job_id)
        os.makedirs(out_dir, exist_ok=True)

        result.df_pareto.to_csv(os.path.join(out_dir, "pareto_front.csv"), index=False)
        run_data = {
            "mode":         result.mode,
            "n_evals":      result.n_evals,
            "wall_seconds": result.wall_seconds,
            "param_keys":   result.param_keys,
            "pareto":       result.df_pareto.to_dict(orient="records"),
            "hv_history":   result.hv_history,
            "convergence":  result.convergence,
            "eval_log":     result.eval_log[:500],
        }
        with open(os.path.join(out_dir, "run_results.json"), "w") as f:
            json.dump(run_data, f, indent=2, default=str)

        log("Generating plots …")
        plot_paths = generate_all_plots(result, model, signals, t_eval, out_dir)
        log(f"Plots done: {len(plot_paths)} files")

        job["status"]     = "done"
        job["result"]     = run_data
        job["plot_files"] = [os.path.basename(p) for p in plot_paths if p]
        job["out_dir"]    = out_dir
        log(f"Job complete | evals={result.n_evals} | wall={result.wall_seconds:.1f}s")

    except Exception as e:
        import traceback
        log(f"[ERROR] {e}\n{traceback.format_exc()}")
        job["status"] = "failed"; job["error"] = str(e)
    finally:
        logq.put("__DONE__")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/")
def root():
    return {"message": "Ride Optimisation API v2 running."}


@app.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...), key: str = Form(...)):
    name = f"{key}_{uuid.uuid4().hex[:8]}_{file.filename}"
    dest = UPLOAD_DIR / name
    with open(dest, "wb") as f:
        f.write(await file.read())
    return {"key": key, "server_path": str(dest), "filename": file.filename}


@app.post("/run")
async def start_run(req: RunRequest, background_tasks: BackgroundTasks):
    job_id = uuid.uuid4().hex[:12]
    _jobs[job_id] = {
        "status": "queued", "mode": req.mode,
        "logs": [], "result": None, "plot_files": [],
        "error": None, "created_at": time.time(),
        "est_seconds": None, "n_evals_total": None,
    }
    _log_queues[job_id] = queue.Queue()
    threading.Thread(target=_run_job, args=(job_id, req), daemon=True).start()
    return {"job_id": job_id, "status": "queued"}


@app.get("/status/{job_id}")
def get_status(job_id: str):
    if job_id not in _jobs:
        raise HTTPException(404, "Job not found")
    j = _jobs[job_id]
    return {
        "job_id":        job_id,
        "status":        j["status"],
        "error":         j["error"],
        "est_seconds":   j.get("est_seconds"),
        "n_evals_total": j.get("n_evals_total"),
        "plot_count":    len(j["plot_files"]),
    }


@app.get("/stream/{job_id}")
async def stream_logs(job_id: str):
    if job_id not in _jobs:
        raise HTTPException(404, "Job not found")
    logq = _log_queues[job_id]

    async def gen():
        while True:
            try:
                msg = logq.get(timeout=0.3)
                if msg == "__DONE__":
                    yield "data: __DONE__\n\n"; break
                yield f"data: {msg}\n\n"
            except queue.Empty:
                yield ": keep-alive\n\n"
                await asyncio.sleep(0.1)
                if _jobs[job_id]["status"] in ("done", "failed"):
                    while not logq.empty():
                        m = logq.get_nowait()
                        if m != "__DONE__":
                            yield f"data: {m}\n\n"
                    yield "data: __DONE__\n\n"; break

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/results/{job_id}")
def get_results(job_id: str):
    if job_id not in _jobs:
        raise HTTPException(404, "Job not found")
    j = _jobs[job_id]
    if j["status"] != "done":
        return {"status": j["status"], "error": j["error"]}
    return {"status": "done", "result": j["result"]}


@app.get("/plots/{job_id}")
def list_plots(job_id: str):
    if job_id not in _jobs:
        raise HTTPException(404, "Job not found")
    return {"plots": _jobs[job_id]["plot_files"]}


@app.get("/plot/{job_id}/{filename}")
def serve_plot(job_id: str, filename: str):
    if job_id not in _jobs:
        raise HTTPException(404, "Job not found")
    path = os.path.join(_jobs[job_id].get("out_dir",""), "plots", filename)
    if not os.path.isfile(path):
        raise HTTPException(404, f"Plot not found: {filename}")
    return FileResponse(path, media_type="image/png")


@app.get("/download/{job_id}")
def download_csv(job_id: str):
    if job_id not in _jobs:
        raise HTTPException(404, "Job not found")
    path = os.path.join(_jobs[job_id].get("out_dir",""), "pareto_front.csv")
    if not os.path.isfile(path):
        raise HTTPException(404, "CSV not ready")
    return FileResponse(path, media_type="text/csv",
                        filename=f"pareto_front_{job_id}.csv")


@app.get("/download-all/{job_id}")
def download_all(job_id: str):
    if job_id not in _jobs:
        raise HTTPException(404, "Job not found")
    out_dir = _jobs[job_id].get("out_dir", "")
    if not out_dir or not os.path.isdir(out_dir):
        raise HTTPException(404, "Results not ready")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(out_dir):
            for fname in files:
                fpath   = os.path.join(root, fname)
                arcname = os.path.relpath(fpath, out_dir)
                zf.write(fpath, arcname)
    buf.seek(0)
    return StreamingResponse(
        buf, media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename=results_{job_id}.zip"},
    )
