"""HTTP server for the diffusion renderer web demo.

Serves the Vite-built frontend, the inverse-render asset folder, and a
POST /render endpoint that runs the diffusion forward renderer.

Usage:
    uv run serve.py                          # default: samples/garment.zprj
    uv run serve.py samples/250000_coat.zprj
"""

import argparse
import cgi
import datetime
import io
import json
import os
import queue
import shutil
import subprocess
import threading
import traceback
from http.server import HTTPServer, SimpleHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import torch
from PIL import Image


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
INVERSE_DIR = os.path.join(PROJECT_ROOT, "asset", "inverse")
INVERSE_UPLOAD_DIR = os.path.join(PROJECT_ROOT, "output", "inverse_uploads")
HDRI_DIR = os.path.join(PROJECT_ROOT, "examples", "hdri")
HDRI_UPLOAD_DIR = os.path.join(PROJECT_ROOT, "output", "hdri_uploads")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "web_capture")
GBUFFER_NAMES = ["basecolor", "normal", "depth", "roughness", "metallic"]
DEFAULT_HDR = os.path.join(PROJECT_ROOT, "examples", "hdri", "sunny_vondelpark_1k.hdr")

_save_lock = threading.Lock()
_render_lock = threading.Lock()
_forward_render = None
_inverse_render = None


def _reset_session_state():
    """Wipe all per-sid upload dirs at server startup. Per-tab sids already
    isolate uploads between clients within a running server (see _sid_upload_dir);
    this is belt-and-suspenders so stale upload dirs from prior server runs
    don't accumulate on disk. Bundled presets in INVERSE_DIR are untouched."""
    if os.path.isdir(INVERSE_UPLOAD_DIR):
        shutil.rmtree(INVERSE_UPLOAD_DIR, ignore_errors=True)

# SSE progress: per-job subscriber queues + global render queue state
# _job_subscribers: {job_id: [Queue, ...]}
# _render_queue: [job_id, ...] — ordered list of jobs waiting or running
_job_subscribers: dict[str, list[queue.Queue]] = {}
_render_queue: list[str] = []   # index 0 = currently running
_sse_lock = threading.Lock()


def _send_to_job(job_id: str, event: dict):
    """Send an event to all SSE subscribers of a specific job."""
    data = "data: " + json.dumps(event) + "\n\n"
    with _sse_lock:
        dead = []
        for q in _job_subscribers.get(job_id, []):
            try:
                q.put_nowait(data)
            except queue.Full:
                dead.append(q)
        for q in dead:
            _job_subscribers[job_id].remove(q)


def _enqueue_job(job_id: str):
    """Register a new job and notify its subscribers of queue position."""
    with _sse_lock:
        _render_queue.append(job_id)
        pos = len(_render_queue) - 1  # 0 = running, 1+ = waiting
    if pos > 0:
        _send_to_job(job_id, {"phase": "queued", "position": pos})


def _start_job(job_id: str):
    """Mark job as started (it's at front of queue)."""
    _send_to_job(job_id, {"phase": "start", "step": 0, "total": 0})


def _finish_job(job_id: str):
    """Remove job from queue."""
    with _sse_lock:
        try:
            _render_queue.remove(job_id)
        except ValueError:
            pass


def _resolve_hdr_path(name):
    """Find an .hdr file by base name in bundled dir or uploads."""
    if not name:
        return None
    fname = name if name.lower().endswith(".hdr") else name + ".hdr"
    for base in (HDRI_DIR, HDRI_UPLOAD_DIR):
        path = os.path.join(base, fname)
        if os.path.isfile(path):
            return path
    return None


def _get_forward_render():
    """Lazy import forward_render to avoid heavy torch imports at startup."""
    global _forward_render
    if _forward_render is None:
        print("Loading diffusion renderer module (first request)...")
        from render_zprj import forward_render
        _forward_render = forward_render
        print("Diffusion renderer module ready.")
    return _forward_render


def _get_inverse_render():
    """Lazy import inverse_render."""
    global _inverse_render
    if _inverse_render is None:
        print("Loading inverse renderer module (first request)...")
        from render_inverse import inverse_render
        _inverse_render = inverse_render
        print("Inverse renderer module ready.")
    return _inverse_render


def _sanitize_sid(raw):
    """Validate the per-tab session id sent by the client. Alphanumeric and
    hyphen only (covers UUIDs and the JS fallback). Returns '' if invalid —
    a client without a valid sid only sees bundled presets, never uploads."""
    if not raw:
        return ""
    safe = "".join(c for c in raw if c.isalnum() or c == "-")
    return safe[:64] if safe == raw and 1 <= len(raw) <= 64 else ""


def _sid_upload_dir(sid):
    """Per-sid upload root, or None when sid is empty/invalid."""
    return os.path.join(INVERSE_UPLOAD_DIR, sid) if sid else None


def _list_bg_presets(sid=""):
    """Bundled presets + uploads that belong to this sid only."""
    names = []
    bases = [INVERSE_DIR]
    sd = _sid_upload_dir(sid)
    if sd:
        bases.append(sd)
    for base in bases:
        if not os.path.isdir(base):
            continue
        for name in sorted(os.listdir(base)):
            if os.path.isfile(os.path.join(base, name, "rgb_input.png")):
                if name not in names:
                    names.append(name)
    return names


def _resolve_inverse_dir(name, sid=""):
    """Find the directory for an inverse preset (bundled or this-sid upload)."""
    bases = [INVERSE_DIR]
    sd = _sid_upload_dir(sid)
    if sd:
        bases.append(sd)
    for base in bases:
        d = os.path.join(base, name)
        if os.path.isdir(d):
            return d
    return None


def _sanitize_preset_name(raw):
    """Make a safe directory name from a user-supplied basename."""
    stem = os.path.splitext(os.path.basename(raw or "upload"))[0]
    safe = "".join(c if (c.isalnum() or c in "-_") else "_" for c in stem)
    safe = safe.strip("_") or "upload"
    return safe[:48]


def build_web_client(web_dir):
    dist_dir = os.path.join(web_dir, "dist")
    # On Windows the node tool wrappers carry a .cmd extension; the bare name is
    # a Unix shell script that CreateProcess can't launch (WinError 193).
    npm = "npm.cmd" if os.name == "nt" else "npm"
    if not os.path.isdir(os.path.join(web_dir, "node_modules")):
        print("Installing web dependencies ...")
        subprocess.run([npm, "install"], cwd=web_dir, check=True)
    src_mtime = max(
        os.path.getmtime(os.path.join(dp, f))
        for dp, _, fns in os.walk(os.path.join(web_dir, "src"))
        for f in fns
    ) if os.path.isdir(os.path.join(web_dir, "src")) else 0
    idx_mtime = os.path.getmtime(os.path.join(web_dir, "index.html")) if os.path.exists(os.path.join(web_dir, "index.html")) else 0
    pub_mtime = max(
        os.path.getmtime(os.path.join(dp, f))
        for dp, _, fns in os.walk(os.path.join(web_dir, "public"))
        for f in fns if not f.endswith('.zprj')
    ) if os.path.isdir(os.path.join(web_dir, "public")) else 0
    dist_mtime = os.path.getmtime(os.path.join(dist_dir, "index.html")) if os.path.exists(os.path.join(dist_dir, "index.html")) else 0
    if max(src_mtime, idx_mtime, pub_mtime) > dist_mtime:
        print("Building web client ...")
        vite_name = "vite.cmd" if os.name == "nt" else "vite"
        local_vite = os.path.join(web_dir, "node_modules", ".bin", vite_name)
        if os.path.isfile(local_vite):
            vite_cmd = [local_vite, "build"]
        else:
            npx = "npx.cmd" if os.name == "nt" else "npx"
            vite_cmd = [npx, "vite", "build"]
        subprocess.run(vite_cmd, cwd=web_dir, check=True)
    return dist_dir


class Handler(SimpleHTTPRequestHandler):
    serve_dir = None
    hdr_path = DEFAULT_HDR
    device = "cuda"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=Handler.serve_dir, **kwargs)

    def end_headers(self):
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def _send_text(self, code, msg):
        body = msg.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_file(self, file_path, content_type):
        if not os.path.isfile(file_path):
            self._send_text(404, "Not found")
            return
        try:
            with open(file_path, "rb") as f:
                data = f.read()
        except OSError as e:
            self._send_text(500, str(e))
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        # SSE progress stream
        if self.path == "/progress" or self.path.startswith("/progress?"):
            return self._handle_progress_sse()

        # List background presets (bundled + this-sid uploads only)
        if self.path == "/bg_presets" or self.path.startswith("/bg_presets?"):
            from urllib.parse import urlparse, parse_qs
            sid = _sanitize_sid(parse_qs(urlparse(self.path).query).get("sid", [""])[0])
            presets = _list_bg_presets(sid)
            body = json.dumps(presets).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        # Serve /inverse/<name>/<file> from bundled asset/inverse/ or this-sid uploads.
        # A client without a valid sid can only fetch bundled presets.
        if self.path.startswith("/inverse/"):
            from urllib.parse import urlparse, parse_qs
            parsed = urlparse(self.path)
            sid = _sanitize_sid(parse_qs(parsed.query).get("sid", [""])[0])
            rel = parsed.path[len("/inverse/"):]
            bases = [INVERSE_DIR]
            sd = _sid_upload_dir(sid)
            if sd:
                bases.append(sd)
            for base in bases:
                file_path = os.path.normpath(os.path.join(base, rel))
                if file_path.startswith(base) and os.path.isfile(file_path):
                    return self._serve_file(file_path, "image/png")
            self._send_text(404, "Not found")
            return

        # Serve /hdri/<name>.hdr from examples/hdri/ (or uploads)
        if self.path.startswith("/hdri/"):
            rel = self.path[len("/hdri/"):].split("?")[0]
            for base in (HDRI_DIR, HDRI_UPLOAD_DIR):
                file_path = os.path.normpath(os.path.join(base, rel))
                if file_path.startswith(base) and os.path.isfile(file_path):
                    return self._serve_file(file_path, "application/octet-stream")
            self._send_text(404, "Not found")
            return

        return super().do_GET()

    def _handle_progress_sse(self):
        """Long-lived SSE endpoint scoped to a single job_id."""
        from urllib.parse import urlparse, parse_qs
        qs = parse_qs(urlparse(self.path).query)
        job_id = (qs.get("job_id") or [None])[0]
        if not job_id:
            self._send_text(400, "Missing job_id")
            return

        q = queue.Queue(maxsize=128)
        with _sse_lock:
            _job_subscribers.setdefault(job_id, []).append(q)
            # Immediately inform client of current queue position if already enqueued
            if job_id in _render_queue:
                pos = _render_queue.index(job_id)
                if pos > 0:
                    q.put_nowait("data: " + json.dumps({"phase": "queued", "position": pos}) + "\n\n")

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        try:
            while True:
                try:
                    chunk = q.get(timeout=25)
                    self.wfile.write(chunk.encode("utf-8"))
                    self.wfile.flush()
                    if '"phase": "done"' in chunk:
                        break
                except queue.Empty:
                    self.wfile.write(b": keep-alive\n\n")
                    self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            with _sse_lock:
                subs = _job_subscribers.get(job_id, [])
                try:
                    subs.remove(q)
                except ValueError:
                    pass
                if not subs:
                    _job_subscribers.pop(job_id, None)

    def do_POST(self):
        if self.path == "/render":
            return self._handle_render()
        if self.path == "/upload_hdr":
            return self._handle_upload_hdr()
        if self.path == "/upload_bg":
            return self._handle_upload_bg()
        self._send_text(404, "Not found")

    def _handle_render(self):
        ctype = self.headers.get("Content-Type", "")
        if not ctype.startswith("multipart/form-data"):
            self._send_text(400, "Expected multipart/form-data")
            return
        job_id = ""
        try:
            fs = cgi.FieldStorage(
                fp=self.rfile, headers=self.headers,
                environ={"REQUEST_METHOD": "POST", "CONTENT_TYPE": ctype},
            )
            hdr_name = fs.getfirst("hdr", "")
            bg_preset = fs.getfirst("bg_preset", "")
            mode = fs.getfirst("mode", "still")
            job_id = fs.getfirst("job_id", "")
            if mode not in ("still", "rotate_light"):
                self._send_text(400, f"Unsupported mode: {mode}")
                return
            if not job_id:
                self._send_text(400, "Missing job_id")
                return
            _enqueue_job(job_id)

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = os.path.join(OUTPUT_DIR, f"{timestamp}_{job_id[:8]}")
            os.makedirs(save_dir, exist_ok=True)

            # Save raw PNGs (also keep bytes in memory for inference)
            gbuf_pngs = {}
            for name in GBUFFER_NAMES:
                if name in fs:
                    raw = fs[name].file.read()
                    gbuf_pngs[name] = raw
                    with open(os.path.join(save_dir, f"{name}.png"), "wb") as f:
                        f.write(raw)
            if not gbuf_pngs:
                self._send_text(400, "No G-buffers provided")
                return

            # Resolve HDR path
            hdr_path = _resolve_hdr_path(hdr_name) if hdr_name else Handler.hdr_path

            # Decode G-buffers → tensors
            gbuffers = {}
            h = w = None
            for name, raw in gbuf_pngs.items():
                img = Image.open(io.BytesIO(raw)).convert("RGB")
                t = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0)
                gbuffers[name] = t
                if h is None:
                    h, w = t.shape[:2]
            # Fill missing G-buffers (e.g. metallic) with zeros, mark as drop_cond
            drop_conds = []
            for name in GBUFFER_NAMES:
                if name not in gbuffers:
                    gbuffers[name] = torch.zeros(h, w, 3, dtype=torch.float32)
                    drop_conds.append(name)

            meta = {
                "mode": mode,
                "hdr": hdr_name or None,
                "hdr_path": os.path.relpath(hdr_path, PROJECT_ROOT) if hdr_path else None,
                "bg_preset": bg_preset or None,
                "files": list(gbuf_pngs.keys()),
                "drop_conds": drop_conds,
                "resolution": [w, h],
            }
            with open(os.path.join(save_dir, "meta.json"), "w") as f:
                json.dump(meta, f, indent=2)

            print(f"[/render] job={job_id} inputs saved → {os.path.relpath(save_dir)} ({w}x{h}) mode={mode} hdr={hdr_name!r} drop={drop_conds}")

            # Run forward render (serialize — GPU is shared)
            forward_render = _get_forward_render()
            frames_holder = {"frames": None}
            def on_sample(_i, _seed, frames):
                frames_holder["frames"] = frames

            def on_step(sample_idx, num_samples, step, total):
                _send_to_job(job_id, {
                    "phase": "denoising",
                    "sample": sample_idx + 1,
                    "num_samples": num_samples,
                    "step": step,
                    "total": total,
                })

            with _render_lock:
                print(f"[/render] job={job_id} started")
                _start_job(job_id)
                forward_render(
                    gbuffers, hdr_path, device=Handler.device,
                    rotate_light=(mode == "rotate_light"),
                    num_samples=1,
                    on_sample=on_sample,
                    on_step=on_step,
                    drop_conds=drop_conds or None,
                )
            _send_to_job(job_id, {"phase": "done"})
            _finish_job(job_id)

            frames = frames_holder["frames"]
            if not frames:
                self._send_text(500, "Forward render produced no image")
                return

            if mode == "rotate_light":
                from utils.utils_render import save_video
                mp4_path = os.path.join(save_dir, "result.mp4")
                save_video(frames, mp4_path, fps=10)
                print(f"[/render] result → {os.path.relpath(mp4_path)} ({len(frames)} frames)")
                with open(mp4_path, "rb") as f:
                    data = f.read()
                content_type = "video/mp4"
            else:
                pil_img = frames[0]
                result_path = os.path.join(save_dir, "result.png")
                pil_img.save(result_path)
                print(f"[/render] result → {os.path.relpath(result_path)}")
                buf = io.BytesIO()
                pil_img.save(buf, format="PNG")
                data = buf.getvalue()
                content_type = "image/png"

            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("X-Saved-Dir", os.path.relpath(save_dir, PROJECT_ROOT))
            self.end_headers()
            self.wfile.write(data)
        except Exception as e:
            traceback.print_exc()
            if job_id:
                _send_to_job(job_id, {"phase": "error", "message": str(e)})
                _finish_job(job_id)
            self._send_text(500, f"Render failed: {e}")

    def _handle_upload_bg(self):
        """Receive an RGB image, run inverse_render, save G-buffers as a preset
        under this client's per-tab sid. Only the uploading client sees it."""
        ctype = self.headers.get("Content-Type", "")
        if not ctype.startswith("multipart/form-data"):
            self._send_text(400, "Expected multipart/form-data")
            return
        job_id = ""
        try:
            fs = cgi.FieldStorage(
                fp=self.rfile, headers=self.headers,
                environ={"REQUEST_METHOD": "POST", "CONTENT_TYPE": ctype},
            )
            if "image" not in fs:
                self._send_text(400, "Missing 'image' field")
                return
            job_id = fs.getfirst("job_id", "")
            if not job_id:
                self._send_text(400, "Missing job_id")
                return
            sid = _sanitize_sid(fs.getfirst("sid", ""))
            if not sid:
                self._send_text(400, "Missing or invalid sid")
                return

            item = fs["image"]
            raw = item.file.read()
            orig_name = os.path.basename(item.filename or "upload.png")
            preset_name = _sanitize_preset_name(orig_name)
            existing = set(_list_bg_presets(sid))
            if preset_name in existing:
                base = preset_name
                i = 2
                while f"{base}_{i}" in existing:
                    i += 1
                preset_name = f"{base}_{i}"

            sid_dir = _sid_upload_dir(sid)
            os.makedirs(sid_dir, exist_ok=True)
            preset_dir = os.path.join(sid_dir, preset_name)
            os.makedirs(preset_dir, exist_ok=True)

            src_ext = os.path.splitext(orig_name)[1].lower() or ".png"
            if src_ext not in (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"):
                src_ext = ".png"
            src_path = os.path.join(preset_dir, f"_source{src_ext}")
            with open(src_path, "wb") as f:
                f.write(raw)

            _enqueue_job(job_id)

            print(f"[/upload_bg] sid={sid[:8]} job={job_id[:8]} preset={preset_name} ({len(raw)} bytes)")

            inverse_render = _get_inverse_render()

            def on_pass(pi, np_, name):
                _send_to_job(job_id, {
                    "phase": "inverse_pass",
                    "pass_idx": pi,
                    "num_passes": np_,
                    "pass_name": name,
                })

            def on_step(pi, np_, step, total):
                _send_to_job(job_id, {
                    "phase": "denoising",
                    "sample": pi + 1,
                    "num_samples": np_,
                    "step": pi * total + step,
                    "total": np_ * total,
                })

            with _render_lock:
                _start_job(job_id)
                results = inverse_render(
                    src_path,
                    passes=GBUFFER_NAMES,
                    device=Handler.device,
                    on_pass=on_pass,
                    on_step=on_step,
                )

            for name, frames in results.items():
                Image.fromarray(frames[0]).save(os.path.join(preset_dir, f"{name}.png"))

            _send_to_job(job_id, {"phase": "done"})
            _finish_job(job_id)

            print(f"[/upload_bg] preset saved → {os.path.relpath(preset_dir)}")
            body = json.dumps({
                "name": preset_name,
                "passes": list(results.keys()),
            }).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except Exception as e:
            traceback.print_exc()
            if job_id:
                _send_to_job(job_id, {"phase": "error", "message": str(e)})
                _finish_job(job_id)
            self._send_text(500, f"Inverse render failed: {e}")

    def _handle_upload_hdr(self):
        ctype = self.headers.get("Content-Type", "")
        if not ctype.startswith("multipart/form-data"):
            self._send_text(400, "Expected multipart/form-data")
            return
        try:
            fs = cgi.FieldStorage(
                fp=self.rfile, headers=self.headers,
                environ={"REQUEST_METHOD": "POST", "CONTENT_TYPE": ctype},
            )
            if "hdr" not in fs:
                self._send_text(400, "Missing 'hdr' field")
                return
            item = fs["hdr"]
            raw = item.file.read()
            os.makedirs(HDRI_UPLOAD_DIR, exist_ok=True)
            # Sanitize filename
            fname = os.path.basename(item.filename or "uploaded.hdr")
            if not fname.lower().endswith(".hdr"):
                fname += ".hdr"
            out_path = os.path.join(HDRI_UPLOAD_DIR, fname)
            with open(out_path, "wb") as f:
                f.write(raw)
            name = os.path.splitext(fname)[0]
            print(f"[/upload_hdr] saved {fname} ({len(raw)} bytes)")
            body = json.dumps({"name": name, "path": os.path.relpath(out_path, PROJECT_ROOT)}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except Exception as e:
            traceback.print_exc()
            self._send_text(500, f"Upload failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="Diffusion renderer web demo server")
    parser.add_argument("input", nargs="?", default="samples/garment.zprj",
                        help="Path to .zprj file")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--dev", action="store_true",
                        help="Dev mode: skip build, use 'npm run dev' in web/ for frontend")
    parser.add_argument("--hdr", default=DEFAULT_HDR,
                        help="HDR environment map for forward rendering")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    web_dir = os.path.join(PROJECT_ROOT, "web")

    zprj_path = os.path.abspath(args.input)
    if not os.path.exists(zprj_path):
        print(f"Error: {zprj_path} not found")
        return

    Handler.hdr_path = os.path.abspath(args.hdr)
    Handler.device = args.device

    if args.dev:
        Handler.serve_dir = web_dir
        dest = os.path.join(web_dir, "public", "sample.zprj")
        shutil.copy2(zprj_path, dest)
        print(f"Copied {args.input} → web/public/sample.zprj")
        print("Dev mode: run 'npm run dev' in web/ for frontend with HMR")
    else:
        dist_dir = build_web_client(web_dir)
        dest = os.path.join(dist_dir, "sample.zprj")
        shutil.copy2(zprj_path, dest)
        print(f"Copied {args.input} → {os.path.relpath(dest)}")
        Handler.serve_dir = dist_dir

    print(f"HDR: {Handler.hdr_path}")
    print(f"Inverse asset dir: {INVERSE_DIR}")
    _reset_session_state()
    print(f"Cleared all BG upload dirs at startup: {INVERSE_UPLOAD_DIR}")
    server = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    print(f"Server running at http://localhost:{args.port}")
    print("Note: diffusion model will load lazily on first /render request.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.server_close()


if __name__ == "__main__":
    main()
