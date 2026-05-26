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
import shutil
import subprocess
import threading
import traceback
from http.server import HTTPServer, SimpleHTTPRequestHandler

import numpy as np
import torch
from PIL import Image


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
INVERSE_DIR = os.path.join(PROJECT_ROOT, "asset", "inverse")
HDRI_DIR = os.path.join(PROJECT_ROOT, "examples", "hdri")
HDRI_UPLOAD_DIR = os.path.join(PROJECT_ROOT, "output", "hdri_uploads")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "web_capture")
GBUFFER_NAMES = ["basecolor", "normal", "depth", "roughness", "metallic"]
DEFAULT_HDR = os.path.join(PROJECT_ROOT, "examples", "hdri", "sunny_vondelpark_1k.hdr")

_save_lock = threading.Lock()
_render_lock = threading.Lock()
_forward_render = None


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


def build_web_client(web_dir):
    dist_dir = os.path.join(web_dir, "dist")
    if not os.path.isdir(os.path.join(web_dir, "node_modules")):
        print("Installing web dependencies ...")
        subprocess.run(["npm", "install"], cwd=web_dir, check=True)
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
        local_vite = os.path.join(web_dir, "node_modules", ".bin", "vite")
        vite_cmd = [local_vite, "build"] if os.path.isfile(local_vite) else ["npx", "vite", "build"]
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
        # Serve /inverse/* from asset/inverse/
        if self.path.startswith("/inverse/"):
            rel = self.path[len("/inverse/"):].split("?")[0]
            file_path = os.path.normpath(os.path.join(INVERSE_DIR, rel))
            if not file_path.startswith(INVERSE_DIR):
                self._send_text(403, "Forbidden")
                return
            return self._serve_file(file_path, "image/png")

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

    def do_POST(self):
        if self.path == "/render":
            return self._handle_render()
        if self.path == "/upload_hdr":
            return self._handle_upload_hdr()
        self._send_text(404, "Not found")

    def _handle_render(self):
        ctype = self.headers.get("Content-Type", "")
        if not ctype.startswith("multipart/form-data"):
            self._send_text(400, "Expected multipart/form-data")
            return
        try:
            fs = cgi.FieldStorage(
                fp=self.rfile, headers=self.headers,
                environ={"REQUEST_METHOD": "POST", "CONTENT_TYPE": ctype},
            )
            hdr_name = fs.getfirst("hdr", "")
            bg_preset = fs.getfirst("bg_preset", "")
            mode = fs.getfirst("mode", "still")
            if mode not in ("still", "rotate_light"):
                self._send_text(400, f"Unsupported mode: {mode}")
                return

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = os.path.join(OUTPUT_DIR, timestamp)
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

            print(f"[/render] inputs saved → {os.path.relpath(save_dir)} ({w}x{h}) mode={mode} hdr={hdr_name!r} drop={drop_conds}")
            print("[/render] running diffusion forward render...")

            # Run forward render (serialize — GPU is shared)
            forward_render = _get_forward_render()
            frames_holder = {"frames": None}
            def on_sample(_i, _seed, frames):
                frames_holder["frames"] = frames

            with _render_lock:
                forward_render(
                    gbuffers, hdr_path, device=Handler.device,
                    rotate_light=(mode == "rotate_light"),
                    num_samples=1,
                    on_sample=on_sample,
                    drop_conds=drop_conds or None,
                )

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
            self._send_text(500, f"Render failed: {e}")

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
    server = HTTPServer(("0.0.0.0", args.port), Handler)
    print(f"Server running at http://localhost:{args.port}")
    print("Note: diffusion model will load lazily on first /render request.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.server_close()


if __name__ == "__main__":
    main()
