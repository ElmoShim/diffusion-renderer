"""HTTP server for the G-buffer web viewer.

Builds the Vite app if needed, copies the sample .zprj into the serve
directory, then serves everything.

Usage:
    uv run serve.py                          # default: samples/garment.zprj
    uv run serve.py samples/250000_coat.zprj
"""

import argparse
import os
import shutil
import subprocess
from http.server import HTTPServer, SimpleHTTPRequestHandler


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
        subprocess.run(["npx", "vite", "build"], cwd=web_dir, check=True)
    return dist_dir


class Handler(SimpleHTTPRequestHandler):
    serve_dir = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=Handler.serve_dir, **kwargs)


def main():
    parser = argparse.ArgumentParser(description="G-buffer web viewer server")
    parser.add_argument("input", nargs="?", default="samples/garment.zprj",
                        help="Path to .zprj file")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--dev", action="store_true",
                        help="Dev mode: skip build, use 'npm run dev' in web/ for frontend")
    args = parser.parse_args()

    web_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")

    zprj_path = os.path.abspath(args.input)
    if not os.path.exists(zprj_path):
        print(f"Error: {zprj_path} not found")
        return

    if args.dev:
        Handler.serve_dir = web_dir
        # In dev mode, copy zprj to public/ for Vite dev server to serve
        dest = os.path.join(web_dir, "public", "sample.zprj")
        shutil.copy2(zprj_path, dest)
        print(f"Copied {args.input} → web/public/sample.zprj")
        print(f"Dev mode: run 'npm run dev' in web/ for frontend with HMR")
    else:
        dist_dir = build_web_client(web_dir)
        # Copy zprj to dist/ after build (not public/ to avoid triggering rebuilds)
        dest = os.path.join(dist_dir, "sample.zprj")
        shutil.copy2(zprj_path, dest)
        print(f"Copied {args.input} → {os.path.relpath(dest)}")
        Handler.serve_dir = dist_dir

    server = HTTPServer(("0.0.0.0", args.port), Handler)
    print(f"Server running at http://localhost:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.server_close()


if __name__ == "__main__":
    main()
