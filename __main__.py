"""
Entry point for ``python -m .`` or ``python __main__.py``.

Usage
-----
    python .                        # Web server (dev, auto-reload)
    python . --mode desktop         # Native desktop window
    python . --mode server          # Web server (production)
    python . --host 0.0.0.0        # Bind to all interfaces
    python . --port 9000           # Custom port
    python . --workers 4           # Multi-worker production mode

Environment variables (via .env or PMA_ prefix) override defaults.
"""

import argparse
import sys
import os

# Ensure project root on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="pma",
        description="Personal Memory Assistant – local-first RAG for your files",
    )
    p.add_argument(
        "--mode",
        choices=["server", "desktop"],
        default="server",
        help="Run as web server (default) or native desktop window",
    )
    p.add_argument("--host", default=None, help="Bind address (default: from config)")
    p.add_argument("--port", type=int, default=None, help="Port (default: from config)")
    p.add_argument("--workers", type=int, default=1, help="Worker processes (prod only, default: 1)")
    p.add_argument("--reload", action="store_true", help="Enable auto-reload (dev)")
    p.add_argument("--no-reload", dest="reload", action="store_false")
    p.set_defaults(reload=None)  # auto-detect from dev_mode
    return p.parse_args()


def run_server(args: argparse.Namespace) -> None:
    import uvicorn
    from app.config import settings

    host = args.host or settings.host
    port = args.port or settings.port
    reload = args.reload if args.reload is not None else settings.dev_mode

    uvicorn_kwargs: dict = {
        "app": "app.main:app",
        "host": host,
        "port": port,
        "log_level": settings.log_level.lower(),
    }

    if reload:
        # Dev mode: single process with hot-reload
        uvicorn_kwargs["reload"] = True
        uvicorn_kwargs["reload_dirs"] = [os.path.dirname(os.path.abspath(__file__))]
    elif args.workers > 1:
        # Production multi-worker (no reload)  
        uvicorn_kwargs["workers"] = args.workers
        uvicorn_kwargs["access_log"] = False
    else:
        # Single production process
        uvicorn_kwargs["access_log"] = False

    print(f"  PMA server -> http://{host}:{port}")
    print(f"  Mode: {'dev (reload)' if reload else f'production ({args.workers} worker(s))'}")
    print()
    uvicorn.run(**uvicorn_kwargs)


def run_desktop(args: argparse.Namespace) -> None:
    """Launch via desktop.py which handles the native window."""
    from desktop import main as desktop_main

    # Pass host/port overrides through env so desktop.py picks them up
    if args.host:
        os.environ.setdefault("PMA_HOST", args.host)
    if args.port:
        os.environ.setdefault("PMA_PORT", str(args.port))

    desktop_main()


def main() -> None:
    args = parse_args()
    if args.mode == "desktop":
        run_desktop(args)
    else:
        run_server(args)


if __name__ == "__main__":
    main()
