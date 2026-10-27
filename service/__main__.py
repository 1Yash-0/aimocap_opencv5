"""Run the web service: ``python -m service [--host HOST] [--port PORT]``."""

from __future__ import annotations

import argparse

import uvicorn

from service.app import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="AIMoCap web service")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(create_app(), host=args.host, port=args.port, proxy_headers=True, forwarded_allow_ips="127.0.0.1")


if __name__ == "__main__":
    main()
