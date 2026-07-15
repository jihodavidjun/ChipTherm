#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark HotSpot worker counts for ChipTherm extension generation.")
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument("--samples-per-case", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", nargs="+", type=int, default=[1, 4, 8, 12, 16])
    parser.add_argument("--sample-uids", nargs="+", default=None)
    parser.add_argument("--hotspot-executable", type=Path, default=None)
    parser.add_argument("--hotspot-timeout-s", type=float, default=None)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--results-json", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    results = []
    for workers in args.workers:
        command = [
            "python3",
            "scripts/build_chiptherm_extension.py",
            "--pilot",
            "--samples-per-case",
            str(args.samples_per_case),
            "--out-root",
            str(args.out_root),
            "--seed",
            str(args.seed),
            "--run-hotspot",
            "--resume",
            "--retry-failed",
            "--max-retries",
            str(args.max_retries),
            "--hotspot-workers",
            str(workers),
        ]
        if args.hotspot_executable is not None:
            command.extend(["--hotspot-executable", str(args.hotspot_executable)])
        if args.hotspot_timeout_s is not None:
            command.extend(["--hotspot-timeout-s", str(args.hotspot_timeout_s)])
        if args.sample_uids:
            command.extend(["--sample-uids", *args.sample_uids])
        print(shlex.join(command))
        if args.dry_run:
            continue
        start = time.perf_counter()
        completed = subprocess.run(command, check=False)
        elapsed = time.perf_counter() - start
        results.append({"workers": workers, "runtime_s": elapsed, "return_code": completed.returncode})
    payload = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "results": results}
    output = args.results_json or (args.out_root / "worker_benchmark.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if all(item["return_code"] == 0 for item in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
