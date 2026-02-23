from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _git_head() -> str:
    completed = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        return "unknown"
    return completed.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate artifact manifest for reproducibility.")
    parser.add_argument("--paths", nargs="+", required=True)
    parser.add_argument("--output-json", default="artifacts/eval/manifest.json")
    args = parser.parse_args()

    records: list[dict[str, object]] = []
    for raw in args.paths:
        path = Path(raw)
        if not path.exists():
            records.append({"path": str(path), "exists": False})
            continue
        records.append(
            {
                "path": str(path),
                "exists": True,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )

    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": _git_head(),
        "records": records,
    }

    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote manifest to {output}")


if __name__ == "__main__":
    main()

