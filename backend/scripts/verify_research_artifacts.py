"""Verify released research artifacts against the SHA-256 manifest."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    manifest = json.loads((ROOT / "experiments/artifacts.json").read_text())
    total = 0
    for artifact in manifest["artifacts"]:
        path = ROOT / artifact["path"]
        if not path.is_file():
            raise SystemExit(f'Missing artifact: {artifact["path"]}')
        if hashlib.sha256(path.read_bytes()).hexdigest() != artifact["sha256"]:
            raise SystemExit(f'Checksum mismatch: {artifact["path"]}')
        total += path.stat().st_size
    print(f'Verified {len(manifest["artifacts"])} artifacts ({total:,} bytes).')


if __name__ == "__main__":
    main()
