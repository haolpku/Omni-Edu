#!/usr/bin/env python3
"""Download the ErrorRadar images referenced by the pool."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

_COMMON_ROOT = Path(__file__).resolve()
while _COMMON_ROOT.name != "data_preparation":
    _COMMON_ROOT = _COMMON_ROOT.parent
if str(_COMMON_ROOT) not in sys.path:
    sys.path.insert(0, str(_COMMON_ROOT))
from common import pools

CAPABILITY = "diagnostic_reasoning"
DATASET = "ErrorRadar"
POOL = pools.cleaned_dir(CAPABILITY, DATASET) / "kept.jsonl"
OUT = pools.cleaned_dir(CAPABILITY, DATASET) / "media"
MANIFEST = pools.cleaned_dir(CAPABILITY, DATASET) / "image_manifest.jsonl"
REPO = pools.DATA_PREPARATION.parent

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--pool", type=Path, default=POOL)
parser.add_argument("--out", type=Path, default=OUT)
parser.add_argument("--manifest", type=Path, default=MANIFEST)
parser.add_argument("--workers", type=int, default=64)
args = parser.parse_args()

rows = [json.loads(line) for line in args.pool.open(encoding="utf-8") if line.strip()]
urls = sorted({row["content_image"] for row in rows if row.get("content_image")})
out = args.out
out.mkdir(parents=True, exist_ok=True)
opener = urllib.request.build_opener()


def get(url):
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with opener.open(req, timeout=10) as x:
            data = x.read()
        h = hashlib.sha256(data).hexdigest()
        ext = (
            ".png"
            if data.startswith(b"\x89PNG")
            else ".jpg" if data.startswith(b"\xff\xd8") else ".bin"
        )
        p = out / h[:2] / (h + ext)
        p.parent.mkdir(exist_ok=True)
        p.write_bytes(data)
        return {
            "url": url,
            "status": "ok",
            "sha256": h,
            "path": str(p.relative_to(REPO)),
            "bytes": len(data),
        }
    except Exception as e:
        return {"url": url, "status": "failed", "error": repr(e)}


with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
    res = list(ex.map(get, urls))
with args.manifest.open("w", encoding="utf-8") as f:
    for x in res:
        f.write(json.dumps(x, ensure_ascii=False) + "\n")
print(
    {
        "urls": len(res),
        "ok": sum(x["status"] == "ok" for x in res),
        "failed": sum(x["status"] != "ok" for x in res),
    }
)
