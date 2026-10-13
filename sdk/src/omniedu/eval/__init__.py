"""Benchmark evaluation for the Omni-Edu models.

``omniedu eval`` runs the twelve benchmarks of the paper and, where a benchmark
needs one, its judge.  The modules here are the shared plumbing every benchmark
uses:

* :mod:`omniedu.eval.io`        -- JSONL reads/writes and the resume rule
* :mod:`omniedu.eval.media`     -- images -> ``data:`` URLs
* :mod:`omniedu.eval.sampling`  -- sampling knobs and the summary block
* :mod:`omniedu.eval.engine`    -- endpoint rotation, retries, bounded fan-out
* :mod:`omniedu.eval.registry`  -- the benchmark registry (``benchmarks.json``)
* :mod:`omniedu.eval.resources` -- packaged assets (prompts, keep-sets, vendor)

Benchmark-specific code lives under :mod:`omniedu.eval.benchmarks` (one module
per benchmark) and the judging pipeline under :mod:`omniedu.eval.judge`.
"""

from .client import EndpointPool, RateLimiter, normalize_base_url
from .engine import Sink, chat, endpoints, fan_out, marker_failed
from .io import append_jsonl, compact_jsonl, iter_jsonl, load_jsonl, resume
from .media import data_url, sniff_mime
from .registry import Benchmark, load_registry, resolve
from .sampling import Sampling, add_sampling_args, sampling_from_args, strip_think

__all__ = [
    "Benchmark",
    "EndpointPool",
    "RateLimiter",
    "Sampling",
    "Sink",
    "add_sampling_args",
    "append_jsonl",
    "chat",
    "compact_jsonl",
    "data_url",
    "endpoints",
    "fan_out",
    "iter_jsonl",
    "load_jsonl",
    "load_registry",
    "marker_failed",
    "normalize_base_url",
    "resolve",
    "resume",
    "sampling_from_args",
    "sniff_mime",
    "strip_think",
]
