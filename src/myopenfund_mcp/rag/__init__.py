"""The `rag` toolset.

Importing this package (and so `rag.backend` or `rag.tools`, whichever comes first) turns the ONNX
runtime's telemetry off for the process, before anything can import the runtime: `rag_engine` loads
it, and so does `qdrant_client` itself, through `fastembed`, when that package is installed. Left
alone, the runtime of the `[rag]` extra sends usage events to its publisher over HTTPS and keeps a
device id under `$HOME/.cache/Microsoft`. `setdefault`: a value set by the operator is kept.
"""
import os

os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")
