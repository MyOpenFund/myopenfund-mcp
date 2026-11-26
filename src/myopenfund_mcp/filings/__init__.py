"""Filings: SEC filing documents fetched, indexed and searched on the user's own disk (no-vault mode).

Importing this package turns the ONNX runtime's telemetry off for the process, before anything can
import the runtime (`rag_engine`, and `qdrant_client` through `fastembed`), as the `rag` package
does. `setdefault`: a value set by the operator is kept.
"""

import os

os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")

#: `server_info.filings.graph` until the graph tools exist
GRAPH_NOT_AVAILABLE = "not available (OLAF after the V0)"
