"""The ONNX runtime of the [rag] extra sends usage events to its publisher unless ORT_DISABLE_TELEMETRY=1.
The server sets the variable for its own process before anything can import the runtime."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_IMPORT_SPY = """
import builtins, json, os, sys
seen, real = {}, builtins.__import__
def spy(name, *args, **kwargs):
    root = name.split(".")[0]
    if root in ("onnxruntime", "fastembed", "qdrant_client", "rag_engine"):
        seen.setdefault(root, os.environ.get("ORT_DISABLE_TELEMETRY"))
    return real(name, *args, **kwargs)
builtins.__import__ = spy
import myopenfund_mcp.cli  # the entry point: it must not import the runtime before the toolset does
assert not seen, seen
import MODULE
print(json.dumps({"seen": seen, "final": os.environ.get("ORT_DISABLE_TELEMETRY")}))
"""


@pytest.mark.parametrize(("module", "given", "expected"), [
    ("myopenfund_mcp.rag.backend", None, "1"),
    ("myopenfund_mcp.rag.tools", None, "1"),
    ("myopenfund_mcp.rag.backend", "0", "0"),  # the operator's choice is kept
])
def test_the_onnx_runtime_telemetry_is_off_before_anything_can_import_the_runtime(module, given, expected):
    """The ONNX runtime sends usage events to its publisher unless ORT_DISABLE_TELEMETRY=1: the server
    sets it before `rag_engine` or `qdrant_client` (which loads fastembed, hence the runtime) is imported."""
    env = {name: value for name, value in os.environ.items() if name != "ORT_DISABLE_TELEMETRY"}
    if given is not None:
        env["ORT_DISABLE_TELEMETRY"] = given
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(Path(__file__).resolve().parent.parent / "src"),
                                                      env.get("PYTHONPATH")]))
    done = subprocess.run([sys.executable, "-c", _IMPORT_SPY.replace("MODULE", module)], env=env, check=False,
                          capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr[-2000:]
    report = json.loads(done.stdout.strip().splitlines()[-1])
    assert report["final"] == expected
    assert {"rag_engine", "qdrant_client"} <= set(report["seen"])
    assert set(report["seen"].values()) == {expected}, report["seen"]  # set before each of them was imported
