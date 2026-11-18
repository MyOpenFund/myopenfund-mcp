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


def _run(script: str, given: str | None) -> dict:
    env = {name: value for name, value in os.environ.items() if name != "ORT_DISABLE_TELEMETRY"}
    if given is not None:
        env["ORT_DISABLE_TELEMETRY"] = given
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(Path(__file__).resolve().parent.parent / "src"),
                                                      env.get("PYTHONPATH")]))
    done = subprocess.run([sys.executable, "-c", script], env=env, check=False, capture_output=True, text=True,
                          timeout=120)
    assert done.returncode == 0, done.stderr[-2000:]
    return json.loads(done.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("module", ["myopenfund_mcp.rag.backend", "myopenfund_mcp.rag.tools"])
def test_the_onnx_runtime_telemetry_is_off_before_anything_can_import_the_runtime(module):
    """The ONNX runtime sends usage events to its publisher unless ORT_DISABLE_TELEMETRY=1: the server
    sets it before `rag_engine` or `qdrant_client` (which loads fastembed, hence the runtime) is imported."""
    report = _run(_IMPORT_SPY.replace("MODULE", module), None)
    assert report["final"] == "1"
    assert {"rag_engine", "qdrant_client"} <= set(report["seen"])
    assert set(report["seen"].values()) == {"1"}, report["seen"]  # set before each of them was imported


# The same spy, but the runtime and the libraries that load it are refused: the value an operator set
# to allow the telemetry is checked without ever loading the runtime with the telemetry on.
_NO_RUNTIME = """
import builtins, json, os
real = builtins.__import__
def spy(name, *args, **kwargs):
    if name.split(".")[0] in ("onnxruntime", "fastembed", "qdrant_client", "rag_engine"):
        raise AssertionError("the runtime must not be loaded by this test: " + name)
    return real(name, *args, **kwargs)
builtins.__import__ = spy
import myopenfund_mcp.rag  # the package whose import sets the variable
print(json.dumps({"final": os.environ.get("ORT_DISABLE_TELEMETRY")}))
"""


@pytest.mark.parametrize(("given", "expected"), [("0", "0"), ("1", "1"), ("", ""), (None, "1")])
def test_a_value_set_by_the_operator_is_kept(given, expected):
    assert _run(_NO_RUNTIME, given) == {"final": expected}


def test_the_tests_themselves_run_with_the_telemetry_off():
    """`tests/conftest.py` sets the variable first, and the CI workflow sets it for the whole job."""
    root = Path(__file__).resolve().parent.parent
    statements = [line for line in (root / "tests" / "conftest.py").read_text().splitlines()
                  if line.strip() and not line.startswith("#")]
    assert statements[:2] == ["import os", 'os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")']
    assert 'ORT_DISABLE_TELEMETRY: "1"' in (root / ".github" / "workflows" / "tests.yml").read_text()
