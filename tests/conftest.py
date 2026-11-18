import os

# First statement, before any import that could load the ONNX runtime (qdrant_client loads it through
# fastembed): the tests never send the runtime's usage events nor leave its device id under $HOME.
os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")

import pytest


@pytest.fixture
def anyio_backend():
    return "asyncio"
