from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
import pytest


ROOT = Path(__file__).resolve().parents[1]
root_str = str(ROOT)
if root_str not in sys.path:
    sys.path.insert(0, root_str)


@pytest.fixture(autouse=True)
def mock_httpx_client(monkeypatch):
    mock_client = MagicMock()
    mock_client.aclose = AsyncMock()
    
    # Enable async context manager support if needed
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    
    monkeypatch.setattr("httpx.AsyncClient", MagicMock(return_value=mock_client))
    return mock_client
