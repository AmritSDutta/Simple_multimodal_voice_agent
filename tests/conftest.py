import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RESOURCES = Path(__file__).parent / "resources"


def image_to_base64(path: str) -> str:
    import base64
    import pathlib
    return base64.b64encode(pathlib.Path(path).read_bytes()).decode("utf-8")


@pytest.fixture
def resources_path():
    return RESOURCES


@pytest.fixture
def image_to_base64_fixture():
    return image_to_base64


@pytest.fixture
def custom_settings():
    """
    Fixture to override settings via environment variables during tests.
    Usage:
        async def test_something(custom_settings):
            custom_settings.set("REASONING_NODE_PREFERENCE", "genai")
            # ... test code using the custom setting
    """
    original_env = os.environ.copy()

    class SettingsManager:
        def set(self, key: str, value: str):
            os.environ[key] = value

        def set_multiple(self, **kwargs):
            for key, value in kwargs.items():
                os.environ[key] = value

    manager = SettingsManager()
    yield manager

    # Restore original environment after test
    os.environ.clear()
    os.environ.update(original_env)
