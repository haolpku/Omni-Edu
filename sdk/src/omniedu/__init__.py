"""Omni-Edu model-calling toolkit.

    from omniedu import OmniEdu

    model = OmniEdu.from_pretrained("4B", base_url="http://127.0.0.1:8000/v1")
    print(model.chat(text="求解 x^2 - 5x + 6 = 0", task="solve_reasoned").text)
"""

from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _distribution_version

from .api import OmniEdu
from .tasks import Task, get_task, list_tasks, register_task
from .types import ChatResponse, GenerationParams, Message

# The installed distribution metadata is the single source of truth, so the
# version only lives in pyproject.toml.  The fallback covers running straight
# from a source checkout that was never installed.
try:
    __version__ = _distribution_version("omniedu")
except _PackageNotFoundError:
    __version__ = "0.0.0+source"

__all__ = [
    "ChatResponse",
    "GenerationParams",
    "Message",
    "OmniEdu",
    "Task",
    "__version__",
    "get_task",
    "list_tasks",
    "register_task",
]
