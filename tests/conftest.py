import logging
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from errant.annotator import Annotator  # type: ignore[import-untyped]
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ToolCallPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.usage import RequestUsage

from joel.edits import load_annotator


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture(scope="session")
def annotator() -> Annotator:
    """ERRANT with spaCy's en_core_web_sm, loaded once per test session."""
    return load_annotator()


@pytest.fixture
def restore_root_logger() -> Iterator[None]:
    """Undo configure_logging(), which rebinds the root logger to this test's stderr."""
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    yield
    root.handlers[:] = handlers
    root.setLevel(level)


@pytest.fixture
def fake_ffmpeg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Callable[[str], None]:
    """Make ``ffmpeg`` on PATH a shell script with the given body, and nothing else."""

    def install(script: str) -> None:
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir(exist_ok=True)
        ffmpeg = bin_dir / "ffmpeg"
        ffmpeg.write_text(f"#!/bin/sh\n{script}\n")
        ffmpeg.chmod(0o755)
        monkeypatch.setenv("PATH", str(bin_dir))

    return install


def user_prompt(messages: list[ModelMessage]) -> str:
    """The user prompt text in a model call's messages."""
    return "\n".join(
        str(part.content)
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, UserPromptPart)
    )


class ScriptedModel:
    """A model that gives canned structured answers, one per call, in order.

    Every call costs 100 input and 10 output tokens.
    """

    def __init__(self, answers: tuple[dict[str, object], ...]) -> None:
        self._answers = list(answers)
        self.calls: list[list[ModelMessage]] = []
        self.model = FunctionModel(self._respond)

    def _respond(self, messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        self.calls.append(messages)
        answer = self._answers.pop(0)
        return ModelResponse(
            parts=[ToolCallPart(info.output_tools[0].name, answer)],
            usage=RequestUsage(input_tokens=100, output_tokens=10),
        )

    def prompt(self, call: int = 0) -> str:
        """The user prompt text the model received on the given call."""
        return user_prompt(self.calls[call])


@pytest.fixture
def scripted_model() -> Callable[..., ScriptedModel]:
    def make(*answers: dict[str, object]) -> ScriptedModel:
        return ScriptedModel(answers)

    return make
