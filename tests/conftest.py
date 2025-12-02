# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from unittest.mock import patch

import pytest

REASONING_TRACE_MOCK_PATH = "nemoguardrails.actions.llm.generation.get_and_clear_reasoning_trace_contextvar"


@pytest.fixture(autouse=True)
def reset_reasoning_trace_var():
    """Reset reasoning_trace_var before each test to prevent state leakage."""
    from nemoguardrails.context import reasoning_trace_var

    reasoning_trace_var.set(None)
    yield
    reasoning_trace_var.set(None)


@pytest.fixture(scope="session", autouse=True)
def register_fake_provider():
    """Register the fake LLM provider for testing."""

    from langchain_core.language_models import BaseChatModel

    from nemoguardrails.llm.providers import (
        register_chat_provider,
        register_llm_provider,
    )
    from tests.utils import FakeLLM

    class FakeChatModel(BaseChatModel):
        """Fake chat model for testing that returns a simple response."""

        @property
        def _llm_type(self) -> str:
            return "fake-chat"

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            from langchain_core.messages import AIMessage
            from langchain_core.outputs import ChatGeneration, ChatResult

            message = AIMessage(content="Hello there! I'm a fake bot. How can I help you?")
            generation = ChatGeneration(message=message)
            return ChatResult(generations=[generation])

    # Register both LLM and Chat providers
    register_llm_provider("fake", FakeLLM)
    register_chat_provider("fake", FakeChatModel)

    yield

    # Clean up
    from nemoguardrails.llm.providers.providers import _chat_providers, _llm_providers

    _llm_providers.pop("fake", None)
    _chat_providers.pop("fake", None)


def pytest_configure(config):
    patch("prompt_toolkit.PromptSession", autospec=True).start()
