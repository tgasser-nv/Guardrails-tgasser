# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from abc import ABC, abstractmethod

from nemoguardrails import RailsConfig
from nemoguardrails.rails.llm.config import Model, Rails, TaskPrompt
from nemoguardrails.rails.llm.options import (
    GenerationOptions,
    GenerationResponse,
)
from nemoguardrails.streaming import StreamingHandler


class GuardrailsEngineBase(ABC):
    """Abstract Base Class for all engines.
    Designed for compatibility with a subset of LLMRails used in prototyping
    """

    def __init__(self, engine_name: str, rails_config: RailsConfig) -> None:
        """Initialize a new engine with a rails configuration object
        Unpack only the necessary fields to run input/output rails
        Use `engine_name` to select this engine from HTTP header
        """

        self.engine_name: str = engine_name
        # Use model name for easy lookups
        self.models: dict[str, Model] = {model.model: model for model in rails_config.models}
        self.rails: Rails = rails_config.rails

        # Prompts are optional in RailsConfig, but we can't call any models without them
        if not rails_config.prompts:
            raise RuntimeError(f"No prompts provided in {rails_config}")

        # Use the `task` field as easy lookup
        self.prompts: dict[str, TaskPrompt] = {prompt.task: prompt for prompt in rails_config.prompts}

    @abstractmethod
    async def generate_async(
        self,
        messages: list[dict],
        options: GenerationOptions | None = None,
        streaming_handler: StreamingHandler | None = None,
    ) -> GenerationResponse:
        """
        Generate an asynchronous response from Guardrails
        """
        pass
