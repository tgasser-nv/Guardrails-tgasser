# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""OpenAI-compatible API models for the Guardrails server."""

from typing import List, Optional

from pydantic import BaseModel, Field


# OpenAI-compatible models
class OpenAIMessage(BaseModel):
    role: str = Field(description="The role of the message (system, user, assistant, etc.)")
    content: str = Field(description="The content of the message")


class OpenAICompletionRequest(BaseModel):
    model: str = Field(description="The model to use for completion")
    messages: List[OpenAIMessage] = Field(description="The list of messages in the conversation")
    temperature: Optional[float] = Field(default=1.0, ge=0, le=2, description="Sampling temperature")
    max_tokens: Optional[int] = Field(default=None, ge=1, description="Maximum number of tokens to generate")
    stream: Optional[bool] = Field(default=False, description="Whether to stream the response")
    top_p: Optional[float] = Field(default=1.0, ge=0, le=1, description="Nucleus sampling parameter")
    frequency_penalty: Optional[float] = Field(default=0.0, ge=-2, le=2, description="Frequency penalty")
    presence_penalty: Optional[float] = Field(default=0.0, ge=-2, le=2, description="Presence penalty")
    stop: Optional[List[str]] = Field(default=None, description="Stop sequences")
    n: Optional[int] = Field(default=1, ge=1, description="Number of completions to generate")
    user: Optional[str] = Field(default=None, description="User identifier")


class OpenAIChoice(BaseModel):
    index: int = Field(description="The index of the choice")
    message: OpenAIMessage = Field(description="The message content")
    finish_reason: Optional[str] = Field(default="stop", description="Reason for finishing")


class OpenAIUsage(BaseModel):
    prompt_tokens: int = Field(default=0, description="Number of tokens in the prompt")
    completion_tokens: int = Field(default=0, description="Number of tokens in the completion")
    total_tokens: int = Field(default=0, description="Total number of tokens")


class OpenAICompletionResponse(BaseModel):
    id: str = Field(description="Unique identifier for the completion")
    object: str = Field(default="chat.completion", description="Object type")
    created: int = Field(description="Unix timestamp of creation")
    model: str = Field(description="Model used for completion")
    choices: List[OpenAIChoice] = Field(description="List of completion choices")
    usage: Optional[OpenAIUsage] = Field(default=None, description="Token usage information")


class OpenAIModel(BaseModel):
    id: str = Field(description="Model identifier")
    object: str = Field(default="model", description="Object type")
    created: int = Field(description="Unix timestamp of creation")
    owned_by: str = Field(default="nvidia", description="Owner of the model")
    guardrails_config: Optional[str] = Field(default=None, description="Guardrails configuration directory name")


class OpenAIModelsResponse(BaseModel):
    object: str = Field(default="list", description="Object type")
    data: List[OpenAIModel] = Field(description="List of available models")
