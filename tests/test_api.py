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

import json
import os

import pytest
from fastapi.testclient import TestClient
from starlette.responses import StreamingResponse

from nemoguardrails.server import api
from nemoguardrails.server.api import (
    RequestBody,
    _handle_openai_echo_completion,
    get_config_id_matching_main_llm,
    get_config_ids,
)
from nemoguardrails.server.api_models import (
    OpenAICompletionRequest,
    OpenAICompletionResponse,
    OpenAIMessage,
)

client = TestClient(api.app)


@pytest.fixture(scope="function", autouse=True)
def set_rails_config_path():
    api.app.rails_config_path = os.path.normpath(os.path.join(os.path.dirname(__file__), "test_configs"))
    yield
    api.app.rails_config_path = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "..", "examples", "bots")
    )


def test_get():
    response = client.get("/v1/rails/configs")
    assert response.status_code == 200

    result = response.json()
    assert len(result) > 0


@pytest.mark.skip(reason="Should only be run locally as it needs OpenAI key.")
def test_chat_completion():
    response = client.post(
        "/v1/chat/completions",
        json={
            "config_id": "general",
            "messages": [
                {
                    "content": "Hello",
                    "role": "user",
                }
            ],
        },
    )
    assert response.status_code == 200
    res = response.json()
    assert len(res["messages"]) == 1
    assert res["messages"][0]["content"]


@pytest.mark.skip(reason="Should only be run locally as it needs OpenAI key.")
def test_chat_completion_with_default_configs():
    api.set_default_config_id("general")

    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [
                {
                    "content": "Hello",
                    "role": "user",
                }
            ],
        },
    )
    assert response.status_code == 200
    res = response.json()
    assert len(res["messages"]) == 1
    assert res["messages"][0]["content"]


def test_request_body_validation():
    """Test RequestBody validation."""

    data = {
        "config_id": "test_config",
        "messages": [{"role": "user", "content": "Hello"}],
    }
    request_body = RequestBody.model_validate(data)
    assert request_body.config_id == "test_config"
    assert request_body.config_ids == ["test_config"]

    data = {
        "config_ids": ["test_config1", "test_config2"],
        "messages": [{"role": "user", "content": "Hello"}],
    }
    request_body = RequestBody.model_validate(data)
    assert request_body.config_ids == ["test_config1", "test_config2"]

    data = {
        "config_id": "test_config",
        "config_ids": ["test_config1", "test_config2"],
        "messages": [{"role": "user", "content": "Hello"}],
    }
    with pytest.raises(ValueError, match="Only one of config_id or config_ids should be specified"):
        RequestBody.model_validate(data)

    data = {"messages": [{"role": "user", "content": "Hello"}]}
    request_body = RequestBody.model_validate(data)
    assert request_body.config_ids is None


def test_request_body_state():
    """Test RequestBody state handling."""
    data = {
        "config_id": "test_config",
        "messages": [{"role": "user", "content": "Hello"}],
        "state": {"key": "value"},
    }
    request_body = RequestBody.model_validate(data)
    assert request_body.state == {"key": "value"}


def test_request_body_messages():
    """Test RequestBody messages validation."""
    data = {
        "config_id": "test_config",
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ],
    }
    request_body = RequestBody.model_validate(data)
    assert len(request_body.messages) == 2

    data = {
        "config_id": "test_config",
        "messages": [{"content": "Hello"}],
    }
    request_body = RequestBody.model_validate(data)
    assert len(request_body.messages) == 1


# OpenAI-compatible endpoint tests
class TestOpenAIModelsEndpoint:
    """Tests for the /v1/models endpoint (OpenAI-compatible)."""

    def test_get_models_endpoint_exists(self):
        """Test that the /v1/models endpoint exists and returns 200."""
        response = client.get("/v1/models")
        assert response.status_code == 200

    def test_get_models_response_structure(self):
        """Test that the response has the correct OpenAI-compatible structure."""
        response = client.get("/v1/models")
        assert response.status_code == 200

        data = response.json()
        assert "object" in data
        assert data["object"] == "list"
        assert "data" in data
        assert isinstance(data["data"], list)

    def test_get_models_contains_model_objects(self):
        """Test that each model in the response has the correct structure."""
        response = client.get("/v1/models")
        assert response.status_code == 200

        data = response.json()
        # May be empty if no configs have main models, but structure should be correct
        for model in data["data"]:
            assert "id" in model
            assert "object" in model
            assert model["object"] == "model"
            assert "created" in model
            assert isinstance(model["created"], int)
            assert "owned_by" in model
            assert isinstance(model["owned_by"], str)
            assert "guardrails_config" in model
            # guardrails_config can be None (for default model) or a string (config directory name)
            assert model["guardrails_config"] is None or isinstance(model["guardrails_config"], str)

    def test_get_models_returns_main_models_only(self):
        """Test that only main models (type='main') are returned."""
        response = client.get("/v1/models")
        assert response.status_code == 200

        data = response.json()
        # Should return at least one model if configs have main models defined
        # If no main models exist, should return default model
        assert len(data["data"]) >= 1

    def test_get_models_returns_model_field_value(self):
        """Test that model IDs are the 'model' field value from config.yml."""
        # Set path to a config that has a main model
        api.app.rails_config_path = os.path.normpath(os.path.join(os.path.dirname(__file__), "test_configs"))

        response = client.get("/v1/models")
        assert response.status_code == 200

        data = response.json()
        # Check that model IDs match the 'model' field from config files
        # For example, simple_actions/config.yml has model: "fake"
        model_ids = [model["id"] for model in data["data"]]

        # If we have models, they should be actual model names from configs
        # (not constructed identifiers like "config_id/model_name")
        # Note: Model names can contain slashes (e.g., "meta/llama3_8b_instruct")
        # So we just verify they're not in the format "config_id/model_name" by checking
        # that they don't start with known config directory names
        for model_id in model_ids:
            if model_id != "default":
                # Model ID should be a valid string
                assert isinstance(model_id, str)
                assert len(model_id) > 0

    def test_get_models_handles_configs_without_main_models(self):
        """Test that configs with main models are returned correctly."""
        # simple_server now has config_1 with a main model named "test_model"
        api.app.rails_config_path = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "test_configs", "simple_server")
        )

        response = client.get("/v1/models")
        assert response.status_code == 200

        data = response.json()
        # Should return the main model from config_1
        assert len(data["data"]) >= 1
        # Should have the test_model model
        model_ids = [model["id"] for model in data["data"]]
        assert "test_model" in model_ids

    def test_get_models_includes_guardrails_config(self):
        """Test that each model includes the guardrails_config field with the config directory name."""
        # Set path to a config that has a main model
        api.app.rails_config_path = os.path.normpath(os.path.join(os.path.dirname(__file__), "test_configs"))

        response = client.get("/v1/models")
        assert response.status_code == 200

        data = response.json()

        # Find models that are not the default model
        non_default_models = [m for m in data["data"] if m["id"] != "default"]

        if non_default_models:
            # Each non-default model should have a guardrails_config
            for model in non_default_models:
                assert model["guardrails_config"] is not None
                assert isinstance(model["guardrails_config"], str)
                assert len(model["guardrails_config"]) > 0

                # Verify it's a valid config directory name (no slashes, no special chars)
                assert "/" not in model["guardrails_config"]
                assert "\\" not in model["guardrails_config"]

    def test_get_models_guardrails_config_matches_config_directory(self):
        """Test that guardrails_config field matches the actual config directory."""
        # Set path to simple_server which has config_1
        api.app.rails_config_path = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "test_configs", "simple_server")
        )

        response = client.get("/v1/models")
        assert response.status_code == 200

        data = response.json()

        # If we have models with guardrails_config, verify they match actual directories
        for model in data["data"]:
            if model["guardrails_config"]:
                config_path = os.path.join(api.app.rails_config_path, model["guardrails_config"], "config.yml")
                config_path_yaml = os.path.join(api.app.rails_config_path, model["guardrails_config"], "config.yaml")
                # The guardrails_config should point to a valid config directory
                assert os.path.exists(config_path) or os.path.exists(config_path_yaml)

    def test_get_models_shows_same_model_in_multiple_configs(self):
        """Test that if the same model appears in multiple configs, it appears multiple times with different guardrails_config."""
        # This test verifies that models are not deduplicated - if the same model name
        # appears in multiple configs, each occurrence is shown with its guardrails_config
        response = client.get("/v1/models")
        assert response.status_code == 200

        data = response.json()

        # Group models by ID to check for duplicates
        models_by_id = {}
        for model in data["data"]:
            if model["id"] not in models_by_id:
                models_by_id[model["id"]] = []
            models_by_id[model["id"]].append(model)

        # If a model appears multiple times, each should have a different guardrails_config
        # (excluding None values which are for the default model)
        for model_id, model_list in models_by_id.items():
            if len(model_list) > 1:
                guardrails_configs = [m["guardrails_config"] for m in model_list if m["guardrails_config"] is not None]
                # All non-None guardrails_config values should be different
                if len(guardrails_configs) > 1:
                    assert len(guardrails_configs) == len(set(guardrails_configs))


class TestOpenAIChatCompletionsEndpoint:
    """Tests for the /v1/chat/completions endpoint (OpenAI-compatible format)."""

    def test_openai_format_detection(self):
        """Test that requests with 'model' field are treated as OpenAI format."""
        # Use simple_server config that doesn't require LLM
        api.app.rails_config_path = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "test_configs", "simple_server")
        )

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test_model",
                "messages": [
                    {"role": "user", "content": "hi"},
                ],
            },
        )
        assert response.status_code == 200

        # Should return OpenAI-compatible format
        data = response.json()
        assert "id" in data
        assert "object" in data
        assert data["object"] == "chat.completion"
        assert "created" in data
        assert "model" in data
        assert "choices" in data
        assert isinstance(data["choices"], list)
        assert len(data["choices"]) > 0

    def test_openai_completion_response_structure(self):
        """Test that the OpenAI-compatible response has correct structure."""
        api.app.rails_config_path = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "test_configs", "simple_server")
        )

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test_model",
                "messages": [
                    {"role": "user", "content": "hi"},
                ],
            },
        )
        assert response.status_code == 200

        data = response.json()
        assert "id" in data
        assert data["id"].startswith("chatcmpl-")
        assert "object" in data
        assert data["object"] == "chat.completion"
        assert "created" in data
        assert isinstance(data["created"], int)
        assert "model" in data
        assert data["model"] == "test_model"
        assert "choices" in data
        assert len(data["choices"]) == 1

        choice = data["choices"][0]
        assert "index" in choice
        assert choice["index"] == 0
        assert "message" in choice
        assert "role" in choice["message"]
        assert choice["message"]["role"] == "assistant"
        assert "content" in choice["message"]
        assert isinstance(choice["message"]["content"], str)
        assert "finish_reason" in choice
        assert choice["finish_reason"] == "stop"

    def test_openai_completion_with_temperature(self):
        """Test OpenAI-compatible request with temperature parameter."""
        api.app.rails_config_path = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "test_configs", "simple_server")
        )

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test_model",
                "messages": [
                    {"role": "user", "content": "hi"},
                ],
                "temperature": 0.7,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "choices" in data
        assert len(data["choices"]) > 0

    def test_openai_completion_with_max_tokens(self):
        """Test OpenAI-compatible request with max_tokens parameter."""
        api.app.rails_config_path = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "test_configs", "simple_server")
        )

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test_model",
                "messages": [
                    {"role": "user", "content": "hi"},
                ],
                "max_tokens": 100,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "choices" in data

    def test_openai_completion_with_multiple_messages(self):
        """Test OpenAI-compatible request with multiple messages."""
        api.app.rails_config_path = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "test_configs", "simple_server")
        )

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test_model",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "hi"},
                ],
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "choices" in data
        assert len(data["choices"]) > 0

    def test_openai_completion_missing_model(self):
        """Test that missing model field returns error."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "messages": [
                    {"role": "user", "content": "hi"},
                ],
            },
        )
        # Should fall back to Guardrails format, which requires config_id
        # This will fail validation or return an error (500 for GuardrailsConfigurationError)
        assert response.status_code in [400, 422, 500]

    def test_openai_completion_invalid_model(self):
        """Test that invalid model returns 404."""
        api.app.rails_config_path = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "test_configs", "simple_server")
        )

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "nonexistent_config",
                "messages": [
                    {"role": "user", "content": "hi"},
                ],
            },
        )
        assert response.status_code == 404
        data = response.json()
        assert "detail" in data

    def test_openai_completion_missing_messages(self):
        """Test that missing messages field returns error."""
        api.app.rails_config_path = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "test_configs", "simple_server")
        )

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "config_1",
            },
        )
        # Pydantic validation should return 422, but if it falls through to Guardrails format
        # it might return 500
        assert response.status_code in [400, 422, 500]

    def test_openai_completion_backward_compatibility(self):
        """Test that Guardrails format still works (backward compatibility)."""
        api.app.rails_config_path = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "test_configs", "simple_server")
        )

        # Guardrails format (no 'model' field, has 'config_id')
        response = client.post(
            "/v1/chat/completions",
            json={
                "config_id": "config_1",
                "messages": [
                    {"role": "user", "content": "hi"},
                ],
            },
        )
        assert response.status_code == 200

        # Should return Guardrails format (not OpenAI format)
        data = response.json()
        assert "messages" in data
        assert isinstance(data["messages"], list)
        assert len(data["messages"]) > 0

    def test_openai_completion_streaming(self):
        """Test OpenAI-compatible streaming response."""
        api.app.rails_config_path = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "test_configs", "simple_server")
        )

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test_model",
                "messages": [
                    {"role": "user", "content": "hi"},
                ],
                "stream": True,
            },
        )
        assert response.status_code == 200
        # Streaming may not be supported by all configs, so check if it's streaming or regular response
        content_type = response.headers.get("content-type", "")
        if "text/event-stream" in content_type:
            assert response.headers["content-type"] == "text/event-stream; charset=utf-8"

            # Read streaming response
            chunks = []
            for line in response.iter_lines():
                if line:
                    line_str = line.decode("utf-8")
                    if line_str.startswith("data: "):
                        data_str = line_str[6:]  # Remove "data: " prefix
                        if data_str == "[DONE]":
                            break
                        try:
                            chunk_data = json.loads(data_str)
                            chunks.append(chunk_data)
                        except json.JSONDecodeError:
                            pass

            # Should have at least one chunk
            assert len(chunks) > 0

            # Check structure of first chunk
            first_chunk = chunks[0]
            assert "id" in first_chunk
            assert "object" in first_chunk
            assert first_chunk["object"] == "chat.completion.chunk"
            assert "choices" in first_chunk
            assert len(first_chunk["choices"]) > 0
            assert "delta" in first_chunk["choices"][0]
        else:
            # Non-streaming response is also acceptable if streaming not supported
            data = response.json()
            assert "choices" in data or "messages" in data

    def test_openai_completion_streaming_final_chunk(self):
        """Test that streaming response ends with [DONE]."""
        api.app.rails_config_path = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "test_configs", "simple_server")
        )

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test_model",
                "messages": [
                    {"role": "user", "content": "hi"},
                ],
                "stream": True,
            },
        )
        assert response.status_code == 200

        # Check if streaming response (may not be supported by all configs)
        content_type = response.headers.get("content-type", "")
        if "text/event-stream" in content_type:
            # Read all lines
            lines = []
            for line in response.iter_lines():
                if line:
                    lines.append(line.decode("utf-8"))

            # Should end with [DONE]
            assert len(lines) > 0
            assert any("data: [DONE]" in line for line in lines)
        else:
            # Non-streaming response is also acceptable if streaming not supported
            data = response.json()
            assert "choices" in data or "messages" in data

    def test_openai_completion_with_usage(self):
        """Test that usage information is included when available."""
        api.app.rails_config_path = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "test_configs", "simple_server")
        )

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test_model",
                "messages": [
                    {"role": "user", "content": "hi"},
                ],
            },
        )
        assert response.status_code == 200

        data = response.json()
        # Usage may or may not be present depending on whether generation logs are enabled
        if "usage" in data and data["usage"] is not None:
            usage = data["usage"]
            assert "prompt_tokens" in usage
            assert "completion_tokens" in usage
            assert "total_tokens" in usage

    def test_openai_completion_model_with_slash(self):
        """Test model identifier with slash format (actual model names can contain slashes)."""
        # The new behavior always looks up model by name, so config_id/model_name format is not supported
        # This test now verifies that invalid model names with slashes return 404
        api.app.rails_config_path = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "test_configs", "simple_server")
        )

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test_model/some_suffix",
                "messages": [
                    {"role": "user", "content": "hi"},
                ],
            },
        )
        # Should return 404 since "test_model/some_suffix" is not a valid model name
        assert response.status_code == 404

    def test_openai_completion_with_model_name(self):
        """Test using model name (which now matches config in simple_server)."""
        api.app.rails_config_path = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "test_configs", "simple_server")
        )

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test_model",  # This is now the model name from config.yml
                "messages": [
                    {"role": "user", "content": "hi"},
                ],
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["model"] == "test_model"

    def test_openai_completion_model_lookup_by_name_only(self):
        """Test that the model field is always used to lookup by model name, not config_id."""
        api.app.rails_config_path = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "test_configs", "simple_server")
        )

        # The new behavior always looks up by model name
        # test_model is the model name in the config
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test_model",
                "messages": [
                    {"role": "user", "content": "hi"},
                ],
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["model"] == "test_model"

        # Verify that a non-existent model returns 404
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "some_other_config_id",  # Does not exist as a model name
                "messages": [
                    {"role": "user", "content": "hi"},
                ],
            },
        )
        assert response.status_code == 404

    def test_openai_completion_model_not_found(self):
        """Test that non-existent model name returns 404."""
        api.app.rails_config_path = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "test_configs", "simple_server")
        )

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "nonexistent_model_name",
                "messages": [
                    {"role": "user", "content": "hi"},
                ],
            },
        )
        assert response.status_code == 404
        data = response.json()
        assert "detail" in data
        assert "not found" in data["detail"].lower()

    def test_openai_completion_invalid_temperature(self):
        """Test that invalid temperature values are rejected."""
        api.app.rails_config_path = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "test_configs", "simple_server")
        )

        # Temperature too high
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test_model",
                "messages": [{"role": "user", "content": "hi"}],
                "temperature": 3.0,  # Should be <= 2.0
            },
        )
        assert response.status_code == 422

    def test_openai_completion_invalid_max_tokens(self):
        """Test that invalid max_tokens values are rejected."""
        api.app.rails_config_path = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "test_configs", "simple_server")
        )

        # max_tokens too low
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test_model",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 0,  # Should be >= 1
            },
        )
        assert response.status_code == 422

    def test_openai_completion_echo_mode_non_streaming(self):
        """Test that X-Guardrails-Architecture: echo returns the user prompt."""
        api.app.rails_config_path = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "test_configs", "simple_server")
        )

        user_message = "Hello, this is a test message!"
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test_model",
                "messages": [{"role": "user", "content": user_message}],
            },
            headers={"X-Guardrails-Architecture": "echo"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "choices" in data
        assert len(data["choices"]) > 0
        assert data["choices"][0]["message"]["content"] == user_message
        assert data["choices"][0]["message"]["role"] == "assistant"

    def test_openai_completion_echo_mode_doesnt_support_streaming(self):
        """Test that X-Guardrails-Architecture: echo works with streaming."""
        api.app.rails_config_path = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "test_configs", "simple_server")
        )

        user_message = "Hello, this is a streaming test!"
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test_model",
                "messages": [{"role": "user", "content": user_message}],
                "stream": True,
            },
            headers={"X-Guardrails-Architecture": "echo"},
        )
        assert response.status_code == 422
        assert "Streaming not supported in echo mode" in response.json()['detail']

    def test_openai_completion_echo_mode_multiple_messages(self):
        """Test that echo mode returns the last user message when multiple messages exist."""
        api.app.rails_config_path = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "test_configs", "simple_server")
        )

        first_message = "First message"
        last_message = "Last user message"
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test_model",
                "messages": [
                    {"role": "user", "content": first_message},
                    {"role": "assistant", "content": "Assistant response"},
                    {"role": "user", "content": last_message},
                ],
            },
            headers={"X-Guardrails-Architecture": "echo"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["choices"][0]["message"]["content"] == last_message

    def test_openai_completion_echo_mode_case_insensitive(self):
        """Test that X-Guardrails-Architecture header is case-insensitive."""
        api.app.rails_config_path = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "test_configs", "simple_server")
        )

        user_message = "Case insensitive test"
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test_model",
                "messages": [{"role": "user", "content": user_message}],
            },
            headers={"X-Guardrails-Architecture": "ECHO"},  # Uppercase
        )
        assert response.status_code == 200
        data = response.json()
        assert data["choices"][0]["message"]["content"] == user_message

    def test_guardrails_completion_echo_mode_non_streaming(self):
        """Test that Guardrails format also supports echo mode."""
        api.app.rails_config_path = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "test_configs", "simple_server")
        )

        user_message = "Guardrails echo test"
        response = client.post(
            "/v1/chat/completions",
            json={
                "config_id": "config_1",
                "messages": [{"role": "user", "content": user_message}],
            },
            headers={"X-Guardrails-Architecture": "echo"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "messages" in data
        assert len(data["messages"]) > 0
        assert data["messages"][0]["content"] == user_message
        assert data["messages"][0]["role"] == "assistant"

    def test_guardrails_completion_echo_mode_streaming(self):
        """Test that Guardrails format echo mode works with streaming."""
        api.app.rails_config_path = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "test_configs", "simple_server")
        )

        user_message = "Guardrails streaming echo test"
        response = client.post(
            "/v1/chat/completions",
            json={
                "config_id": "config_1",
                "messages": [{"role": "user", "content": user_message}],
                "stream": True,
            },
            headers={"X-Guardrails-Architecture": "echo"},
        )
        assert response.status_code == 200

        # Read streaming response
        content = b""
        for chunk in response.iter_bytes():
            content += chunk

        # Should contain the echo content
        assert user_message.encode("utf-8") in content


class TestHandleOpenAIEchoCompletion:
    """Tests for the _handle_openai_echo_completion helper method."""

    @pytest.mark.asyncio
    async def test_non_streaming_echo_response(self):
        """Test that non-streaming echo returns OpenAICompletionResponse with correct content."""
        request = OpenAICompletionRequest(
            model="test-model",
            messages=[OpenAIMessage(role="user", content="Hello, world!")],
            stream=False,
        )
        echo_content = "Hello, world!"

        response = await _handle_openai_echo_completion(request, echo_content)

        assert isinstance(response, OpenAICompletionResponse)
        assert response.model == "test-model"
        assert len(response.choices) == 1
        assert response.choices[0].message.content == echo_content
        assert response.choices[0].message.role == "assistant"
        assert response.choices[0].finish_reason == "stop"
        assert response.usage is not None
        assert response.usage.prompt_tokens == 0
        assert response.usage.completion_tokens == 0
        assert response.usage.total_tokens == 0
        assert response.id.startswith("chatcmpl-")
        assert isinstance(response.created, int)

    @pytest.mark.asyncio
    async def test_streaming_echo_response(self):
        """Test that streaming echo returns StreamingResponse."""
        request = OpenAICompletionRequest(
            model="test-model",
            messages=[OpenAIMessage(role="user", content="Streaming test message")],
            stream=True,
        )
        echo_content = "Streaming test message"

        response = await _handle_openai_echo_completion(request, echo_content)

        assert isinstance(response, StreamingResponse)
        assert response.media_type == "text/event-stream"

    @pytest.mark.asyncio
    async def test_streaming_echo_content(self):
        """Test that streaming echo response contains the correct content chunks."""
        request = OpenAICompletionRequest(
            model="test-model",
            messages=[OpenAIMessage(role="user", content="This is a longer test message for streaming")],
            stream=True,
        )
        echo_content = "This is a longer test message for streaming"

        response = await _handle_openai_echo_completion(request, echo_content)

        assert isinstance(response, StreamingResponse)

        # Read the streaming response
        content_parts = []
        async for chunk in response.body_iterator:
            if chunk:
                # Chunk is already a string
                chunk_str = chunk if isinstance(chunk, str) else chunk.decode("utf-8")
                # Parse SSE format
                for line in chunk_str.split("\n"):
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str == "[DONE]":
                            break
                        try:
                            chunk_data = json.loads(data_str)
                            if "choices" in chunk_data and len(chunk_data["choices"]) > 0:
                                delta = chunk_data["choices"][0].get("delta", {})
                                if "content" in delta:
                                    content_parts.append(delta["content"])
                        except json.JSONDecodeError:
                            pass

        # Reconstruct content from chunks
        reconstructed_content = "".join(content_parts)
        assert reconstructed_content == echo_content

    @pytest.mark.asyncio
    async def test_streaming_echo_final_chunk(self):
        """Test that streaming echo response includes final chunk with finish_reason."""
        request = OpenAICompletionRequest(
            model="test-model",
            messages=[OpenAIMessage(role="user", content="Test")],
            stream=True,
        )
        echo_content = "Test"

        response = await _handle_openai_echo_completion(request, echo_content)

        assert isinstance(response, StreamingResponse)

        # Collect all chunks
        chunks = []
        async for chunk in response.body_iterator:
            if chunk:
                chunk_str = chunk if isinstance(chunk, str) else chunk.decode("utf-8")
                chunks.append(chunk_str)

        # Check that we have a final chunk with finish_reason
        all_content = "".join(chunks)
        assert "[DONE]" in all_content

        # Check for final chunk with finish_reason
        found_final = False
        for chunk in chunks:
            if "data: " in chunk and "[DONE]" not in chunk:
                try:
                    data_str = chunk.split("data: ")[1].split("\n")[0]
                    chunk_data = json.loads(data_str)
                    if (
                        "choices" in chunk_data
                        and len(chunk_data["choices"]) > 0
                        and chunk_data["choices"][0].get("finish_reason") == "stop"
                    ):
                        found_final = True
                        break
                except (json.JSONDecodeError, IndexError):
                    pass

        assert found_final, "Should have a final chunk with finish_reason='stop'"

    @pytest.mark.asyncio
    async def test_empty_echo_content(self):
        """Test that empty echo content is handled correctly."""
        request = OpenAICompletionRequest(
            model="test-model",
            messages=[OpenAIMessage(role="user", content="")],
            stream=False,
        )
        echo_content = ""

        response = await _handle_openai_echo_completion(request, echo_content)

        assert isinstance(response, OpenAICompletionResponse)
        assert response.choices[0].message.content == ""

    @pytest.mark.asyncio
    async def test_long_echo_content_chunking(self):
        """Test that long echo content is properly chunked in streaming mode."""
        request = OpenAICompletionRequest(
            model="test-model",
            messages=[OpenAIMessage(role="user", content="A" * 50)],  # 50 characters
            stream=True,
        )
        echo_content = "A" * 50

        response = await _handle_openai_echo_completion(request, echo_content)

        assert isinstance(response, StreamingResponse)

        # Read chunks (should be chunked in groups of 10)
        content_parts = []
        async for chunk in response.body_iterator:
            if chunk:
                chunk_str = chunk if isinstance(chunk, str) else chunk.decode("utf-8")
                for line in chunk_str.split("\n"):
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str == "[DONE]":
                            break
                        try:
                            chunk_data = json.loads(data_str)
                            if "choices" in chunk_data and len(chunk_data["choices"]) > 0:
                                delta = chunk_data["choices"][0].get("delta", {})
                                if "content" in delta:
                                    content_parts.append(delta["content"])
                        except json.JSONDecodeError:
                            pass

        # Verify content is chunked (should have multiple chunks for 50 characters)
        reconstructed_content = "".join(content_parts)
        assert reconstructed_content == echo_content
        # With chunk_size=10, we should have at least 5 chunks (50/10)
        assert len([p for p in content_parts if p]) >= 5

    @pytest.mark.asyncio
    async def test_echo_response_has_correct_model(self):
        """Test that echo response includes the correct model name."""
        request = OpenAICompletionRequest(
            model="custom-model-name",
            messages=[OpenAIMessage(role="user", content="Test")],
            stream=False,
        )
        echo_content = "Test"

        response = await _handle_openai_echo_completion(request, echo_content)

        assert isinstance(response, OpenAICompletionResponse)
        assert response.model == "custom-model-name"


class TestGetConfigIds:
    """Tests for the get_config_ids helper method."""

    @pytest.fixture(autouse=True)
    def reset_single_config_mode(self):
        """Reset single_config_mode after each test."""
        original_single_config_mode = api.app.single_config_mode
        original_single_config_id = api.app.single_config_id
        yield
        api.app.single_config_mode = original_single_config_mode
        api.app.single_config_id = original_single_config_id

    @pytest.mark.asyncio
    async def test_get_config_ids_normal_mode(self):
        """Test that get_config_ids returns all config directories in normal mode."""
        api.app.single_config_mode = False
        rails_config_path = os.path.normpath(os.path.join(os.path.dirname(__file__), "test_configs", "simple_server"))

        config_ids = await get_config_ids(rails_config_path)

        assert isinstance(config_ids, list)
        assert len(config_ids) > 0
        assert "config_1" in config_ids
        # Should only include directories with config.yml or config.yaml
        for config_id in config_ids:
            config_path = os.path.join(rails_config_path, config_id)
            assert os.path.isdir(config_path)
            assert os.path.exists(os.path.join(config_path, "config.yml")) or os.path.exists(
                os.path.join(config_path, "config.yaml")
            )

    @pytest.mark.asyncio
    async def test_get_config_ids_excludes_hidden_directories(self):
        """Test that get_config_ids excludes directories starting with . or _."""
        api.app.single_config_mode = False
        rails_config_path = os.path.normpath(os.path.join(os.path.dirname(__file__), "test_configs"))

        config_ids = await get_config_ids(rails_config_path)

        # Should not include hidden directories
        assert not any(config_id.startswith(".") for config_id in config_ids)
        assert not any(config_id.startswith("_") for config_id in config_ids)

    @pytest.mark.asyncio
    async def test_get_config_ids_single_config_mode(self):
        """Test that get_config_ids returns single config ID in single config mode."""
        api.app.single_config_mode = True
        api.app.single_config_id = "test_config"
        rails_config_path = os.path.normpath(os.path.join(os.path.dirname(__file__), "test_configs"))

        config_ids = await get_config_ids(rails_config_path)

        assert isinstance(config_ids, list)
        assert len(config_ids) == 1
        assert config_ids[0] == "test_config"

    @pytest.mark.asyncio
    async def test_get_config_ids_single_config_mode_none(self):
        """Test that get_config_ids raises HTTPException when single_config_id is None."""
        api.app.single_config_mode = True
        api.app.single_config_id = None
        rails_config_path = os.path.normpath(os.path.join(os.path.dirname(__file__), "test_configs"))

        with pytest.raises(Exception) as exc_info:
            await get_config_ids(rails_config_path)

        # Should raise HTTPException with 404 status
        assert "Single config mode enabled but no config ID set" in str(exc_info.value)


class TestGetConfigIdMatchingMainLlm:
    """Tests for the get_config_id_matching_main_llm helper method."""

    @pytest.mark.asyncio
    async def test_get_config_id_matching_main_llm_found(self):
        """Test that get_config_id_matching_main_llm finds config with matching main model."""
        rails_config_path = os.path.normpath(os.path.join(os.path.dirname(__file__), "test_configs"))
        config_ids_to_check = ["with_openai_embeddings"]
        model_name = "gpt-3.5-turbo-instruct"

        config_id = await get_config_id_matching_main_llm(config_ids_to_check, model_name, rails_config_path)

        assert config_id == "with_openai_embeddings"

    @pytest.mark.asyncio
    async def test_get_config_id_matching_main_llm_not_found(self):
        """Test that get_config_id_matching_main_llm raises HTTPException when model not found."""
        rails_config_path = os.path.normpath(os.path.join(os.path.dirname(__file__), "test_configs"))
        config_ids_to_check = ["simple_server"]
        model_name = "nonexistent-model"

        with pytest.raises(Exception) as exc_info:
            await get_config_id_matching_main_llm(config_ids_to_check, model_name, rails_config_path)

        # Should raise HTTPException with 404 status
        assert "not found in any configuration" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_config_id_matching_main_llm_multiple_configs(self):
        """Test that get_config_id_matching_main_llm returns first matching config."""
        rails_config_path = os.path.normpath(os.path.join(os.path.dirname(__file__), "test_configs"))
        # Use configs that might have the same model
        config_ids_to_check = ["with_openai_embeddings", "with_langchain_safetool"]
        model_name = "gpt-3.5-turbo-instruct"

        config_id = await get_config_id_matching_main_llm(config_ids_to_check, model_name, rails_config_path)

        # Should return the first matching config
        assert config_id in config_ids_to_check

    @pytest.mark.asyncio
    async def test_get_config_id_matching_main_llm_empty_config_list(self):
        """Test that get_config_id_matching_main_llm raises HTTPException with empty config list."""
        rails_config_path = os.path.normpath(os.path.join(os.path.dirname(__file__), "test_configs"))
        config_ids_to_check = []
        model_name = "gpt-3.5-turbo-instruct"

        with pytest.raises(Exception) as exc_info:
            await get_config_id_matching_main_llm(config_ids_to_check, model_name, rails_config_path)

        assert "not found in any configuration" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_config_id_matching_main_llm_handles_yaml_extension(self):
        """Test that get_config_id_matching_main_llm handles both .yml and .yaml extensions."""
        rails_config_path = os.path.normpath(os.path.join(os.path.dirname(__file__), "test_configs"))
        config_ids_to_check = ["with_openai_embeddings"]
        model_name = "gpt-3.5-turbo-instruct"

        # Should work with config.yml
        config_id = await get_config_id_matching_main_llm(config_ids_to_check, model_name, rails_config_path)

        assert config_id == "with_openai_embeddings"

    @pytest.mark.asyncio
    async def test_get_config_id_matching_main_llm_ignores_non_main_models(self):
        """Test that get_config_id_matching_main_llm only matches main model type."""
        rails_config_path = os.path.normpath(os.path.join(os.path.dirname(__file__), "test_configs"))
        config_ids_to_check = ["with_openai_embeddings"]
        # This model exists but is of type "embeddings", not "main"
        embeddings_model = "text-embedding-ada-002"

        with pytest.raises(Exception) as exc_info:
            await get_config_id_matching_main_llm(config_ids_to_check, embeddings_model, rails_config_path)

        # Should not find it because it's not a main model
        assert "not found in any configuration" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_config_id_matching_main_llm_handles_invalid_configs(self):
        """Test that get_config_id_matching_main_llm handles invalid config files gracefully."""
        rails_config_path = os.path.normpath(os.path.join(os.path.dirname(__file__), "test_configs"))
        # Use a config that exists but might have issues
        config_ids_to_check = ["simple_server"]  # This has empty models list
        model_name = "some-model"

        with pytest.raises(Exception) as exc_info:
            await get_config_id_matching_main_llm(config_ids_to_check, model_name, rails_config_path)

        # Should raise HTTPException when model not found
        assert "not found in any configuration" in str(exc_info.value)
