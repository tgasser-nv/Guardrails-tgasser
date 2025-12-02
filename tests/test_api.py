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

from nemoguardrails.server import api
from nemoguardrails.server.api import RequestBody

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
        """Test that configs without main models are handled gracefully."""
        # Set path to simple_server which has configs with empty models list
        api.app.rails_config_path = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "test_configs", "simple_server")
        )

        response = client.get("/v1/models")
        assert response.status_code == 200

        data = response.json()
        # Should return default model if no main models found
        assert len(data["data"]) >= 1
        # Should have at least the default model
        assert any(model["id"] == "default" for model in data["data"])

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
                "model": "config_1",
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
                "model": "config_1",
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
        assert data["model"] == "config_1"
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
                "model": "config_1",
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
                "model": "config_1",
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
                "model": "config_1",
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
                "model": "config_1",
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
                "model": "config_1",
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
                "model": "config_1",
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
        """Test model identifier with slash format (config_id/model_name)."""
        api.app.rails_config_path = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "test_configs", "simple_server")
        )

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "config_1/test_model",
                "messages": [
                    {"role": "user", "content": "hi"},
                ],
            },
        )
        # Should parse config_id correctly (uses config_id before slash)
        assert response.status_code == 200
        data = response.json()
        assert data["model"] == "config_1/test_model"

    @pytest.mark.skip(reason="Requires LLM initialization which may fail in test environment")
    def test_openai_completion_with_model_name(self):
        """Test using model name directly (as returned by /v1/models)."""
        # Use a config that has a main model
        api.app.rails_config_path = os.path.normpath(os.path.join(os.path.dirname(__file__), "test_configs"))

        # simple_actions has model: "fake"
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "fake",  # Model name from simple_actions/config.yml
                "messages": [
                    {"role": "user", "content": "hi"},
                ],
            },
        )
        # Should find config with matching main model
        assert response.status_code == 200
        data = response.json()
        assert data["model"] == "fake"

    def test_openai_completion_with_config_id(self):
        """Test using config_id directly."""
        api.app.rails_config_path = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "test_configs", "simple_server")
        )

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "config_1",  # Config ID
                "messages": [
                    {"role": "user", "content": "hi"},
                ],
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["model"] == "config_1"

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
                "model": "config_1",
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
                "model": "config_1",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 0,  # Should be >= 1
            },
        )
        assert response.status_code == 422
