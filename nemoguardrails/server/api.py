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
import asyncio
import contextvars
import importlib.util
import json
import logging
import os
import re
import time
import uuid
import warnings
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Callable, List, Optional

import yaml
from fastapi import (  # type: ignore[reportMissingImports]
    FastAPI,
    HTTPException,
    Request,
)
from fastapi.middleware.cors import CORSMiddleware  # type: ignore[reportMissingImports]
from pydantic import BaseModel, Field, ValidationError, root_validator, validator
from starlette.responses import StreamingResponse  # type: ignore[reportMissingImports]
from starlette.staticfiles import StaticFiles  # type: ignore[reportMissingImports]

from nemoguardrails import LLMRails, RailsConfig, utils
from nemoguardrails.engines.scheduler_engine import SchedulerEngine
from nemoguardrails.rails.llm.options import (
    GenerationLog,
    GenerationOptions,
    GenerationResponse,
)
from nemoguardrails.server.api_models import (
    OpenAIChoice,
    OpenAICompletionRequest,
    OpenAICompletionResponse,
    OpenAIMessage,
    OpenAIModel,
    OpenAIModelsResponse,
    OpenAIUsage,
)
from nemoguardrails.server.datastore.datastore import DataStore
from nemoguardrails.streaming import StreamingHandler

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class GuardrailsApp(FastAPI):
    """Custom FastAPI subclass with additional attributes for Guardrails server."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize custom attributes
        self.default_config_id: Optional[str] = None
        self.rails_config_path: str = ""
        self.disable_chat_ui: bool = False
        self.auto_reload: bool = False
        self.stop_signal: bool = False
        self.single_config_mode: bool = False
        self.single_config_id: Optional[str] = None
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.task: Optional[asyncio.Future] = None
        self.scheduler: Optional[SchedulerEngine] = None

# The list of registered loggers. Can be used to send logs to various
# backends and storage engines.
registered_loggers: List[Callable] = []

api_description = """Guardrails Sever API."""

# The headers for each request
api_request_headers: contextvars.ContextVar = contextvars.ContextVar("headers")

# The datastore that the Server should use.
# This is currently used only for storing threads.
# TODO: refactor to wrap the FastAPI instance inside a RailsServer class
#  and get rid of all the global attributes.
datastore: Optional[DataStore] = None

ARCHITECTURES = {}

@asynccontextmanager
async def lifespan(app: GuardrailsApp):
    # Startup logic here
    """Register any additional challenges, if available at startup."""
    challenges_files = os.path.join(app.rails_config_path, "challenges.json")

    if os.path.exists(challenges_files):
        with open(challenges_files) as f:
            register_challenges(json.load(f))

    # If there is a `config.yml` in the root `app.rails_config_path`, then
    # that means we are in single config mode.
    if os.path.exists(os.path.join(app.rails_config_path, "config.yml")) or os.path.exists(
        os.path.join(app.rails_config_path, "config.yaml")
    ):
        app.single_config_mode = True
        app.single_config_id = os.path.basename(app.rails_config_path)
    else:
        # If we're not in single-config mode, we check if we have a config.py for the
        # server configuration.
        filepath = os.path.join(app.rails_config_path, "config.py")
        if os.path.exists(filepath):
            filename = os.path.basename(filepath)
            spec = importlib.util.spec_from_file_location(filename, filepath)
            if spec is not None and spec.loader is not None:
                config_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(config_module)
            else:
                config_module = None

            # If there is an `init` function, we call it with the reference to the app.
            if config_module is not None and hasattr(config_module, "init"):
                config_module.init(app)

    #
    config_path = os.path.abspath(app.rails_config_path)
    log.info("Loading config from %s", config_path)
    scheduler_rails_config = RailsConfig.from_path(config_path)
    scheduler = SchedulerEngine(scheduler_rails_config, num_workers=4)
    await scheduler.start()
    app.scheduler = scheduler


    # Finally, we register the static frontend UI serving
    if not app.disable_chat_ui:
        FRONTEND_DIR = utils.get_chat_ui_data_path("frontend")

        app.mount(
            "/",
            StaticFiles(
                directory=FRONTEND_DIR,
                html=True,
            ),
            name="chat",
        )
    else:

        @app.get("/")
        async def root_handler():
            return {"status": "ok"}

    if app.auto_reload:
        app.loop = asyncio.get_running_loop()
        # Store the future directly as task
        app.task = app.loop.run_in_executor(None, start_auto_reload_monitoring)

    yield

    # Shut down the scheduler
    await app.scheduler.stop()

    # Shutdown logic here
    if app.auto_reload:
        app.stop_signal = True
        if hasattr(app, "task") and app.task is not None:
            app.task.cancel()
        log.info("Shutting down file observer")
    else:
        pass


app = GuardrailsApp(
    title="Guardrails Server API",
    description=api_description,
    version="0.1.0",
    license_info={"name": "Apache License, Version 2.0"},
    lifespan=lifespan,
)

ENABLE_CORS = os.getenv("NEMO_GUARDRAILS_SERVER_ENABLE_CORS", "false").lower() == "true"
ALLOWED_ORIGINS = os.getenv("NEMO_GUARDRAILS_SERVER_ALLOWED_ORIGINS", "*")

if ENABLE_CORS:
    # Split origins by comma
    origins = ALLOWED_ORIGINS.split(",")

    log.info(f"CORS enabled with the following origins: {origins}")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

app.default_config_id = None

# By default, we use the rails in the examples folder
app.rails_config_path = utils.get_examples_data_path("bots")

# Weather the chat UI is enabled or not.
app.disable_chat_ui = False

# auto reload flag
app.auto_reload = False

# stop signal for observer
app.stop_signal = False

# Whether the server is pointed to a directory containing a single config.
app.single_config_mode = False
app.single_config_id = None


class RequestBody(BaseModel):
    config_id: Optional[str] = Field(
        default=os.getenv("DEFAULT_CONFIG_ID", None),
        description="The id of the configuration to be used. If not set, the default configuration will be used.",
    )
    config_ids: Optional[List[str]] = Field(
        default=None,
        description="The list of configuration ids to be used. If set, the configurations will be combined.",
        # alias="guardrails",
        validate_default=True,
    )
    thread_id: Optional[str] = Field(
        default=None,
        min_length=16,
        max_length=255,
        description="The id of an existing thread to which the messages should be added.",
    )
    messages: Optional[List[dict]] = Field(
        default=None, description="The list of messages in the current conversation."
    )
    context: Optional[dict] = Field(
        default=None,
        description="Additional context data to be added to the conversation.",
    )
    stream: Optional[bool] = Field(
        default=False,
        description="If set, partial message deltas will be sent, like in ChatGPT. "
        "Tokens will be sent as data-only server-sent events as they become "
        "available, with the stream terminated by a data: [DONE] message.",
    )
    options: GenerationOptions = Field(
        default_factory=GenerationOptions,
        description="Additional options for controlling the generation.",
    )
    state: Optional[dict] = Field(
        default=None,
        description="A state object that should be used to continue the interaction.",
    )

    @root_validator(pre=True)
    def ensure_config_id(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if data.get("config_id") is not None and data.get("config_ids") is not None:
                raise ValueError("Only one of config_id or config_ids should be specified")
            if data.get("config_id") is None and data.get("config_ids") is not None:
                data["config_id"] = None
            if data.get("config_id") is None and data.get("config_ids") is None:
                warnings.warn("No config_id or config_ids provided, using default config_id")
        return data

    @validator("config_ids", pre=True, always=True)
    def ensure_config_ids(cls, v, values):
        if v is None and values.get("config_id") and values.get("config_ids") is None:
            # populate config_ids with config_id if only config_id is provided
            return [values["config_id"]]
        return v


class ResponseBody(BaseModel):
    messages: Optional[List[dict]] = Field(default=None, description="The new messages in the conversation")
    llm_output: Optional[dict] = Field(
        default=None,
        description="Contains any additional output coming from the LLM.",
    )
    output_data: Optional[dict] = Field(
        default=None,
        description="The output data, i.e. a dict with the values corresponding to the `output_vars`.",
    )
    log: Optional[GenerationLog] = Field(default=None, description="Additional logging information.")
    state: Optional[dict] = Field(
        default=None,
        description="A state object that should be used to continue the interaction in the future.",
    )


@app.get(
    "/v1/rails/configs",
    summary="Get List of available rails configurations.",
)
async def get_rails_configs():
    """Returns the list of available rails configurations."""

    # In single-config mode, we return a single config.
    if app.single_config_mode:
        # And we use the name of the root folder as the id of the config.
        return [{"id": app.single_config_id}]

    rails_config_path = app.rails_config_path
    if not rails_config_path or not os.path.exists(rails_config_path):
        return []

    # We extract all folder names as config names
    config_ids = [
        f
        for f in os.listdir(rails_config_path)
        if os.path.isdir(os.path.join(rails_config_path, f))
        and f[0] != "."
        and f[0] != "_"
        # We filter out all the configs for which there is no `config.yml` file.
        and (
            os.path.exists(os.path.join(rails_config_path, f, "config.yml"))
            or os.path.exists(os.path.join(rails_config_path, f, "config.yaml"))
        )
    ]

    return [{"id": config_id} for config_id in config_ids]


@app.get(
    "/v1/models",
    response_model=OpenAIModelsResponse,
    summary="List available models (OpenAI-compatible).",
)
async def get_openai_models():
    """Returns a list of available models in OpenAI-compatible format.

    Only returns models with type='main' from each config's config.yml file.
    The model ID is the 'model' field value from the main model configuration.
    """
    models = []
    base_timestamp = int(datetime.now().timestamp())
    seen_model_ids = set()

    # Get rails_config_path and validate it
    rails_config_path_raw = app.rails_config_path
    if not rails_config_path_raw:
        return OpenAIModelsResponse(
            data=[
                OpenAIModel(
                    id="default",
                    created=base_timestamp,
                    owned_by="nvidia",
                    guardrails_config=None,
                )
            ]
        )
    # Type assertion: rails_config_path is guaranteed to be a non-empty string after the check
    assert isinstance(rails_config_path_raw, str) and rails_config_path_raw
    rails_config_path: str = rails_config_path_raw

    # Get all available configs
    if app.single_config_mode:
        if app.single_config_id is None:
            return OpenAIModelsResponse(data=[])
        assert app.single_config_id is not None
        config_ids: List[str] = [app.single_config_id]
    else:
        config_ids = [
            f
            for f in os.listdir(rails_config_path)
            if os.path.isdir(os.path.join(rails_config_path, f))
            and f[0] != "."
            and f[0] != "_"
            and (
                os.path.exists(os.path.join(rails_config_path, f, "config.yml"))
                or os.path.exists(os.path.join(rails_config_path, f, "config.yaml"))
            )
        ]

    # Extract main models from each config

    for config_id in config_ids:
        try:
            # Load the config YAML file directly
            config_path = os.path.join(rails_config_path, config_id, "config.yml")
            if not os.path.exists(config_path):
                config_path = os.path.join(rails_config_path, config_id, "config.yaml")

            if not os.path.exists(config_path):
                continue

            with open(config_path, "r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f)

            # Find the main model
            if "models" in config_data and isinstance(config_data["models"], list):
                for model_config in config_data["models"]:
                    if isinstance(model_config, dict) and model_config.get("type") == "main":
                        # Get the model field value
                        model_id = model_config.get("model")
                        if model_id:
                            # Check if we've seen this model_id before
                            # If so, we still add it but with the config_id to show it's available in multiple configs
                            # For now, we'll add each occurrence with its config_id
                            models.append(
                                OpenAIModel(
                                    id=model_id,
                                    created=base_timestamp,
                                    owned_by="nvidia",
                                    guardrails_config=config_id,
                                )
                            )
                        break  # Only take the first main model
        except Exception as ex:
            log.warning(f"Could not load config {config_id} for models endpoint: {ex}")
            continue

    # If no models found, return a default model
    if not models:
        models.append(
            OpenAIModel(
                id="default",
                created=base_timestamp,
                owned_by="nvidia",
                guardrails_config=None,
            )
        )

    return OpenAIModelsResponse(data=models)


# One instance of LLMRails per config id
llm_rails_instances: dict[str, LLMRails] = {}
llm_rails_events_history_cache: dict[str, dict] = {}


def _generate_cache_key(config_ids: List[str]) -> str:
    """Generates a cache key for the given config ids."""

    return "-".join((config_ids))  # remove sorted


def _get_rails(config_ids: List[str]) -> LLMRails:
    """Returns the rails instance for the given config id."""

    # If we have a single config id, we just use it as the key
    configs_cache_key = _generate_cache_key(config_ids)

    if configs_cache_key in llm_rails_instances:
        return llm_rails_instances[configs_cache_key]

    # In single-config mode, we only load the main config directory
    if app.single_config_mode:
        if config_ids != [app.single_config_id]:
            raise ValueError(f"Invalid configuration ids: {config_ids}")

        # We set this to an empty string so tha when joined with the root path, we
        # get the same thing.
        config_ids = [""]

    full_llm_rails_config: Optional[RailsConfig] = None

    for config_id in config_ids:
        base_path = os.path.abspath(app.rails_config_path)
        full_path = os.path.normpath(os.path.join(base_path, config_id))

        # @NOTE: (Rdinu) Reject config_ids that contain dangerous characters or sequences
        if re.search(r"[\\/]|(\.\.)", config_id):
            raise ValueError("Invalid config_id.")

        if os.path.commonprefix([full_path, base_path]) != base_path:
            raise ValueError("Access to the specified path is not allowed.")

        rails_config = RailsConfig.from_path(full_path)

        if not full_llm_rails_config:
            full_llm_rails_config = rails_config
        else:
            full_llm_rails_config += rails_config

    if full_llm_rails_config is None:
        raise ValueError("No valid rails configuration found.")

    llm_rails = LLMRails(config=full_llm_rails_config, verbose=True)
    llm_rails_instances[configs_cache_key] = llm_rails

    # If we have a cache for the events, we restore it
    llm_rails.events_history_cache = llm_rails_events_history_cache.get(configs_cache_key, {})

    return llm_rails


@app.post(
    "/v1/chat/completions",
    response_model_exclude_none=True,
)
async def chat_completion(request: Request):
    """Chat completion endpoint supporting both Guardrails and OpenAI-compatible formats.

    If the request contains a 'model' field, it's treated as an OpenAI-compatible request.
    Otherwise, it's treated as a Guardrails request with config_id/config_ids.
    """
    # Parse the request body
    body_data = await request.json()

    # Check if this is an OpenAI-compatible request (has 'model' field)
    # If 'model' is present, treat as OpenAI format even if 'messages' is missing
    # (will fail validation with proper error)
    is_openai_format = "model" in body_data

    if is_openai_format:
        # Handle OpenAI-compatible request
        return await _handle_openai_completion(body_data, request)
    else:
        # Handle Guardrails request format
        try:
            return await _handle_guardrails_completion(body_data, request)
        except GuardrailsConfigurationError as e:
            # Convert GuardrailsConfigurationError to HTTP error
            raise HTTPException(status_code=500, detail=str(e))


def _extract_last_user_message(messages: List[dict]) -> str:
    """Extract the content from the last user message in the messages list."""
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "user":
            content = msg.get("content", "")
            return content if isinstance(content, str) else str(content)
    return ""


async def _handle_openai_echo_completion(
    openai_request: OpenAICompletionRequest,
    echo_content: str,
) -> StreamingResponse | OpenAICompletionResponse:
    """Handle OpenAI-compatible echo completion request.

    Returns the echo_content as the assistant response, supporting both
    streaming and non-streaming modes.

    Args:
        openai_request: The OpenAI completion request object
        echo_content: The content to echo back as the assistant response

    Returns:
        StreamingResponse for streaming requests, OpenAICompletionResponse for non-streaming
    """
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:29]}"
    created_timestamp = int(datetime.now().timestamp())

    if openai_request.stream:
        # Streaming echo response
        async def echo_stream_generator():
            # Send echo content as chunks
            chunk_size = 10  # Send 10 characters at a time for testing
            for i in range(0, len(echo_content), chunk_size):
                chunk_text = echo_content[i : i + chunk_size]
                chunk_data = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created_timestamp,
                    "model": openai_request.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": chunk_text},
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(chunk_data)}\n\n"

            # Send final chunk
            final_data = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created_timestamp,
                "model": openai_request.model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop",
                    }
                ],
            }
            yield f"data: {json.dumps(final_data)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(echo_stream_generator(), media_type="text/event-stream")
    else:
        # Non-streaming echo response
        response = OpenAICompletionResponse(
            id=completion_id,
            created=created_timestamp,
            model=openai_request.model,
            choices=[
                OpenAIChoice(
                    index=0,
                    message=OpenAIMessage(role="assistant", content=echo_content),
                    finish_reason="stop",
                )
            ],
            usage=OpenAIUsage(
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
            ),
        )
        return response


async def _handle_openai_streaming_completion(
    openai_request: OpenAICompletionRequest,
    llm_rails: LLMRails,
    messages: List[dict],
    options: GenerationOptions,
) -> StreamingResponse:
    """Handle OpenAI-compatible streaming chat completion request.

    Args:
        openai_request: The OpenAI completion request object
        llm_rails: The LLMRails instance to use for generation
        messages: List of messages in internal format
        options: Generation options

    Returns:
        StreamingResponse with OpenAI-compatible SSE format
    """
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:29]}"
    created_timestamp = int(datetime.now().timestamp())

    async def openai_stream_generator():
        streaming_handler = StreamingHandler()

        # Start the generation
        asyncio.create_task(
            llm_rails.generate_async(
                messages=messages,
                streaming_handler=streaming_handler,
                options=options,
            )
        )

        # Convert to OpenAI SSE format
        async for chunk in streaming_handler:
            if chunk is None:
                continue

            # Handle both string chunks and dict chunks from StreamingHandler
            if isinstance(chunk, dict):
                chunk_text = chunk.get("text", "")
                if chunk_text is None or chunk_text == "":
                    continue
            elif isinstance(chunk, str):
                chunk_text = chunk
            else:
                chunk_text = str(chunk)

            # Format as OpenAI SSE
            chunk_data = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created_timestamp,
                "model": openai_request.model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": chunk_text},
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(chunk_data)}\n\n"

        # Send final chunk with finish_reason
        final_data = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created_timestamp,
            "model": openai_request.model,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }
            ],
        }
        yield f"data: {json.dumps(final_data)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(openai_stream_generator(), media_type="text/event-stream")


async def _handle_openai_completion(body_data: dict, request: Request):
    """Handle OpenAI-compatible chat completion request."""
    try:
        openai_request = OpenAICompletionRequest(**body_data)
    except ValidationError as e:
        # Pydantic validation errors should return 422
        raise HTTPException(status_code=422, detail=f"Validation error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid request format: {str(e)}")

    if openai_request.stream:
        raise HTTPException(status_code=400, detail="Streaming not supported")

    # Check for X-Guardrails-Architecture header for echo mode
    architecture_header = request.headers.get("X-Guardrails-Architecture")
    if architecture_header:
        log.debug("Requested architecture %s", architecture_header)

    if architecture_header == "echo":
        # Extract last user message
        messages_list = [{"role": msg.role, "content": msg.content} for msg in openai_request.messages]
        echo_content = _extract_last_user_message(messages_list)
        return await _handle_openai_echo_completion(openai_request, echo_content)

    messages = await list_of_dict_openai_messages(openai_request)
    options = await openai_generation_options(openai_request)

    if architecture_header == "scheduler":
        generation_response = await app.scheduler.generate_async(messages=messages, options=options)
        response = await _openai_response(openai_request, generation_response)
        return response


    # log.info("Got OpenAI-compatible request for model %s", openai_request.model)
    for logger in registered_loggers:
        asyncio.get_event_loop().create_task(logger({"endpoint": "/v1/chat/completions", "body": body_data}))

    # Save the request headers in a context variable.
    api_request_headers.set(request.headers)

    rails_config_path_raw = app.rails_config_path
    if not rails_config_path_raw:
        raise HTTPException(status_code=404, detail="Rails configuration path not set")
    # Type assertion: rails_config_path is guaranteed to be a non-empty string after the check
    assert isinstance(rails_config_path_raw, str) and rails_config_path_raw
    rails_config_path: str = rails_config_path_raw

    config_ids_to_check = await get_config_ids(rails_config_path)

    # Find first config with matching main model
    config_id = await get_config_id_matching_main_llm(config_ids_to_check, openai_request.model, rails_config_path)

    # Get the rails instance
    try:
        llm_rails = _get_rails([config_id])
    except ValueError as ex:
        log.exception(ex)
        raise HTTPException(status_code=404, detail=f"Configuration '{config_id}' not found")



    try:
        if openai_request.stream and llm_rails.config.streaming_supported and llm_rails.main_llm_supports_streaming:
            return await _handle_openai_streaming_completion(
                openai_request=openai_request,
                llm_rails=llm_rails,
                messages=messages,
                options=options,
            )
        else:
            # Non-streaming response
            res = await llm_rails.generate_async(messages=messages, options=options)

            # Extract the response message
            if isinstance(res, GenerationResponse):
                if isinstance(res.response, list) and len(res.response) > 0:
                    bot_message_content = res.response[0]
                    if isinstance(bot_message_content, str):
                        content = bot_message_content
                    elif isinstance(bot_message_content, dict):
                        content = bot_message_content.get("content", "")
                    else:
                        content = str(bot_message_content)
                else:
                    content = str(res.response) if isinstance(res.response, str) else ""
            else:
                content = str(res.get("content", "")) if isinstance(res, dict) else str(res)

            # Extract token usage if available
            usage = None
            if isinstance(res, GenerationResponse) and res.log is not None:
                stats = getattr(res.log, "stats", None)
                if stats is not None:
                    usage = OpenAIUsage(
                        prompt_tokens=getattr(stats, "llm_calls_total_prompt_tokens", None) or 0,
                        completion_tokens=getattr(stats, "llm_calls_total_completion_tokens", None) or 0,
                        total_tokens=getattr(stats, "llm_calls_total_tokens", None) or 0,
                    )

            # Create OpenAI-compatible response
            completion_id = f"chatcmpl-{uuid.uuid4().hex[:29]}"
            created_timestamp = int(datetime.now().timestamp())

            response = OpenAICompletionResponse(
                id=completion_id,
                created=created_timestamp,
                model=openai_request.model,
                choices=[
                    OpenAIChoice(
                        index=0,
                        message=OpenAIMessage(role="assistant", content=content),
                        finish_reason="stop",
                    )
                ],
                usage=usage,
            )

            return response

    except Exception as ex:
        log.exception(ex)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(ex)}")


async def _openai_response(openai_request: OpenAICompletionRequest, generation_response: GenerationResponse) -> OpenAICompletionResponse:
    # Create OpenAI-compatible response
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:29]}"
    created_timestamp = int(datetime.now().timestamp())

    response = OpenAICompletionResponse(
        id=completion_id,
        created=created_timestamp,
        model=openai_request.model,
        choices=[
            OpenAIChoice(
                index=0,
                message=OpenAIMessage(role="assistant", content=generation_response.response),
                finish_reason="stop",
            )
        ],
        usage=None,
    )
    return response


async def openai_generation_options(openai_request: OpenAICompletionRequest) -> GenerationOptions:
    # Prepare generation options from OpenAI parameters
    llm_params = {}
    if openai_request.temperature is not None:
        llm_params["temperature"] = openai_request.temperature
    if openai_request.max_tokens is not None:
        llm_params["max_tokens"] = openai_request.max_tokens
    if openai_request.top_p is not None:
        llm_params["top_p"] = openai_request.top_p
    if openai_request.frequency_penalty is not None:
        llm_params["frequency_penalty"] = openai_request.frequency_penalty
    if openai_request.presence_penalty is not None:
        llm_params["presence_penalty"] = openai_request.presence_penalty
    if openai_request.stop is not None:
        llm_params["stop"] = openai_request.stop

    options = GenerationOptions(llm_params=llm_params if llm_params else None)
    return options


async def list_of_dict_openai_messages(openai_request: OpenAICompletionRequest) -> list[Any]:
    # Convert OpenAI messages to internal format
    messages = []
    for msg in openai_request.messages:
        messages.append({"role": msg.role, "content": msg.content})
    return messages


async def get_config_id_matching_main_llm(
    config_ids_to_check: List[str],
    model_or_config_id: str,
    rails_config_path: str,
) -> str:
    """Find the config ID that has a main model matching the given model name.

    Args:
        config_ids_to_check: List of config IDs to search through
        model_or_config_id: The model name to match against main models
        rails_config_path: Path to the rails configuration directory

    Returns:
        The config ID that contains a main model matching model_or_config_id

    Raises:
        HTTPException: If no matching config is found
    """
    for candidate_config_id in config_ids_to_check:
        try:
            candidate_config_path = os.path.join(rails_config_path, candidate_config_id, "config.yml")
            if not os.path.exists(candidate_config_path):
                candidate_config_path = os.path.join(rails_config_path, candidate_config_id, "config.yaml")

            if os.path.exists(candidate_config_path):
                with open(candidate_config_path, "r", encoding="utf-8") as f:
                    config_data = yaml.safe_load(f)

                if "models" in config_data and isinstance(config_data["models"], list):
                    for model_config in config_data["models"]:
                        if isinstance(model_config, dict) and model_config.get("type") == "main":
                            if model_config.get("model") == model_or_config_id:
                                return candidate_config_id
        except Exception:
            continue

    raise HTTPException(
        status_code=404,
        detail=f"Model '{model_or_config_id}' not found in any configuration",
    )


async def get_config_ids(rails_config_path: str) -> List[str]:
    """Get list of available config IDs.

    In single config mode, returns the single config ID.
    Otherwise, scans the rails_config_path directory for subdirectories
    containing config.yml or config.yaml files.

    Args:
        rails_config_path: Path to the rails configuration directory

    Returns:
        List of config IDs (directory names)

    Raises:
        HTTPException: If single config mode is enabled but no config ID is set
    """
    if app.single_config_mode:
        if app.single_config_id is None:
            raise HTTPException(
                status_code=404,
                detail="Single config mode enabled but no config ID set",
            )
        assert app.single_config_id is not None
        config_ids_to_check: List[str] = [app.single_config_id]
    else:
        config_ids_to_check = [
            f
            for f in os.listdir(rails_config_path)
            if os.path.isdir(os.path.join(rails_config_path, f))
            and f[0] != "."
            and f[0] != "_"
            and (
                os.path.exists(os.path.join(rails_config_path, f, "config.yml"))
                or os.path.exists(os.path.join(rails_config_path, f, "config.yaml"))
            )
        ]
    return config_ids_to_check


async def _handle_guardrails_completion(body_data: dict, request: Request):
    """Handle Guardrails chat completion request (original format)."""
    try:
        body = RequestBody(**body_data)
    except ValidationError as e:
        # Pydantic validation errors should return 422
        raise HTTPException(status_code=422, detail=f"Validation error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid request format: {str(e)}")

    # Check for X-Guardrails-Architecture header for echo mode
    architecture_header = request.headers.get("X-Guardrails-Architecture", "").lower()
    if architecture_header == "echo":
        # Extract last user message
        messages = body.messages or []
        echo_content = _extract_last_user_message(messages)

        if body.stream:
            # Streaming echo response
            async def echo_stream_generator():
                # Send echo content as chunks
                chunk_size = 10  # Send 10 characters at a time for testing
                for i in range(0, len(echo_content), chunk_size):
                    chunk_text = echo_content[i : i + chunk_size]
                    yield chunk_text
                yield ""  # Final empty chunk to signal completion

            return StreamingResponse(echo_stream_generator())
        else:
            # Non-streaming echo response
            bot_message = {"role": "assistant", "content": echo_content}
            return ResponseBody(messages=[bot_message])

    log.info("Got request for config %s", body.config_id)
    for logger in registered_loggers:
        asyncio.get_event_loop().create_task(
            logger({"endpoint": "/v1/chat/completions", "body": json.dumps(body_data)})
        )

    # Save the request headers in a context variable.
    api_request_headers.set(request.headers)

    # Use Request config_ids if set, otherwise use the FastAPI default config.
    # If neither is available we can't generate any completions as we have no config_id
    config_ids = body.config_ids
    if not config_ids:
        if app.default_config_id:
            config_ids = [app.default_config_id]
        else:
            raise GuardrailsConfigurationError("No request config_ids provided and server has no default configuration")

    try:
        llm_rails = _get_rails(config_ids)
    except ValueError as ex:
        log.exception(ex)
        return ResponseBody(
            messages=[
                {
                    "role": "assistant",
                    "content": f"Could not load the {config_ids} guardrails configuration. "
                    f"An internal error has occurred.",
                }
            ]
        )

    try:
        messages = body.messages or []
        if body.context:
            messages.insert(0, {"role": "context", "content": body.context})

        # If we have a `thread_id` specified, we need to look up the thread
        datastore_key = None

        if body.thread_id:
            if datastore is None:
                raise RuntimeError("No DataStore has been configured.")

            # We make sure the `thread_id` meets the minimum complexity requirement.
            if len(body.thread_id) < 16:
                return ResponseBody(
                    messages=[
                        {
                            "role": "assistant",
                            "content": "The `thread_id` must have a minimum length of 16 characters.",
                        }
                    ]
                )

            # Fetch the existing thread messages. For easier management, we prepend
            # the string `thread-` to all thread keys.
            datastore_key = "thread-" + body.thread_id
            thread_messages = json.loads(await datastore.get(datastore_key) or "[]")

            # And prepend them.
            messages = thread_messages + messages

        if body.stream and llm_rails.config.streaming_supported and llm_rails.main_llm_supports_streaming:
            # Create the streaming handler instance
            streaming_handler = StreamingHandler()

            # Start the generation
            asyncio.create_task(
                llm_rails.generate_async(
                    messages=messages,
                    streaming_handler=streaming_handler,
                    options=body.options,
                    state=body.state,
                )
            )

            # TODO: Add support for thread_ids in streaming mode

            return StreamingResponse(streaming_handler)
        else:
            res = await llm_rails.generate_async(messages=messages, options=body.options, state=body.state)

            if isinstance(res, GenerationResponse):
                bot_message_content = res.response[0]
                # Ensure bot_message is always a dict
                if isinstance(bot_message_content, str):
                    bot_message = {"role": "assistant", "content": bot_message_content}
                else:
                    bot_message = bot_message_content
            else:
                assert isinstance(res, dict)
                bot_message = res

            # If we're using threads, we also need to update the data before returning
            # the message.
            if body.thread_id and datastore is not None and datastore_key is not None:
                await datastore.set(datastore_key, json.dumps(messages + [bot_message]))

            result = ResponseBody(messages=[bot_message])

            # If we have additional GenerationResponse fields, we return as well
            if isinstance(res, GenerationResponse):
                result.llm_output = res.llm_output
                result.output_data = res.output_data
                result.log = res.log
                result.state = res.state

            return result

    except Exception as ex:
        log.exception(ex)
        return ResponseBody(messages=[{"role": "assistant", "content": "Internal server error."}])


# By default, there are no challenges
challenges = []


def register_challenges(additional_challenges: List[dict]):
    """Register additional challenges

    Args:
        additional_challenges: The new challenges to be registered.
    """
    challenges.extend(additional_challenges)


@app.get(
    "/v1/challenges",
    summary="Get list of available challenges.",
)
async def get_challenges():
    """Returns the list of available challenges for red teaming."""

    return challenges


def register_datastore(datastore_instance: DataStore):
    """Registers a DataStore to be used by the server."""
    global datastore

    datastore = datastore_instance


def register_logger(logger: Callable):
    """Register an additional logger"""
    registered_loggers.append(logger)


def start_auto_reload_monitoring():
    """Start a thread that monitors the config folder for changes."""
    try:
        from watchdog.events import (
            FileSystemEventHandler,  # type: ignore[reportMissingImports]
        )
        from watchdog.observers import Observer  # type: ignore[reportMissingImports]

        class Handler(FileSystemEventHandler):
            def on_any_event(self, event):
                if event.is_directory:
                    return None

                elif event.event_type == "created" or event.event_type == "modified":
                    log.info(f"Watchdog received {event.event_type} event for file {event.src_path}")

                    # Compute the relative path
                    src_path_str = str(event.src_path)
                    rel_path = os.path.relpath(src_path_str, app.rails_config_path)

                    # The config_id is the first component
                    parts = rel_path.split(os.path.sep)
                    config_id = parts[0]

                    if (
                        not parts[-1].startswith(".")
                        and ".ipynb_checkpoints" not in parts
                        and os.path.isfile(src_path_str)
                    ):
                        # We just remove the config from the cache so that a new one is used next time
                        if config_id in llm_rails_instances:
                            instance = llm_rails_instances[config_id]
                            del llm_rails_instances[config_id]
                            if instance:
                                val = instance.events_history_cache
                                # We save the events history cache, to restore it on the new instance
                                llm_rails_events_history_cache[config_id] = val

                            log.info(f"Configuration {config_id} has changed. Clearing cache.")

        observer = Observer()
        event_handler = Handler()
        observer.schedule(event_handler, app.rails_config_path, recursive=True)
        observer.start()
        try:
            while not app.stop_signal:
                time.sleep(5)
        finally:
            observer.stop()
            observer.join()

    except ImportError:
        # Since this is running in a separate thread, we just print the error.
        print("The auto-reload feature requires `watchdog`. Please install using `pip install watchdog`.")
        # Force close everything.
        os._exit(-1)


def set_default_config_id(config_id: str):
    app.default_config_id = config_id


class GuardrailsConfigurationError(Exception):
    """Exception raised for errors in the configuration."""

    pass


# # Register a nicer error message for 422 error
# def register_exception(app: FastAPI):
#     @app.exception_handler(RequestValidationError)
#     async def validation_exception_handler(
#         request: Request, exc: RequestValidationError
#     ):
#         exc_str = f"{exc}".replace("\n", " ").replace("   ", " ")
#         # or logger.error(f'{exc}')
#         log.error(request, exc_str)
#         content = {"status_code": 10422, "message": exc_str, "data": None}
#         return JSONResponse(
#             content=content, status_code=status.HTTP_422_UNPROCESSABLE_ENTITY
#         )
#
#
# register_exception(app)
