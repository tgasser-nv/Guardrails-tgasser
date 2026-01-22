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

import asyncio
import json
import logging
import os
import time
import uuid
from typing import AsyncIterator, Union
from urllib.parse import urljoin

import aiohttp
from fastapi import HTTPException

from nemoguardrails import RailsConfig
from nemoguardrails.engines.async_worker_pool_engine_models import (
    AsyncWorkerPoolEngineJob,
)
from nemoguardrails.engines.guardrails_engine_base import GuardrailsEngineBase
from nemoguardrails.rails.llm.config import Model
from nemoguardrails.rails.llm.options import GenerationOptions, GenerationResponse
from nemoguardrails.streaming import END_OF_STREAM, StreamingHandler

MAX_QUEUE_SIZE = 1000

# Get chunk window size from environment or use default
# This controls how many chunks are buffered before performing content safety checks
DEFAULT_CHUNK_WINDOW_SIZE = int(os.environ.get("GUARDRAILS_CHUNK_WINDOW_SIZE", "200"))

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(formatter)

log.addHandler(console_handler)


class AsyncWorkerPoolEngine(GuardrailsEngineBase):
    """Workflow engine using an asynchronous queue and pool of workers.
    Implements hard-coded Content-safety input and output rails
    """

    def __init__(
        self,
        rails_config: RailsConfig,
        num_workers: int,
        chunk_window_size: int = DEFAULT_CHUNK_WINDOW_SIZE,
    ) -> None:
        """Create a new scheduler engine

        Args:
            rails_config: The rails configuration
            num_workers: Number of worker tasks in the pool
            chunk_window_size: Number of chunks to buffer before running content safety checks (default: 200)
        """

        # Queue work from the FastAPI interface
        self.request_queue = asyncio.Queue(maxsize=MAX_QUEUE_SIZE)

        # Worker tasks to call content-safety input, generation, and content-safety output
        self.workers: list[asyncio.Task] = []

        # Scheduler attributes
        self.is_running = False
        self.num_workers = num_workers
        self.chunk_window_size = chunk_window_size

        self.rails_config = rails_config

        # Run common base-class init. Stores self.engine_name, self.models, self.rails, and self.prompts
        super().__init__("scheduler_engine", rails_config)

    async def start(self):
        """Starts a pool of workers"""
        for idx in range(self.num_workers):
            task = asyncio.create_task(self._worker_loop(idx))
            self.workers.append(task)

        self.is_running = True

    async def stop(self):
        """Shuts down the worker pool"""
        for task in self.workers:
            task.cancel()
        await asyncio.gather(*self.workers, return_exceptions=True)
        self.is_running = False

    async def _get_model_by_type(self, model_type: str) -> Model:
        """Returns a single model whose type matches the given type"""

        matching_models = [model for model in self.models.values() if model.type == model_type]
        num_models = len(matching_models)
        if num_models != 1:
            raise Exception(f"Expected one model with type {model_type}, got {num_models}: {matching_models}")
        return matching_models[0]

    async def _get_content_safety_model(self) -> Model:
        """Return the Model used for content safety"""
        return await self._get_model_by_type("content_safety")

    async def _get_main_model(self) -> Model:
        """Return the Model used for content safety"""
        return await self._get_model_by_type("main")

    async def _is_content_safety_input_safe(self, job: AsyncWorkerPoolEngineJob) -> bool:
        """Make an input-rail content-safety request

        Example: https://build.nvidia.com/nvidia/llama-3_1-nemoguard-8b-content-safety?snippet_tab=Shell
        """
        model = await self._get_content_safety_model()
        prompt_template = self.prompts["content_safety_check_input $model=content_safety"]
        if prompt_template.content is None:
            raise ValueError("Prompt template content is None")
        prompt = prompt_template.content.replace("{{ user_input }}", job.messages[-1]["content"])

        base_url = None
        if model.parameters and model.parameters["base_url"]:
            base_url = model.parameters["base_url"]
        elif model.engine == "nim":
            base_url = "https://integrate.api.nvidia.com"
        elif model.engine == "openai":
            base_url = "https://api.openai.com"

        if not base_url:
            raise HTTPException(
                status_code=404,
                detail="No base_url provided, and it could not be inferred",
            )

        endpoint = "/v1/chat/completions"
        url = urljoin(base_url, endpoint)
        api_key = os.environ.get("NVIDIA_API_KEY")

        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        body = {"model": model.model, "messages": job.messages}

        async with aiohttp.ClientSession(headers=headers) as session:
            async with session.post(url, json=body) as response:
                # 10:47:45 gr.1      | {"User Safety": "safe", "Response Safety": "safe"}

                try:
                    response_dict = await response.json()
                    content_safety_text = response_dict["choices"][0]["message"]["content"]
                    content_safety_response = json.loads(content_safety_text)

                    is_request_safe = content_safety_response.get("User Safety", "unsafe") == "safe"
                    return is_request_safe

                except Exception as e:
                    raise HTTPException(status_code=404, detail=str(e))

    async def _is_content_safety_output_safe(self, job: AsyncWorkerPoolEngineJob, llm_response: str) -> bool:
        """Make an output-rail content-safety request

        Example: https://build.nvidia.com/nvidia/llama-3_1-nemoguard-8b-content-safety?snippet_tab=Shell
        """

        model = await self._get_content_safety_model()
        prompt_template = self.prompts["content_safety_check_output $model=content_safety"]
        if prompt_template.content is None:
            raise ValueError("Prompt template content is None")
        prompt = prompt_template.content.replace("{{ user_input }}", job.messages[-1]["content"])
        prompt = prompt.replace("{{ bot_response }}", llm_response)

        base_url = None
        if model.parameters and model.parameters["base_url"]:
            base_url = model.parameters["base_url"]
        elif model.engine == "nim":
            base_url = "https://integrate.api.nvidia.com"
        elif model.engine == "openai":
            base_url = "https://api.openai.com"

        if not base_url:
            raise HTTPException(
                status_code=404,
                detail="No base_url provided, and it could not be inferred",
            )

        endpoint = "/v1/chat/completions"
        url = urljoin(base_url, endpoint)
        api_key = os.environ.get("NVIDIA_API_KEY")

        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        body_messages = job.messages + [{"role": "assistant", "content": llm_response}]
        body = {"model": model.model, "messages": body_messages}

        async with aiohttp.ClientSession(headers=headers) as session:
            async with session.post(url, json=body) as response:
                try:
                    response_dict = await response.json()
                    content_safety_text = response_dict["choices"][0]["message"]["content"]
                    content_safety_response = json.loads(content_safety_text)

                    is_request_safe = content_safety_response.get("User Safety", "unsafe") == "safe"
                    is_response_safe = content_safety_response.get("Response Safety", "unsafe") == "safe"
                    return is_request_safe and is_response_safe

                except Exception as e:
                    raise HTTPException(status_code=404, detail=str(e))

    async def _app_llm_response(self, job: AsyncWorkerPoolEngineJob) -> str:
        """Generate a response from the application LLM)

        Example: https://build.nvidia.com/nvidia/llama-3_1-nemoguard-8b-content-safety?snippet_tab=Shell
        """
        model = await self._get_main_model()
        prompt = job.messages

        base_url = None
        if model.parameters and model.parameters["base_url"]:
            base_url = model.parameters["base_url"]
        elif model.engine == "nim":
            base_url = "https://integrate.api.nvidia.com"
        elif model.engine == "openai":
            base_url = "https://api.openai.com"
        if not base_url:
            raise HTTPException(
                status_code=404,
                detail="No base_url provided, and it could not be inferred",
            )

        endpoint = "/v1/chat/completions"
        url = urljoin(base_url, endpoint)
        api_key = os.environ.get("NVIDIA_API_KEY")

        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        body = {"model": model.model, "messages": job.messages}

        async with aiohttp.ClientSession(headers=headers) as session:
            async with session.post(url, json=body) as response:
                try:
                    response_dict = await response.json()
                    content_text = response_dict["choices"][0]["message"]["content"]
                    return content_text

                except Exception as e:
                    raise HTTPException(status_code=404, detail=str(e))

    async def _app_llm_response_streaming(self, job: AsyncWorkerPoolEngineJob) -> AsyncIterator[str]:
        """Generate a streaming response from the application LLM (yields chunks)

        Yields:
            String chunks from the LLM response
        """
        model = await self._get_main_model()
        prompt = job.messages

        base_url = None
        if model.parameters and model.parameters["base_url"]:
            base_url = model.parameters["base_url"]
        elif model.engine == "nim":
            base_url = "https://integrate.api.nvidia.com"
        elif model.engine == "openai":
            base_url = "https://api.openai.com"
        if not base_url:
            raise HTTPException(
                status_code=404,
                detail="No base_url provided, and it could not be inferred",
            )

        endpoint = "/v1/chat/completions"
        url = urljoin(base_url, endpoint)
        api_key = os.environ.get("NVIDIA_API_KEY")

        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        body = {"model": model.model, "messages": job.messages, "stream": True}

        async with aiohttp.ClientSession(headers=headers) as session:
            async with session.post(url, json=body) as response:
                try:
                    async for line in response.content:
                        line_str = line.decode("utf-8").strip()
                        if not line_str or line_str.startswith(":"):
                            continue

                        # Remove "data: " prefix if present
                        if line_str.startswith("data: "):
                            line_str = line_str[6:]

                        # Check for end of stream
                        if line_str == "[DONE]":
                            break

                        try:
                            chunk_dict = json.loads(line_str)
                            delta = chunk_dict.get("choices", [{}])[0].get("delta", {})
                            content = delta.get("content", "")

                            if content:
                                yield content

                        except json.JSONDecodeError:
                            # Skip malformed JSON lines
                            continue

                except Exception as e:
                    raise HTTPException(status_code=404, detail=str(e))

    async def _worker_loop(self, worker_id: int):
        while True:
            # asyncio guarantees workers get a unique item from the queue
            job = await self.request_queue.get()

            # Check if this is a streaming job
            if job.streaming_handler is not None:
                await self._process_streaming_job(worker_id, job)
            else:
                await self._process_non_streaming_job(worker_id, job)

            self.request_queue.task_done()

    async def _process_non_streaming_job(self, worker_id: int, job: AsyncWorkerPoolEngineJob):
        """Process a non-streaming job"""
        log.info("Worker #%d running job %s", worker_id, job)

        try:
            log.info("Worker #%d checking content-safety input", worker_id)
            is_input_safe = await self._is_content_safety_input_safe(job)
            if not is_input_safe:
                generation_response = GenerationResponse(response="I'm sorry I can't help you with that")
                job.future.set_result(generation_response)
                return

            log.info("Worker #%d generating response", worker_id)
            app_llm_response = await self._app_llm_response(job)

            log.info("Worker #%d checking content-safety output", worker_id)
            is_output_safe = await self._is_content_safety_output_safe(job, app_llm_response)
            if not is_output_safe:
                generation_response = GenerationResponse(response="I'm sorry I can't help you with that")
                job.future.set_result(generation_response)
                return

            generation_response = GenerationResponse(response=app_llm_response)
            job.future.set_result(generation_response)

        except Exception as e:
            log.exception("Worker #%d encountered an error %s", worker_id, e)
            job.future.set_exception(e)

    async def _process_streaming_job(self, worker_id: int, job: AsyncWorkerPoolEngineJob):
        """Process a streaming job with buffered content safety checks

        This method:
        1. Checks content safety of the input
        2. Streams the LLM response while buffering chunks
        3. Performs content safety checks on windows of N chunks
        4. Stops streaming and returns error message if any window fails safety check
        """
        log.info("Worker #%d running streaming job %s", worker_id, job)

        streaming_handler = job.streaming_handler

        # Type guard to ensure streaming_handler is not None
        if streaming_handler is None:
            log.error("Worker #%d: streaming_handler is None for streaming job", worker_id)
            job.future.set_exception(ValueError("Streaming handler is required for streaming jobs"))
            return

        try:
            # Step 1: Check content safety of input
            log.info("Worker #%d checking content-safety input", worker_id)
            is_input_safe = await self._is_content_safety_input_safe(job)
            if not is_input_safe:
                # Push error message and end stream
                await streaming_handler.push_chunk("I'm sorry I can't help you with that")
                await streaming_handler.push_chunk(END_OF_STREAM)  # type: ignore
                job.future.set_result(None)
                return

            # Step 2 & 3: Stream LLM response with buffered content safety checks
            log.info(
                "Worker #%d generating streaming response with buffered safety checks",
                worker_id,
            )

            chunk_buffer = []
            full_response = ""
            safety_check_failed = False

            async for chunk in self._app_llm_response_streaming(job):
                # Return all the chunks immediately if stream_first is enabled
                if self.rails_config.rails.output.streaming.stream_first:
                    await streaming_handler.push_chunk(chunk)

                # Add chunk to buffer
                chunk_buffer.append(chunk)
                full_response += chunk

                # Check if we've accumulated enough chunks for a safety check
                if len(chunk_buffer) >= self.chunk_window_size:
                    # Concatenate buffered chunks for safety check
                    buffered_text = "".join(chunk_buffer)

                    log.info(
                        "Worker #%d checking content-safety output for %d chunks (%d chars)",
                        worker_id,
                        len(chunk_buffer),
                        len(buffered_text),
                    )

                    # Perform content safety check on the accumulated buffer
                    is_output_safe = await self._is_content_safety_output_safe(job, full_response)

                    if not is_output_safe:
                        log.warning(
                            "Worker #%d: Output failed content safety check at %d total chars",
                            worker_id,
                            len(full_response),
                        )
                        safety_check_failed = True
                        # Stop streaming and send error message
                        await streaming_handler.push_chunk(
                            "\n\n[Content filtered: Response violated safety guidelines]"
                        )
                        await streaming_handler.push_chunk(END_OF_STREAM)  # type: ignore
                        job.future.set_result(None)
                        return

                    # Safety check passed, push all buffered chunks to the streaming handler
                    for buffered_chunk in chunk_buffer:
                        await streaming_handler.push_chunk(buffered_chunk)

                    # Clear the buffer
                    chunk_buffer = []

            # Handle any remaining chunks in the buffer after streaming completes
            if chunk_buffer and not safety_check_failed:
                # Final safety check on remaining chunks
                log.info(
                    "Worker #%d checking content-safety output for final %d chunks",
                    worker_id,
                    len(chunk_buffer),
                )

                is_output_safe = await self._is_content_safety_output_safe(job, full_response)

                if not is_output_safe:
                    log.warning(
                        "Worker #%d: Final output failed content safety check",
                        worker_id,
                    )
                    # Don't send the remaining buffer, just send error and end
                    await streaming_handler.push_chunk("\n\n[Content filtered: Response violated safety guidelines]")
                    await streaming_handler.push_chunk(END_OF_STREAM)  # type: ignore
                    job.future.set_result(None)
                    return

                # Safety check passed, push remaining buffered chunks
                for buffered_chunk in chunk_buffer:
                    await streaming_handler.push_chunk(buffered_chunk)

            # Signal end of stream
            await streaming_handler.push_chunk(END_OF_STREAM)  # type: ignore
            log.info("Worker #%d completed streaming job successfully", worker_id)
            job.future.set_result(None)

        except Exception as e:
            log.exception("Worker #%d encountered an error %s", worker_id, e)
            await streaming_handler.push_chunk(END_OF_STREAM)  # type: ignore
            job.future.set_exception(e)

    async def generate_async(
        self,
        messages: list[dict],
        options: GenerationOptions | None = None,
        streaming_handler: StreamingHandler | None = None,
    ) -> GenerationResponse:
        """
        Create a Job for the Scheduler to process downstream, with Future for response.
        Await the future and return it
        """

        loop = asyncio.get_event_loop()
        future = loop.create_future()
        request_time = int(time.time())
        job_id = str(uuid.uuid4())

        # Create a new job. The `work_timestamp` and `completed_timestamp` are None since it's only queued
        job = AsyncWorkerPoolEngineJob(
            job_id=job_id,
            messages=messages,
            options=options,
            queue_timestamp=request_time,
            future=future,
        )

        log.info("Queueing job, %d waiting in queue", self.request_queue.qsize())
        await self.request_queue.put(job)

        try:
            result = await future

            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    def stream_async(
        self,
        prompt: str | None = None,
        messages: list[dict] | None = None,
        options: GenerationOptions | None = None,
        state: dict | None = None,
        include_generation_metadata: bool = False,
        generator: AsyncIterator[str] | None = None,
    ) -> AsyncIterator[Union[str, dict]]:
        """Simplified interface for getting directly the streamed tokens from the LLM.

        This method creates a streaming handler and submits a job to the worker pool
        to process the request with streaming support.

        Args:
            prompt: Optional prompt string (not currently used, kept for signature compatibility)
            messages: List of message dictionaries to process
            options: Optional generation options
            state: Optional state dictionary (not currently used, kept for signature compatibility)
            include_generation_metadata: Whether to include generation metadata in chunks
            generator: Optional external generator (not currently used, kept for signature compatibility)

        Returns:
            AsyncIterator that yields string chunks or dicts with metadata
        """
        if messages is None:
            raise ValueError("messages parameter is required for stream_async")

        streaming_handler = StreamingHandler(include_generation_metadata=include_generation_metadata)

        # Create a properly managed task with exception handling
        async def _generation_task():
            try:
                loop = asyncio.get_event_loop()
                future = loop.create_future()
                request_time = int(time.time())
                job_id = str(uuid.uuid4())

                # Create a new job with streaming handler
                job = AsyncWorkerPoolEngineJob(
                    job_id=job_id,
                    messages=messages,
                    options=options,
                    queue_timestamp=request_time,
                    future=future,
                    streaming_handler=streaming_handler,
                )

                log.info(
                    "Queueing streaming job, %d waiting in queue",
                    self.request_queue.qsize(),
                )
                await self.request_queue.put(job)

                # Wait for the job to complete
                await future

            except Exception as e:
                # If an exception occurs during generation, push it to the streaming handler
                log.error(f"Error in streaming generation task: {e}", exc_info=True)
                error_message = str(e)
                error_dict = {"error": error_message}
                error_payload = json.dumps(error_dict)
                await streaming_handler.push_chunk(error_payload)
                await streaming_handler.push_chunk(END_OF_STREAM)  # type: ignore

        task = asyncio.create_task(_generation_task())

        # Store task reference to prevent garbage collection and ensure proper cleanup
        if not hasattr(self, "_active_tasks"):
            self._active_tasks = set()
        self._active_tasks.add(task)

        # Clean up task when it's done
        def task_done_callback(task):
            self._active_tasks.discard(task)

        task.add_done_callback(task_done_callback)

        async def wrapped_iterator():
            try:
                async for chunk in streaming_handler:
                    if chunk is not None:
                        yield chunk
            finally:
                await task

        return wrapped_iterator()
