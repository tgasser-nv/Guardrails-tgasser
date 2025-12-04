import json

from nemoguardrails.engines.guardrails_engine_base import GuardrailsEngineBase
from nemoguardrails import RailsConfig

from nemoguardrails.engines.scheduler_engine_models import SchedulerEngineJob
from nemoguardrails.rails.llm.options import (
    GenerationOptions,
    GenerationResponse,
)
from nemoguardrails.streaming import StreamingHandler
from fastapi import HTTPException

import time

import uuid
import aiohttp
import asyncio
import logging
import os

from urllib.parse import urljoin

from nemoguardrails.rails.llm.config import Model

MAX_QUEUE_SIZE = 1000

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(formatter)

log.addHandler(console_handler)


class SchedulerEngine(GuardrailsEngineBase):
    """Workflow engine using a synchronous scheduler to allocate work to async-based API requesters"""


    def __init__(self, rails_config: RailsConfig, num_workers: int) -> None:
        """Create a new scheduler engine"""

        # Queue work from the FastAPI interface
        self.request_queue = asyncio.Queue(maxsize=MAX_QUEUE_SIZE)

        # Worker tasks to call content-safety input, generation, and content-safety output
        self.workers: list[asyncio.Task] = []

        # Scheduler attributes
        self.is_running = False
        self.num_workers = num_workers

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

    async def _is_content_safety_input_safe(self, job: SchedulerEngineJob) -> bool:
        """Make an input-rail content-safety request

        Example: https://build.nvidia.com/nvidia/llama-3_1-nemoguard-8b-content-safety?snippet_tab=Shell
        """
        model = await self._get_content_safety_model()
        prompt_template = self.prompts["content_safety_check_input $model=content_safety"]
        prompt = prompt_template.content.replace("{{ user_input }}", job.messages[-1]["content"])

        base_url = None
        if model.parameters and model.parameters["base_url"]:
            base_url = model.parameters["base_url"]
        elif model.engine == "nim":
            base_url = "https://integrate.api.nvidia.com"
        elif model.engine == "openai":
            base_url = "https://api.openai.com"

        if not base_url:
            raise HTTPException(status_code=404, detail="No base_url provided, and it could not be inferred")

        endpoint = "/v1/chat/completions"
        url = urljoin(base_url, endpoint)
        api_key = os.environ.get("NVIDIA_API_KEY")

        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        body = {"model": model.model,
                "messages": job.messages}

        async with aiohttp.ClientSession(headers=headers) as session:
            async with session.post(url, json=body) as response:

                # 10:47:45 gr.1      | {"User Safety": "safe", "Response Safety": "safe"}

                try:
                    response_dict = await response.json()
                    content_safety_text = response_dict['choices'][0]['message']['content']
                    content_safety_response = json.loads(content_safety_text)

                    is_request_safe = content_safety_response.get("User Safety", "unsafe") == "safe"
                    return is_request_safe

                except Exception as e:
                    raise HTTPException(status_code=404, detail=str(e))

    async def _is_content_safety_output_safe(self, job: SchedulerEngineJob, llm_response: str) -> bool:
        """Make an output-rail content-safety request

        Example: https://build.nvidia.com/nvidia/llama-3_1-nemoguard-8b-content-safety?snippet_tab=Shell
        """

        model = await self._get_content_safety_model()
        prompt_template = self.prompts["content_safety_check_output $model=content_safety"]
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
            raise HTTPException(status_code=404, detail="No base_url provided, and it could not be inferred")

        endpoint = "/v1/chat/completions"
        url = urljoin(base_url, endpoint)
        api_key = os.environ.get("NVIDIA_API_KEY")

        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        body_messages = job.messages + [{"role": "assistant", "content": llm_response}]
        body = {"model": model.model,
                "messages": body_messages}

        async with aiohttp.ClientSession(headers=headers) as session:
            async with session.post(url, json=body) as response:
                try:
                    response_dict = await response.json()
                    content_safety_text = response_dict['choices'][0]['message']['content']
                    content_safety_response = json.loads(content_safety_text)

                    is_request_safe = content_safety_response.get("User Safety", "unsafe") == "safe"
                    is_response_safe = content_safety_response.get("Response Safety", "unsafe") == "safe"
                    return is_request_safe and is_response_safe

                except Exception as e:
                    raise HTTPException(status_code=404, detail=str(e))

    async def _app_llm_response(self, job: SchedulerEngineJob) -> str:
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
            raise HTTPException(status_code=404, detail="No base_url provided, and it could not be inferred")

        endpoint = "/v1/chat/completions"
        url = urljoin(base_url, endpoint)
        api_key = os.environ.get("NVIDIA_API_KEY")

        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        body = {"model": model.model,
                "messages": job.messages}

        async with aiohttp.ClientSession(headers=headers) as session:
            async with session.post(url, json=body) as response:

                try:
                    response_dict = await response.json()
                    content_text = response_dict['choices'][0]['message']['content']
                    return content_text

                except Exception as e:
                    raise HTTPException(status_code=404, detail=str(e))


    async def _worker_loop(self, worker_id: int):
        while True:

            # asyncio guarantees workers get a unique item from the queue
            job = await self.request_queue.get()
            log.info("Worker #%d running job %s", worker_id, job)

            log.info("Worker #%d checking content-safety input", worker_id)
            is_input_safe = await self._is_content_safety_input_safe(job)
            if not is_input_safe:
                return "I'm sorry I can't help you with that"

            log.info("Worker #%d generating response", worker_id)
            app_llm_response = await self._app_llm_response(job)

            log.info("Worker #%d checking content-safety output", worker_id)
            is_output_safe = await self._is_content_safety_output_safe(job, app_llm_response)
            if not is_output_safe:
                return "I'm sorry I can't help you with that"

            job.future.set_result(app_llm_response)


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
        request_time = time.time()
        job_id = uuid.uuid4()

        # Create a new job. The `work_timestamp` and `completed_timestamp` are None since it's only queued
        job = SchedulerEngineJob(job_id=job_id, messages=messages, options=options, queue_timestamp=request_time, future=future)

        await self.request_queue.put(job)

        try:
            result = await future
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))



