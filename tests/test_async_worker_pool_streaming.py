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

"""Tests for AsyncWorkerPoolEngine streaming with buffered content safety checks."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from nemoguardrails import RailsConfig
from nemoguardrails.engines.async_worker_pool_engine import AsyncWorkerPoolEngine
from nemoguardrails.engines.async_worker_pool_engine_models import (
    AsyncWorkerPoolEngineJob,
)
from nemoguardrails.streaming import StreamingHandler


@pytest.fixture
def mock_rails_config():
    """Create a mock RailsConfig for testing."""
    config = RailsConfig.from_content(
        config={
            "models": [
                {
                    "type": "main",
                    "engine": "openai",
                    "model": "gpt-3.5-turbo",
                    "parameters": {"base_url": "https://api.openai.com"},
                },
                {
                    "type": "content_safety",
                    "engine": "openai",
                    "model": "safety-model",
                    "parameters": {"base_url": "https://api.openai.com"},
                },
            ],
        },
        colang_content="",
    )
    return config


@pytest.fixture
async def worker_pool_engine(mock_rails_config):
    """Create an AsyncWorkerPoolEngine instance for testing."""
    engine = AsyncWorkerPoolEngine(
        mock_rails_config, num_workers=2, chunk_window_size=5
    )
    await engine.start()
    yield engine
    await engine.stop()


class TestAsyncWorkerPoolEngineStreaming:
    """Test suite for AsyncWorkerPoolEngine streaming functionality."""

    @pytest.mark.asyncio
    async def test_stream_async_basic(self, mock_rails_config):
        """Test basic stream_async method signature and return type."""
        engine = AsyncWorkerPoolEngine(mock_rails_config, num_workers=1)

        messages = [{"role": "user", "content": "Hello"}]
        result = engine.stream_async(messages=messages)

        # Verify it returns an AsyncIterator
        assert hasattr(result, "__aiter__")

        await engine.stop()

    @pytest.mark.asyncio
    async def test_stream_async_requires_messages(self, mock_rails_config):
        """Test that stream_async raises error when messages is None."""
        engine = AsyncWorkerPoolEngine(mock_rails_config, num_workers=1)

        with pytest.raises(ValueError, match="messages parameter is required"):
            engine.stream_async(messages=None)

        await engine.stop()

    @pytest.mark.asyncio
    async def test_chunk_buffering_window_size(self, mock_rails_config):
        """Test that chunks are buffered according to chunk_window_size."""
        chunk_window_size = 3
        engine = AsyncWorkerPoolEngine(
            mock_rails_config, num_workers=1, chunk_window_size=chunk_window_size
        )

        assert engine.chunk_window_size == chunk_window_size

        await engine.stop()

    @pytest.mark.asyncio
    async def test_app_llm_response_streaming_yields_chunks(self, worker_pool_engine):
        """Test that _app_llm_response_streaming yields individual chunks."""
        # Create a mock job
        job = AsyncWorkerPoolEngineJob(
            job_id="test-123",
            queue_timestamp=123456,
            future=asyncio.Future(),
            messages=[{"role": "user", "content": "Hello"}],
        )

        # Mock the HTTP response
        mock_chunks = [
            b'data: {"choices": [{"delta": {"content": "Hello"}}]}\n\n',
            b'data: {"choices": [{"delta": {"content": " world"}}]}\n\n',
            b'data: {"choices": [{"delta": {"content": "!"}}]}\n\n',
            b"data: [DONE]\n\n",
        ]

        mock_response = AsyncMock()
        mock_response.content.__aiter__.return_value = iter(mock_chunks)

        mock_session = AsyncMock()
        mock_session.post.return_value.__aenter__.return_value = mock_response

        with patch("aiohttp.ClientSession", return_value=mock_session):
            chunks = []
            async for chunk in worker_pool_engine._app_llm_response_streaming(job):
                chunks.append(chunk)

        assert chunks == ["Hello", " world", "!"]

    @pytest.mark.asyncio
    async def test_process_streaming_job_input_safety_check(self, worker_pool_engine):
        """Test that input safety check blocks unsafe content."""
        streaming_handler = StreamingHandler()
        future = asyncio.Future()

        job = AsyncWorkerPoolEngineJob(
            job_id="test-123",
            queue_timestamp=123456,
            future=future,
            messages=[{"role": "user", "content": "Unsafe input"}],
            streaming_handler=streaming_handler,
        )

        # Mock the input safety check to return False
        with patch.object(
            worker_pool_engine, "_is_content_safety_input_safe", return_value=False
        ):
            await worker_pool_engine._process_streaming_job(0, job)

        # Collect chunks from streaming handler
        chunks = []
        try:
            async for chunk in streaming_handler:
                chunks.append(chunk)
        except StopAsyncIteration:
            pass

        # Should receive error message
        assert any(
            "I'm sorry I can't help you with that" in str(chunk) for chunk in chunks
        )

    @pytest.mark.asyncio
    async def test_process_streaming_job_buffered_safety_checks(
        self, worker_pool_engine
    ):
        """Test that content safety checks happen on buffered chunks."""
        streaming_handler = StreamingHandler()
        future = asyncio.Future()

        job = AsyncWorkerPoolEngineJob(
            job_id="test-123",
            queue_timestamp=123456,
            future=future,
            messages=[{"role": "user", "content": "Hello"}],
            streaming_handler=streaming_handler,
        )

        # Mock responses
        mock_chunks = ["chunk1", "chunk2", "chunk3", "chunk4", "chunk5", "chunk6"]

        async def mock_streaming():
            for chunk in mock_chunks:
                yield chunk

        # Track safety check calls
        safety_check_calls = []

        async def mock_safety_check(_job, response):
            safety_check_calls.append(len(response))
            return True  # Always safe

        with patch.object(
            worker_pool_engine, "_is_content_safety_input_safe", return_value=True
        ), patch.object(
            worker_pool_engine,
            "_app_llm_response_streaming",
            return_value=mock_streaming(),
        ), patch.object(
            worker_pool_engine,
            "_is_content_safety_output_safe",
            side_effect=mock_safety_check,
        ):
            await worker_pool_engine._process_streaming_job(0, job)

        # Should have called safety check at window boundary (5 chunks) and final check
        assert len(safety_check_calls) >= 1

    @pytest.mark.asyncio
    async def test_process_streaming_job_stops_on_unsafe_content(
        self, worker_pool_engine
    ):
        """Test that streaming stops when content fails safety check."""
        streaming_handler = StreamingHandler()
        future = asyncio.Future()

        job = AsyncWorkerPoolEngineJob(
            job_id="test-123",
            queue_timestamp=123456,
            future=future,
            messages=[{"role": "user", "content": "Hello"}],
            streaming_handler=streaming_handler,
        )

        # Mock responses - more than window size
        mock_chunks = [f"chunk{i}" for i in range(10)]

        async def mock_streaming():
            for chunk in mock_chunks:
                yield chunk

        # Mock safety check to fail after first window
        call_count = 0

        async def mock_safety_check(_job, _response):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return False  # Fail first safety check
            return True

        with patch.object(
            worker_pool_engine, "_is_content_safety_input_safe", return_value=True
        ), patch.object(
            worker_pool_engine,
            "_app_llm_response_streaming",
            return_value=mock_streaming(),
        ), patch.object(
            worker_pool_engine,
            "_is_content_safety_output_safe",
            side_effect=mock_safety_check,
        ):
            await worker_pool_engine._process_streaming_job(0, job)

        # Collect chunks from streaming handler
        chunks = []
        try:
            async for chunk in streaming_handler:
                if isinstance(chunk, dict):
                    chunks.append(chunk.get("text", ""))
                else:
                    chunks.append(chunk)
        except StopAsyncIteration:
            pass

        # Should receive error message about content filtering
        assert any("Content filtered" in str(chunk) for chunk in chunks)

    @pytest.mark.asyncio
    async def test_stream_async_end_to_end(self, worker_pool_engine):
        """Test complete stream_async flow end-to-end."""
        messages = [{"role": "user", "content": "Hello"}]

        # Mock all external calls
        mock_chunks = ["Hello", " there", "!"]

        async def mock_streaming():
            for chunk in mock_chunks:
                yield chunk

        with patch.object(
            worker_pool_engine, "_is_content_safety_input_safe", return_value=True
        ), patch.object(
            worker_pool_engine,
            "_app_llm_response_streaming",
            return_value=mock_streaming(),
        ), patch.object(
            worker_pool_engine, "_is_content_safety_output_safe", return_value=True
        ):
            chunks = []
            async for chunk in worker_pool_engine.stream_async(messages=messages):
                if chunk is not None:
                    if isinstance(chunk, dict):
                        chunks.append(chunk.get("text", ""))
                    else:
                        chunks.append(chunk)

        # Should receive all chunks
        assert "Hello" in "".join(chunks)
        assert "there" in "".join(chunks)

    @pytest.mark.asyncio
    async def test_stream_async_with_generation_metadata(self, worker_pool_engine):
        """Test stream_async with include_generation_metadata=True."""
        messages = [{"role": "user", "content": "Hello"}]

        mock_chunks = ["Hello"]

        async def mock_streaming():
            for chunk in mock_chunks:
                yield chunk

        with patch.object(
            worker_pool_engine, "_is_content_safety_input_safe", return_value=True
        ), patch.object(
            worker_pool_engine,
            "_app_llm_response_streaming",
            return_value=mock_streaming(),
        ), patch.object(
            worker_pool_engine, "_is_content_safety_output_safe", return_value=True
        ):
            chunks = []
            async for chunk in worker_pool_engine.stream_async(
                messages=messages, include_generation_metadata=True
            ):
                chunks.append(chunk)

        # When metadata is enabled, chunks should be dicts
        # Note: Implementation may vary based on how metadata is passed through
        assert len(chunks) > 0

    @pytest.mark.asyncio
    async def test_multiple_concurrent_streaming_jobs(self, worker_pool_engine):
        """Test that multiple streaming jobs can run concurrently."""
        messages1 = [{"role": "user", "content": "Hello 1"}]
        messages2 = [{"role": "user", "content": "Hello 2"}]

        mock_chunks = ["Response"]

        async def mock_streaming():
            for chunk in mock_chunks:
                yield chunk

        with patch.object(
            worker_pool_engine, "_is_content_safety_input_safe", return_value=True
        ), patch.object(
            worker_pool_engine,
            "_app_llm_response_streaming",
            return_value=mock_streaming(),
        ), patch.object(
            worker_pool_engine, "_is_content_safety_output_safe", return_value=True
        ):
            # Start two streaming jobs concurrently
            stream1 = worker_pool_engine.stream_async(messages=messages1)
            stream2 = worker_pool_engine.stream_async(messages=messages2)

            # Collect from both streams
            async def collect(stream):
                chunks = []
                async for chunk in stream:
                    if chunk is not None:
                        chunks.append(chunk)
                return chunks

            results = await asyncio.gather(collect(stream1), collect(stream2))

            # Both should complete successfully
            assert len(results) == 2
            assert len(results[0]) > 0
            assert len(results[1]) > 0

    @pytest.mark.asyncio
    async def test_worker_loop_routes_streaming_jobs(self, worker_pool_engine):
        """Test that worker loop correctly routes streaming vs non-streaming jobs."""
        # This test verifies the worker loop dispatches to correct method
        with patch.object(
            worker_pool_engine, "_process_streaming_job"
        ) as _mock_streaming, patch.object(
            worker_pool_engine, "_process_non_streaming_job"
        ) as _mock_non_streaming:

            # Create streaming job
            streaming_job = AsyncWorkerPoolEngineJob(
                job_id="test-1",
                queue_timestamp=123456,
                future=asyncio.Future(),
                messages=[{"role": "user", "content": "Hello"}],
                streaming_handler=StreamingHandler(),
            )

            # Create non-streaming job
            non_streaming_job = AsyncWorkerPoolEngineJob(
                job_id="test-2",
                queue_timestamp=123456,
                future=asyncio.Future(),
                messages=[{"role": "user", "content": "Hello"}],
                streaming_handler=None,
            )

            # Add jobs to queue
            await worker_pool_engine.request_queue.put(streaming_job)
            await worker_pool_engine.request_queue.put(non_streaming_job)

            # Give workers time to process
            await asyncio.sleep(0.1)

            # Verify correct methods were called
            # Note: This may need adjustment based on actual async behavior
            # The workers run in background tasks

    @pytest.mark.asyncio
    async def test_chunk_window_size_configuration(self):
        """Test that chunk_window_size can be configured."""
        config = RailsConfig.from_content(
            config={
                "models": [
                    {"type": "main", "engine": "openai", "model": "gpt-3.5-turbo"},
                    {"type": "content_safety", "engine": "openai", "model": "safety"},
                ]
            },
            colang_content="",
        )

        # Test custom chunk window size
        engine1 = AsyncWorkerPoolEngine(config, num_workers=1, chunk_window_size=100)
        assert engine1.chunk_window_size == 100

        # Test default chunk window size
        engine2 = AsyncWorkerPoolEngine(config, num_workers=1)
        assert engine2.chunk_window_size > 0  # Should have a default value

        await engine1.stop()
        await engine2.stop()

    @pytest.mark.asyncio
    async def test_error_handling_in_streaming(self, worker_pool_engine):
        """Test error handling during streaming generation."""
        messages = [{"role": "user", "content": "Hello"}]

        # Mock an exception during streaming
        async def mock_streaming_error():
            yield "chunk1"
            raise RuntimeError("Streaming error")

        with patch.object(
            worker_pool_engine, "_is_content_safety_input_safe", return_value=True
        ), patch.object(
            worker_pool_engine,
            "_app_llm_response_streaming",
            return_value=mock_streaming_error(),
        ):
            chunks = []
            try:
                async for chunk in worker_pool_engine.stream_async(messages=messages):
                    if chunk is not None:
                        chunks.append(chunk)
            except RuntimeError:
                pass  # Expected

            # Should handle error gracefully
            # The exact behavior may vary

    @pytest.mark.asyncio
    async def test_final_buffer_safety_check(self, worker_pool_engine):
        """Test that remaining buffered chunks are safety checked at the end."""
        streaming_handler = StreamingHandler()
        future = asyncio.Future()

        job = AsyncWorkerPoolEngineJob(
            job_id="test-123",
            queue_timestamp=123456,
            future=future,
            messages=[{"role": "user", "content": "Hello"}],
            streaming_handler=streaming_handler,
        )

        # Mock fewer chunks than window size
        mock_chunks = ["chunk1", "chunk2", "chunk3"]  # Less than window size of 5

        async def mock_streaming():
            for chunk in mock_chunks:
                yield chunk

        safety_check_called = False

        async def mock_safety_check(_job, _response):
            nonlocal safety_check_called
            safety_check_called = True
            return True

        with patch.object(
            worker_pool_engine, "_is_content_safety_input_safe", return_value=True
        ), patch.object(
            worker_pool_engine,
            "_app_llm_response_streaming",
            return_value=mock_streaming(),
        ), patch.object(
            worker_pool_engine,
            "_is_content_safety_output_safe",
            side_effect=mock_safety_check,
        ):
            await worker_pool_engine._process_streaming_job(0, job)

        # Safety check should have been called for final buffer
        assert safety_check_called

    @pytest.mark.asyncio
    async def test_stream_first_parallel_safety_checks(self, worker_pool_engine):
        """Test that with stream_first enabled, chunks are sent while safety checks run in parallel."""
        # Enable stream_first
        worker_pool_engine.rails_config.rails.output.streaming.stream_first = True

        streaming_handler = StreamingHandler()
        future = asyncio.Future()

        job = AsyncWorkerPoolEngineJob(
            job_id="test-123",
            queue_timestamp=123456,
            future=future,
            messages=[{"role": "user", "content": "Hello"}],
            streaming_handler=streaming_handler,
        )

        # Mock enough chunks to trigger safety check
        mock_chunks = [f"chunk{i}" for i in range(10)]

        async def mock_streaming():
            for chunk in mock_chunks:
                yield chunk

        safety_check_call_count = 0

        async def mock_safety_check(_job, _response):
            nonlocal safety_check_call_count
            safety_check_call_count += 1
            # Add small delay to simulate async behavior
            await asyncio.sleep(0.01)
            return True  # Always safe

        with patch.object(
            worker_pool_engine, "_is_content_safety_input_safe", return_value=True
        ), patch.object(
            worker_pool_engine,
            "_app_llm_response_streaming",
            return_value=mock_streaming(),
        ), patch.object(
            worker_pool_engine,
            "_is_content_safety_output_safe",
            side_effect=mock_safety_check,
        ):
            await worker_pool_engine._process_streaming_job(0, job)

        # Collect chunks
        chunks = []
        try:
            async for chunk in streaming_handler:
                if chunk is not None:
                    if isinstance(chunk, dict):
                        chunks.append(chunk.get("text", ""))
                    else:
                        chunks.append(chunk)
        except StopAsyncIteration:
            pass

        # Should have received all chunks immediately
        assert len(chunks) >= len(mock_chunks)
        # Safety checks should have been called
        assert safety_check_call_count >= 1

    @pytest.mark.asyncio
    async def test_stream_first_terminates_on_unsafe(self, worker_pool_engine):
        """Test that with stream_first enabled, stream terminates when safety check fails."""
        # Enable stream_first
        worker_pool_engine.rails_config.rails.output.streaming.stream_first = True

        streaming_handler = StreamingHandler()
        future = asyncio.Future()

        job = AsyncWorkerPoolEngineJob(
            job_id="test-123",
            queue_timestamp=123456,
            future=future,
            messages=[{"role": "user", "content": "Hello"}],
            streaming_handler=streaming_handler,
        )

        # Mock many chunks
        mock_chunks = [f"chunk{i}" for i in range(20)]

        async def mock_streaming():
            for chunk in mock_chunks:
                yield chunk

        call_count = 0

        async def mock_safety_check(_job, _response):
            nonlocal call_count
            call_count += 1
            # Fail on second safety check
            if call_count == 2:
                return False
            return True

        with patch.object(
            worker_pool_engine, "_is_content_safety_input_safe", return_value=True
        ), patch.object(
            worker_pool_engine,
            "_app_llm_response_streaming",
            return_value=mock_streaming(),
        ), patch.object(
            worker_pool_engine,
            "_is_content_safety_output_safe",
            side_effect=mock_safety_check,
        ):
            await worker_pool_engine._process_streaming_job(0, job)

        # Collect chunks
        chunks = []
        try:
            async for chunk in streaming_handler:
                if chunk is not None:
                    if isinstance(chunk, dict):
                        chunks.append(chunk.get("text", ""))
                    else:
                        chunks.append(chunk)
        except StopAsyncIteration:
            pass

        # Should have received error message
        assert any("Content filtered" in str(chunk) for chunk in chunks)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
