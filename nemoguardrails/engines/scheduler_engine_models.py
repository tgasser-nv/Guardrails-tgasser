

import asyncio

from pydantic import BaseModel
from dataclasses import dataclass, field
from nemoguardrails.rails.llm.options import GenerationOptions


# # Cna't use a Pydantic model here because it can't use Futures
# class SchedulerEngineJob(BaseModel):
#     """Pydantic class in which to store incoming work items"""
#
#     job_id: str
#     messages: list[dict]
#     options: GenerationOptions | None
#
#     queue_timestamp: int  # Queue epoch time
#     work_timestamp: int | None  # Work epoch time (set after popping from work queue)
#     completed_timestamp: int | None   # Completed epoch time (set after workflow completes)
#     future: asyncio.Future  # Future with request result
#
#     # Validate any re-assigned fields
#     model_config = {
#         "validate_assignment": True
#     }

@dataclass
class SchedulerEngineJob:
    """Dataclass in which to store incoming work items"""

    job_id: str
    queue_timestamp: int
    future: asyncio.Future

    # Use default_factory for mutable types (lists/dicts)
    messages: list[dict] = field(default_factory=list)

    # Optional fields default to None
    options: GenerationOptions | None = None
    work_timestamp: int | None = None
    completed_timestamp: int | None = None
