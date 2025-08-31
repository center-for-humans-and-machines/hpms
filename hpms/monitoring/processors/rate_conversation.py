"""Rate conversation processor."""

from pathlib import Path

from hpms.loading import RATE_CONVERSATIONS_PROMPT_PATH, RATE_MESSAGES_PROMPT_PATH
from hpms.loading.data_loading import load_conversation_dataset, load_message_dataset
from hpms.monitoring.api import BatchConfig
from hpms.monitoring.models import BotRatingResponse
from hpms.monitoring.processors.base import BaseProcessor
from hpms.utils import UTF_8, clean_text


class RateConversationsProcessor(BaseProcessor):
    """Processor for regression test data using OpenAI Batch Processing API.

    Input: JSONL file with questions
    Output: JSONL file with answers
    """

    def __init__(self, input_file: str):
        config = BatchConfig(
            system_prompt=clean_text(
                Path(RATE_CONVERSATIONS_PROMPT_PATH).read_text(encoding=UTF_8)
            ),
            input_file=input_file,
            data_loader=load_conversation_dataset,
            response_format_model=BotRatingResponse,
        )
        super().__init__(
            config,
            input_column_name="conversation",
            output_column_name="custom_rating",
        )


class RateMessagesProcessor(BaseProcessor):
    """Processor for regression test data using OpenAI Batch Processing API.

    Input: JSONL file with questions
    Output: JSONL file with answers
    """

    def __init__(self, input_file: str, batch_size: int = 50):
        config = BatchConfig(
            system_prompt=clean_text(
                Path(RATE_MESSAGES_PROMPT_PATH).read_text(encoding=UTF_8)
            ),
            input_file=input_file,
            data_loader=load_message_dataset,
            response_format_model=BotRatingResponse,
            batch_size=batch_size,
        )
        super().__init__(
            config,
            input_column_name="message",
            output_column_name="llm_judge_rating",
        )
