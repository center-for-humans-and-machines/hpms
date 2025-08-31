"""Regression test processor for OpenAI Batch Processing API."""

from pathlib import Path

from hpms.loading import (
    COMPANION_SYSTEM_PROMPT_PATH,
    DATASET_2_CONVERSATION_STARTERS,
    load_regression_test_prompts,
)
from hpms.monitoring.api import BatchConfig
from hpms.monitoring.constants import RegTestColumns
from hpms.monitoring.processors.base import BaseProcessor
from hpms.utils import UTF_8, clean_text


class RegTestProcessor(BaseProcessor):
    """Processor for regression test data using OpenAI Batch Processing API.

    Input: JSONL file with questions
    Output: JSONL file with answers
    """

    def __init__(self):
        config = BatchConfig(
            system_prompt=clean_text(
                Path(COMPANION_SYSTEM_PROMPT_PATH).read_text(encoding=UTF_8)
            ),
            input_file=DATASET_2_CONVERSATION_STARTERS,
            data_loader=load_regression_test_prompts,
        )
        super().__init__(
            config,
            input_column_name=RegTestColumns.QUESTION,
            output_column_name=RegTestColumns.ANSWER,
        )
