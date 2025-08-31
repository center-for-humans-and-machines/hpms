"""Entrypoint for the API used in monitoring."""

from hpms.monitoring.api.client import create_client
from hpms.monitoring.api.config import BatchConfig

__all__ = ["create_client", "BatchConfig"]
