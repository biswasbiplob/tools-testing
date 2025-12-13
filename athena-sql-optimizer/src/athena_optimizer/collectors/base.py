"""Base collector with resource management."""

from abc import ABC, abstractmethod
from typing import Optional
import boto3
from botocore.client import BaseClient

from ..models import OptimizerConfig


class BaseCollector(ABC):
    """Base class for AWS collectors with proper resource management."""

    def __init__(self, config: OptimizerConfig):
        self.config = config
        self._session: Optional[boto3.Session] = None
        self._client: Optional[BaseClient] = None
        self._closed = False

    @property
    def session(self) -> boto3.Session:
        """Get or create boto3 session."""
        if self._session is None:
            session_kwargs = {"region_name": self.config.region}
            if self.config.aws_profile:
                session_kwargs["profile_name"] = self.config.aws_profile
            self._session = boto3.Session(**session_kwargs)
        return self._session

    @property
    @abstractmethod
    def client_name(self) -> str:
        """AWS service name (e.g., 'athena', 'glue')."""
        pass

    @property
    def client(self) -> BaseClient:
        """Get or create service client."""
        if self._client is None:
            self._client = self.session.client(self.client_name)
        return self._client

    def close(self):
        """Clean up resources."""
        if not self._closed:
            # boto3 clients don't have explicit close in older versions
            # but we can clear references to help GC
            self._client = None
            self._session = None
            self._closed = True

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False

    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.close()
        except:
            pass  # Ignore errors in __del__
