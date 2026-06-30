"""Base interface for transaction data sources."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class TransactionDataSource(ABC):
    """Load transaction data without exposing storage-specific details."""

    @abstractmethod
    def load(
        self,
        start_at: str | pd.Timestamp | None = None,
        end_at: str | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Return a cleaned DataFrame that matches the input data contract."""
