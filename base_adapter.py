"""Base interface for RL model inference adapters."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class TradeSignal:
    """Output of an RL adapter's predict() call."""
    ticker: str
    action: int          # signed share count (+ buy, - sell, 0 hold)
    raw_action: float    # continuous action from model before scaling
    confidence: float    # |raw_action| as a rough proxy


@dataclass
class PredictionResult:
    """Batch prediction result for all tickers."""
    signals: list[TradeSignal] = field(default_factory=list)
    turbulence: float = 0.0
    turbulence_sell_all: bool = False
    metadata: dict = field(default_factory=dict)


class BaseRLAdapter(ABC):
    """Common interface for loading a trained RL model and running inference.

    Subclasses must implement:
        load_model()       — load checkpoint + any normalization artefacts
        build_observation() — market data → model-ready observation vector
        predict()          — observation → PredictionResult
    """

    @abstractmethod
    def load_model(self, checkpoint_path: str, **kwargs) -> None:
        ...

    @abstractmethod
    def build_observation(
        self,
        cash: float,
        shares: np.ndarray,
        close_prices: np.ndarray,
        tech_features: np.ndarray,
    ) -> np.ndarray:
        ...

    @abstractmethod
    def predict(
        self,
        cash: float,
        shares: np.ndarray,
        close_prices: np.ndarray,
        tech_features: np.ndarray,
        tickers: list[str],
        turbulence: float = 0.0,
    ) -> PredictionResult:
        ...
