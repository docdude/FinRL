"""FinRL PPO inference adapter using state_dict loading.

Reconstructs the ActorPPO architecture locally — no finrl/elegantrl package needed.
Only requires torch.

Model spec (DOW-29, trained via FinRL StockTradingEnv):
    state_dim  = 322   (1 cash + 2 turb + 3*29 price/stocks/cd + 8*29 tech)
    action_dim = 29
    net_dims   = [32, 16]
    activation = ReLU
    output     = tanh

State encoding (matches FinRL StockTradingEnv.get_state / AlpacaPaperTrading.get_state):
    [cash * CASH_NORM,
     sigmoid_sign(turbulence, thresh) * 2^-5,
     turbulence_bool,
     price * 2^-6,
     stocks * 2^-6,
     stocks_cd,
     tech * 2^-7]

No VecNormalize — normalization is baked into the state encoding.

CASH_NORM = 2^-12 * (1_000_000 / initial_capital)
    e.g. $100K account → CASH_NORM ≈ 0.00244
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .base_adapter import BaseRLAdapter, PredictionResult, TradeSignal

log = logging.getLogger(__name__)


# ── Standalone network (inference-only) ─────────────────────────────────────

def _build_mlp(dims: list[int]) -> nn.Sequential:
    """MLP with ReLU (matching FinRL's build_mlp)."""
    layers = []
    for i in range(len(dims) - 1):
        layers.extend([nn.Linear(dims[i], dims[i + 1]), nn.ReLU()])
    del layers[-1]  # remove trailing ReLU after output layer
    return nn.Sequential(*layers)


class _ActorPPO(nn.Module):
    """Minimal FinRL ActorPPO for inference."""

    def __init__(self, state_dim: int, net_dims: list[int], action_dim: int):
        super().__init__()
        self.net = _build_mlp(dims=[state_dim, *net_dims, action_dim])
        # FinRL uses nn.Parameter (not buffer) for action_std_log
        self.action_std_log = nn.Parameter(
            torch.zeros((1, action_dim)), requires_grad=True
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).tanh()


# ── Helpers ─────────────────────────────────────────────────────────────────

def _sigmoid_sign(ary: np.ndarray, thresh: float) -> np.ndarray:
    """Sigmoid-based soft sign, matching FinRL's AlpacaPaperTrading.sigmoid_sign."""
    def sigmoid(x):
        return 1 / (1 + np.exp(-x * np.e)) - 0.5
    return sigmoid(ary / thresh) * thresh


# ── Adapter ─────────────────────────────────────────────────────────────────

DOW_29_TICKERS = [
    "AAPL", "AMGN", "AMZN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX",
    "DIS", "GS", "HD", "HON", "IBM", "INTC", "JNJ", "JPM",
    "KO", "MCD", "MMM", "MRK", "MSFT", "NKE", "PG", "TRV",
    "UNH", "V", "VZ", "WMT",
]


class FinRLAdapter(BaseRLAdapter):
    """Load a FinRL-trained ActorPPO from a state_dict checkpoint.

    Parameters
    ----------
    state_dim, action_dim, net_dims : model architecture (must match training).
    initial_capital : account starting capital — used to compute CASH_NORM.
    max_stock : action scaling factor (action * max_stock → share count).
    min_action : minimum |shares| to emit a trade signal.
    turbulence_thresh : if turbulence >= this, emit sell-all signal.
        Notebook default is 30 but deployed override is 99 — turbulence
        regularly sits ~35, so 30 freezes trading.
    device : 'cpu' or 'cuda:N'.
    """

    def __init__(
        self,
        state_dim: int = 322,
        action_dim: int = 29,
        net_dims: list[int] | None = None,
        initial_capital: float = 100_000,
        max_stock: int = 100,
        min_action: int = 10,
        buy_cost_pct: float = 0.001,
        sell_cost_pct: float = 0.001,
        turbulence_thresh: float = 99.0,
        device: str = "cpu",
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.net_dims = net_dims or [32, 16]
        self.initial_capital = initial_capital
        self.cash_norm = 2**-12 * (1_000_000 / initial_capital)
        self.max_stock = max_stock
        self.min_action = min_action
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.turbulence_thresh = turbulence_thresh
        self.device = device

        self.actor: _ActorPPO | None = None

        # Cooldown tracker — persists across predict() calls
        self._stocks_cd: np.ndarray | None = None

    # ── BaseRLAdapter interface ─────────────────────────────────────────

    def load_model(self, checkpoint_path: str, **kwargs) -> None:
        self.actor = _ActorPPO(
            state_dim=self.state_dim,
            net_dims=self.net_dims,
            action_dim=self.action_dim,
        ).to(self.device)

        sd = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.actor.load_state_dict(sd, strict=True)
        self.actor.eval()
        n_params = sum(p.numel() for p in self.actor.parameters())
        log.info("Loaded FinRL actor state_dict from %s (%d params)", checkpoint_path, n_params)

        # Initialize cooldown tracker
        self._stocks_cd = np.zeros(self.action_dim, dtype=np.float32)

    def build_observation(
        self,
        cash: float,
        shares: np.ndarray,
        close_prices: np.ndarray,
        tech_features: np.ndarray,
        turbulence: float = 0.0,
    ) -> np.ndarray:
        """Construct state matching FinRL StockTradingEnv / AlpacaPaperTrading.get_state().

        State layout (322-dim for 29 stocks):
            [cash * CASH_NORM,                          # (1,)
             sigmoid_sign(turbulence) * 2^-5,           # (1,)
             turbulence_bool,                            # (1,)
             close_prices * 2^-6,                        # (29,)
             shares * 2^-6,                              # (29,)
             stocks_cd,                                   # (29,)
             tech_features * 2^-7]                       # (232,)

        Note: tech_features should be raw (unscaled) values from
        AlpacaProcessor — the 2^-7 scaling is applied here.
        """
        turb_scaled = (_sigmoid_sign(
            np.array([turbulence], dtype=np.float32),
            self.turbulence_thresh,
        ) * 2**-5).astype(np.float32)

        turb_bool = np.array(
            [1.0 if turbulence > self.turbulence_thresh else 0.0],
            dtype=np.float32,
        )

        amount = np.array([cash * self.cash_norm], dtype=np.float32)
        scale = np.float32(2**-6)

        if self._stocks_cd is None:
            self._stocks_cd = np.zeros(len(shares), dtype=np.float32)

        state = np.concatenate([
            amount,
            turb_scaled,
            turb_bool,
            close_prices.astype(np.float32) * scale,
            shares.astype(np.float32) * scale,
            self._stocks_cd,
            tech_features.astype(np.float32) * np.float32(2**-7),
        ]).astype(np.float32)

        state = np.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)
        return state

    def predict(
        self,
        cash: float,
        shares: np.ndarray,
        close_prices: np.ndarray,
        tech_features: np.ndarray,
        tickers: list[str],
        turbulence: float = 0.0,
    ) -> PredictionResult:
        if self.actor is None:
            raise RuntimeError("Call load_model() before predict()")

        turb_sell_all = turbulence > self.turbulence_thresh

        obs = self.build_observation(cash, shares, close_prices, tech_features, turbulence)
        state_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            action_t = self.actor(state_t.unsqueeze(0)).squeeze(0)
        raw_actions = action_t.cpu().numpy()
        action_int = (raw_actions * self.max_stock).astype(int)

        # Update cooldown: increment all, reset traded tickers
        self._stocks_cd += 1

        # Resolve final action per ticker — matching notebook order:
        #   1. all sells first  (cash increases from proceeds)
        #   2. all buys second  (using updated cash)
        actions = np.zeros(len(tickers), dtype=int)

        if turb_sell_all:
            # Sell-all mode: liquidate every held position, reset all cooldowns
            for i in range(len(tickers)):
                actions[i] = -int(shares[i]) if shares[i] > 0 else 0
            self._stocks_cd[:] = 0
        else:
            # --- Pass 1: sells (action < -min_action) ---
            sell_indices = np.where(action_int < -self.min_action)[0]
            for i in sell_indices:
                sell_num = min(int(shares[i]), -int(action_int[i]))
                actions[i] = -sell_num
                self._stocks_cd[i] = 0

            # Credit sell proceeds into available cash (conservative: deduct sell cost)
            available_cash = cash
            for i in sell_indices:
                available_cash += (-actions[i]) * close_prices[i] * (1 - self.sell_cost_pct)

            # --- Pass 2: buys (action > min_action) ---
            buy_indices = np.where(action_int > self.min_action)[0]
            for i in buy_indices:
                price = close_prices[i]
                if price <= 0 or available_cash <= 0:
                    actions[i] = 0
                    continue
                affordable = int(available_cash // price)
                buy_num = min(affordable, abs(int(action_int[i])))
                actions[i] = buy_num
                available_cash -= buy_num * price * (1 + self.buy_cost_pct)
                self._stocks_cd[i] = 0

        signals = []
        for i, ticker in enumerate(tickers):
            signals.append(TradeSignal(
                ticker=ticker,
                action=int(actions[i]),
                raw_action=float(raw_actions[i]),
                confidence=abs(float(raw_actions[i])),
            ))

        return PredictionResult(
            signals=signals,
            turbulence=turbulence,
            turbulence_sell_all=turb_sell_all,
            metadata={
                "checkpoint": "finrl",
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
                "cash_norm": self.cash_norm,
            },
        )

    def reset_cooldown(self):
        """Reset the stocks_cd cooldown tracker (e.g., at start of trading day)."""
        if self._stocks_cd is not None:
            self._stocks_cd[:] = 0
