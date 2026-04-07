"""RL model trading loop.

Connects a BaseRLAdapter (model inference) to AlpacaBroker (order execution)
with market-aware scheduling and two data pipeline modes:

- 'intraday': 1-minute bars via AlpacaProcessor.fetch_latest_data()
              Turbulence from VIXY close. For FinRL-style models.
- 'daily':    Daily bars with full feature pipeline (download → clean →
              indicators → VIX → turbulence). For ElegantRL-style models.

Usage:

    from strategies.rl import FinRLAdapter
    from strategies.rl.trade_loop import RLTradeLoop
    from alpaca_trading import AlpacaBroker

    adapter = FinRLAdapter()
    adapter.load_model("rl_models/finRL/papertrading_erl_v2_deploy/actor.pth")

    broker = AlpacaBroker(paper_trading=True)
    broker.connect()

    loop = RLTradeLoop(
        adapter=adapter,
        broker=broker,
        tickers=DOW_29_TICKERS,
        tech_indicators=INDICATORS,
        api_key=os.environ["ALPACA_API_KEY"],
        api_secret=os.environ["ALPACA_API_SECRET"],
        data_mode="intraday",
        trade_interval_seconds=60,
    )
    loop.run()           # continuous intraday loop
    # or
    loop.run_once()      # single cycle (e.g. daily cron job)
"""
from __future__ import annotations

import datetime
import logging
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from alpaca_trading import AlpacaBroker, AlpacaProcessor
from alpaca_trading.alpaca_health_monitor import BrokerHealthMonitor
from broker_framework.broker_interface import OrderSide

from .base_adapter import BaseRLAdapter, PredictionResult

log = logging.getLogger(__name__)

INDICATORS = [
    "macd", "boll_ub", "boll_lb", "rsi_30",
    "cci_30", "dx_30", "close_30_sma", "close_60_sma",
]


@dataclass
class TradeCycleSummary:
    """Result of a single trading cycle."""
    timestamp: str
    cash: float
    total_asset: float
    num_positions: int
    turbulence: float
    turbulence_sell_all: bool
    buys: int
    sells: int
    orders_placed: int
    orders_failed: int
    errors: list[str] = field(default_factory=list)


class RLTradeLoop:
    """Execute RL model predictions through AlpacaBroker.

    Parameters
    ----------
    adapter : BaseRLAdapter
        Loaded RL adapter (call load_model() before passing).
    broker : AlpacaBroker
        Connected AlpacaBroker instance.
    tickers : list[str]
        Sorted ticker list (must match model training order).
    tech_indicators : list[str]
        Technical indicator names (must match training).
    api_key, api_secret : str
        Alpaca API credentials (for AlpacaProcessor data fetching).
    data_mode : str
        'intraday' (1-min bars) or 'daily' (daily bars with turbulence).
    trade_interval_seconds : int
        Sleep between trading cycles (default 60 for 1-min bars).
    close_buffer_seconds : int
        Stop trading this many seconds before market close (default 120).
    cancel_open_on_start : bool
        Cancel all open orders when run() starts (default True).
    lookback_days : int
        Lookback for daily data pipeline (default 400, needs >= 252
        trading days for turbulence covariance calculation).
    """

    def __init__(
        self,
        adapter: BaseRLAdapter,
        broker: AlpacaBroker,
        tickers: list[str],
        tech_indicators: list[str] | None = None,
        api_key: str | None = None,
        api_secret: str | None = None,
        data_mode: str = "intraday",
        trade_interval_seconds: int = 60,
        close_buffer_seconds: int = 120,
        cancel_open_on_start: bool = True,
        lookback_days: int = 400,
        health_monitor: BrokerHealthMonitor | None = None,
    ):
        self.adapter = adapter
        self.broker = broker
        self.tickers = sorted(tickers)
        self.tech_indicators = tech_indicators or INDICATORS
        self.data_mode = data_mode
        self.trade_interval_seconds = trade_interval_seconds
        self.close_buffer_seconds = close_buffer_seconds
        self.cancel_open_on_start = cancel_open_on_start
        self.lookback_days = lookback_days

        # Health monitoring
        self.health_monitor = health_monitor or BrokerHealthMonitor()
        self.health_monitor.record_connection(True)

        # Data processor for market data fetching
        self.processor = AlpacaProcessor(API_KEY=api_key, API_SECRET=api_secret)

        # Runtime state
        self._running = False
        self.trade_log: list[TradeCycleSummary] = []

    # ── Public API ──────────────────────────────────────────────────────

    def run(self) -> list[TradeCycleSummary]:
        """Main trading loop.

        1. Cancel stale open orders (optional)
        2. Wait for market to open
        3. Loop: fetch data → predict → execute → sleep
        4. Stop when market is about to close

        Returns list of TradeCycleSummary for each completed cycle.
        """
        if self.cancel_open_on_start:
            self._cancel_open_orders()

        self._await_market_open()
        self._running = True
        log.info("Trading loop started (mode=%s, interval=%ds, buffer=%ds)",
                 self.data_mode, self.trade_interval_seconds, self.close_buffer_seconds)

        while self._running:
            ttc = self._time_to_close()
            if ttc < self.close_buffer_seconds:
                log.info("Market closing in %.0fs (buffer=%ds), stopping loop",
                         ttc, self.close_buffer_seconds)
                break

            # Check broker health before trading
            if self.health_monitor.should_fallback():
                log.warning("Broker health fallback active (%s), skipping cycle",
                            self.health_monitor.get_fallback_reason())
            else:
                try:
                    summary = self.run_once()
                    self.trade_log.append(summary)
                except Exception as e:
                    log.error("Trade cycle failed: %s", e, exc_info=True)

            if self._running:
                time.sleep(self.trade_interval_seconds)

        self._running = False
        log.info("Trading loop finished — %d cycles completed", len(self.trade_log))
        return self.trade_log

    def run_once(self) -> TradeCycleSummary:
        """Execute a single trading cycle: fetch → predict → execute.

        Can be called standalone (e.g. from a cron job) or repeatedly
        inside run().
        """
        errors: list[str] = []

        # 1. Fetch market data
        try:
            prices, tech, turb = self._fetch_data()
        except Exception as e:
            log.error("Data fetch failed: %s", e)
            raise

        # 2. Sync portfolio from broker
        cash, shares = self._sync_portfolio()

        # 3. Predict
        result = self.adapter.predict(
            cash=cash,
            shares=shares,
            close_prices=prices,
            tech_features=tech,
            tickers=self.tickers,
            turbulence=turb,
        )

        buy_count = sum(1 for s in result.signals if s.action > 0)
        sell_count = sum(1 for s in result.signals if s.action < 0)
        log.info("Prediction: buys=%d, sells=%d, turb=%.2f, sell_all=%s",
                 buy_count, sell_count, turb, result.turbulence_sell_all)

        # 4. Execute orders (sells first, then buys — matching notebook)
        placed, failed = self._execute_signals(result, prices)

        # 5. Build summary
        # Re-sync to get updated totals
        cash_after, shares_after = self._sync_portfolio()
        total_asset = cash_after + float(np.dot(prices, shares_after))

        summary = TradeCycleSummary(
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            cash=cash_after,
            total_asset=total_asset,
            num_positions=int((shares_after > 0).sum()),
            turbulence=turb,
            turbulence_sell_all=result.turbulence_sell_all,
            buys=buy_count,
            sells=sell_count,
            orders_placed=placed,
            orders_failed=failed,
            errors=errors,
        )
        log.info("Cycle done: total=$%.2f, positions=%d, placed=%d, failed=%d",
                 total_asset, summary.num_positions, placed, failed)
        return summary

    def stop(self):
        """Signal the trading loop to stop after the current cycle."""
        self._running = False
        log.info("Stop requested")

    # ── Data fetching ───────────────────────────────────────────────────

    def _fetch_data(self) -> tuple[np.ndarray, np.ndarray, float]:
        """Fetch latest market data. Returns (prices, tech, turbulence)."""
        if self.data_mode == "intraday":
            return self._fetch_intraday()
        elif self.data_mode == "daily":
            return self._fetch_daily()
        else:
            raise ValueError(f"Unknown data_mode: {self.data_mode!r}")

    def _fetch_intraday(self) -> tuple[np.ndarray, np.ndarray, float]:
        """1-minute bar pipeline (FinRL-style).

        Uses AlpacaProcessor.fetch_latest_data() which returns:
        - latest_price: (num_tickers,) close prices
        - latest_tech:  (num_tickers * num_indicators,) flattened tech
        - latest_turb:  VIXY close price array
        """
        prices, tech, turb_arr = self.processor.fetch_latest_data(
            ticker_list=self.tickers,
            time_interval="1Min",
            tech_indicator_list=self.tech_indicators,
        )
        # turb_arr is VIXY close — extract scalar
        turb = float(turb_arr[-1]) if hasattr(turb_arr, "__len__") and len(turb_arr) > 0 else float(turb_arr)
        return np.asarray(prices, dtype=np.float32), np.asarray(tech, dtype=np.float32), turb

    def _fetch_daily(self) -> tuple[np.ndarray, np.ndarray, float]:
        """Daily bar pipeline (ElegantRL-style).

        Full pipeline: download → clean → indicators → VIX → turbulence.
        Tech features include vix and turbulence appended (matching
        ElegantRL's training state encoding).
        """
        end_date = datetime.date.today().isoformat()
        start_date = (
            datetime.date.today() - datetime.timedelta(days=int(self.lookback_days * 1.5))
        ).isoformat()

        df = self.processor.download_data(
            self.tickers, start_date, end_date, time_interval="1Day"
        )
        df = self.processor.clean_data(df)
        df = self.processor.add_technical_indicator(df, self.tech_indicators)

        # VIX
        try:
            df = self.processor.add_vix(df)
            if "VIXY" in df.columns and "vix" not in df.columns:
                df = df.rename(columns={"VIXY": "vix"})
        except Exception as e:
            log.warning("VIX unavailable (%s), filling zeros", e)
            df["vix"] = 0.0

        # Turbulence (covariance-based)
        try:
            df = self.processor.add_turbulence(df)
        except Exception as e:
            log.warning("Turbulence calculation failed (%s), filling zeros", e)
            df["turbulence"] = 0.0

        df = df.fillna(0).replace([np.inf, -np.inf], 0)

        # Extract latest day arrays
        prices, tech, turb_raw = self._df_to_latest_arrays(df)
        return prices, tech, turb_raw

    def _df_to_latest_arrays(
        self, df: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Extract latest-day arrays from a daily DataFrame.

        Returns:
            close_prices  : (num_tickers,)
            tech_features : (num_tickers * num_indicators + market_features,)
                            Market features = [vix, turbulence] if available.
            turbulence_raw: float (un-scaled, for circuit breaker)
        """
        tickers = self.tickers
        indicators = self.tech_indicators

        if not pd.api.types.is_datetime64_any_dtype(df["date"]):
            df["date"] = pd.to_datetime(df["date"])

        dates = sorted(df["date"].unique())
        latest_date = dates[-1]

        # Close prices
        close_pivot = df.pivot_table(
            index="date", columns="tic", values="close", aggfunc="first"
        )
        close_pivot = close_pivot.reindex(columns=tickers)
        close_prices = close_pivot.loc[latest_date].values.astype(np.float32)

        # Tech indicators per stock
        available = [t for t in indicators if t in df.columns]
        tech_arys = []
        for tech_name in available:
            pivot = df.pivot_table(
                index="date", columns="tic", values=tech_name, aggfunc="first"
            )
            pivot = pivot.reindex(columns=tickers)
            tech_arys.append(pivot.loc[latest_date].values.astype(np.float32))

        # Stack (num_tickers, num_indicators) then flatten
        tech_matrix = np.stack(tech_arys, axis=-1)  # (num_tickers, num_indicators)
        tech_flat = tech_matrix.reshape(-1)

        # Market-wide features (vix, turbulence)
        market = []
        if "vix" in df.columns:
            vix_val = df[df["date"] == latest_date].groupby("date")["vix"].first().values[0]
            market.append(np.float32(vix_val))
        if "turbulence" in df.columns:
            turb_val = df[df["date"] == latest_date].groupby("date")["turbulence"].first().values[0]
            market.append(np.float32(turb_val))
            turb_raw = float(turb_val)
        else:
            turb_raw = 0.0

        tech_features = np.concatenate(
            [tech_flat, np.array(market, dtype=np.float32)]
        )

        close_prices = np.nan_to_num(close_prices, nan=0.0)
        tech_features = np.nan_to_num(tech_features, nan=0.0)

        log.info("Latest date: %s | prices: %s | tech: %s | turb: %.2f",
                 str(latest_date)[:10], close_prices.shape, tech_features.shape, turb_raw)
        return close_prices, tech_features, turb_raw

    # ── Portfolio sync ──────────────────────────────────────────────────

    def _sync_portfolio(self) -> tuple[float, np.ndarray]:
        """Read cash and per-ticker share counts from broker.

        Returns (cash, shares) where shares is a numpy array aligned
        with self.tickers.
        """
        account = self.broker.get_account_info()
        cash = account.cash

        positions = self.broker.get_positions()
        shares = np.zeros(len(self.tickers), dtype=np.float32)
        for pos in positions:
            if pos.symbol in self.tickers:
                idx = self.tickers.index(pos.symbol)
                shares[idx] = abs(pos.qty)

        log.info("Portfolio synced: cash=$%.2f, positions=%d",
                 cash, int((shares > 0).sum()))
        return cash, shares

    # ── Order execution ─────────────────────────────────────────────────

    def _execute_signals(
        self,
        result: PredictionResult,
        prices: np.ndarray | None = None,
    ) -> tuple[int, int]:
        """Convert PredictionResult into broker orders.

        Matches notebook AlpacaPaperTrading.trade() execution order:
        1. Turbulence sell-all: liquidate ALL broker positions (not just
           model universe — matches notebook list_positions() approach).
        2. Normal mode sells first (clamped to held shares).
        3. Refresh cash from broker after sells settle.
        4. Normal mode buys (clamped to affordable using data-fetch prices,
           matching notebook's ``min(cash // self.price[i], abs(action))``).

        Parameters
        ----------
        result : PredictionResult from adapter.predict()
        prices : close prices from the data fetch (used for buy
            affordability, matching the notebook's ``self.price``).
            Falls back to live broker quote when unavailable.

        Returns (orders_placed, orders_failed).
        """
        placed = 0
        failed = 0

        # ── Turbulence sell-all (notebook: list ALL positions, sell each) ──
        if result.turbulence_sell_all:
            positions = self.broker.get_positions()
            for pos in positions:
                qty = abs(int(float(pos.qty)))
                if qty <= 0:
                    continue
                t0 = time.time()
                order_id = self.broker.place_order(
                    symbol=pos.symbol, qty=qty, side=OrderSide.SELL,
                )
                latency_ms = (time.time() - t0) * 1000
                if order_id:
                    self.health_monitor.record_api_call(True, latency_ms)
                    log.info("TURB SELL-ALL %d %s (order %s)",
                             qty, pos.symbol, order_id)
                    placed += 1
                else:
                    self.health_monitor.record_api_call(False, latency_ms)
                    log.warning("TURB SELL-ALL FAILED %d %s",
                                qty, pos.symbol)
                    failed += 1
            return placed, failed

        # ── Normal mode ─────────────────────────────────────────────────
        # Fresh portfolio state for sell-clamping safety net
        cash, shares = self._sync_portfolio()
        ticker_to_idx = {t: i for i, t in enumerate(self.tickers)}

        # --- Pass 1: Sells (notebook: for index in np.where(action < -min_action)) ---
        for signal in result.signals:
            if signal.action >= 0:
                continue
            idx = ticker_to_idx.get(signal.ticker)
            held = int(shares[idx]) if idx is not None else 0
            qty = min(abs(signal.action), held)
            if qty <= 0:
                continue
            t0 = time.time()
            order_id = self.broker.place_order(
                symbol=signal.ticker, qty=qty, side=OrderSide.SELL,
            )
            latency_ms = (time.time() - t0) * 1000
            if order_id:
                self.health_monitor.record_api_call(True, latency_ms)
                log.info("SELL %d %s (order %s)", qty, signal.ticker, order_id)
                placed += 1
                shares[idx] -= qty
            else:
                self.health_monitor.record_api_call(False, latency_ms)
                log.warning("SELL FAILED %d %s", qty, signal.ticker)
                failed += 1

        # Refresh cash after sells (notebook: self.cash = float(self.alpaca.get_account().cash))
        cash, _ = self._sync_portfolio()

        # --- Pass 2: Buys (notebook: min(cash // self.price[i], abs(action))) ---
        for signal in result.signals:
            if signal.action <= 0:
                continue
            idx = ticker_to_idx.get(signal.ticker)

            # Use data-fetch prices for affordability (matches notebook's
            # self.price[index]).  Fall back to live quote if unavailable.
            price = 0.0
            if prices is not None and idx is not None and idx < len(prices):
                price = float(prices[idx])
            if price <= 0:
                try:
                    quote = self.broker.get_quote(signal.ticker)
                    price = (
                        float(quote.get('ask') or quote.get('bid') or 0)
                        if quote else 0
                    )
                except Exception:
                    pass
            if price <= 0:
                log.warning("BUY SKIP %s: no valid price", signal.ticker)
                continue

            # Clamp to affordable (notebook: min(tmp_cash // price, abs(action)))
            tmp_cash = max(cash, 0.0)
            buy_num = min(int(tmp_cash // price), abs(signal.action))
            if buy_num <= 0:
                continue

            t0 = time.time()
            order_id = self.broker.place_order(
                symbol=signal.ticker, qty=buy_num, side=OrderSide.BUY,
            )
            latency_ms = (time.time() - t0) * 1000
            if order_id:
                self.health_monitor.record_api_call(True, latency_ms)
                log.info("BUY %d %s (order %s)", buy_num, signal.ticker, order_id)
                placed += 1
                cash -= price * buy_num
            else:
                self.health_monitor.record_api_call(False, latency_ms)
                log.warning("BUY FAILED %d %s", buy_num, signal.ticker)
                failed += 1

        return placed, failed

    # ── Market scheduling ───────────────────────────────────────────────

    def _await_market_open(self):
        """Block until market is open, polling every 60 seconds."""
        clock = self.broker.trading_client.get_clock()
        if clock.is_open:
            log.info("Market is open")
            return

        next_open = clock.next_open.replace(tzinfo=datetime.timezone.utc)
        now = clock.timestamp.replace(tzinfo=datetime.timezone.utc)
        wait_min = int((next_open - now).total_seconds() / 60)
        log.info("Market closed. ~%d minutes until open.", wait_min)

        while not self.broker.trading_client.get_clock().is_open:
            if not self._running and not self.cancel_open_on_start:
                # If stop() was called while waiting, exit
                break
            time.sleep(60)

        log.info("Market opened")

    def _time_to_close(self) -> float:
        """Seconds until market close."""
        clock = self.broker.trading_client.get_clock()
        close_ts = clock.next_close.replace(tzinfo=datetime.timezone.utc).timestamp()
        now_ts = clock.timestamp.replace(tzinfo=datetime.timezone.utc).timestamp()
        return close_ts - now_ts

    def _cancel_open_orders(self):
        """Cancel all open orders."""
        orders = self.broker.get_orders(status="open")
        if not orders:
            log.info("No open orders to cancel")
            return
        for order in orders:
            cancelled = self.broker.cancel_order(order.order_id)
            if cancelled:
                log.info("Cancelled stale order %s (%s %s %s)",
                         order.order_id, order.side.value, order.qty, order.symbol)
            else:
                log.warning("Failed to cancel order %s", order.order_id)
