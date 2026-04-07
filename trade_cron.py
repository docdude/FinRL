"""CLI entry point for cron-based RL model trading.

Called by crontab (installed from Model Management UI):
    python -m strategies.rl.trade_cron --account elegantrl_split4 --once
    python -m strategies.rl.trade_cron --account finrl_dow29 --continuous
"""
from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger(__name__)

# Ticker lists (must match training data)
DOW_28 = [
    "AAPL", "AMGN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX",
    "DIS", "GS", "HD", "HON", "IBM", "INTC", "JNJ", "JPM",
    "KO", "MCD", "MMM", "MRK", "MSFT", "NKE", "PG", "TRV",
    "UNH", "V", "VZ", "WMT",
]
DOW_29 = [
    "AAPL", "AMGN", "AMZN", "AXP", "BA", "CAT", "CRM", "CSCO",
    "CVX", "DIS", "GS", "HD", "HON", "IBM", "INTC", "JNJ",
    "JPM", "KO", "MCD", "MMM", "MRK", "MSFT", "NKE", "PG",
    "TRV", "UNH", "V", "VZ", "WMT",
]
TICKER_MAP = {"DOW_28": DOW_28, "DOW_29": DOW_29}


def main():
    parser = argparse.ArgumentParser(
        description="RL model trading via cron")
    parser.add_argument(
        "--account", required=True,
        help="Account name from rl_accounts.yaml")
    parser.add_argument(
        "--once", action="store_true",
        help="Run a single trade cycle (daily models)")
    parser.add_argument(
        "--continuous", action="store_true",
        help="Run continuous intraday loop")
    parser.add_argument(
        "--auto-execute", action="store_true",
        help="Execute orders (otherwise log-only)")
    parser.add_argument(
        "--user-id", required=True,
        help="User ID for loading per-user credentials")
    args = parser.parse_args()

    if not args.once and not args.continuous:
        args.once = True  # default to single cycle

    # Load account config
    from alpaca_trading.account_manager import (
        AlpacaAccountManager,
    )

    config_path = Path("config/rl_accounts.yaml")
    mgr = AlpacaAccountManager(config_path)
    cfg = mgr.get_config(args.account)
    if cfg is None:
        log.error("Account %s not found in config", args.account)
        sys.exit(1)

    tickers = TICKER_MAP.get(cfg.tickers, DOW_28)
    suffix = cfg.env_suffix

    # Load user credentials from per-user store
    from rl_web_frontend.user_context import load_all_user_credentials
    user_creds = load_all_user_credentials(args.user_id)
    if not user_creds or suffix not in user_creds:
        log.error("No credentials for suffix %s in user %s. "
                  "Save them via Settings → Alpaca Credentials.",
                  suffix, args.user_id)
        sys.exit(1)
    api_key = user_creds[suffix]["api_key"]
    api_secret = user_creds[suffix]["api_secret"]

    # Load adapter
    if cfg.model_type == "elegantrl":
        from strategies.rl.elegantrl_adapter import (
            ElegantRLAdapter,
        )
        adapter = ElegantRLAdapter()
        adapter.load_model(
            checkpoint_path=cfg.checkpoint,
            vec_normalize_path=(
                cfg.vec_normalize if cfg.vec_normalize
                else None
            ),
        )
    elif cfg.model_type == "ensemble":
        from strategies.rl.ensemble_adapter import (
            EnsembleAdapter,
        )
        adapter = EnsembleAdapter(
            regime_weights=cfg.regime_weights or None,
            regime_mode=cfg.regime_mode,
        )
        adapter.load_models(
            checkpoint_paths=[m.checkpoint for m in cfg.models],
            vec_normalize_paths=[
                m.vec_normalize or None for m in cfg.models
            ],
            model_labels=[m.label for m in cfg.models],
        )
    elif cfg.model_type == "finrl":
        from strategies.rl.finrl_adapter import FinRLAdapter
        adapter = FinRLAdapter(
            state_dim=getattr(cfg, 'state_dim', 322),
            action_dim=getattr(cfg, 'action_dim', 29),
            net_dims=getattr(cfg, 'net_dims', None),
            initial_capital=getattr(cfg, 'initial_capital', 100_000),
            max_stock=getattr(cfg, 'max_stock', 100),
            turbulence_thresh=getattr(cfg, 'turbulence_thresh', 99.0),
        )
        adapter.load_model(checkpoint_path=cfg.checkpoint)
    else:
        log.error("Unknown model_type: %s", cfg.model_type)
        sys.exit(1)

    # Connect broker
    broker = mgr.get_broker(args.account, credentials=user_creds)
    log.info(
        "Loaded %s adapter for account %s (paper=%s)",
        cfg.model_type, args.account, cfg.paper,
    )

    # Create trade loop
    from strategies.rl.trade_loop import (
        RLTradeLoop, INDICATORS,
    )

    loop = RLTradeLoop(
        adapter=adapter,
        broker=broker,
        tickers=tickers,
        tech_indicators=INDICATORS,
        api_key=api_key,
        api_secret=api_secret,
        data_mode=cfg.data_mode,
        trade_interval_seconds=cfg.trade_interval,
    )

    if args.once:
        log.info("Running single trade cycle for %s",
                 args.account)

        if args.auto_execute:
            summary = loop.run_once()
            log.info(
                "Cycle result: placed=%d failed=%d "
                "total=$%.2f",
                summary.orders_placed,
                summary.orders_failed,
                summary.total_asset,
            )
        else:
            # Inference only — log signals but don't trade
            prices, tech, turb = loop._fetch_data()
            cash, shares = loop._sync_portfolio()
            result = adapter.predict(
                cash=cash, shares=shares,
                close_prices=prices,
                tech_features=tech,
                tickers=tickers, turbulence=turb,
            )
            n_buys = sum(
                1 for s in result.signals
                if s.action > 0)
            n_sells = sum(
                1 for s in result.signals
                if s.action < 0)
            log.info(
                "Inference-only: buys=%d, sells=%d, "
                "turb=%.2f, sell_all=%s",
                n_buys, n_sells, turb,
                result.turbulence_sell_all,
            )
            for s in result.signals:
                if s.action != 0:
                    log.info(
                        "  %s %d %s (raw=%.4f)",
                        "BUY" if s.action > 0 else "SELL",
                        abs(s.action), s.ticker,
                        s.raw_action,
                    )

    elif args.continuous:
        log.info(
            "Starting continuous trading loop for %s",
            args.account)
        if not args.auto_execute:
            log.warning(
                "Continuous mode without --auto-execute "
                "is not useful. Enabling auto-execute.")

        # Register signal handlers for graceful shutdown
        def _handle_signal(signum, frame):
            sig_name = signal.Signals(signum).name
            log.info("Received %s, requesting graceful shutdown...",
                     sig_name)
            loop.stop()

        signal.signal(signal.SIGTERM, _handle_signal)
        signal.signal(signal.SIGINT, _handle_signal)

        loop.run()

    log.info("Done.")


if __name__ == "__main__":
    main()
