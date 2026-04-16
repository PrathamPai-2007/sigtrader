

def _compute_atr(candles: list[Candle], period: int = 14) -> float:
    """Compute ATR as average of last `period` true ranges. Returns 0.0 when insufficient data."""
    if len(candles) < 2:
        return 0.0
    trs: list[float] = []
    prev_close = candles[0].close
    for c in candles[1:]:
        tr = max(c.high - c.low, abs(c.high - prev_close), abs(c.low - prev_close))
        trs.append(tr)
        prev_close = c.close
    if not trs:
        return 0.0
    tail = trs[-period:] if len(trs) >= period else trs
    return sum(tail) / len(tail)


def _ema_value(closes: list[float], period: int) -> float:
    """Compute EMA of a close series. Returns last close when insufficient data."""
    if not closes:
        return 0.0
    if len(closes) < period:
        return closes[-1]
    k = 2.0 / (period + 1)
    ema = sum(closes[:period]) / period
    for c in closes[period:]:
        ema = c * k + ema * (1 - k)
    return ema


def _funding_momentum_local(history: list[float], window: int = 4) -> float:
    """Compute slope of last `window` funding rate values, normalised to [-1, 1]."""
    from futures_analyzer.analysis.scoring.utils import _clamp as _cl
    slope_scale = load_app_config().strategy.funding_momentum_slope_scale
    tail = history[-window:] if len(history) >= window else history
    n = len(tail)
    if n < 2:
        return 0.0
    xs = list(range(n))
    x_mean = (n - 1) / 2.0
    y_mean = sum(tail) / n
    num = sum((xs[i] - x_mean) * (tail[i] - y_mean) for i in range(n))
    den = sum((xs[i] - x_mean) ** 2 for i in range(n))
    if den == 0:
        return 0.0
    slope = num / den
    return _cl(slope * slope_scale, -1.0, 1.0)


def compute_all_indicators(
    entry: list[Candle],
    trigger: list[Candle],
    context: list[Candle],
    higher: list[Candle],
    market_meta: MarketMeta,
    indicator_params=None,
    signal_transforms=None,
    config: AppConfig | None = None,
) -> IndicatorBundle:
    """Compute all indicators in a single coordinated pass over the four timeframe candle lists.

    Returns a fully-populated IndicatorBundle. Safe neutral defaults are used for any
    indicator when candle data is insufficient; a warning string is appended to the
    returned bundle's `warnings` list.
    """
    from futures_analyzer.analysis.scoring.utils import _clamp as _cl
    warnings_list: list[str] = []

    # --- Per-timeframe ATR ---
    entry_atr = _compute_atr(entry)
    trigger_atr = _compute_atr(trigger)
    context_atr = _compute_atr(context)
    higher_atr = _compute_atr(higher)

    # --- Config resolution ---
    if indicator_params is None or signal_transforms is None:
        cfg = config or load_app_config()
        _strategy = cfg.strategy
        _ip = indicator_params if indicator_params is not None else _strategy.indicator_params
        _st = signal_transforms if signal_transforms is not None else _strategy.signal_transforms
    else:
        _ip = indicator_params
        _st = signal_transforms
    _ema_fast = _ip.ema_fast_period
    _ema_slow = _ip.ema_slow_period
    _trend_tanh_scale = _st.trend_tanh_scale
    _momentum_tanh_scale = _st.momentum_tanh_scale
    _vol_window = _ip.volume_window
    _roc_period = _ip.roc_period

    # --- Trend: EMA(fast) vs EMA(slow) slope, normalised via tanh ---
    def _trend_ema(candles: list[Candle], label: str) -> float:
        if len(candles) < _ema_slow:
            warnings_list.append(
                f"Insufficient candles for {label} trend (need {_ema_slow}, got {len(candles)}); defaulting to 0.0"
            )
            return 0.0
        closes = [c.close for c in candles]
        ema_f = _ema_value(closes, _ema_fast)
        ema_s = _ema_value(closes, _ema_slow)
        if ema_s == 0.0:
            return 0.0
        raw = (ema_f - ema_s) / ema_s
        return math.tanh(raw * _trend_tanh_scale)

    higher_trend = _trend_ema(higher, "higher")
    context_trend = _trend_ema(context, "context")

    # --- Momentum: rate of change, normalised via tanh ---
    def _roc_momentum(candles: list[Candle], label: str) -> float:
        need = _roc_period + 1
        if len(candles) < need:
            warnings_list.append(
                f"Insufficient candles for {label} momentum (need {need}, got {len(candles)}); defaulting to 0.0"
            )
            return 0.0
        first = candles[-need].close
        last = candles[-1].close
        if first <= 0:
            return 0.0
        raw = last / first - 1.0
        return _cl(math.tanh(raw * _momentum_tanh_scale), -1.0, 1.0)

    trigger_momentum = _roc_momentum(trigger, "trigger")
    entry_momentum = _roc_momentum(entry, "entry")

    # --- Volume surge: last volume / mean(last N volumes) ---
    def _vol_surge(candles: list[Candle]) -> float:
        if len(candles) < 2:
            return 1.0
        window = candles[-_vol_window:] if len(candles) >= _vol_window else candles
        base = sum(c.volume for c in window) / len(window)
        if base <= 0:
            return 1.0
        return candles[-1].volume / base

    trigger_volume_surge = _vol_surge(trigger)
    entry_volume_surge = _vol_surge(entry)

    # --- Cumulative delta ---
    if trigger:
        cumulative_delta = compute_cumulative_delta(trigger)
    else:
        cumulative_delta = 0.0
        warnings_list.append("No trigger candles for cumulative_delta; defaulting to 0.0")

    # --- Oscillators on entry TF ---
    _atr_period = _ip.atr_period
    if len(entry) >= _atr_period + 1:
        rsi_14 = rsi(entry, period=_atr_period)
    else:
        rsi_14 = 50.0
        warnings_list.append(
            f"Insufficient entry candles for RSI (need {_atr_period + 1}, got {len(entry)}); defaulting to 50.0"
        )

    macd_result = macd(entry)
    macd_histogram = macd_result.histogram if macd_result.histogram is not None else 0.0

    stoch_result = stochastic(entry)
    stoch_k = stoch_result.value

    bb_result = bollinger_bands(entry)
    bb_position = bb_result.get("position", 0.5)
    bb_bandwidth_pct = bb_result.get("bandwidth", 0.0)

    # --- Swing highs/lows from trigger candles ---
    pivot_n = _ip.pivot_n

    if len(trigger) >= (2 * pivot_n + 1):
        highs = [c.high for c in trigger]
        lows = [c.low for c in trigger]
        raw_pivots = _swing_pivots(highs, n=pivot_n)
        swing_highs_raw = [
            val for idx, val in raw_pivots
            if val == max(highs[max(0, idx - pivot_n): idx + pivot_n + 1])
        ]
        raw_pivots_low = _swing_pivots(lows, n=pivot_n)
        swing_lows_raw = [
            val for idx, val in raw_pivots_low
            if val == min(lows[max(0, idx - pivot_n): idx + pivot_n + 1])
        ]
        swing_highs = sorted(swing_highs_raw)
        swing_lows = sorted(swing_lows_raw)
    else:
        swing_highs = []
        swing_lows = []
        warnings_list.append(
            f"Insufficient trigger candles for swing pivots (need {2 * pivot_n + 1}, got {len(trigger)}); defaulting to empty lists"
        )

    # --- Market structure ---
    if len(trigger) >= 7:
        market_structure = compute_market_structure(trigger)
    else:
        market_structure = "mixed"
        warnings_list.append(
            f"Insufficient trigger candles for market_structure (need 7, got {len(trigger)}); defaulting to 'mixed'"
        )

    # --- Liquidity sweeps ---
    if len(trigger) >= 7:
        liquidity_sweeps = detect_liquidity_sweeps(trigger)
    else:
        liquidity_sweeps = []
        warnings_list.append(
            f"Insufficient trigger candles for liquidity_sweeps (need 7, got {len(trigger)}); defaulting to []"
        )

    # --- VWAP bands from trigger candles ---
    if trigger:
        vwap_bands = compute_vwap_bands(trigger)
        vwap_val = vwap_bands["vwap"]
        vwap_upper_1sd = vwap_bands["upper_1sd"]
        vwap_lower_1sd = vwap_bands["lower_1sd"]
        vwap_upper_2sd = vwap_bands["upper_2sd"]
        vwap_lower_2sd = vwap_bands["lower_2sd"]
    else:
        last_close = 0.0
        vwap_val = last_close
        vwap_upper_1sd = last_close
        vwap_lower_1sd = last_close
        vwap_upper_2sd = last_close
        vwap_lower_2sd = last_close
        warnings_list.append("No trigger candles for VWAP bands; defaulting to 0.0")

    # --- Volume profile from trigger candles ---
    if trigger:
        vp = volume_profile(trigger)
        poc = vp["poc"]
        vah = vp["vah"]
        val = vp["val"]
    else:
        last_close = trigger[-1].close if trigger else 0.0
        poc = last_close
        vah = last_close
        val = last_close
        warnings_list.append("No trigger candles for volume_profile; defaulting to last close")

    # --- RSI divergence on entry TF ---
    min_div_bars = 14 + 30 + 3  # period + lookback + pivot_n
    if len(entry) >= min_div_bars:
        div_detected, div_type, div_strength = rsi_divergence(entry)
    else:
        div_detected, div_type, div_strength = False, "none", 0.0
        warnings_list.append(
            f"Insufficient entry candles for RSI divergence (need {min_div_bars}, got {len(entry)}); defaulting to none"
        )

    rsi_divergence_type = div_type
    rsi_divergence_strength = div_strength

    # --- OI / funding from market_meta ---
    funding_rate = getattr(market_meta, "funding_rate", None)
    oi_change_pct = getattr(market_meta, "open_interest_change_pct", None)
    funding_history = getattr(market_meta, "funding_rate_history", None) or []
    funding_momentum_val = _funding_momentum_local(funding_history)

    # --- Order book ---
    order_book_imbalance = getattr(market_meta, "order_book_imbalance", 0.0) or 0.0
    bid_ask_spread_pct = 0.0  # placeholder

    return IndicatorBundle(
        entry_atr=entry_atr,
        trigger_atr=trigger_atr,
        context_atr=context_atr,
        higher_atr=higher_atr,
        higher_trend=higher_trend,
        context_trend=context_trend,
        trigger_momentum=trigger_momentum,
        entry_momentum=entry_momentum,
        trigger_volume_surge=trigger_volume_surge,
        entry_volume_surge=entry_volume_surge,
        cumulative_delta=cumulative_delta,
        rsi_14=rsi_14,
        macd_histogram=macd_histogram,
        stoch_k=stoch_k,
        bb_position=bb_position,
        bb_bandwidth_pct=bb_bandwidth_pct,
        swing_highs=swing_highs,
        swing_lows=swing_lows,
        market_structure=market_structure,
        liquidity_sweeps=liquidity_sweeps,
        vwap=vwap_val,
        vwap_upper_1sd=vwap_upper_1sd,
        vwap_lower_1sd=vwap_lower_1sd,
        vwap_upper_2sd=vwap_upper_2sd,
        vwap_lower_2sd=vwap_lower_2sd,
        poc=poc,
        vah=vah,
        val=val,
        rsi_divergence_type=rsi_divergence_type,
        rsi_divergence_strength=rsi_divergence_strength,
        funding_rate=funding_rate,
        oi_change_pct=oi_change_pct,
        funding_momentum=funding_momentum_val,
        order_book_imbalance=order_book_imbalance,
        bid_ask_spread_pct=bid_ask_spread_pct,
        warnings=warnings_list,
    )
