"""
Phase 4 — Integrated forecast/risk dashboard.

One predictive distribution, two consequences: the same quantile forecast
drives both the price fan chart and the derived VaR/margin figures below it.
No explicit "trader" vs "clearing" split — the interactivity (position size,
date range) lets whoever's looking pull out what they need.

Lookback window for margin is 250 trading days (~1 calendar year), matching
standard industry practice for commodity margin calibration.

DATA SOURCE (current): reads existing backtest/panel outputs directly.
TODO(data layer): swap load_* functions for the shared data/API layer once
it exists — isolated here so nothing else needs to change.

Run: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import risk

MARGIN_WINDOW_DAYS = 250  # ~1 calendar year, standard commodity margin lookback

st.set_page_config(page_title="DE_LU Power Price Risk", layout="wide")


@st.cache_data
def load_backtest():
    return pd.read_parquet("backtest_predictions.parquet")


@st.cache_data
def load_panel():
    return pd.read_parquet("de_lu_features.parquet")


def render_intro():
    st.title("German Power Price — Forecast & Risk")
    st.markdown(
        "This tool predicts the hourly price of electricity in Germany, one day "
        "ahead — and shows what that prediction means in financial terms if you "
        "hold a position in the market. **The two are the same underlying model, "
        "just read two different ways:** as a price forecast, and as the amount "
        "of money you'd need to set aside to cover the risk of holding that price "
        "exposure."
    )
    with st.expander("New here? A 60-second guide to the numbers below"):
        st.markdown(
            """
**Why isn't there just one number?** Power prices can move a lot, and
sometimes a lot more than expected — think of a cold, windless week when
renewable output drops and demand rises. So instead of one guess, the model
gives a *range* of plausible prices for each hour, along with how confident
it is.

**What the fan chart shows:** the shaded area is that range — how wide it
is tells you how uncertain the model currently is. The dark center line is
its single best guess (the "median"). The black line is what the price
*actually* turned out to be, so you can see how the forecast performed with
hindsight.

**What "VaR" and "initial margin" mean:** if you held a position of a given
size (adjustable in the sidebar), how much money could you plausibly lose in
a single day? VaR answers that at a 99% confidence level — a loss beyond
that number should only happen about 1 day in 100. Initial margin takes
that number and stretches it slightly, to cover the time it would
realistically take to close out a position if something went wrong.

**What the margin path chart shows:** the same VaR figure, recalculated
every day using the past year of price history. Watch what happens to it
around a real price spike — it jumps up and *stays* elevated for months
afterward, because the shock stays inside the year-long lookback until it
eventually ages out. This is a well-known effect called procyclicality:
the same volatility that makes a position risky also makes it more
expensive to hold, right when it hurts most.
            """
        )
    st.divider()


def render_header(window: pd.DataFrame, position_mwh: float):
    latest = window.iloc[-1]
    band_width = latest["q95"] - latest["q05"]

    pnl = position_mwh * window["actual"].diff(24).dropna()
    var99 = risk.historical_var(pnl, 0.99) if len(pnl) > 10 else float("nan")
    im = risk.initial_margin(pnl, alpha=0.99, mpor_days=2) if len(pnl) > 10 else float("nan")

    exceptions = ((window["actual"] < window["q05"]) | (window["actual"] > window["q95"])).mean()

    st.subheader("Current snapshot")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Price forecast", f"{latest['q50']:.0f} EUR/MWh",
             help="The model's single best-guess price for the most recent hour "
                  "in your selected window.")
    c2.metric("Forecast uncertainty", f"±{band_width/2:.0f} EUR/MWh",
             help="Half the width of the model's 90% confidence range. Bigger "
                  "number = the model is less sure right now (typically during "
                  "volatile periods).")
    c3.metric(f"Potential 1-day loss ({position_mwh:.0f} MWh)", f"{var99:.0f} EUR",
             help="Value at Risk (99%): the loss level that should only be "
                  "exceeded about 1 day in 100, for a position of this size.")
    c4.metric(f"Collateral to post ({position_mwh:.0f} MWh)", f"{im:.0f} EUR",
             help="Initial margin: the potential loss figure above, stretched "
                  "to cover a realistic close-out period — a simplified version "
                  "of what a clearing house would require as collateral.")
    c5.metric("Forecast reliability", f"{exceptions:.1%} breach rate",
             help="How often the actual price fell outside the model's 90% "
                  "predicted range in this window. Ideally close to 10% — much "
                  "higher means the model is overconfident; much lower means "
                  "it's overly cautious.")


def render_fan_chart(window: pd.DataFrame):
    st.subheader("Price forecast over time")
    st.caption(
        "Shaded bands = the range of prices the model considered plausible, at "
        "the time. Black line = what actually happened. Red × marks = hours "
        "where the actual price broke outside the model's 90% range."
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=window.index, y=window["q95"], line=dict(width=0),
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=window.index, y=window["q05"], line=dict(width=0),
                             fill="tonexty", fillcolor="rgba(31,119,180,0.15)",
                             name="90% likely range"))
    fig.add_trace(go.Scatter(x=window.index, y=window["q75"], line=dict(width=0),
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=window.index, y=window["q25"], line=dict(width=0),
                             fill="tonexty", fillcolor="rgba(31,119,180,0.35)",
                             name="50% likely range"))
    fig.add_trace(go.Scatter(x=window.index, y=window["q50"], line=dict(color="#1f77b4", width=2),
                             name="model's best guess"))
    fig.add_trace(go.Scatter(x=window.index, y=window["actual"], line=dict(color="#222", width=1.3),
                             name="what actually happened"))

    breaches = window[(window["actual"] < window["q05"]) | (window["actual"] > window["q95"])]
    if len(breaches):
        fig.add_trace(go.Scatter(x=breaches.index, y=breaches["actual"], mode="markers",
                                 marker=dict(color="#c1121f", size=6, symbol="x"),
                                 name="surprise (broke the range)"))

    fig.update_layout(height=420, yaxis_title="EUR per MWh", hovermode="x unified",
                      legend=dict(orientation="h", y=1.15),
                      margin=dict(t=30, b=20))
    st.plotly_chart(fig, use_container_width=True)


def render_margin_path(window: pd.DataFrame, position_mwh: float):
    st.subheader("Collateral requirement over time")

    window_hours = MARGIN_WINDOW_DAYS * 24
    pnl = position_mwh * window["actual"].diff(24).dropna()
    if len(pnl) < window_hours:
        days_have = len(pnl) // 24
        st.info(
            f"This chart needs about {MARGIN_WINDOW_DAYS} days of price history to "
            f"compute a rolling collateral figure (you currently have ~{days_have} "
            "days selected). Widen the date range in the sidebar to see it."
        )
        return

    im_path = risk.rolling_initial_margin(
        pd.Series(pnl.values, index=pnl.index), window=window_hours, alpha=0.99, mpor_days=2
    )

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=window.index, y=window["actual"], line=dict(color="#999", width=0.6),
                             name="actual price", opacity=0.6), secondary_y=False)
    fig.add_trace(go.Scatter(x=im_path.index, y=im_path.values, line=dict(color="#c1121f", width=2),
                             name=f"collateral needed ({position_mwh:.0f} MWh)"), secondary_y=True)
    fig.update_yaxes(title_text="Price (EUR/MWh)", secondary_y=False)
    fig.update_yaxes(title_text="Collateral (EUR)", secondary_y=True)
    fig.update_layout(height=350, hovermode="x unified", margin=dict(t=20, b=20),
                      legend=dict(orientation="h", y=1.15))
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "The red line recalculates every day using the trailing year of price "
        "history. A price shock pushes it up quickly — and it stays elevated "
        "for months afterward, since the shock remains inside that trailing "
        "window until enough time passes for it to age out."
    )


def main():
    render_intro()

    preds = load_backtest()

    with st.sidebar:
        st.header("Adjust the view")
        position_mwh = st.slider(
            "Position size (MWh)", 1, 500, 50, step=1,
            help="How large a hypothetical electricity position to evaluate. "
                 "All euro figures below scale with this."
        )
        min_date, max_date = preds.index.min().date(), preds.index.max().date()
        default_start = max(min_date, max_date - pd.Timedelta(days=400))
        date_range = st.slider(
            "Date range", min_value=min_date, max_value=max_date,
            value=(default_start, max_date),
            help="Which period of history to look at. This is validated, historical "
                 "backtest data, not a live forecast."
        )
        st.caption(
            "This shows how the model would have performed historically. It is "
            "not a live, real-time forecast."
        )

    start, end = pd.Timestamp(date_range[0], tz=preds.index.tz), pd.Timestamp(date_range[1], tz=preds.index.tz)
    window = preds.loc[start:end]

    if window.empty:
        st.warning("No data in the selected range — try widening it.")
        return

    render_header(window, position_mwh)
    st.divider()
    render_fan_chart(window)
    st.divider()
    render_margin_path(window, position_mwh)


if __name__ == "__main__":
    main()