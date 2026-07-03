"""
Quantile gradient boosting.

Takes the panel data_pipeline.py builds (de_lu_features.parquet) and walk forward
backtests it.
Prediction of how far off the price is from a recent reference level (price_roll168_mean)

Run:
    python forecast.py                              # residual mode, default
    python forecast.py --target-mode level          # old level mode, for comparison
    python forecast.py --train-window-years 2       # rolling window instead of expanding
    python forecast.py --step-days 90               # coarser/faster backtest
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance

TZ = "Europe/Brussels"
TARGET = "price"
REFERENCE = "price_roll168_mean"          # lag-safe level anchor for residual mode
DEFAULT_QUANTILES = [0.05, 0.25, 0.50, 0.75, 0.95]


# --------------------------------------------------------------------------- #
# Data                                                                        #
# --------------------------------------------------------------------------- #
def load_data(path: str, target_mode: str):
    """Load the panel and split it into X / y.

    Rows with no target just get dropped. In residual mode I also drop the
    warm-up rows where the reference isn't populated yet (first week of data,
    basically). Whatever NaNs are left inside X I leave alone on purpose —
    HistGradientBoosting handles missing values natively, so there's no reason
    to impute anything and pretend I know a value I don't.
    """
    df = pd.read_parquet(path)
    if TARGET not in df.columns:
        raise SystemExit(f"'{TARGET}' column missing — is this the pipeline output?")
    df = df.sort_index()
    df = df[df[TARGET].notna()]

    ref = None
    if target_mode == "residual":
        if REFERENCE not in df.columns:
            raise SystemExit(f"residual mode needs '{REFERENCE}' in the panel.")
        df = df[df[REFERENCE].notna()]
        ref = df[REFERENCE]

    feature_cols = [c for c in df.columns if c != TARGET]
    return df[feature_cols], df[TARGET], ref, feature_cols


# --------------------------------------------------------------------------- #
# Model                                                                       #
# --------------------------------------------------------------------------- #
def _make_model(q: float) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="quantile",
        quantile=q,
        learning_rate=0.05,
        max_iter=300,
        max_leaf_nodes=63,
        min_samples_leaf=50,
        l2_regularization=1.0,
        early_stopping=False,        # keep retrains deterministic between blocks
        random_state=0,
    )


def fit_predict_block(X_tr, y_tr_model, X_te, quantiles, ref_te=None,
                      conformal=False, calib_frac=0.2):
    """Fit one model per quantile and predict on the held-out test block.

    y_tr_model is whatever the models are actually being asked to learn —
    either raw price, or price-minus-reference if we're in residual mode. If
    ref_te is passed in, I add it back at the end to lift the residual
    predictions up to an actual price level. Predictions get sorted row-wise
    at the end so quantiles never cross each other.

    conformal=True turns on the per-quantile shift correction. Here's the
    story on that: I originally had symmetric CQR in here, which was fine for
    interval coverage but it was applying one shared correction across
    quantiles that were actually miscalibrated in different directions and by
    different amounts. So instead: hold out the most recent calib_frac slice
    of the training window, fit on the rest, and for each quantile separately
    work out the shift s that would make calib_frac's empirical coverage match
    the nominal q, then apply that same shift to the test predictions. Each
    quantile line gets corrected on its own terms rather than one blanket
    adjustment for all of them — this is what actually fixed the q50 (and a
    few interior-quantile) miscalibration I was seeing.

    One caveat worth remembering for the writeup: this doesn't inherit the
    same finite-sample coverage guarantee that paired-interval CQR has (that
    proof depends specifically on taking the max of two-sided errors). What I
    get instead is a direct fix for systematic per-quantile bias, which is the
    actual failure mode I was seeing here — all quantiles pulled the same
    direction, not just "interval too narrow." Worth double-checking against
    the paired version if the formal guarantee ends up mattering.
    """
    cols = [f"q{int(q * 100):02d}" for q in quantiles]

    if not conformal:
        te = {q: _make_model(q).fit(X_tr, y_tr_model).predict(X_te) for q in quantiles}
    else:
        n = len(X_tr)
        k = min(max(int(n * calib_frac), 500), n - 500)  # size of the calibration slice
        X_pt, y_pt = X_tr.iloc[:-k], y_tr_model.iloc[:-k]   # the "proper" training data (older)
        X_ca, y_ca = X_tr.iloc[-k:], y_tr_model.iloc[-k:]   # calibration slice (most recent)
        yca = y_ca.values

        models = {q: _make_model(q).fit(X_pt, y_pt) for q in quantiles}
        cal = {q: models[q].predict(X_ca) for q in quantiles}
        te = {q: models[q].predict(X_te) for q in quantiles}

        # For each quantile, find the shift s such that q-fraction of the
        # calibration residuals fall at or below it, then nudge the test
        # prediction by that amount. For q=0.5 this collapses to just the
        # median residual — same fix I validated on the median originally,
        # just generalized so every quantile gets its own correction.
        for q in quantiles:
            shift = np.quantile(yca - cal[q], q)
            te[q] = te[q] + shift

    preds = np.column_stack([te[q] for q in quantiles])
    if ref_te is not None:
        preds = preds + ref_te.values[:, None]     # back to price level from residual
    preds = np.sort(preds, axis=1)                 # enforce monotone quantiles
    return pd.DataFrame(preds, index=X_te.index, columns=cols)
# --------------------------------------------------------------------------- #
# Walk-forward backtest                                                        #
# --------------------------------------------------------------------------- #
def walk_forward(X, y, ref, quantiles, init_train_years=2, step_days=30,
                 train_window_years=0, conformal=False):
    """The actual walk-forward loop: retrain every step_days, predict the next block.

    train_window_years=0 means expanding window — use everything seen so far.
    Set it >0 for a rolling window instead, which keeps training data closer in
    time to whatever you're predicting — a cheap partial defense against the
    non-stationarity problem described up top.
    """
    test_start = X.index.min() + pd.DateOffset(years=init_train_years)
    anchors = pd.date_range(test_start, X.index.max(), freq=f"{step_days}D", tz=TZ)

    out = []
    for i, anchor in enumerate(anchors):
        block_end = anchor + pd.Timedelta(days=step_days)
        train_mask = X.index < anchor
        if train_window_years > 0:
            train_mask &= X.index >= (anchor - pd.DateOffset(years=train_window_years))
        test_mask = (X.index >= anchor) & (X.index < block_end)
        if test_mask.sum() < 24 or train_mask.sum() == 0:
            continue  # trailing stub block, less than a day of data — not worth it

        y_tr = y[train_mask]
        ref_te = None
        if ref is not None:
            y_tr = y_tr - ref[train_mask]          # this is the deviation the model learns
            ref_te = ref[test_mask]
        block = fit_predict_block(X[train_mask], y_tr, X[test_mask],
                                  quantiles, ref_te, conformal=conformal)
        block["actual"] = y[test_mask]
        out.append(block)
        print(f"  [{i+1:>3}/{len(anchors)}] {anchor.date()} "
              f"train={train_mask.sum():>6}  test={test_mask.sum():>4}")

    if not out:
        raise SystemExit("No test blocks produced — check init_train_years vs data span.")
    return pd.concat(out).sort_index()


# --------------------------------------------------------------------------- #
# Feature importance (diagnostic)                                              #
# --------------------------------------------------------------------------- #
def feature_importance(X, y, ref, feature_cols, eval_frac=0.2, n_repeats=5, seed=0):
    """Permutation importance for the median (q50) model, checked on the most recent slice.

    Trains on the earlier chunk, evaluates on the last eval_frac — deliberately
    the current regime, since that's the part I actually care about — then
    shuffles each feature one at a time and sees how much worse median MAE
    gets. The units are EUR/MWh of MAE damage: a value of 12 means "shuffling
    this feature costs about 12 EUR/MWh of accuracy," so bigger = the model is
    leaning on it more.

    If ref is set, this runs in residual space, same as training — the ranking
    ends up telling you what actually drives the deviation from the level
    anchor, not the raw price.
    """
    n = len(X)
    cut = int(n * (1 - eval_frac))
    X_tr, X_te = X.iloc[:cut], X.iloc[cut:]
    y_tr, y_te = y.iloc[:cut].copy(), y.iloc[cut:].copy()
    if ref is not None:
        y_tr = y_tr - ref.iloc[:cut]
        y_te = y_te - ref.iloc[cut:]

    model = _make_model(0.5).fit(X_tr, y_tr)
    r = permutation_importance(model, X_te, y_te, n_repeats=n_repeats,
                               random_state=seed,
                               scoring="neg_mean_absolute_error")
    imp = pd.DataFrame({"importance": r.importances_mean,
                        "std": r.importances_std}, index=feature_cols)
    return imp.sort_values("importance", ascending=False)


# --------------------------------------------------------------------------- #
# Metrics                                                                      #
# --------------------------------------------------------------------------- #
def pinball_loss(y, q_pred, q):
    d = y - q_pred
    return float(np.mean(np.maximum(q * d, (q - 1) * d)))


def evaluate(results: pd.DataFrame, quantiles, spike_quantile=0.95):
    """Score everything overall, then split calm-vs-spike hours since that split
    is the whole point of this project — nobody cares if you nail the calm
    hours, the desk cares about the spikes.
    """
    y = results["actual"].values
    qcols = [f"q{int(q * 100):02d}" for q in quantiles]

    thr = float(np.quantile(y, spike_quantile))   # threshold only used for reporting the split
    spike = y >= thr

    def block(mask, label):
        yy = y[mask]
        rep = {"label": label, "n_hours": int(mask.sum())}
        if "q50" in qcols:
            med = results.loc[mask, "q50"].values
            rep["median_MAE"] = float(np.mean(np.abs(yy - med)))
            rep["median_RMSE"] = float(np.sqrt(np.mean((yy - med) ** 2)))
        pin = {c: pinball_loss(yy, results.loc[mask, c].values, q)
               for q, c in zip(quantiles, qcols)}
        rep["pinball_per_q"] = pin
        rep["pinball_avg"] = float(np.mean(list(pin.values())))
        return rep

    all_mask = np.ones(len(y), dtype=bool)
    report = {
        "spike_threshold_eur": thr,
        "overall": block(all_mask, "overall"),
        "calm": block(~spike, f"calm (< {thr:.0f})"),
        "spike": block(spike, f"spike (>= {thr:.0f})"),
    }

    cov = {c: {"nominal": q, "empirical": float(np.mean(y <= results[c].values))}
           for q, c in zip(quantiles, qcols)}
    report["calibration"] = cov
    lo, hi = qcols[0], qcols[-1]
    report["interval"] = {
        "band": f"{lo}-{hi}",
        "nominal": quantiles[-1] - quantiles[0],
        "empirical": float(np.mean((y >= results[lo].values) & (y <= results[hi].values))),
    }
    return report


def print_report(rep):
    print("\n" + "=" * 60)
    print("BACKTEST SUMMARY")
    print("=" * 60)
    for key in ("overall", "calm", "spike"):
        b = rep[key]
        line = f"\n[{b['label']}]  n={b['n_hours']:,}"
        if "median_MAE" in b:
            line += f"   median MAE={b['median_MAE']:.1f}  RMSE={b['median_RMSE']:.1f}"
        line += f"   pinball_avg={b['pinball_avg']:.2f}"
        print(line)

    print("\nCalibration (empirical vs nominal — closer is better):")
    for c, d in rep["calibration"].items():
        flag = "  <-- off" if abs(d["empirical"] - d["nominal"]) > 0.03 else ""
        print(f"   {c}:  nominal {d['nominal']:.2f}   empirical {d['empirical']:.3f}{flag}")
    it = rep["interval"]
    print(f"\nInterval {it['band']}: nominal {it['nominal']:.2f}  empirical {it['empirical']:.3f}")
    print("=" * 60)


# --------------------------------------------------------------------------- #
# Entry point                                                                  #
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="DE_LU quantile price backtest")
    ap.add_argument("--parquet", default="de_lu_features.parquet")
    ap.add_argument("--target-mode", choices=["residual", "level"], default="residual",
                    help="residual = predict price minus rolling reference (default)")
    ap.add_argument("--init-train-years", type=int, default=2)
    ap.add_argument("--step-days", type=int, default=30)
    ap.add_argument("--train-window-years", type=int, default=0,
                    help="0 = expanding window; >0 = rolling window of N years")
    ap.add_argument("--conformal", action="store_true",
                    help="apply per-quantile conformal shift so coverage matches nominal by construction")
    ap.add_argument("--quantiles", type=float, nargs="+", default=DEFAULT_QUANTILES)
    ap.add_argument("--importance", action=argparse.BooleanOptionalAction, default=True,
                    help="print permutation feature importance (use --no-importance to skip)")
    ap.add_argument("--out-dir", default=".")
    args = ap.parse_args()

    q = sorted(args.quantiles)
    X, y, ref, feats = load_data(args.parquet, args.target_mode)
    print(f"panel: {len(X):,} rows, {len(feats)} features  |  target-mode={args.target_mode}")
    print(f"quantiles: {q}")
    win = "expanding" if args.train_window_years == 0 else f"{args.train_window_years}y rolling"
    cflag = "  +conformal(CQR)" if args.conformal else ""
    print(f"walk-forward (init {args.init_train_years}y, step {args.step_days}d, {win}{cflag}):")

    results = walk_forward(X, y, ref, q, args.init_train_years,
                           args.step_days, args.train_window_years,
                           conformal=args.conformal)
    report = evaluate(results, q)
    print_report(report)

    out = Path(args.out_dir)
    results.to_parquet(out / "backtest_predictions.parquet")
    with open(out / "backtest_metrics.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nsaved -> {out / 'backtest_predictions.parquet'}  ({results.shape})")
    print(f"saved -> {out / 'backtest_metrics.json'}")

    if args.importance:
        print("\n" + "=" * 60)
        print("FEATURE IMPORTANCE  (permutation, median model, recent 20%)")
        print("Δ MAE in EUR/MWh when the feature is shuffled — higher = more used")
        print("=" * 60)
        imp = feature_importance(X, y, ref, feats)
        for name, row in imp.iterrows():
            print(f"   {name:20s} {row['importance']:8.2f}  (± {row['std']:.2f})")
        imp.to_csv(out / "feature_importance.csv")
        print(f"\nsaved -> {out / 'feature_importance.csv'}")


if __name__ == "__main__":
    main()