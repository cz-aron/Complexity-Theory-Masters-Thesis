"""Functions for Calculating Complexity Theory Indicators based on van den End (2019)"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kurtosis, skew


## Autocorrelation
def ar1_phi(x: np.ndarray) -> float:
    """
    The function calculates: N_t = c + rho * N_{t-1} + epsilon_t
    It expects x to be a one-dimensional time series: a sequence of numeric observations ordered in time.
    And obtains the coefficient of the AR(1) model.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]  # drop non-finite values
    if len(x) < 3:
        return np.nan
    x0 = x[:-1]  # lagged series
    x1 = x[1:]  # current series
    if np.std(x0) == 0:
        return np.nan
    X = np.column_stack([np.ones(len(x0)), x0])
    beta, *_ = np.linalg.lstsq(X, x1, rcond=None)
    return float(beta[1])  # estimated AR(1) coefficient (phi/rho)


## Decay Rate - Calculated from phi -> practical modification since sovereign spreads can become NEGATIVE
def decay_rate_from_phi(phi: float) -> float:
    """
    tau is the AR(1)-based decay rate, defined as tau = -ln(phi), where phi is the AR(1) persistence coefficient (rho).
    It is only defined for 0 < phi < 1. tau is always positive: 
        values close to 0 mean very SLOW recovery (high persistence, stronger critical slowing down),
        while larger tau values mean FASTER recovery (lower persistence).
    In this framework, lower tau indicates greater fragility.

    Methodological rule used in this project
    ----------------------------------------
    1) Keep the rolling AR(1) estimate as fitted (no clipping to [0,1], no absolute-value transform).
    2) If rho is outside the admissible domain (rho <= 0 or rho >= 1), set tau to NaN.
    3) Downstream tail-signal logic uses pandas quantiles (which ignore NaN), and NaN comparisons
       evaluate to False, so out-of-domain tau windows do not generate slowdown signals.
    """
    if not np.isfinite(phi):
        return np.nan
    
    # tau = -ln(phi) is defined for 0 < phi < 1, therefore:
    if phi <= 0 or phi >= 1:
        return np.nan  
    
    return float(-np.log(phi))    


## Decary Rate - Calculated from N_t / N_0 ratio following Van den End (2019) paper -> for use cases where the state variable is non-negative at all times
def decay_rate_from_N_ratio(
    x,
    signed: bool = False,
    min_positive: float = 1e-12,
    nonpositive_policy: str = "abs",
) -> float:
    """
    Rolling-window decay rate based on the N_t / N_0 logic.

    Half-life:
        h = -t * ln(2) / ln(N_t / N_0)

    Decay rate:
        tau = ln(2) / h

    If signed=False, the function uses abs(h), following the idea of using
    the absolute half-life.

    Parameters
    ----------
    x : array-like
        Rolling window of the state variable, e.g. Spain-Germany spread.
    signed : bool
        If True, returns signed decay rate.
        If False, returns decay rate based on absolute half-life.
    min_positive : float
        Small positive floor to avoid log(0).
    nonpositive_policy : {"abs", "nan"}
        - "abs": use absolute endpoint values, abs(N0) and abs(Nt).
        - "nan": return NaN if N0 <= 0 or Nt <= 0.

    Returns
    -------
    float
        Decay rate for the rolling window.
    """

    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) < 2:
        return np.nan

    if nonpositive_policy not in {"abs", "nan"}:
        raise ValueError("nonpositive_policy must be either 'abs' or 'nan'.")

    N0, Nt = x[0], x[-1]
    t = len(x) - 1

    # ------------------------------------------------------------
    # Handle non-positive endpoints
    # ------------------------------------------------------------
    if nonpositive_policy == "abs":
        # Van den End defines the slowdown indicator through the half-life of the state variable and reports the empirical slowdown measure using the absolute value of the half-life. However, the paper does not provide a code-level treatment of cases where the logarithmic ratio is undefined. In the present application, the Spain--Germany spread can become zero or negative, so the raw ratio $N_t/N_0$ is not always valid. To keep the indicator computable, the endpoint values are transformed using absolute values before calculating the ratio. The resulting decay rate is then based on the absolute half-life. This should be interpreted as a practical adaptation of the original indicator to sovereign spreads that may cross zero.
        # Nt - N0 = -30-100 = -130
        N0 = max(abs(N0), min_positive)
        Nt = max(abs(Nt), min_positive)

    elif nonpositive_policy == "nan":
        if N0 <= 0 or Nt <= 0:
            return np.nan

    ratio = Nt / N0

    if ratio <= 0 or not np.isfinite(ratio):
        return np.nan

    log_ratio = np.log(ratio)

    # If Nt == N0, half-life is infinite and decay rate is zero
    if np.isclose(log_ratio, 0.0):
        return 0.0

    # ------------------------------------------------------------
    # Calculating half-life and decay rate
    # ------------------------------------------------------------
    half_life_signed = -t * np.log(2) / log_ratio

    if signed:
        tau = np.log(2) / half_life_signed
    else:
        tau = np.log(2) / abs(half_life_signed)

    return float(tau)


## Flickering
def sarle_bimodality(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 4:
        return np.nan
    g = skew(x, bias=False)
    k = kurtosis(x, fisher=False, bias=False)
    if not np.isfinite(k) or k == 0:
        return np.nan
    return float((g**2 + 1) / k)


## Function for calculating Rolling Complexity Indicators
def rolling_complexity_indicators(
    df: pd.DataFrame,
    value_col: str = "state_variable",
    date_col: str = "observation_date",
    window: int = 100,
    min_periods: int = 100,
    decay_method: str = "ar1_phi", # or "N_ratio"
    signed_decay: bool = False, # added parameter to control whether decay rate is signed or absolute
    tau_nonpositive_policy: str = "abs",
) -> pd.DataFrame:
    
    # Build series with timestamp index
    if date_col in df.columns:
        s = pd.Series(df[value_col].to_numpy(), index=pd.to_datetime(df[date_col]))
    else:
        s = df[value_col].copy()

    roll = s.rolling(window=window, min_periods=min_periods)

    ar1 = roll.apply(ar1_phi, raw=True).abs().rename("Autocorrelation (ρ)") # .abs() for van den end more accurate replication

    # Decay rate calculation for potentially NEGATIVE state variables
    if decay_method == "ar1_phi":
        # Calculating decay rate from AR(1) coefficient
        # Important: decay_rate_from_phi returns NaN when AR(1) rho is outside (0,1).
        decay = ar1.map(decay_rate_from_phi).rename("Slowdown Indicator (τ)")
    
    # Decay rate calculation for NON-NEGATIVE state variables
    elif decay_method == "N_ratio":
        decay = roll.apply(
            lambda x: decay_rate_from_N_ratio(
                x,
                signed=signed_decay,
                nonpositive_policy=tau_nonpositive_policy,
            ),
            raw=True,
        ).rename("Slowdown Indicator (τ)")

    # Flickering
    bimod = roll.apply(sarle_bimodality, raw=True).rename("Flickering (β)")

    # Variance
    variance = roll.var(ddof=1).rename("Variance (σ2)")

    # Skewness
    skewness = roll.skew().rename("Skewness (γ)")

    out = pd.concat([ar1, decay, bimod, variance, skewness], axis=1)
    out.index.name = "observation_date"
    return out


## Calculating tail signals for complexity indicators based on x% tail thresholds, with optional lookback window for validation against crisis dates
def tail_signals(
    df: pd.DataFrame,
    crisis_dates,
    tail: float = 0.1,
    lookback: int = 100,
    check_all_signals: bool = False,
    lower_tail_cols=("Slowdown Indicator (τ)",),
    upper_tail_cols=("Autocorrelation (ρ)", "Variance (σ2)", "Skewness (γ)", "Flickering (β)"),
):
    data = df.copy()
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)

    thresholds = {}
    for col in lower_tail_cols:
        thresholds[col] = data[col].quantile(tail)
    for col in upper_tail_cols:
        thresholds[col] = data[col].quantile(1 - tail)

    signals = pd.DataFrame(index=data.index)
    for col in lower_tail_cols:
        signals[col] = data[col] <= thresholds[col]
    for col in upper_tail_cols:
        signals[col] = data[col] >= thresholds[col]

    if check_all_signals:
        valid = signals.copy()
    else:
        crisis_dates = pd.to_datetime(pd.Index(crisis_dates))
        valid = pd.DataFrame(False, index=data.index, columns=signals.columns)

        for cd in crisis_dates:
            window_idx = data.index[data.index <= cd][-lookback:]
            valid.loc[window_idx, :] |= signals.loc[window_idx, :]

    summary = pd.DataFrame(
        {
            "total_signals": signals.sum(),
            "valid_signals": valid.sum(),
        }
    )
    summary["valid_share"] = summary["valid_signals"] / summary["total_signals"]

    return thresholds, signals, valid, summary


## Aggregating valid  signals into a simple Early Warning Score (EWS)
def ews_score_from_valid(
    valid: pd.DataFrame,
    cols: list[str] | tuple[str, ...] | None = None,
    name: str = "EWS",
) -> pd.Series:
    """
    Simple Early Warning Score (EWS): count of valid boolean signals per date.

    If 5 indicators are used, the score is in [0, 5].
    """
    if valid is None or valid.empty:
        raise ValueError("valid must be a non-empty DataFrame of booleans.")

    if cols is None:
        cols = list(valid.columns)

    missing = [c for c in cols if c not in valid.columns]
    if missing:
        raise ValueError(f"Columns not found in valid: {missing}")

    ews = valid.loc[:, cols].fillna(False).astype(bool).sum(axis=1).astype(int)
    ews.name = name
    return ews


## Setting Plotting Theme
def set_theme_like_example():
    sns.set_theme(
        style="darkgrid",
        context="notebook",
        font_scale=1.1,
        rc={
            "axes.facecolor": "#EAEAF2",
            "figure.facecolor": "white",
            "grid.color": "white",
            "grid.linewidth": 1.0,
            "axes.edgecolor": "#EAEAF2",
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
        },
    )


## Plotting function for plotting the spread level with EWS overlay
def plot_spread_with_ews_overlay(
    spread: pd.Series | pd.DataFrame,
    ews_score: pd.Series,
    crisis_date: pd.Timestamp | str | None = None,
    start_date: pd.Timestamp | str | None = None,
    end_date: pd.Timestamp | str | None = None,
    min_overlay_score: int = 3,
    spread_label: str = "Spread",
    extreme_indicator__cutoff=0.1,
    title: str = "Spread with EWS Overlay",
    look_after_crisis_date: bool = True,
    look_after_date: pd.Timestamp | str | None = None,
    input_title:str = "Spain-Germany 5Y Benchmark Spread with EWS Overlay"
):
    set_theme_like_example()

    if isinstance(spread, pd.DataFrame):
        if spread.shape[1] != 1:
            raise ValueError("If spread is a DataFrame, it must contain exactly one column.")
        spread_series = spread.iloc[:, 0].copy()
        if spread_label == "Spread":
            spread_label = str(spread.columns[0])
    else:
        spread_series = spread.copy()

    if not isinstance(spread_series.index, pd.DatetimeIndex):
        spread_series.index = pd.to_datetime(spread_series.index)

    ews = ews_score.copy()
    if not isinstance(ews.index, pd.DatetimeIndex):
        ews.index = pd.to_datetime(ews.index)

    data = pd.DataFrame({"spread": spread_series}).join(ews.rename("EWS"), how="left")
    data["EWS"] = data["EWS"].fillna(0).astype(int)

    crisis_dt = pd.to_datetime(crisis_date) if crisis_date is not None else None

    if look_after_crisis_date:
        if crisis_dt is None:
            raise ValueError("When look_after_crisis_date=True, pass crisis_date.")
        post_end = pd.to_datetime(look_after_date) if look_after_date is not None else (crisis_dt + pd.DateOffset(months=6))
        left = pd.to_datetime(start_date) if start_date is not None else data.index.min()
        data = data.loc[left:post_end]
    else:
        left = pd.to_datetime(start_date) if start_date is not None else None
        right = pd.to_datetime(end_date) if end_date is not None else None
        data = data.loc[left:right]

    fig, ax1 = plt.subplots(figsize=(13, 5))
    ax1.plot(data.index, data["spread"], color="#1D4ED8", lw=2, label=spread_label)

    if crisis_dt is not None:
        ax1.axvline(crisis_dt, color="#DC2626", ls="--", lw=1.2, alpha=0.85, label="Crisis date")

    # EWS overlay ONLY before/at crisis date
    pre_mask = data.index <= crisis_dt if crisis_dt is not None else pd.Series(True, index=data.index)
    high_mask = (data["EWS"] >= min_overlay_score) & pre_mask
    if high_mask.any():
        ax1.scatter(
            data.index[high_mask],
            data.loc[high_mask, "spread"],
            color="#B91C1C",
            s=45,
            zorder=4,
            label=f"EWS >= {min_overlay_score} (pre-crisis)",
        )

    ax1.set_xlabel("")
    ax1.set_ylabel(spread_label)
    ax1.grid(alpha=0.25)
    ax1.set_title(input_title)

    # EWS axis only for pre-crisis part
    ax2 = ax1.twinx()
    ews_pre = data["EWS"].where(pre_mask, np.nan)
    ax2.step(data.index, ews_pre, where="post", color="#6B7280", lw=1.6, alpha=0.9, label="EWS score (pre-crisis)")
    ax2.axhline(min_overlay_score, color="#6B7280", ls=":", lw=1.0, alpha=0.8)
    ax2.set_ylabel("EWS score")
    ax2.set_ylim(-0.2, max(min_overlay_score + 1, int(data["EWS"].max()) + 0.8))

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left")

    plt.tight_layout()
    plt.show()

    return data


## Plotting each indicator with selected thresholds, signals, and valid signals, with crisis date marked optionally
def plot_all_indicators(
    rolling_complexity_inds: pd.DataFrame,
    crisis_date: pd.Timestamp,
    start_date: pd.Timestamp | str | None = None,
    end_date: pd.Timestamp | str | None = None,
    thresholds: dict | None = None,
    signals: pd.DataFrame | None = None,
    valid: pd.DataFrame | None = None,
):
    set_theme_like_example()

    data = rolling_complexity_inds
    if start_date is not None or end_date is not None:
        data = data.loc[
            pd.to_datetime(start_date) if start_date is not None else None : pd.to_datetime(end_date)
            if end_date is not None
            else None
        ]

    for col in data.columns:
        ax = data[col].plot(title=col, linewidth=2.0, color="#4C72B0")
        ax.axvline(crisis_date, color="red", linestyle="--", linewidth=1.2, alpha=0.8)

        if thresholds is not None and col in thresholds:
            ax.axhline(thresholds[col], color="black", linestyle=":", linewidth=1.2, alpha=0.8)

        if signals is not None and col in signals.columns:
            sig_mask = signals.loc[data.index, col].fillna(False)
            if sig_mask.any():
                sig_idx = data.index[sig_mask]
                ax.scatter(
                    sig_idx.to_pydatetime(),
                    data.loc[sig_idx, col].to_numpy(dtype=float),
                    color="orange",
                    s=30,
                    label="signal",
                )

        if valid is not None and col in valid.columns:
            val_mask = valid.loc[data.index, col].fillna(False)
            if val_mask.any():
                val_idx = data.index[val_mask]
                ax.scatter(
                    val_idx.to_pydatetime(),
                    data.loc[val_idx, col].to_numpy(dtype=float),
                    color="green",
                    s=40,
                    label="valid",
                )

        ax.set_xlabel("")
        if signals is not None or valid is not None:
            ax.legend()
        plt.show()