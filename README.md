# Complexity Theory Master's Thesis

This repository contains the empirical work for a master's thesis on whether
complexity theory indicators can detect early signs of market fragility in
sovereign yield spreads. The main application is the Spain-Germany 5-year benchmark spread during the European sovereign debt crisis.

The project combines three related pieces of analysis:

- Construction of Spain-Germany sovereign spread series from benchmark yields.
- Rolling complexity indicators and aggregate early-warning signals (EWS).
- Regime classification of sovereign spread stress using Markov-switching and
  hidden Markov style models.

## Repository Structure

```text
.
|-- Complexity_Theory_Indicators/
|-- Data/
|-- Masters_Thesis/
|-- Research_Papers/
|-- Sovereign_Debt_Crisis/
|-- LICENSE
|-- README.md
`-- REQUIREMENTS.txt
```

## Folder Contents

### `Complexity_Theory_Indicators/`

This folder contains the reusable Python helper code for calculating and
plotting the complexity theory indicators used throughout the notebooks.

Files:

- `Complexity_Theory_Functions_Master.py`

  - Main utility module for the thesis analysis.
  - Implements rolling complexity indicators based on the framework of van den
    End (2019).
  - Contains functions for:
    - AR(1) persistence estimation through `ar1_phi`.
    - Decay-rate and slowdown calculations through `decay_rate_from_phi` and
      `decay_rate_from_N_ratio`.
    - Sarle's bimodality coefficient through `sarle_bimodality`, used as the
      flickering indicator.
    - Rolling indicator generation through `rolling_complexity_indicators`.
    - Tail-based signal extraction through `tail_signals`.
    - Aggregating individual indicator signals into an EWS score through
      `ews_score_from_valid`.
    - Plot styling and visual diagnostics through `set_theme_like_example`,
      `plot_spread_with_ews_overlay`, `plot_all_indicators`
  - The indicators produced by the module include autocorrelation, slowdown,
    flickering, variance, and skewness.

### `Data/`

This folder contains the yield and spread data used by the notebooks.

Files:

- `Spread_Data.xlsx`

  - Main Excel workbook containing raw or source time series.
  - Sheets currently include benchmark yields for Spain, Germany, Great
    Britain, and Hungary, plus market and policy-rate series such as `SP500`,
    `EURO OVERNIGHT DEPOSIT`, `ECB MRO`, `ECBMLF_1`, and `ECBMLF_2`.
- `Spain_Benchmark_Yields.csv`

  - Daily Spanish benchmark government bond yields.
  - Contains 5-year, 10-year, and 30-year benchmark yield columns.
- `Germany_Benchmark_Yields.csv`

  - Daily German benchmark government bond yields.
  - Contains 5-year, 10-year, and 30-year benchmark yield columns.
- `Spain_Germany_Spreads_5Y.csv`

  - Precomputed 5-year Spain-Germany spread series.
  - The spread is constructed as Spanish 5-year yield minus German 5-year yield.
  - This is the central state variable for most of the crisis and EWS analysis.
- `Spain_Germany_Spreads_10Y.csv`

  - Precomputed 10-year Spain-Germany spread series.
  - Used as an additional maturity for comparison and robustness.
- `Spain_Germany_Spreads_30Y.csv`

  - Precomputed 30-year Spain-Germany spread series.
  - Used as a longer-maturity comparison series.

### `Sovereign_Debt_Crisis/`

This folder contains the main empirical notebooks for the sovereign debt crisis
application.

Files:

- `Sovereign_Debt_Crisis.ipynb`

  - Main notebook for the complexity-indicator and early-warning-signal
    analysis.
  - Loads the benchmark yield and spread data.
  - Constructs Spain-Germany 5-year, 10-year, and 30-year sovereign spreads.
  - Defines and studies key European sovereign debt crisis episodes, including:
    - Greek fiscal shock.
    - SMP launch.
    - Spain/Italy contagion.
    - LTRO liquidity episode.
    - Spanish banking assistance.
  - Computes rolling complexity indicators for the Spain-Germany 5-year spread.
  - Converts individual indicator extremes into binary tail signals.
  - Aggregates those binary signals into an EWS score.
  - Builds event-window diagnostics around crisis episodes, including whether
    EWS signals appear in the 100-day, 50-day, 25-day, 10-day, and 5-day
    windows before each episode.
  - Produces heatmaps for:
    - Presence of EWS warnings before crisis episodes.
    - Persistence of EWS warnings as a percentage of each pre-event window.
  - Runs a Welch two-sided t-test comparing future spread changes after EWS
    signal dates with future spread changes after randomly selected non-signal
    dates.
  - Produces merged plots showing crisis dates, spread behavior, and EWS
    overlays.
  - Concludes that EWS signals are more informative for episodes driven by
    endogenous fragility in Spanish spreads than for episodes driven mainly by
    external shocks or discretionary policy actions.
- `Sovereign_Spreads_and_HMM.ipynb`

  - Notebook focused on regime classification and Markov-switching analysis.
  - Loads and constructs Spain-Germany sovereign spread data.
  - Fits a two-state Markov-switching AR(1) model to the 5-year spread.
  - Examines whether the spread series is suitable for a hidden Markov or
    Markov-switching approach.
  - Uses distribution plots, ACF, and PACF diagnostics to inspect spread levels
    and spread changes.
  - Compares the interpretation of a Gaussian HMM with a Markov-switching
    autoregressive model.
  - Labels regimes as stress or calm based on average spread levels.
  - Plots the spread and underlying Spanish/German yields colored by inferred
    regime.
  - Uses the Spanish banking assistance episode as a visual benchmark for
    whether the model classifies the crisis period plausibly.
  - Runs residual diagnostics, including Ljung-Box tests for residual
    autocorrelation and ARCH-LM tests for volatility clustering.
  - Concludes that the model is useful as a regime-dating tool, while residual
    autocorrelation and heteroskedasticity suggest that richer specifications
    could improve model fit.

### `Research_Papers/`

This folder contains background literature used to motivate the methods and
interpretation.

Files:

- `Complexity_Theory_Van_den_End.pdf`

  - Core methodological reference for the complexity indicators used in the
    project.
  - Motivates the use of indicators such as autocorrelation, slowdown,
    variance, skewness, and flickering as signs of rising systemic fragility.

### `Masters_Thesis/`

This folder contains final thesis and presentation outputs.

Files:

- `Master's_Thesis_Marriott_Czachesz.pdf`

  - Final written thesis document.
  - Presents the theoretical motivation, methodology, empirical analysis, and
    conclusions.
- `Presentation_Marriott_Czachesz.pdf`

  - Thesis presentation slide deck.
  - Summarizes the research question, methods, results, and conclusions

## Setup

Create and activate a virtual environment, then install the requirements:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r REQUIREMENTS.txt
```

On macOS or Linux, activate the environment with:

```bash
source .venv/bin/activate
```

Then start Jupyter:

```bash
jupyter notebook
```

## Suggested Reading Order

1. `Masters_Thesis/Master's_Thesis_Marriott_Czachesz.pdf`

   - Best for the full research narrative.
2. `Sovereign_Debt_Crisis/Sovereign_Debt_Crisis.ipynb`

   - Best for the main early-warning-signal workflow.
3. `Sovereign_Debt_Crisis/Sovereign_Spreads_and_HMM.ipynb`

   - Best for the complementary regime-classification analysis.
4. `Complexity_Theory_Indicators/Complexity_Theory_Functions_Master.py`

   - Best for understanding the reusable indicator and plotting code.

## Methodological Summary

The project treats the Spain-Germany sovereign spread as a state variable that
may show signs of instability before crisis episodes. Rolling windows are used
to compute complexity indicators. Extreme values of these indicators are
converted into binary warning signals using tail thresholds. The individual
signals are then summed into an aggregate EWS score. Dates where the aggregate
score exceeds the chosen warning threshold are interpreted as periods of
elevated market fragility.

The HMM and Markov-switching notebook complements this by asking whether the
spread can be statistically separated into calm and stress regimes. This is not
treated as a replacement for the complexity-indicator framework, but as an
additional way to assess whether the crisis periods identified in the thesis
correspond to distinct spread regimes.
