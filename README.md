# One Screener — hosted dashboard

Public, read-only deploy of the One Screener IDX signal dashboard, for
Streamlit Community Cloud.

This repository is a generated mirror: it carries only what the dashboard needs
to run — the app, the `idxcore` reader package, `requirements.txt`, and a slim
read-only DuckDB store (`data/idx_slim.duckdb`, the last ~180 bars per ticker
plus precomputed calibration and quality summaries). The full price store and
the data pipeline live in the private source repository and never travel here.

Run locally:

    streamlit run app/dashboard.py

Deploy: point Streamlit Community Cloud at `app/dashboard.py`.

Prices are raw and unadjusted, from Yahoo Finance. Nothing here is investment
advice, and no win rate is shown that has not been measured.
