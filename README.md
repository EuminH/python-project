# ⚡ Sports Betting Intelligence

A multi-sport value-betting dashboard built with Streamlit. It finds **+EV (positive expected value)** bets across MLB, the FIFA World Cup, and Wimbledon (ATP + WTA), builds parlays, runs an ML model for the Premier League, and tracks your bets — all in one dark, live dashboard.

## Features

- **Value finder** — scans **FanDuel + DraftKings** for today & tomorrow, de-vigs the lines into a fair probability, and tells you exactly what to bet (pick, book, odds, EV, ½-Kelly stake).
- **Best parlays** — auto-built legs spanning the risk spectrum (most likely to hit → highest EV).
- **ML model** — Gradient Boosting on 6 Premier League seasons + StatsBomb xG, with probability calibration and a walk-forward backtest.
- **Live odds** — best lines across major US sportsbooks via The Odds API, plus ESPN scores.
- **Tools** — odds calculator, Kelly staking, parlay builder, arbitrage checker, and a bet tracker with P&L.

## Live data

Live odds come from [The Odds API](https://the-odds-api.com) (free tier). You need your own API key — it is read from `ODDS_API_KEY`.

## Run locally

```bash
pip install -r requirements.txt
echo "ODDS_API_KEY=your_key_here" > .env
streamlit run app.py
```

Open http://localhost:8501.

## Deploy to Streamlit Community Cloud (free)

1. Push this repo to GitHub (already done if you're reading this on GitHub).
2. Go to **https://share.streamlit.io** and sign in with GitHub.
3. **New app** → pick this repo, branch `main`, main file `app.py`.
4. Open **Advanced settings → Secrets** and paste:
   ```toml
   ODDS_API_KEY = "your_the_odds_api_key_here"
   ```
5. **Deploy.** You'll get a public `https://<your-app>.streamlit.app` URL.

> The API key lives only in Streamlit's Secrets store — never in the repo. `.env` and `.streamlit/secrets.toml` are gitignored.

## Note

For research and educational use. Betting involves risk; only the +EV plays are mathematically favorable, and even those carry variance.
