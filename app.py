import os
import streamlit as st

# Bridge Streamlit Cloud secrets -> environment BEFORE importing data modules
# (live_data / value_betting read ODDS_API_KEY at import time). Locally this is a
# no-op and the .env file is used instead.
try:
    if "ODDS_API_KEY" in st.secrets:
        os.environ["ODDS_API_KEY"] = str(st.secrets["ODDS_API_KEY"])
except Exception:
    pass

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import json, datetime
from dotenv import load_dotenv

load_dotenv()  # local .env fallback (does not override secrets already set above)

from live_data import get_live_odds, get_best_lines, get_espn_scores, get_espn_standings, get_quota
from ml_betting import load_data, build_features, backtest, vig, kelly_fraction, FEATURE_COLS, predict_upcoming
from sports_betting import load_bets, save_bets, american_to_decimal, american_implied_prob, payout, profit

st.set_page_config(page_title="Sports Betting Intelligence", page_icon="⚡", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background-color: #0d0d1a !important;
    font-family: 'Inter', sans-serif;
}
[data-testid="stSidebar"] {
    background-color: #111127 !important;
    border-right: 1px solid #1e1e3a;
}
[data-testid="stSidebar"] .stRadio label {
    color: #94a3b8 !important;
    font-size: 13px !important;
    padding: 6px 0;
}
[data-testid="stSidebar"] .stRadio label:hover { color: #f97316 !important; }
section[data-testid="stSidebar"] > div { padding-top: 1rem; }
h1,h2,h3 { color: #e2e8f0 !important; font-weight: 600 !important; }
.stMetric { background: #1a1a2e; border-radius: 10px; padding: 1rem; border: 1px solid #1e1e3a; }
.stMetric label { color: #64748b !important; font-size: 11px !important; text-transform: uppercase; letter-spacing: .08em; }
.stMetric [data-testid="stMetricValue"] { color: #e2e8f0 !important; font-size: 22px !important; }
.stDataFrame { border-radius: 10px; overflow: hidden; }
[data-testid="stDataFrameResizable"] { border: 1px solid #1e1e3a !important; }
.stTabs [data-baseweb="tab"] { color: #64748b !important; }
.stTabs [aria-selected="true"] { color: #f97316 !important; border-bottom-color: #f97316 !important; }
.stSelectbox label, .stSlider label, .stNumberInput label { color: #94a3b8 !important; font-size: 13px !important; }
.stForm { background: #1a1a2e; border-radius: 10px; padding: 1.2rem; border: 1px solid #1e1e3a; }
div[data-testid="metric-container"] { background: #1a1a2e; border-radius: 10px; padding: 12px 16px; border: 1px solid #1e1e3a; }

/* Top nav bar */
.topbar {
    display: flex; align-items: center; justify-content: space-between;
    padding: 10px 0 20px 0; border-bottom: 1px solid #1e1e3a; margin-bottom: 20px;
}
.topbar-left { display: flex; align-items: center; gap: 12px; }
.sport-badge {
    background: #1e1e3a; color: #f97316; font-size: 12px; font-weight: 700;
    letter-spacing: .1em; padding: 4px 10px; border-radius: 6px;
}
.topbar-sub { color: #64748b; font-size: 12px; letter-spacing: .05em; }
.topbar-date { color: #64748b; font-size: 12px; }

/* Intelligence header */
.intel-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16162a 100%);
    border: 1px solid #1e1e3a; border-radius: 12px;
    padding: 20px 24px; margin-bottom: 16px;
}
.intel-label { color: #f97316; font-size: 11px; font-weight: 700; letter-spacing: .1em; margin-bottom: 4px; }
.intel-date { color: #e2e8f0; font-size: 28px; font-weight: 700; margin-bottom: 16px; }
.intel-stats { display: flex; gap: 32px; flex-wrap: wrap; }
.intel-stat { display: flex; flex-direction: column; gap: 2px; }
.intel-stat-label { color: #64748b; font-size: 10px; text-transform: uppercase; letter-spacing: .08em; }
.intel-stat-value { font-size: 20px; font-weight: 700; }
.intel-stat-sub { color: #64748b; font-size: 11px; }
.val-green { color: #22c55e; }
.val-orange { color: #f97316; }
.val-blue { color: #7c3aed; }
.val-white { color: #e2e8f0; }
.tracked-badge { color: #22c55e; font-size: 12px; }

/* Win probability bars */
.prob-bar-wrap { margin-bottom: 8px; }
.prob-bar-label { display: flex; justify-content: space-between; align-items: center; margin-bottom: 3px; }
.prob-bar-name { color: #e2e8f0; font-size: 13px; }
.prob-bar-pct { font-size: 13px; font-weight: 600; }
.prob-bar-track { background: #1e1e3a; border-radius: 4px; height: 8px; overflow: hidden; }
.prob-bar-fill-h { background: #ef4444; height: 100%; border-radius: 4px; }
.prob-bar-fill-m { background: #f97316; height: 100%; border-radius: 4px; }
.prob-bar-fill-l { background: #64748b; height: 100%; border-radius: 4px; }

/* Best play card */
.best-play-card {
    background: #1a1a2e; border: 1px solid #1e1e3a; border-radius: 12px; padding: 18px;
}
.best-play-label { color: #f97316; font-size: 10px; font-weight: 700; letter-spacing: .1em; margin-bottom: 4px; }
.best-play-sub { color: #64748b; font-size: 11px; margin-bottom: 14px; }
.matchup-row { display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }
.player-name { color: #e2e8f0; font-size: 15px; font-weight: 600; }
.player-elo { color: #64748b; font-size: 11px; }
.win-pct { color: #f97316; font-size: 36px; font-weight: 800; text-align: center; margin: 6px 0 2px; }
.win-label { color: #64748b; font-size: 11px; text-align: center; margin-bottom: 12px; }
.stat-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 8px; margin-top: 10px; }
.stat-cell { background: #111127; border-radius: 8px; padding: 8px 10px; }
.stat-cell-label { color: #64748b; font-size: 10px; text-transform: uppercase; letter-spacing: .06em; }
.stat-cell-val { color: #e2e8f0; font-size: 14px; font-weight: 600; margin-top: 2px; }
.stat-cell-val.orange { color: #f97316; }
.stat-cell-val.green { color: #22c55e; }

/* Insights */
.insight-row {
    display: flex; align-items: center; gap: 10px; padding: 10px 14px;
    background: #111127; border-radius: 8px; margin-bottom: 6px; font-size: 13px; color: #e2e8f0;
}

/* Match cards */
.match-card {
    background: #1a1a2e; border: 1px solid #1e1e3a; border-radius: 10px;
    padding: 14px 16px; margin-bottom: 10px;
}
.match-card-header { display: flex; gap: 8px; align-items: center; margin-bottom: 10px; }
.match-tag-atp { background: #7c3aed22; color: #a78bfa; font-size: 10px; font-weight: 700; padding: 2px 7px; border-radius: 4px; }
.match-tag-wta { background: #f9731622; color: #fb923c; font-size: 10px; font-weight: 700; padding: 2px 7px; border-radius: 4px; }
.match-tag-epl { background: #22c55e22; color: #4ade80; font-size: 10px; font-weight: 700; padding: 2px 7px; border-radius: 4px; }
.match-tag-nfl { background: #3b82f622; color: #60a5fa; font-size: 10px; font-weight: 700; padding: 2px 7px; border-radius: 4px; }
.match-venue { color: #64748b; font-size: 11px; }
.match-teams-row { display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; }
.match-team { color: #e2e8f0; font-size: 15px; font-weight: 600; }
.match-vs { color: #64748b; font-size: 11px; }
.match-bar { height: 3px; border-radius: 2px; background: linear-gradient(90deg, #f97316 0%, #7c3aed 100%); margin-bottom: 8px; }
.match-footer { display: flex; justify-content: space-between; font-size: 11px; color: #64748b; }
.match-edge { color: #22c55e; font-weight: 600; }
.match-edge-neg { color: #ef4444; font-weight: 600; }

/* ELO table */
.elo-row { display: flex; align-items: center; gap: 10px; padding: 6px 0; border-bottom: 1px solid #1e1e3a; }
.elo-rank { color: #64748b; font-size: 12px; width: 16px; }
.elo-name { color: #e2e8f0; font-size: 13px; flex: 1; }
.elo-bar-wrap { flex: 2; background: #1e1e3a; height: 6px; border-radius: 3px; overflow: hidden; }
.elo-bar-fill { height: 100%; border-radius: 3px; background: linear-gradient(90deg, #f97316, #ef4444); }
.elo-val { color: #f97316; font-size: 13px; font-weight: 700; min-width: 40px; text-align: right; }

/* Surface / tour split */
.split-card { background: #1a1a2e; border: 1px solid #1e1e3a; border-radius: 10px; padding: 14px; }
.split-label { color: #64748b; font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: .08em; margin-bottom: 10px; }
.split-row { display: flex; align-items: center; gap: 8px; margin-bottom: 6px; }
.split-badge { width: 24px; height: 24px; border-radius: 6px; display: flex; align-items: center; justify-content: center; font-size: 10px; font-weight: 700; }
.split-badge-h { background: #3b82f622; color: #60a5fa; }
.split-badge-c { background: #f9731622; color: #fb923c; }
.split-badge-g { background: #22c55e22; color: #4ade80; }
.split-badge-atp { background: #7c3aed22; color: #a78bfa; }
.split-badge-wta { background: #f9731622; color: #fb923c; }
.split-name { color: #94a3b8; font-size: 13px; flex: 1; }
.split-val { color: #e2e8f0; font-size: 15px; font-weight: 700; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div style="color:#f97316;font-size:18px;font-weight:700;padding:8px 16px 4px;">⚡ BET INTEL</div>', unsafe_allow_html=True)
    st.markdown('<div style="color:#64748b;font-size:11px;padding:0 16px 16px;">Sports Betting Dashboard</div>', unsafe_allow_html=True)
    st.divider()
    page = st.radio("Navigation", [
        "🏠 Daily Intelligence",
        "🎯 Side Bets",
        "📖 How It Works",
        "🤖 ML Model",
        "📡 Live Odds",
        "🔢 Odds Calculator",
        "📊 Kelly Staking",
        "🎰 Parlay Builder",
        "🧩 Build My Parlay",
        "📋 Bet Tracker",
    ], label_visibility="collapsed")


now = datetime.datetime.now()
date_str = now.strftime("%A, %b %-d, %Y")


# ── Daily Intelligence ─────────────────────────────────────────────────────
if page == "🏠 Daily Intelligence":
    import datetime as _dt
    from value_betting import (fetch_events, value_bets, build_parlay_suite, kelly_stake,
                               snapshot_closing, SPORTS, SPORT_TAGS)

    today = _dt.date.today()
    tomorrow = today + _dt.timedelta(days=1)
    range_str = f"{today.strftime('%a %b %-d')} – {tomorrow.strftime('%a %b %-d')}"

    # Sportsbook selector — which book's prices to bet at (fair % uses the full consensus)
    book_choice = st.sidebar.radio("Sportsbook to bet at",
                                   ["Best of both", "FanDuel only", "DraftKings only"],
                                   help="Which book you'll actually place bets at. Fair % always uses the full multi-book consensus; this only changes which price (and EV) you're shown.")
    BET_BOOKS = {"Best of both": None, "FanDuel only": ["fanduel"], "DraftKings only": ["draftkings"]}[book_choice]
    book_short = {"Best of both": "FD · DK", "FanDuel only": "FanDuel", "DraftKings only": "DraftKings"}[book_choice]
    book_sub = {"Best of both": "best line taken", "FanDuel only": "FanDuel prices", "DraftKings only": "DraftKings prices"}[book_choice]
    topbar_books = {"Best of both": "FANDUEL + DRAFTKINGS", "FanDuel only": "FANDUEL", "DraftKings only": "DRAFTKINGS"}[book_choice]

    # Top bar
    st.markdown(f"""
    <div class="topbar">
        <div class="topbar-left">
            <div style="width:10px;height:10px;border-radius:50%;background:#22c55e"></div>
            <span class="sport-badge">VALUE FINDER</span>
            <span class="topbar-sub">· {topbar_books} · TODAY & TOMORROW · 8-BOOK CONSENSUS</span>
        </div>
        <div class="topbar-date">{range_str}</div>
    </div>
    """, unsafe_allow_html=True)

    bankroll = st.sidebar.number_input("Bankroll ($)", 50.0, 1_000_000.0, 1000.0, 50.0)
    min_ev_pct = st.sidebar.slider("Min EV to recommend (%)", 0.0, 10.0, 0.0, 0.5)
    min_leg_prob = st.sidebar.slider("Min leg probability for parlays (%)", 0, 90, 0, 5,
                                     help="Only build parlays from legs at least this likely to hit. Raise it for safer, more-likely-to-hit parlays.")
    market_filter = st.sidebar.multiselect("Markets", ["ML", "Spread", "Total"],
                                           default=["ML", "Spread", "Total"],
                                           help="ML = moneyline (who wins). Spread = handicap. Total = over/under.")

    # Cache the raw API fetch only — switching book/market/sliders re-computes locally
    @st.cache_data(ttl=300, show_spinner="Scanning the books for value…")
    def load_events(_d):
        ev = fetch_events(days=2)
        snapshot_closing(ev)   # keep latest pre-kickoff prices for CLV grading
        return ev

    raw = load_events(f"{today}-v2")
    data = value_bets(raw, min_ev=-1.0, bet_books=BET_BOOKS)
    if set(market_filter) != {"ML", "Spread", "Total"}:
        data = {s: [b for b in v if b.get("market", "ML") in market_filter]
                for s, v in data.items()}
    parlays = build_parlay_suite(data, min_leg_prob=min_leg_prob / 100)

    # API quota meter (The Odds API free tier = 500 credits/month)
    _q = get_quota()
    if _q:
        _pct = max(0.0, min(1.0, _q["remaining"] / (_q["remaining"] + _q["used"]))) if (_q["remaining"] + _q["used"]) else 0
        st.sidebar.markdown(
            f'<div style="margin-top:8px;padding:8px 10px;background:#111127;border-radius:8px;border:1px solid #1e1e3a">'
            f'<div style="color:#64748b;font-size:10px;text-transform:uppercase;letter-spacing:.06em">API credits left</div>'
            f'<div style="color:{"#22c55e" if _q["remaining"]>100 else "#f59e0b" if _q["remaining"]>25 else "#ef4444"};font-size:16px;font-weight:700">{_q["remaining"]:.0f}</div>'
            f'<div style="background:#1e1e3a;border-radius:3px;height:5px;margin-top:4px"><div style="background:#f97316;height:100%;border-radius:3px;width:{_pct*100:.0f}%"></div></div>'
            f'<div style="color:#475569;font-size:10px;margin-top:3px">used {_q["used"]:.0f} · as of {_q["ts"][11:16]}</div></div>',
            unsafe_allow_html=True)

    # Flatten + split
    all_priced = [b for bets in data.values() for b in bets]
    recs = sorted([b for b in all_priced if b["ev"] * 100 >= max(min_ev_pct, 0.0001)],
                  key=lambda x: x["ev"], reverse=True)
    n_games = len({(b["sport"], b["match"]) for b in all_priced})
    best_ev = recs[0]["ev_per_100"] if recs else 0.0
    total_stake = sum(kelly_stake(b["fair_prob"], b["decimal"], bankroll) for b in recs)

    # ── Header ──────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="intel-header">
        <div style="display:flex;justify-content:space-between;align-items:flex-start">
            <div>
                <div class="intel-label">VALUE BETS · TODAY & TOMORROW</div>
                <div class="intel-date">{today.strftime('%A, %B %-d')}</div>
            </div>
            <div class="tracked-badge">● {len(recs)} +EV bets · {n_games} games scanned</div>
        </div>
        <div class="intel-stats">
            <div class="intel-stat">
                <div class="intel-stat-label">+EV Bets</div>
                <div class="intel-stat-value val-green">{len(recs)}</div>
                <div class="intel-stat-sub">to place now</div>
            </div>
            <div class="intel-stat">
                <div class="intel-stat-label">Best Edge</div>
                <div class="intel-stat-value val-orange">+{best_ev:.1f}%</div>
                <div class="intel-stat-sub">EV per $100</div>
            </div>
            <div class="intel-stat">
                <div class="intel-stat-label">Games Scanned</div>
                <div class="intel-stat-value val-blue">{n_games}</div>
                <div class="intel-stat-sub">4 sports · 2 books</div>
            </div>
            <div class="intel-stat">
                <div class="intel-stat-label">Suggested Stake</div>
                <div class="intel-stat-value val-white">${total_stake:,.0f}</div>
                <div class="intel-stat-sub">½-Kelly · ${bankroll:,.0f} roll</div>
            </div>
            <div class="intel-stat">
                <div class="intel-stat-label">Betting at</div>
                <div class="intel-stat-value val-white">{book_short}</div>
                <div class="intel-stat-sub">{book_sub}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── WHAT TO BET ─────────────────────────────────────────────────────────
    st.markdown('<div style="color:#64748b;font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.08em;margin:6px 0 12px">✅ What to bet — recommended +EV plays</div>', unsafe_allow_html=True)

    if recs:
        head = (
            '<tr style="border-bottom:1px solid #1e1e3a;color:#64748b">'
            '<th style="text-align:left;padding:6px 8px 6px 0;font-weight:500">SPORT</th>'
            '<th style="text-align:left;padding:6px 8px;font-weight:500">BET (PICK)</th>'
            '<th style="text-align:center;padding:6px 8px;font-weight:500">MARKET</th>'
            '<th style="text-align:left;padding:6px 8px;font-weight:500">MATCH</th>'
            '<th style="text-align:center;padding:6px 8px;font-weight:500">BOOK</th>'
            '<th style="text-align:right;padding:6px 8px;font-weight:500">ODDS</th>'
            '<th style="text-align:right;padding:6px 8px;font-weight:500">FAIR%</th>'
            '<th style="text-align:right;padding:6px 8px;font-weight:500">EV/$100</th>'
            '<th style="text-align:right;padding:6px 8px;font-weight:500">½-KELLY STAKE</th>'
            '<th style="text-align:right;padding:6px 0;font-weight:500">KICKOFF</th></tr>'
        )
        body = ""
        for b in recs[:25]:
            am = f"+{b['american']}" if b['american'] > 0 else str(b['american'])
            book_color = "#1493ff" if "Draft" in b['book'] else "#1a9c4c"
            stake = kelly_stake(b["fair_prob"], b["decimal"], bankroll)
            ev_color = "#22c55e" if b["ev_per_100"] >= 3 else "#4ade80"
            body += (
                '<tr style="border-bottom:1px solid #111127">'
                f'<td style="color:#94a3b8;padding:7px 8px 7px 0;font-size:11px">{SPORT_TAGS.get(b["sport"],"")} {b["sport"]}</td>'
                f'<td style="color:#e2e8f0;padding:7px 8px;font-weight:600">{b["pick"][:22]}</td>'
                f'<td style="text-align:center;padding:7px 8px"><span style="background:#7c3aed22;color:#a78bfa;padding:2px 7px;border-radius:4px;font-size:10px;font-weight:600">{b.get("market","ML")}</span></td>'
                f'<td style="color:#64748b;padding:7px 8px;font-size:11px">{b["match"][:30]}</td>'
                f'<td style="text-align:center;padding:7px 8px"><span style="background:{book_color}22;color:{book_color};padding:2px 7px;border-radius:4px;font-size:11px;font-weight:600">{b["book"]}</span></td>'
                f'<td style="color:#f97316;text-align:right;padding:7px 8px;font-weight:700">{am}</td>'
                f'<td style="color:#94a3b8;text-align:right;padding:7px 8px">{b["fair_prob"]*100:.1f}%</td>'
                f'<td style="color:{ev_color};text-align:right;padding:7px 8px;font-weight:700">+${b["ev_per_100"]:.2f}</td>'
                f'<td style="color:#e2e8f0;text-align:right;padding:7px 8px">${stake:.2f}</td>'
                f'<td style="color:#64748b;text-align:right;padding:7px 0;font-size:11px">{b["date"][5:]} {b["time"][:5]}</td></tr>'
            )
        st.markdown(
            '<div style="background:#1a1a2e;border:1px solid #1e1e3a;border-radius:12px;padding:16px;overflow-x:auto">'
            '<table style="width:100%;border-collapse:collapse;font-size:12px">' + head + body + '</table></div>',
            unsafe_allow_html=True)
        st.caption(f"Fair % = median power-devig consensus of up to 8 US books (corrects longshot bias). EV = edge at your book's price. Stake = ½-Kelly on ${bankroll:,.0f}.")

        # ── Inspect & log a bet ─────────────────────────────────────────────
        with st.expander("🔍 Inspect a pick — full math, every book's price, log the bet"):
            opts = {f"{i+1}. {b['pick']} [{b.get('market','ML')}] · {b['match'][:26]} · {b['book']}": b
                    for i, b in enumerate(recs[:25])}
            sel = st.selectbox("Pick to inspect", list(opts.keys()))
            b = opts[sel]
            stake_sug = kelly_stake(b["fair_prob"], b["decimal"], bankroll)
            imp_at_book = 1 / b["decimal"] * 100

            ic1, ic2 = st.columns([1.1, 0.9])
            with ic1:
                st.markdown(
                    '<div style="background:#111127;border-radius:10px;padding:14px;font-size:13px;line-height:2;color:#cbd5e1">'
                    f'<b style="color:#e2e8f0">The math for this pick</b><br>'
                    f'Your book\'s implied probability: <b>{imp_at_book:.1f}%</b> (1 ÷ {b["decimal"]:.3f})<br>'
                    f'Consensus fair probability: <b style="color:#a78bfa">{b["fair_prob"]*100:.1f}%</b> (median of {b["n_books"]} books, power de-vig)<br>'
                    f'Edge: <b style="color:#22c55e">{(b["fair_prob"]-1/b["decimal"])*100:+.1f} points</b><br>'
                    f'EV = {b["fair_prob"]:.3f} × {b["decimal"]:.3f} − 1 = <b style="color:#22c55e">{b["ev_per_100"]:+.2f} per $100</b><br>'
                    f'Market width (book disagreement): <b>{b["width"]:.1f}%</b> — {"wide, treat with caution" if b["width"] > 8 else "tight, consensus is solid"}'
                    '</div>', unsafe_allow_html=True)
            with ic2:
                rows_html = "".join(
                    f'<div style="display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid #1a1a2e">'
                    f'<span style="color:{"#f97316" if t == b["book"] else "#94a3b8"};font-size:12px">{t}{" ← bet here" if t == b["book"] else ""}</span>'
                    f'<span style="color:#e2e8f0;font-size:12px;font-weight:600">{"+" if a > 0 else ""}{a}</span></div>'
                    for t, a in sorted(b["all_prices"].items(), key=lambda kv: -kv[1]))
                st.markdown(
                    '<div style="background:#111127;border-radius:10px;padding:14px">'
                    '<div style="color:#64748b;font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.06em;margin-bottom:6px">Every book\'s price</div>'
                    + rows_html + '</div>', unsafe_allow_html=True)

            lc1, lc2 = st.columns([1, 1])
            log_stake = lc1.number_input("Stake to log ($)", 1.0, 100000.0,
                                         float(max(1.0, round(stake_sug, 2))), 1.0)
            if lc2.button("➕ Log this bet to Bet Tracker", use_container_width=True):
                bets_file = load_bets()
                bets_file.append({
                    "id": int(datetime.datetime.now().timestamp()),
                    "date": datetime.date.today().isoformat(),
                    "sport": b["sport"], "event": b["match"],
                    "pick": b["pick"], "odds": int(b["american"]),
                    "stake": float(log_stake), "book": b["book"],
                    "result": "pending", "payout": 0.0,
                    "market": b.get("market", "ML"),
                    "logged_dec": b["decimal"], "fair_prob": b["fair_prob"],
                    "commence": b.get("commence", ""),
                })
                save_bets(bets_file)
                st.success(f"Logged: {b['pick']} ({b['american']:+d} @ {b['book']}) — ${log_stake:.2f}. Track it in 📋 Bet Tracker.")
    else:
        st.markdown('<div style="color:#94a3b8;font-size:13px;padding:18px;background:#1a1a2e;border-radius:10px;border:1px solid #1e1e3a">No bets clear your EV threshold right now — with the 8-book consensus, real edges are rarer but trustworthy. Lower the "Min EV" slider or check back; lines move all day.</div>', unsafe_allow_html=True)

    # ── BEST PARLAYS ────────────────────────────────────────────────────────
    st.markdown('<div style="height:22px"></div>', unsafe_allow_html=True)
    leg_note = f" · legs ≥ {min_leg_prob}% to hit" if min_leg_prob > 0 else ""
    st.markdown(f'<div style="color:#64748b;font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.08em;margin-bottom:12px">🎰 Best parlays — by probability & EV{leg_note}</div>', unsafe_allow_html=True)

    if parlays:
        pcols = st.columns(len(parlays))
        for col, p in zip(pcols, parlays):
            with col:
                ev_color = "#22c55e" if p["ev"] > 0 else "#ef4444"
                ev_sign = "+" if p["ev"] >= 0 else ""
                legs_html = ""
                for l in p["legs"]:
                    am = f"+{l['american']}" if l['american'] > 0 else str(l['american'])
                    legs_html += (
                        '<div style="display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid #111127">'
                        f'<span style="color:#e2e8f0;font-size:12px">{SPORT_TAGS.get(l["sport"],"")} {l["pick"][:18]}</span>'
                        f'<span style="color:#94a3b8;font-size:12px">{am} · {l["book"][:2].upper()}</span></div>'
                    )
                st.markdown(
                    '<div class="best-play-card">'
                    f'<div class="best-play-label">{p["label"]}</div>'
                    f'<div class="best-play-sub">{p["note"]}</div>'
                    '<div style="display:flex;justify-content:space-between;margin:10px 0 4px">'
                    f'<div><div style="color:#64748b;font-size:10px;text-transform:uppercase">Hit prob</div><div style="color:#e2e8f0;font-size:20px;font-weight:700">{p["combined_prob"]*100:.1f}%</div></div>'
                    f'<div style="text-align:right"><div style="color:#64748b;font-size:10px;text-transform:uppercase">Payout</div><div style="color:#f97316;font-size:20px;font-weight:700">{p["combined_decimal"]:.1f}x</div></div>'
                    '</div>'
                    f'<div style="height:3px;border-radius:2px;background:linear-gradient(90deg,#f97316,#7c3aed);margin:6px 0 10px"></div>'
                    f'{legs_html}'
                    '<div style="display:flex;justify-content:space-between;margin-top:10px">'
                    f'<span style="color:#64748b;font-size:11px">{p["combined_american"]} · {len(p["legs"])} legs</span>'
                    f'<span style="color:{ev_color};font-size:13px;font-weight:700">EV {ev_sign}${p["ev_per_100"]:.1f}/$100</span>'
                    '</div></div>',
                    unsafe_allow_html=True)
        st.caption("Hit prob assumes independent legs (de-vigged fair odds). Payout = $1 → $X. A +EV parlay needs every leg +EV; the 'Most Likely' parlay favors win-rate over EV.")
    else:
        msg = "Not enough qualifying legs to build a parlay right now."
        if min_leg_prob > 0:
            msg += f" Your Leg-Probability filter ({min_leg_prob}%) may be too high — lower it in the sidebar."
        st.info(msg)

    # ── PER-SPORT EV TABLES ─────────────────────────────────────────────────
    st.markdown('<div style="height:22px"></div>', unsafe_allow_html=True)
    st.markdown('<div style="color:#64748b;font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.08em;margin-bottom:12px">📊 Full EV board — every game, both sides</div>', unsafe_allow_html=True)

    tabs = st.tabs([f"{SPORT_TAGS.get(s,'')} {s}" for s in SPORTS])
    for tab, sport in zip(tabs, SPORTS):
        with tab:
            bets = sorted(data.get(sport, []), key=lambda x: (x["date"], x["time"], -x["ev"]))
            if not bets:
                st.markdown(f'<div style="color:#64748b;font-size:13px;padding:16px;background:#1a1a2e;border-radius:10px;border:1px solid #1e1e3a">No {sport} games today or tomorrow with a multi-book consensus.</div>', unsafe_allow_html=True)
                continue
            head = (
                '<tr style="border-bottom:1px solid #1e1e3a;color:#64748b">'
                '<th style="text-align:left;padding:6px 8px 6px 0;font-weight:500">MATCH</th>'
                '<th style="text-align:left;padding:6px 8px;font-weight:500">PICK</th>'
                '<th style="text-align:center;padding:6px 8px;font-weight:500">MKT</th>'
                '<th style="text-align:center;padding:6px 8px;font-weight:500">BOOK</th>'
                '<th style="text-align:right;padding:6px 8px;font-weight:500">ODDS</th>'
                '<th style="text-align:right;padding:6px 8px;font-weight:500">FAIR%</th>'
                '<th style="text-align:right;padding:6px 8px;font-weight:500">#BKS</th>'
                '<th style="text-align:right;padding:6px 8px;font-weight:500">EV/$100</th>'
                '<th style="text-align:center;padding:6px 8px;font-weight:500">VERDICT</th>'
                '<th style="text-align:right;padding:6px 0;font-weight:500">KICKOFF</th></tr>'
            )
            body = ""
            shown = bets[:80]
            for b in shown:
                am = f"+{b['american']}" if b['american'] > 0 else str(b['american'])
                pos = b["ev"] > 0
                ev_color = "#22c55e" if pos else "#64748b"
                verdict = ('<span style="background:#22c55e22;color:#22c55e;padding:2px 8px;border-radius:4px;font-size:11px;font-weight:600">BET</span>'
                           if pos else '<span style="color:#475569;font-size:11px">pass</span>')
                book_color = "#1493ff" if "Draft" in b['book'] else "#1a9c4c"
                body += (
                    '<tr style="border-bottom:1px solid #111127">'
                    f'<td style="color:#64748b;padding:7px 8px 7px 0;font-size:11px">{b["match"][:26]}</td>'
                    f'<td style="color:#e2e8f0;padding:7px 8px;font-weight:600">{b["pick"][:20]}</td>'
                    f'<td style="text-align:center;padding:7px 8px"><span style="background:#7c3aed22;color:#a78bfa;padding:2px 6px;border-radius:4px;font-size:10px;font-weight:600">{b.get("market","ML")}</span></td>'
                    f'<td style="text-align:center;padding:7px 8px"><span style="color:{book_color};font-size:11px;font-weight:600">{b["book"]}</span></td>'
                    f'<td style="color:#f97316;text-align:right;padding:7px 8px;font-weight:700">{am}</td>'
                    f'<td style="color:#94a3b8;text-align:right;padding:7px 8px">{b["fair_prob"]*100:.1f}%</td>'
                    f'<td style="color:#64748b;text-align:right;padding:7px 8px">{b.get("n_books","")}</td>'
                    f'<td style="color:{ev_color};text-align:right;padding:7px 8px;font-weight:700">{"+" if pos else ""}${b["ev_per_100"]:.2f}</td>'
                    f'<td style="text-align:center;padding:7px 8px">{verdict}</td>'
                    f'<td style="color:#64748b;text-align:right;padding:7px 0;font-size:11px">{b["date"][5:]} {b["time"][:5]}</td></tr>'
                )
            n_bet = sum(1 for b in bets if b["ev"] > 0)
            st.markdown(
                '<div style="background:#1a1a2e;border:1px solid #1e1e3a;border-radius:12px;padding:16px;overflow-x:auto">'
                '<table style="width:100%;border-collapse:collapse;font-size:12px">' + head + body + '</table></div>',
                unsafe_allow_html=True)
            extra = f" (showing first {len(shown)})" if len(bets) > len(shown) else ""
            st.caption(f"{len(bets)} priced outcomes{extra} · {n_bet} rated BET (+EV) · #BKS = books in the consensus for that line.")




elif page == "🎯 Side Bets":
    from value_betting import (fetch_events, side_bets_for_event, kelly_stake,
                               SIDE_MARKETS, SPORTS, SPORT_TAGS, CONSENSUS_BOOKS)
    from live_data import get_event_odds

    sb_today = datetime.date.today()
    sb_book_choice = st.sidebar.radio("Sportsbook to bet at",
                                      ["Best of both", "FanDuel only", "DraftKings only"],
                                      key="sb_book")
    SB_BOOKS = {"Best of both": None, "FanDuel only": ["fanduel"], "DraftKings only": ["draftkings"]}[sb_book_choice]
    sb_bankroll = st.sidebar.number_input("Bankroll ($)", 50.0, 1_000_000.0, 1000.0, 50.0, key="sb_bank")

    _q = get_quota()
    if _q:
        st.sidebar.caption(f"🔋 API credits left: {_q['remaining']:.0f} · each fetch below costs ≈1 credit per market")

    st.markdown(f"""
    <div class="topbar">
        <div class="topbar-left">
            <div style="width:10px;height:10px;border-radius:50%;background:#f97316"></div>
            <span class="sport-badge">SIDE BETS</span>
            <span class="topbar-sub">· GOALS · SCORERS · CORNERS · PROPS · PER-GAME</span>
        </div>
        <div class="topbar-date">{sb_today.strftime('%a %b %-d')}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="color:#e2e8f0;font-size:24px;font-weight:700;margin-bottom:2px">Side bets &amp; props</div>', unsafe_allow_html=True)
    st.markdown('<div style="color:#94a3b8;font-size:14px;margin-bottom:14px">Pick a game, choose the prop markets you care about, and fetch. Where the market can be de-vigged (goals O/U, corners, BTTS, handicaps) you get a real <b>Fair % and EV</b>. Where it can\'t (goalscorers — many players can score), you get the <b>best price vs the median book</b> instead: pure line-shopping, not an EV claim.</div>', unsafe_allow_html=True)

    @st.cache_data(ttl=300, show_spinner="Loading games…")
    def _sb_events(_d):
        return fetch_events(days=2)

    raw = _sb_events(f"{sb_today}-v2")
    sb_sport = st.selectbox("Sport", list(SIDE_MARKETS.keys()),
                            format_func=lambda s: f"{SPORT_TAGS.get(s,'')} {s}")
    events = raw.get(sb_sport, [])

    if sb_sport in ("ATP Wimbledon", "WTA Wimbledon"):
        st.markdown('<div style="background:#f59e0b14;border:1px solid #f59e0b44;border-radius:10px;padding:10px 14px;margin-bottom:10px;color:#fbbf24;font-size:12px">ℹ️ Heads-up: The Odds API has <b>no set-winner market</b> for tennis — only alternate game handicaps and total games, and FanDuel/DraftKings price those sparsely. Soccer prop coverage is much deeper.</div>', unsafe_allow_html=True)

    if not events:
        st.info(f"No {sb_sport} games today or tomorrow.")
    else:
        ev_opts = {f"{e['away_team']} @ {e['home_team']}  ·  {e.get('commence_time','')[:10]} {e.get('commence_time','')[11:16]} UTC": e
                   for e in sorted(events, key=lambda x: x.get("commence_time", ""))}
        sel_game = st.selectbox("Game", list(ev_opts.keys()))
        e0 = ev_opts[sel_game]

        labels = SIDE_MARKETS[sb_sport]
        label_list = list(labels.values())
        chosen_labels = st.multiselect("Prop markets", label_list, default=label_list[:6])
        chosen_keys = [k for k, v in labels.items() if v in chosen_labels]

        fetch_col, _sp = st.columns([1, 2])
        if fetch_col.button(f"⚡ Fetch side bets (≈{max(1,len(chosen_keys))} API credits)", use_container_width=True):
            st.session_state["sb_sel"] = (SPORTS[sb_sport], e0["id"], tuple(sorted(chosen_keys)), sb_sport)

        sel = st.session_state.get("sb_sel")
        if sel and sel[1] == e0["id"] and set(sel[2]) == set(chosen_keys):
            skey, eid, mkeys, slabel = sel

            @st.cache_data(ttl=300, show_spinner="Fetching prop odds…")
            def _sb_fetch(_skey, _eid, _mkeys):
                return get_event_odds(_skey, _eid, ",".join(_mkeys), books=CONSENSUS_BOOKS)

            ev_data = _sb_fetch(skey, eid, mkeys)
            rows = side_bets_for_event(ev_data, slabel, SB_BOOKS)

            if not rows:
                st.markdown('<div style="color:#94a3b8;font-size:13px;padding:16px;background:#1a1a2e;border-radius:10px;border:1px solid #1e1e3a">FanDuel/DraftKings aren\'t pricing these prop markets for this game (common far from kickoff, and normal for tennis). Try closer to game time or another game.</div>', unsafe_allow_html=True)
            else:
                fair_rows = [r for r in rows if r["mode"] == "fair"]
                shop_rows = [r for r in rows if r["mode"] == "shop"]
                pos = [r for r in fair_rows if r["ev"] > 0]

                st.markdown(
                    '<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:10px 0 16px">'
                    f'<div class="stat-cell"><div class="stat-cell-label">Prop outcomes</div><div class="stat-cell-val">{len(rows)}</div></div>'
                    f'<div class="stat-cell"><div class="stat-cell-label">De-viggable (real EV)</div><div class="stat-cell-val">{len(fair_rows)}</div></div>'
                    f'<div class="stat-cell"><div class="stat-cell-label">+EV props</div><div class="stat-cell-val" style="color:{"#22c55e" if pos else "#94a3b8"}">{len(pos)}</div></div>'
                    f'<div class="stat-cell"><div class="stat-cell-label">Price-shop rows</div><div class="stat-cell-val">{len(shop_rows)}</div></div>'
                    '</div>', unsafe_allow_html=True)

                tab_fair, tab_shop = st.tabs(["✅ De-vigged — real EV", "🛒 Scorers & props — best price"])

                with tab_fair:
                    if not fair_rows:
                        st.info("No de-viggable markets returned for this game.")
                    else:
                        head = ('<tr style="border-bottom:1px solid #1e1e3a;color:#64748b">'
                                '<th style="text-align:left;padding:6px 8px 6px 0;font-weight:500">MARKET</th>'
                                '<th style="text-align:left;padding:6px 8px;font-weight:500">PICK</th>'
                                '<th style="text-align:center;padding:6px 8px;font-weight:500">BOOK</th>'
                                '<th style="text-align:right;padding:6px 8px;font-weight:500">ODDS</th>'
                                '<th style="text-align:right;padding:6px 8px;font-weight:500">FAIR%</th>'
                                '<th style="text-align:right;padding:6px 8px;font-weight:500">#BKS</th>'
                                '<th style="text-align:right;padding:6px 8px;font-weight:500">EV/$100</th>'
                                '<th style="text-align:right;padding:6px 8px;font-weight:500">½-KELLY</th>'
                                '<th style="text-align:center;padding:6px 0;font-weight:500">VERDICT</th></tr>')
                        body = ""
                        for r in fair_rows[:60]:
                            am = f"+{r['american']}" if r['american'] > 0 else str(r['american'])
                            posr = r["ev"] > 0
                            evc = "#22c55e" if posr else "#64748b"
                            verdict = ('<span style="background:#22c55e22;color:#22c55e;padding:2px 8px;border-radius:4px;font-size:11px;font-weight:600">BET</span>'
                                       if posr else '<span style="color:#475569;font-size:11px">pass</span>')
                            bc = "#1493ff" if "Draft" in r["book"] else "#1a9c4c"
                            kst = kelly_stake(r["fair_prob"], r["decimal"], sb_bankroll)
                            body += ('<tr style="border-bottom:1px solid #111127">'
                                     f'<td style="padding:7px 8px 7px 0"><span style="background:#7c3aed22;color:#a78bfa;padding:2px 6px;border-radius:4px;font-size:10px;font-weight:600">{r["market"]}</span></td>'
                                     f'<td style="color:#e2e8f0;padding:7px 8px;font-weight:600">{r["pick"][:26]}</td>'
                                     f'<td style="text-align:center;padding:7px 8px"><span style="color:{bc};font-size:11px;font-weight:600">{r["book"]}</span></td>'
                                     f'<td style="color:#f97316;text-align:right;padding:7px 8px;font-weight:700">{am}</td>'
                                     f'<td style="color:#94a3b8;text-align:right;padding:7px 8px">{r["fair_prob"]*100:.1f}%</td>'
                                     f'<td style="color:#64748b;text-align:right;padding:7px 8px">{r["n_books"]}</td>'
                                     f'<td style="color:{evc};text-align:right;padding:7px 8px;font-weight:700">{"+" if posr else ""}${r["ev_per_100"]:.2f}</td>'
                                     f'<td style="color:#e2e8f0;text-align:right;padding:7px 8px">${kst:.2f}</td>'
                                     f'<td style="text-align:center;padding:7px 0">{verdict}</td></tr>')
                        st.markdown('<div style="background:#1a1a2e;border:1px solid #1e1e3a;border-radius:12px;padding:16px;overflow-x:auto">'
                                    '<table style="width:100%;border-collapse:collapse;font-size:12px">' + head + body + '</table></div>',
                                    unsafe_allow_html=True)
                        st.caption(f"Sorted by EV. #BKS = books in the consensus (1 = single book de-vig — trust it less). Showing {min(len(fair_rows),60)} of {len(fair_rows)}.")

                with tab_shop:
                    if not shop_rows:
                        st.info("No scorer/prop rows returned for this game.")
                    else:
                        head = ('<tr style="border-bottom:1px solid #1e1e3a;color:#64748b">'
                                '<th style="text-align:left;padding:6px 8px 6px 0;font-weight:500">MARKET</th>'
                                '<th style="text-align:left;padding:6px 8px;font-weight:500">PICK</th>'
                                '<th style="text-align:center;padding:6px 8px;font-weight:500">BEST BOOK</th>'
                                '<th style="text-align:right;padding:6px 8px;font-weight:500">ODDS</th>'
                                '<th style="text-align:right;padding:6px 8px;font-weight:500">#BKS</th>'
                                '<th style="text-align:right;padding:6px 8px;font-weight:500">VS MEDIAN</th>'
                                '<th style="text-align:center;padding:6px 0;font-weight:500">CALL</th></tr>')
                        body = ""
                        for r in shop_rows[:60]:
                            am = f"+{r['american']}" if r['american'] > 0 else str(r['american'])
                            good = r["shop_edge"] >= 5 and r["n_books"] >= 2
                            ec = "#22c55e" if good else "#64748b"
                            call = ('<span style="background:#22c55e22;color:#22c55e;padding:2px 8px;border-radius:4px;font-size:11px;font-weight:600">BEST PRICE</span>'
                                    if good else '<span style="color:#475569;font-size:11px">—</span>')
                            bc = "#1493ff" if "Draft" in r["book"] else "#1a9c4c"
                            body += ('<tr style="border-bottom:1px solid #111127">'
                                     f'<td style="padding:7px 8px 7px 0"><span style="background:#f9731622;color:#fb923c;padding:2px 6px;border-radius:4px;font-size:10px;font-weight:600">{r["market"]}</span></td>'
                                     f'<td style="color:#e2e8f0;padding:7px 8px;font-weight:600">{r["pick"][:26]}</td>'
                                     f'<td style="text-align:center;padding:7px 8px"><span style="color:{bc};font-size:11px;font-weight:600">{r["book"]}</span></td>'
                                     f'<td style="color:#f97316;text-align:right;padding:7px 8px;font-weight:700">{am}</td>'
                                     f'<td style="color:#64748b;text-align:right;padding:7px 8px">{r["n_books"]}</td>'
                                     f'<td style="color:{ec};text-align:right;padding:7px 8px;font-weight:700">+{r["shop_edge"]:.1f}%</td>'
                                     f'<td style="text-align:center;padding:7px 0">{call}</td></tr>')
                        st.markdown('<div style="background:#1a1a2e;border:1px solid #1e1e3a;border-radius:12px;padding:16px;overflow-x:auto">'
                                    '<table style="width:100%;border-collapse:collapse;font-size:12px">' + head + body + '</table></div>',
                                    unsafe_allow_html=True)
                        st.caption(f"VS MEDIAN = how much better this price is than the median book for the same pick — line shopping, not a fair-value EV. Scorer markets can't be de-vigged (many players can score). Showing {min(len(shop_rows),60)} of {len(shop_rows)}.")

                # ── Log a side bet ──────────────────────────────────────────
                with st.expander("➕ Log one of these side bets"):
                    def _sb_label(r, i):
                        am = f"+{r['american']}" if r['american'] > 0 else str(r['american'])
                        tag = f"EV {r['ev_per_100']:+.1f}" if r["mode"] == "fair" else f"+{r['shop_edge']:.1f}% vs median"
                        return f"{i+1}. {r['pick']} · {r['market']} · {am} @ {r['book']} · {tag}"
                    logopts = {_sb_label(r, i): r for i, r in enumerate(rows[:120])}
                    pick_log = st.selectbox("Side bet", list(logopts.keys()))
                    rsel = logopts[pick_log]
                    sug = kelly_stake(rsel["fair_prob"], rsel["decimal"], sb_bankroll) if rsel["mode"] == "fair" else 0.0
                    lc1, lc2 = st.columns(2)
                    sb_stake = lc1.number_input("Stake ($)", 1.0, 100000.0, float(max(1.0, round(sug, 2))), 1.0, key="sb_stake")
                    if lc2.button("Log to Bet Tracker", use_container_width=True, key="sb_log"):
                        bets_file = load_bets()
                        bets_file.append({
                            "id": int(datetime.datetime.now().timestamp()),
                            "date": datetime.date.today().isoformat(),
                            "sport": rsel["sport"], "event": rsel["match"],
                            "pick": f"{rsel['pick']} ({rsel['market']})",
                            "odds": int(rsel["american"]), "stake": float(sb_stake),
                            "book": rsel["book"], "result": "pending", "payout": 0.0,
                            "market": rsel["market"], "logged_dec": rsel["decimal"],
                            "fair_prob": rsel.get("fair_prob"), "commence": rsel.get("commence", ""),
                        })
                        save_bets(bets_file)
                        st.success(f"Logged: {rsel['pick']} ({rsel['american']:+d} @ {rsel['book']}) — ${sb_stake:.2f}")



# ── ML Model ───────────────────────────────────────────────────────────────
elif page == "📖 How It Works":
    st.markdown('<div style="color:#f97316;font-size:11px;font-weight:700;letter-spacing:.1em;margin-top:6px">THE THREE NUMBERS THAT MATTER</div>', unsafe_allow_html=True)
    st.markdown('<div style="color:#e2e8f0;font-size:30px;font-weight:700;margin-bottom:4px">Fair %, EV &amp; Stake</div>', unsafe_allow_html=True)
    st.markdown('<div style="color:#94a3b8;font-size:14px;max-width:760px;margin-bottom:18px">Every recommended bet on the dashboard shows these three numbers. Here is exactly what each one means, the math behind it, and how to act on it. We use one running example the whole way through so you can see how they connect.</div>', unsafe_allow_html=True)

    # Running example callout
    st.markdown("""
    <div style="background:linear-gradient(135deg,#1a1a2e,#16162a);border:1px solid #2a2a4a;border-radius:12px;padding:18px 22px;margin-bottom:22px">
        <div style="color:#f97316;font-size:10px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;margin-bottom:8px">📌 Running example</div>
        <div style="color:#e2e8f0;font-size:15px;line-height:1.6">
        A tennis match. After removing the bookmaker's margin, our model thinks it's a true <b>coin flip — 50/50</b>.
        But <b>FanDuel</b> is pricing your player at <b>+120</b> (decimal <b>2.20</b>).
        Your bankroll is <b>$1,000</b>. Should you bet, and how much? The three numbers answer that.
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        <div style="background:#1a1a2e;border:1px solid #1e1e3a;border-radius:12px;padding:18px;height:100%">
            <div style="color:#7c3aed;font-size:11px;font-weight:700;letter-spacing:.08em;text-transform:uppercase">① Fair %</div>
            <div style="color:#e2e8f0;font-size:18px;font-weight:700;margin:6px 0 10px">The real chance of winning</div>
            <div style="color:#94a3b8;font-size:13px;line-height:1.6;margin-bottom:12px">
            Bookmakers bake a profit margin (the <i>"vig"</i>) into their odds, so their numbers add up to more than 100%.
            <b>Fair %</b> strips that margin with the <b>power method</b> (which corrects the bias books have on longshots),
            then takes the <b>median across up to 8 US books</b> — so it's the market's consensus, not one book's opinion.
            </div>
            <div style="background:#0d0d1a;border-radius:8px;padding:10px 12px;font-family:monospace;font-size:12px;color:#cbd5e1;line-height:1.7">
            implied = 1 / decimal&nbsp;odds<br>
            fair&nbsp;% = implied / (sum of all<br>
            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;outcomes' implied)
            </div>
            <div style="color:#64748b;font-size:12px;line-height:1.6;margin-top:10px">
            <b style="color:#94a3b8">Example:</b> Both sides priced ≈ +120/−140 imply 45.5% + 58.3% = <b style="color:#fb923c">103.8%</b> (the 3.8% is the vig). Normalize → your player's fair chance is <b style="color:#e2e8f0">~50%</b>.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div style="background:#1a1a2e;border:1px solid #1e1e3a;border-radius:12px;padding:18px;height:100%">
            <div style="color:#22c55e;font-size:11px;font-weight:700;letter-spacing:.08em;text-transform:uppercase">② EV (Expected Value)</div>
            <div style="color:#e2e8f0;font-size:18px;font-weight:700;margin:6px 0 10px">Profit per $100, long-run</div>
            <div style="color:#94a3b8;font-size:13px;line-height:1.6;margin-bottom:12px">
            If you made this <b>same</b> bet thousands of times, EV is your average win (or loss) per $100.
            <b style="color:#22c55e">Positive EV</b> = the price pays more than the true odds deserve → profitable over time.
            <b style="color:#ef4444">Negative</b> = the house edge wins. <b>Only bet +EV.</b>
            </div>
            <div style="background:#0d0d1a;border-radius:8px;padding:10px 12px;font-family:monospace;font-size:12px;color:#cbd5e1;line-height:1.7">
            EV&nbsp;per&nbsp;$1 = fair% × decimal − 1<br>
            EV/$100 = EV per $1 × 100
            </div>
            <div style="color:#64748b;font-size:12px;line-height:1.6;margin-top:10px">
            <b style="color:#94a3b8">Example:</b> 0.50 × 2.20 − 1 = <b style="color:#22c55e">+0.10</b> → <b style="color:#22c55e">+$10 per $100</b>.<br>
            You win $120 half the time, lose $100 half the time → average <b style="color:#e2e8f0">+$10</b>. Edge confirmed.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div style="background:#1a1a2e;border:1px solid #1e1e3a;border-radius:12px;padding:18px;height:100%">
            <div style="color:#f97316;font-size:11px;font-weight:700;letter-spacing:.08em;text-transform:uppercase">③ Stake</div>
            <div style="color:#e2e8f0;font-size:18px;font-weight:700;margin:6px 0 10px">How much to actually bet</div>
            <div style="color:#94a3b8;font-size:13px;line-height:1.6;margin-bottom:12px">
            The <b>Kelly Criterion</b> sizes each bet to your edge and bankroll — bet more when the edge is big, less when it's thin.
            We use <b>½-Kelly</b> (half the formula) because full Kelly swings hard; half keeps ~75% of the growth with far less risk of going broke.
            </div>
            <div style="background:#0d0d1a;border-radius:8px;padding:10px 12px;font-family:monospace;font-size:12px;color:#cbd5e1;line-height:1.7">
            b = decimal − 1<br>
            kelly = (b × p − (1−p)) / b<br>
            stake = bankroll × kelly × 0.5
            </div>
            <div style="color:#64748b;font-size:12px;line-height:1.6;margin-top:10px">
            <b style="color:#94a3b8">Example:</b> b=1.2, p=0.50 → kelly = (1.2×.5 − .5)/1.2 = <b>8.3%</b>. Half = 4.2%.<br>
            Stake = $1,000 × 0.042 = <b style="color:#e2e8f0">$41.67</b>.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # How they connect
    st.markdown('<div style="height:18px"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="background:#1a1a2e;border:1px solid #1e1e3a;border-radius:12px;padding:18px 22px">
        <div style="color:#64748b;font-size:10px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;margin-bottom:10px">How they connect</div>
        <div style="color:#cbd5e1;font-size:14px;line-height:1.8">
        <b style="color:#7c3aed">Fair %</b> tells you the <i>truth</i> about the matchup.
        Compare it to the <b style="color:#f97316">price</b> on offer — when the price pays more than the truth deserves, that gap is your <b style="color:#22c55e">EV</b>.
        <b style="color:#f97316">Stake</b> then turns that edge into a dollar amount sized to your bankroll. In short:
        <span style="color:#94a3b8">find the truth → measure the edge → size the bet.</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # How to use the board
    st.markdown('<div style="height:18px"></div>', unsafe_allow_html=True)
    st.markdown('<div style="color:#64748b;font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.08em;margin-bottom:12px">📋 How to use the dashboard</div>', unsafe_allow_html=True)
    steps = [
        ("1", "Set your bankroll", "In the sidebar on the <b>Daily Intelligence</b> page, enter your real bankroll. Every stake scales to it."),
        ("2", "Read the “What to bet” table", "Each row is a <b>+EV</b> play: the pick, which book (FanDuel or DraftKings) has the best price, the odds, the Fair %, the EV per $100, and your ½-Kelly stake."),
        ("3", "Place the bet at the listed book", "The edge only exists at that book's price. Bet the exact stake shown — no more chasing, no less from fear."),
        ("4", "Use the Min-EV slider", "Raise it to see only the strongest edges. If the table is empty, FanDuel and DraftKings agree and there's no value right now — that's normal."),
        ("5", "Consider a parlay (optional)", "The parlay cards combine legs. <b>Most Likely to Hit</b> = favorites (often slightly −EV, shown in red). <b>Balanced / Max Value</b> = genuinely +EV. Higher payout = lower hit chance."),
        ("6", "Log it in Bet Tracker", "Record what you placed so you can track real P&amp;L over time."),
    ]
    for num, title, desc in steps:
        st.markdown(
            '<div style="display:flex;gap:14px;align-items:flex-start;background:#111127;border-radius:10px;padding:12px 16px;margin-bottom:8px">'
            f'<div style="background:#f97316;color:#0d0d1a;width:24px;height:24px;border-radius:6px;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:13px;flex-shrink:0">{num}</div>'
            f'<div><div style="color:#e2e8f0;font-size:14px;font-weight:600">{title}</div><div style="color:#94a3b8;font-size:13px;line-height:1.5;margin-top:2px">{desc}</div></div>'
            '</div>',
            unsafe_allow_html=True)

    # Caveat
    st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="background:#1a1a14;border:1px solid #3a2a14;border-radius:10px;padding:14px 18px">
        <div style="color:#fbbf24;font-size:13px;line-height:1.7">
        ⚠️ <b>+EV is a long-run edge, not a guarantee.</b> A +EV bet still loses often — a 50% bet loses half the time.
        The math only pays off across <b>many</b> bets, which is why disciplined staking matters. Parlay hit-rates assume the legs are independent.
        Bet only what you can afford to lose.
        </div>
    </div>
    """, unsafe_allow_html=True)


elif page == "🤖 ML Model":
    st.title("ML Betting Model")
    st.caption("Gradient Boosting · 6-season training · Probability calibration · Premier League")

    df = load_data()
    df_feat = build_features(df)

    models_dict = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=2, random_state=42),
    }

    st.subheader("Model accuracy (5-fold CV · 2,280 matches)")
    acc_data = []
    for name, m in models_dict.items():
        s = cross_val_score(m, df_feat[FEATURE_COLS], df_feat['result'], cv=5)
        acc_data.append({'Model': name, 'Accuracy': round(s.mean()*100,1), 'Std': round(s.std()*100,1)})
    acc_df = pd.DataFrame(acc_data)
    fig = px.bar(acc_df, x='Model', y='Accuracy', error_y='Std',
                 color='Accuracy', color_continuous_scale=[[0,'#1e1e3a'],[1,'#f97316']],
                 range_y=[40,60])
    fig.update_layout(height=280, margin=dict(t=10,b=10), coloraxis_showscale=False,
                      plot_bgcolor='#0d0d1a', paper_bgcolor='#0d0d1a',
                      font_color='#94a3b8', xaxis=dict(gridcolor='#1e1e3a'),
                      yaxis=dict(gridcolor='#1e1e3a'))
    st.plotly_chart(fig, use_container_width=True)

    threshold = st.slider("Value bet edge threshold", 0.01, 0.20, 0.05, 0.01, format="%0.2f")
    stake = st.number_input("Flat stake per bet ($)", 1.0, 500.0, 10.0, 5.0)

    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=2, random_state=42)
    (bets, flat_staked, flat_return, flat_roi,
     kelly_staked, kelly_return, kelly_roi,
     kelly_curve, trained_model, classes, fcols, test_df, probs) = backtest(df_feat, model, flat_stake=stake, threshold=threshold)

    wins = sum(1 for b in bets if b['win'])
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Value bets", len(bets))
    c2.metric("Win rate", f"{wins/len(bets)*100:.1f}%" if bets else "0%")
    c3.metric("Flat ROI", f"{flat_roi:+.1f}%")
    c4.metric("Flat profit", f"${flat_return-flat_staked:+.2f}")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Calibration")
        if bets:
            bdf = pd.DataFrame(bets)
            bdf['prob_bucket'] = pd.cut(bdf['model_prob'], bins=[0,.3,.4,.5,.6,.7,1.0],
                                         labels=['0-30%','30-40%','40-50%','50-60%','60-70%','70-100%'])
            cal = bdf.groupby('prob_bucket', observed=True)['win'].mean().reset_index()
            cal.columns=['Model prob range','Actual win rate']
            cal['Actual win rate'] *= 100
            fig3 = px.bar(cal, x='Model prob range', y='Actual win rate',
                          color='Actual win rate', color_continuous_scale=[[0,'#ef4444'],[0.5,'#f97316'],[1,'#22c55e']],
                          range_y=[0,100])
            fig3.add_hline(y=33, line_dash="dot", line_color="#64748b", annotation_text="Random (33%)")
            fig3.update_layout(height=280, margin=dict(t=10,b=10), coloraxis_showscale=False,
                               plot_bgcolor='#0d0d1a', paper_bgcolor='#0d0d1a',
                               font_color='#94a3b8', xaxis=dict(gridcolor='#1e1e3a'), yaxis=dict(gridcolor='#1e1e3a'))
            st.plotly_chart(fig3, use_container_width=True)

    with col2:
        st.subheader("Bankroll curve (Kelly)")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(y=kelly_curve, mode='lines', name='Bankroll',
                                   line=dict(color='#f97316', width=2)))
        fig2.add_hline(y=1000, line_dash="dot", line_color="#64748b", annotation_text="Start $1,000")
        fig2.update_layout(height=280, margin=dict(t=10,b=10),
                           plot_bgcolor='#0d0d1a', paper_bgcolor='#0d0d1a',
                           font_color='#94a3b8', xaxis=dict(gridcolor='#1e1e3a'), yaxis=dict(gridcolor='#1e1e3a'),
                           yaxis_title="Bankroll ($)", xaxis_title="Bet #")
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Top value bets")
    if bets:
        top = sorted(bets, key=lambda x: x['edge'], reverse=True)[:15]
        bdf = pd.DataFrame(top)[['outcome','edge','model_prob','odds','win','kelly_stake']]
        bdf.columns = ['Outcome','Edge','Model prob','Odds','Won','Kelly stake']
        bdf['Edge'] = bdf['Edge'].apply(lambda x: f"+{x*100:.1f}%")
        bdf['Model prob'] = bdf['Model prob'].apply(lambda x: f"{x*100:.1f}%")
        bdf['Won'] = bdf['Won'].apply(lambda x: "✓" if x else "✗")
        bdf['Kelly stake'] = bdf['Kelly stake'].apply(lambda x: f"${x:.2f}")
        st.dataframe(bdf, use_container_width=True, hide_index=True)


# ── Live Odds ──────────────────────────────────────────────────────────────
elif page == "📡 Live Odds":
    st.title("Live Odds")
    sport = st.selectbox("Sport", ["nfl","nba","mlb","nhl","epl"], format_func=str.upper)
    col1,col2 = st.columns(2)
    with col1:
        st.subheader("Live scores")
        games = get_espn_scores(sport)
        if games:
            for g in games:
                teams=g['teams']; away=next((t for t in teams if not t['home']),{})
                home=next((t for t in teams if t['home']),{})
                w="✓" if home.get('winner') else ""
                st.markdown(f"**{away.get('name','')} {away.get('score','')} @ {home.get('name','')} {home.get('score','')}** {w} — *{g['status']}*")
        else:
            st.info("No games right now.")
    with col2:
        st.subheader("Best available lines")
        best = get_best_lines(sport)
        if best:
            rows=[{'Matchup':b['matchup'],'Best away':f"{b['best_away']:+d}" if b['best_away'] else 'N/A',
                   'Book':b['best_away_book'],'Best home':f"{b['best_home']:+d}" if b['best_home'] else 'N/A',
                   'Book ':b['best_home_book']} for b in best]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.info("No odds available right now.")
    st.subheader("All book odds")
    for event in get_live_odds(sport):
        home=event.get('home_team',''); away=event.get('away_team','')
        with st.expander(f"{away} @ {home} — {event.get('commence_time','')[:10]}"):
            rows=[]
            for book in event.get('bookmakers',[]):
                markets={m['key']:m for m in book.get('markets',[])}
                h2h=markets.get('h2h',{}); outcomes={o['name']:o['price'] for o in h2h.get('outcomes',[])}
                rows.append({'Book':book['title'],
                             away:f"{outcomes.get(away,'-'):+d}" if isinstance(outcomes.get(away),int) else '-',
                             home:f"{outcomes.get(home,'-'):+d}" if isinstance(outcomes.get(home),int) else '-'})
            if rows: st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ── Premier League ─────────────────────────────────────────────────────────

elif page == "🔢 Odds Calculator":
    st.title("Odds Calculator")
    col1,col2=st.columns(2)
    with col1:
        american=st.number_input("American odds",value=-110,step=5)
        stake=st.number_input("Stake ($)",value=100.0,step=10.0)
        dec=american_to_decimal(int(american)); impl=american_implied_prob(int(american))*100
        pay=payout(stake,int(american)); pro=profit(stake,int(american))
        c1,c2=st.columns(2)
        c1.metric("Decimal odds",f"{dec:.4f}"); c2.metric("Implied prob",f"{impl:.2f}%")
        c1.metric("Payout",f"${pay:.2f}"); c2.metric("Profit",f"${pro:.2f}")
    with col2:
        st.subheader("Arbitrage checker")
        o1=st.number_input("Odds side 1",value=-110,step=5,key="a1"); o2=st.number_input("Odds side 2",value=120,step=5,key="a2")
        imp_sum=american_implied_prob(int(o1))+american_implied_prob(int(o2)); margin=(imp_sum-1)*100
        if imp_sum<1:
            st.success(f"Arbitrage! Margin: {abs(margin):.2f}%")
            total=st.number_input("Total stake ($)",value=200.0,step=10.0)
            for i,o in enumerate([o1,o2]):
                s=total*(american_implied_prob(int(o))/imp_sum)
                st.write(f"Side {i+1} ({int(o):+d}): **${s:.2f}** → **${payout(s,int(o)):.2f}**")
        else:
            st.warning(f"No arbitrage. Book margin: {margin:.2f}%")


# ── Kelly Staking ──────────────────────────────────────────────────────────
elif page == "📊 Kelly Staking":
    st.title("Kelly Criterion")
    st.caption("Size bets optimally based on your edge.")
    col1,col2=st.columns(2)
    with col1:
        american=st.number_input("American odds",value=150,step=5)
        model_prob=st.slider("Your win probability",0.01,0.99,0.55,0.01,format="%.2f")
        bankroll=st.number_input("Bankroll ($)",value=1000.0,step=100.0)
        fraction=st.slider("Kelly fraction (0.5 = half-Kelly)",0.1,1.0,0.5,0.05)
    with col2:
        dec=american_to_decimal(int(american)); b=dec-1; q=1-model_prob
        imp=american_implied_prob(int(american)); edge=model_prob-imp
        kelly=max(0.0,(b*model_prob-q)/b); stake=bankroll*kelly*fraction
        if edge>0:
            st.success(f"Edge: +{edge*100:.2f}%")
            c1,c2=st.columns(2)
            c1.metric("Recommended stake",f"${stake:.2f}"); c2.metric("Potential profit",f"${profit(stake,int(american)):.2f}")
            c1.metric("Full Kelly %",f"{kelly*100:.2f}%"); c2.metric("Half-Kelly %",f"{kelly*fraction*100:.2f}%")
        else:
            st.error(f"No edge ({edge*100:.2f}%) — do not bet.")


# ── Parlay Builder ─────────────────────────────────────────────────────────

elif page == "🎰 Parlay Builder":
    st.title("Parlay Builder")
    st.caption("A parlay is only +EV if every individual leg is +EV.")
    n_legs=st.number_input("Number of legs",2,8,2)
    legs=[]; combined_prob=combined_odds=1.0; all_ev=True; rows=[]
    for i in range(int(n_legs)):
        c1,c2=st.columns(2)
        with c1: american=st.number_input(f"Leg {i+1} odds",value=-110,step=5,key=f"l{i}")
        with c2: prob=st.slider(f"Leg {i+1} prob",0.01,0.99,0.55,0.01,key=f"p{i}")
        dec=american_to_decimal(int(american)); imp=american_implied_prob(int(american))
        edge=prob-imp; ev=prob*(dec-1)-(1-prob)
        if ev<=0: all_ev=False
        combined_prob*=prob; combined_odds*=dec
        rows.append({'Leg':i+1,'Odds':f"{int(american):+d}",'Imp prob':f"{imp*100:.1f}%",'Your prob':f"{prob*100:.1f}%",'Edge':f"{edge*100:+.1f}%",'EV/$1':f"{ev:+.4f}"})
    st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
    combined_ev=combined_prob*(combined_odds-1)-(1-combined_prob)
    c1,c2,c3=st.columns(3)
    c1.metric("Combined prob",f"{combined_prob*100:.3f}%"); c2.metric("Combined odds",f"{combined_odds:.2f}x"); c3.metric("Parlay EV/$1",f"{combined_ev:+.4f}")
    if all_ev and combined_ev>0: st.success("All legs +EV — parlay justified.")
    elif combined_ev>0: st.warning("Parlay +EV overall but some legs are -EV.")
    else: st.error("Parlay is -EV — not recommended.")


# ── Bet Tracker ────────────────────────────────────────────────────────────
elif page == "🧩 Build My Parlay":
    from value_betting import fetch_events, value_bets, kelly_stake, SPORT_TAGS, SPORTS, _american_from_decimal
    from collections import Counter

    bp_today = datetime.date.today()
    bp_tomorrow = bp_today + datetime.timedelta(days=1)
    bp_range = f"{bp_today.strftime('%a %b %-d')} – {bp_tomorrow.strftime('%a %b %-d')}"

    bp_book_choice = st.sidebar.radio("Sportsbook to bet at",
                                      ["Best of both", "FanDuel only", "DraftKings only"],
                                      key="bp_book",
                                      help="Which book you'll place the parlay at. Fair % always uses both books for the truth estimate.")
    BP_BOOKS = {"Best of both": None, "FanDuel only": ["fanduel"], "DraftKings only": ["draftkings"]}[bp_book_choice]
    bp_topbar = {"Best of both": "FANDUEL + DRAFTKINGS", "FanDuel only": "FANDUEL", "DraftKings only": "DRAFTKINGS"}[bp_book_choice]

    st.markdown(f"""
    <div class="topbar">
        <div class="topbar-left">
            <div style="width:10px;height:10px;border-radius:50%;background:#7c3aed"></div>
            <span class="sport-badge">PARLAY LAB</span>
            <span class="topbar-sub">· PICK YOUR LEGS · LIVE {bp_topbar}</span>
        </div>
        <div class="topbar-date">{bp_range}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="color:#e2e8f0;font-size:24px;font-weight:700;margin-bottom:2px">Build your own parlay</div>', unsafe_allow_html=True)
    st.markdown('<div style="color:#94a3b8;font-size:14px;margin-bottom:16px">Pick the games and sides you like from today &amp; tomorrow. The lab combines the de-vigged fair odds and tells you the parlay\'s EV — and whether it\'s worth placing.</div>', unsafe_allow_html=True)

    @st.cache_data(ttl=300, show_spinner="Loading today & tomorrow's games…")
    def _load_events_bp(_d):
        return fetch_events(days=2)

    data = value_bets(_load_events_bp(f"{bp_today}-v2"), min_ev=-1.0, bet_books=BP_BOOKS)
    pool_all = [b for bets in data.values() for b in bets]

    if not pool_all:
        st.info("No games priced on FanDuel + DraftKings right now — check back closer to game time.")
    else:
        sports_present = [s for s in SPORTS if data.get(s)]
        fc1, fc2 = st.columns([1.4, 1])
        chosen_sports = fc1.multiselect("Sports to choose from", sports_present, default=sports_present)
        chosen_markets = fc2.multiselect("Markets", ["ML", "Spread", "Total"], default=["ML", "Spread", "Total"])
        pool = [b for b in pool_all if b["sport"] in chosen_sports
                and b.get("market", "ML") in chosen_markets]
        pool.sort(key=lambda b: (b["sport"], b["date"], b["time"], b["match"]))

        def leg_label(b):
            am = f"+{b['american']}" if b["american"] > 0 else str(b["american"])
            return (f"{SPORT_TAGS.get(b['sport'],'')} {b['match']} → {b['pick']} ({b.get('market','ML')})  "
                    f"[{am} · {b['book']} · fair {b['fair_prob']*100:.0f}% · EV {b['ev_per_100']:+.1f}]  "
                    f"{b['date'][5:]} {b['time'][:5]}")

        labelmap = {leg_label(b): b for b in pool}
        picks = st.multiselect("Add legs (type a team or player to search)", list(labelmap.keys()))
        legs = [labelmap[p] for p in picks]

        cstake, cbank = st.columns(2)
        stake = cstake.number_input("Parlay stake ($)", 1.0, 100000.0, 20.0, 5.0,
                                    help="What you plan to bet — drives the payout below.")
        bankroll = cbank.number_input("Bankroll ($)", 50.0, 1_000_000.0, 1000.0, 50.0,
                                      help="Used to compute the suggested ½-Kelly stake for this parlay.")

        if not legs:
            st.markdown('<div style="color:#64748b;font-size:13px;padding:16px;background:#1a1a2e;border-radius:10px;border:1px solid #1e1e3a">Select one or more legs above to see the combined EV and the verdict.</div>', unsafe_allow_html=True)
        else:
            cp = cd = 1.0
            for l in legs:
                cp *= l["fair_prob"]
                cd *= l["decimal"]
            ev = cp * cd - 1
            payout = stake * cd
            profit = payout - stake
            ev_dollars = stake * ev
            am_combined = _american_from_decimal(cd)
            kelly = kelly_stake(cp, cd, bankroll)   # suggested ½-Kelly stake (0 if no edge)

            # Hard conflict: two legs from the same game AND market (can't both win).
            # Soft warning: same game, different markets (correlated — EV math assumes independence).
            hard_counts = Counter((l["sport"], l["match"], l.get("market", "ML")) for l in legs)
            conflicts = [g for g, c in hard_counts.items() if c > 1]
            game_counts = Counter((l["sport"], l["match"]) for l in legs)
            correlated = [g for g, c in game_counts.items() if c > 1] if not conflicts else []

            # ── Selected legs table ─────────────────────────────────────────
            head = (
                '<tr style="border-bottom:1px solid #1e1e3a;color:#64748b">'
                '<th style="text-align:left;padding:6px 8px 6px 0;font-weight:500">#</th>'
                '<th style="text-align:left;padding:6px 8px;font-weight:500">PICK</th>'
                '<th style="text-align:left;padding:6px 8px;font-weight:500">MATCH</th>'
                '<th style="text-align:center;padding:6px 8px;font-weight:500">BOOK</th>'
                '<th style="text-align:right;padding:6px 8px;font-weight:500">ODDS</th>'
                '<th style="text-align:right;padding:6px 8px;font-weight:500">FAIR%</th>'
                '<th style="text-align:right;padding:6px 8px;font-weight:500">LEG EV</th>'
                '<th style="text-align:right;padding:6px 0;font-weight:500">KICKOFF</th></tr>'
            )
            body = ""
            for i, l in enumerate(legs, 1):
                am = f"+{l['american']}" if l["american"] > 0 else str(l["american"])
                leg_ev_color = "#22c55e" if l["ev"] > 0 else "#64748b"
                book_color = "#1493ff" if "Draft" in l["book"] else "#1a9c4c"
                body += (
                    '<tr style="border-bottom:1px solid #111127">'
                    f'<td style="color:#64748b;padding:7px 8px 7px 0">{i}</td>'
                    f'<td style="color:#e2e8f0;padding:7px 8px;font-weight:600">{SPORT_TAGS.get(l["sport"],"")} {l["pick"][:20]} <span style="background:#7c3aed22;color:#a78bfa;padding:1px 5px;border-radius:3px;font-size:9px;font-weight:600">{l.get("market","ML")}</span></td>'
                    f'<td style="color:#64748b;padding:7px 8px;font-size:11px">{l["match"][:28]}</td>'
                    f'<td style="text-align:center;padding:7px 8px"><span style="color:{book_color};font-size:11px;font-weight:600">{l["book"]}</span></td>'
                    f'<td style="color:#f97316;text-align:right;padding:7px 8px;font-weight:700">{am}</td>'
                    f'<td style="color:#94a3b8;text-align:right;padding:7px 8px">{l["fair_prob"]*100:.1f}%</td>'
                    f'<td style="color:{leg_ev_color};text-align:right;padding:7px 8px">{"+" if l["ev"]>0 else ""}{l["ev_per_100"]:.1f}</td>'
                    f'<td style="color:#64748b;text-align:right;padding:7px 0;font-size:11px">{l["date"][5:]} {l["time"][:5]}</td></tr>'
                )
            st.markdown(
                '<div style="background:#1a1a2e;border:1px solid #1e1e3a;border-radius:12px;padding:16px;overflow-x:auto;margin-top:10px">'
                '<table style="width:100%;border-collapse:collapse;font-size:12px">' + head + body + '</table></div>',
                unsafe_allow_html=True)

            # ── Combined metrics ────────────────────────────────────────────
            ev_color = "#22c55e" if ev > 0 else "#ef4444"
            kelly_color = "#22c55e" if kelly > 0 else "#ef4444"
            st.markdown(
                '<div style="display:grid;grid-template-columns:repeat(6,1fr);gap:10px;margin-top:12px">'
                f'<div class="stat-cell"><div class="stat-cell-label">Legs</div><div class="stat-cell-val">{len(legs)}</div></div>'
                f'<div class="stat-cell"><div class="stat-cell-label">Hit probability</div><div class="stat-cell-val">{cp*100:.2f}%</div></div>'
                f'<div class="stat-cell"><div class="stat-cell-label">Combined odds</div><div class="stat-cell-val orange">{am_combined} ({cd:.2f}x)</div></div>'
                f'<div class="stat-cell"><div class="stat-cell-label">EV / $100</div><div class="stat-cell-val" style="color:{ev_color}">{"+" if ev>=0 else ""}${ev*100:.2f}</div></div>'
                f'<div class="stat-cell"><div class="stat-cell-label">½-Kelly stake</div><div class="stat-cell-val" style="color:{kelly_color}">${kelly:,.2f}</div></div>'
                f'<div class="stat-cell"><div class="stat-cell-label">${stake:.0f} returns</div><div class="stat-cell-val">${payout:,.2f}</div></div>'
                '</div>',
                unsafe_allow_html=True)

            # ── Verdict ─────────────────────────────────────────────────────
            if conflicts:
                vc, icon, vtitle = "#ef4444", "⚠️", "SAME-MARKET CONFLICT"
                vmsg = "You picked two sides of the same market in one game (e.g. both teams' ML, or Over AND Under) — they can't both win, so this isn't a real parlay. Remove one."
            elif ev > 0.03:
                vc, icon, vtitle = "#22c55e", "✅", "WORTH IT — STRONG +EV"
                vmsg = f"+{ev*100:.1f}% expected value. Suggested ½-Kelly stake: <b>${kelly:,.2f}</b> on your ${bankroll:,.0f} bankroll. Hits {cp*100:.1f}% of the time for {cd:.2f}x."
            elif ev > 0:
                vc, icon, vtitle = "#22c55e", "✅", "WORTH IT — SLIGHT +EV"
                vmsg = f"+{ev*100:.1f}% expected value — thin but positive. Suggested ½-Kelly stake: <b>${kelly:,.2f}</b> (small, because the edge is thin). Every leg pulling its weight helps."
            elif ev > -0.03:
                vc, icon, vtitle = "#f59e0b", "⚠️", "MARGINAL — ~BREAK-EVEN"
                vmsg = f"{ev*100:.1f}% EV — the bookmaker's vig almost exactly cancels your edge. Swap a leg for a +EV one to push this positive."
            else:
                vc, icon, vtitle = "#ef4444", "❌", "NOT WORTH IT — NEGATIVE EV"
                vmsg = f"{ev*100:.1f}% EV (≈ ${ev_dollars:.2f} on ${stake:.0f}). The book's margin beats your edge — likely a favorites parlay where the vig compounds. Drop the −EV legs (grey in the table)."

            st.markdown(
                f'<div style="background:{vc}18;border:1px solid {vc}55;border-radius:12px;padding:16px 20px;margin-top:14px">'
                f'<div style="color:{vc};font-size:16px;font-weight:700;margin-bottom:4px">{icon} {vtitle}</div>'
                f'<div style="color:#cbd5e1;font-size:13px;line-height:1.6">{vmsg}</div>'
                '</div>',
                unsafe_allow_html=True)

            if correlated:
                st.markdown(
                    '<div style="background:#f59e0b14;border:1px solid #f59e0b44;border-radius:10px;padding:12px 16px;margin-top:10px">'
                    '<div style="color:#fbbf24;font-size:12px;line-height:1.6">⚠️ Two of your legs come from the <b>same game</b> in different markets '
                    '(e.g. a team\'s ML + the total). They\'re <b>correlated</b>, so the hit% and EV above — which assume independence — are off. '
                    'Books usually make you price this as a same-game parlay with worse odds.</div></div>',
                    unsafe_allow_html=True)

            # Log the whole parlay to the tracker
            if not conflicts and st.button(f"➕ Log this parlay (${stake:.0f} at {am_combined})"):
                bets_file = load_bets()
                bets_file.append({
                    "id": int(datetime.datetime.now().timestamp()),
                    "date": datetime.date.today().isoformat(),
                    "sport": "Parlay",
                    "event": " + ".join(l["pick"] for l in legs)[:90],
                    "pick": f"{len(legs)}-leg parlay",
                    "odds": int(am_combined),
                    "stake": float(stake), "book": bp_book_choice,
                    "result": "pending", "payout": 0.0,
                    "market": "Parlay", "logged_dec": cd, "fair_prob": cp,
                    "commence": min(l.get("commence", "") for l in legs),
                })
                save_bets(bets_file)
                st.success(f"Parlay logged — ${stake:.0f} to win ${payout - stake:,.2f}. Track it in 📋 Bet Tracker.")

            st.caption("Fair % = median power-devig consensus of up to 8 US books. Hit% assumes legs are independent (true across different games). A leg with grey EV is −EV on its own and drags the parlay down.")


elif page == "📋 Bet Tracker":
    st.title("Bet Tracker")
    tab1,tab2=st.tabs(["Log a bet","History & P&L"])
    with tab1:
        with st.form("log_bet"):
            c1,c2=st.columns(2)
            sport=c1.text_input("Sport",placeholder="NFL")
            event=c2.text_input("Event",placeholder="Chiefs vs Eagles")
            pick=c1.text_input("Pick",placeholder="Chiefs -3.5")
            odds=c2.number_input("Odds (American)",value=-110,step=5)
            stake=c1.number_input("Stake ($)",value=50.0,step=10.0)
            book=c2.text_input("Sportsbook",placeholder="FanDuel")
            if st.form_submit_button("Log bet"):
                bets=load_bets()
                bets.append({"id":int(datetime.datetime.now().timestamp()),"date":datetime.date.today().isoformat(),
                              "sport":sport,"event":event,"pick":pick,"odds":int(odds),"stake":stake,
                              "book":book,"result":"pending","payout":0.0})
                save_bets(bets); st.success(f"Logged! Potential profit: ${profit(stake,int(odds)):.2f}")
    with tab2:
        from value_betting import closing_price
        bets=load_bets()

        def bet_clv(b):
            """Closing Line Value % vs the last pre-kickoff price. None if ungraded."""
            if not b.get("logged_dec") or b.get("market") in (None, "Parlay"):
                return None
            close = closing_price(b.get("sport",""), b.get("event",""),
                                  b.get("market","ML"), b.get("pick",""))
            if not close or not close.get("decimal"):
                return None
            return (b["logged_dec"] / close["decimal"] - 1) * 100

        if not bets:
            st.info("No bets logged yet.")
        else:
            pending=[b for b in bets if b['result']=='pending']
            if pending:
                st.subheader("Settle pending")
                for b in pending:
                    c1,c2,c3=st.columns([3,1,1])
                    c1.write(f"**{b['event']}** — {b['pick']} ({b['odds']:+d}) ${b['stake']:.2f}")
                    result=c2.selectbox("Result",["pending","win","loss","push"],key=f"r{b['id']}")
                    if c3.button("Save",key=f"s{b['id']}") and result!="pending":
                        for bet in bets:
                            if bet['id']==b['id']:
                                bet['result']=result
                                bet['payout']=payout(bet['stake'],bet['odds']) if result=='win' else (bet['stake'] if result=='push' else 0.0)
                        save_bets(bets); st.rerun()
            settled=[b for b in bets if b['result']!='pending']
            if settled:
                def clv_str(b):
                    v = bet_clv(b)
                    return f"{v:+.1f}%" if v is not None else "—"
                rows=[{'Date':b['date'],'Event':b['event'],'Pick':b['pick'],'Odds':f"{b['odds']:+d}",
                       'Stake':f"${b['stake']:.2f}",'Result':b['result'],
                       'P&L':f"${b['payout']-b['stake']:+.2f}",'CLV':clv_str(b)} for b in settled]
                st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
                total_staked=sum(b['stake'] for b in settled); total_pnl=sum(b['payout']-b['stake'] for b in settled)
                wins=sum(1 for b in settled if b['result']=='win')
                clvs=[v for v in (bet_clv(b) for b in bets) if v is not None]
                c1,c2,c3,c4=st.columns(4)
                c1.metric("Total staked",f"${total_staked:.2f}"); c2.metric("P&L",f"${total_pnl:+.2f}"); c3.metric("Record",f"{wins}W-{len(settled)-wins}L")
                c4.metric("Avg CLV",f"{sum(clvs)/len(clvs):+.1f}%" if clvs else "—",
                          help="Closing Line Value: how much better your logged price was than the final pre-kickoff price. Consistently positive CLV is the strongest evidence you're beating the market — it shows up long before P&L does.")
                pnl_curve=[0]
                for b in sorted(settled,key=lambda x:x['date']):
                    pnl_curve.append(pnl_curve[-1]+(b['payout']-b['stake']))
                fig=go.Figure()
                fig.add_trace(go.Scatter(y=pnl_curve,mode='lines+markers',line=dict(color='#22c55e' if total_pnl>=0 else '#ef4444',width=2)))
                fig.update_layout(height=250,margin=dict(t=10,b=10),plot_bgcolor='#0d0d1a',paper_bgcolor='#0d0d1a',
                                  font_color='#94a3b8',xaxis=dict(gridcolor='#1e1e3a'),yaxis=dict(gridcolor='#1e1e3a'),
                                  yaxis_title="Cumulative P&L ($)")
                st.plotly_chart(fig,use_container_width=True)
