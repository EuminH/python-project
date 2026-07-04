"""
Value betting engine — finds +EV bets and builds parlays. (build 2)

Method: consensus de-vigging + line shopping.
  1. For each game, pull h2h / spreads / totals odds from up to 8 US books.
  2. Remove each book's vig with the POWER method (solves sum(p^k)=1), which
     corrects the favorite-longshot bias that plain proportional de-vigging has.
  3. Take the MEDIAN fair probability across books -> robust consensus "truth".
  4. Price the bet at the user's chosen book(s) (default FanDuel/DraftKings),
     taking the best available line among them.
  5. EV per $1 = fair_prob * best_decimal - 1.

Window: today + tomorrow. Sports: MLB, FIFA World Cup, ATP + WTA Wimbledon.

Also keeps a rolling snapshot of the latest pre-kickoff prices
(closing_lines.json) so logged bets can be graded on Closing Line Value.
"""

import json
import datetime
import statistics
from live_data import get_live_odds
from sports_betting import american_to_decimal

# Bumped on engine changes; app.py force-reloads this module when the loaded
# copy is older (works around Streamlit Cloud keeping stale modules in memory
# across deploys, which crashed the app with ImportErrors twice).
ENGINE_VERSION = 4

# The Odds API sport key -> ESPN scoreboard key, for the free fallback feed
# used when quota is exhausted. ESPN carries DraftKings moneylines for these.
ESPN_FALLBACK = {
    "baseball_mlb":          "mlb",
    "americanfootball_nfl":  "nfl",
    "basketball_nba":        "nba",
    "icehockey_nhl":         "nhl",
    "soccer_epl":            "epl",
    "soccer_fifa_world_cup": "worldcup",
}

# Books used to BUILD the fair consensus (more books = sharper truth estimate).
# <=10 bookmakers costs the same API credits as 2, so this is free accuracy.
CONSENSUS_BOOKS = ["fanduel", "draftkings", "betmgm", "williamhill_us",
                   "betrivers", "bovada", "pointsbetus", "unibet_us",
                   "espnbet"]   # espnbet only ever appears via the ESPN fallback

# Books the user can actually bet at (default: both).
VALUE_BOOKS = ["fanduel", "draftkings"]

MARKETS = "h2h,spreads,totals"
MARKET_LABELS = {"h2h": "ML", "spreads": "Spread", "totals": "Total"}

CLOSING_FILE = "closing_lines.json"

# Fallback set used if sport discovery is unavailable. discover_sports()
# replaces the contents of SPORTS/SPORT_TAGS/SIDE_MARKETS in place at fetch
# time, so tournaments rotate automatically as seasons start and end.
SPORTS = {
    "MLB":           "baseball_mlb",
    "World Cup":     "soccer_fifa_world_cup",
    "ATP Wimbledon": "tennis_atp_wimbledon",
    "WTA Wimbledon": "tennis_wta_wimbledon",
}

SPORT_TAGS = {
    "MLB": "⚾", "World Cup": "🌍", "ATP Wimbledon": "🎾", "WTA Wimbledon": "🎾",
}

MAX_SPORTS = 5          # each sport costs ~3 credits per board refresh

def _tag_for_key(key):
    if key.startswith("baseball"):
        return "⚾"
    if key.startswith("tennis"):
        return "🎾"
    if key == "soccer_fifa_world_cup":
        return "🌍"
    if key.startswith("soccer"):
        return "⚽"
    if key.startswith("americanfootball"):
        return "🏈"
    if key.startswith("basketball"):
        return "🏀"
    if key.startswith("icehockey"):
        return "🏒"
    return "🏟"


def discover_sports():
    """Refresh SPORTS/SPORT_TAGS/SIDE_MARKETS from The Odds API's free
    active-sports list. Tournaments (tennis events, World Cup) appear only
    while running; league staples fill the remaining slots. Mutates the
    module dicts IN PLACE so every existing import sees the update."""
    from live_data import get_sports_list
    listing = get_sports_list()
    if not listing:
        return SPORTS                      # API down: keep whatever we have
    active = {s["key"]: s for s in listing if s.get("active")}

    picked = {}
    # Tournaments first — they only show up while actually running
    for k, s in sorted(active.items()):
        title = s.get("title", k)
        if k.startswith("tennis_atp_"):
            picked[f"ATP {title.replace('ATP ', '')}"] = k
        elif k.startswith("tennis_wta_"):
            picked[f"WTA {title.replace('WTA ', '')}"] = k
    if "soccer_fifa_world_cup" in active:
        picked["World Cup"] = "soccer_fifa_world_cup"
    # League staples by season priority
    for label, key in [("MLB", "baseball_mlb"), ("NFL", "americanfootball_nfl"),
                       ("NBA", "basketball_nba"), ("NHL", "icehockey_nhl"),
                       ("EPL", "soccer_epl")]:
        if key in active:
            picked[label] = key
    picked = dict(list(picked.items())[:MAX_SPORTS])

    SPORTS.clear()
    SPORTS.update(picked)
    SPORT_TAGS.clear()
    SPORT_TAGS.update({lbl: _tag_for_key(k) for lbl, k in picked.items()})
    SIDE_MARKETS.clear()
    for lbl, k in picked.items():
        if k.startswith("soccer"):
            SIDE_MARKETS[lbl] = dict(_SOCCER_SIDE)
        elif k.startswith("tennis"):
            SIDE_MARKETS[lbl] = dict(_TENNIS_SIDE)
        else:
            SIDE_MARKETS[lbl] = dict(_US_SIDE)
    return SPORTS


def _date_window(days=2):
    today = datetime.date.today()
    return {str(today + datetime.timedelta(days=i)) for i in range(days)}


# ── De-vigging ─────────────────────────────────────────────────────────────

def _devig_power(imps):
    """Power-method de-vig: find k>=1 with sum(p_i^k)=1, fair_i = p_i^k.
    Shrinks longshots more than favorites, correcting favorite-longshot bias.
    Falls back to proportional normalization if the market has no overround."""
    total = sum(imps.values())
    if total <= 1.0:                       # rare arb/underround: just normalize
        return {n: p / total for n, p in imps.items()}
    lo, hi = 1.0, 10.0
    for _ in range(60):                    # bisection on k
        mid = (lo + hi) / 2
        s = sum(p ** mid for p in imps.values())
        if s > 1:
            lo = mid
        else:
            hi = mid
    k = (lo + hi) / 2
    fair = {n: p ** k for n, p in imps.items()}
    t = sum(fair.values())
    return {n: v / t for n, v in fair.items()}


def _book_lines(event, books=CONSENSUS_BOOKS):
    """{(market_key, line_group): {book_key: {(name, point): (dec, american, title)}}}
    line_group identifies the exact line (e.g. spread -1.5 vs -2.0 are separate)."""
    out = {}
    for bm in event.get("bookmakers", []):
        if bm["key"] not in books:
            continue
        for m in bm.get("markets", []):
            mkey = m["key"]
            if mkey not in MARKET_LABELS:
                continue
            outs = [o for o in m.get("outcomes", [])
                    if isinstance(o.get("price"), (int, float))]
            if len(outs) < 2:
                continue
            group = tuple(sorted((o["name"], o.get("point")) for o in outs))
            d = {}
            for o in outs:
                dec = american_to_decimal(int(o["price"]))
                d[(o["name"], o.get("point"))] = (dec, int(o["price"]), bm["title"])
            out.setdefault((mkey, group), {})[bm["key"]] = d
    return out


def _fair_probs(group_books):
    """Median of each book's power-devigged probabilities -> consensus fair prob."""
    names = None
    per_book = []
    for d in group_books.values():
        keys = set(d.keys())
        if names is None:
            names = keys
        if keys != names:
            continue
        imps = {k: 1.0 / d[k][0] for k in keys}
        per_book.append(_devig_power(imps))
    if not per_book or names is None:
        return {}
    fair = {n: statistics.median(f[n] for f in per_book) for n in names}
    t = sum(fair.values())
    return {n: v / t for n, v in fair.items()} if t > 0 else {}


def _pick_name(mkey, name, point):
    if mkey == "h2h" or point is None:
        return name
    if mkey == "spreads":
        return f"{name} {point:+g}"
    return f"{name} {point:g}"            # totals: "Over 8.5"


def _bets_for_event(e, label, bet_books, min_books=2):
    """Bet rows for one event across all markets. Fair prob = consensus of all
    books; the price/EV use only `bet_books` (where the user will bet).
    min_books=1 accepts single-book de-vigs (needed for tennis/soccer
    spreads+totals where books hang different lines); n_books flags them."""
    if e.get("_espn_fallback"):
        # single syndicated book: price at that book, allow 1-book de-vig
        min_books = 1
        bet_books = [bm["key"] for bm in e.get("bookmakers", [])] or bet_books
    lines = _book_lines(e)
    ct = e.get("commence_time", "")
    home, away = e.get("home_team", ""), e.get("away_team", "")
    match = f"{away} vs {home}" if home and away else (away or home)
    rows = []
    for (mkey, group), books_d in lines.items():
        if len(books_d) < min_books:
            continue
        fair = _fair_probs(books_d)
        if not fair:
            continue
        for (name, point), p in fair.items():
            cands = [books_d[bk][(name, point)] for bk in bet_books
                     if bk in books_d and (name, point) in books_d[bk]]
            if not cands:
                continue
            dec, american, book = max(cands, key=lambda t: t[0])
            all_prices = {d[(name, point)][2]: d[(name, point)][1]
                          for d in books_d.values() if (name, point) in d}
            decs = [d[(name, point)][0] for d in books_d.values()
                    if (name, point) in d]
            width = (max(decs) - min(decs)) / min(decs) * 100 if len(decs) > 1 else 0.0
            ev = p * dec - 1
            rows.append({
                "sport": label,
                "match": match,
                "home": home, "away": away,
                "date": ct[:10],
                "time": ct[11:16] + " UTC" if len(ct) >= 16 else "",
                "market": MARKET_LABELS[mkey],
                "pick": _pick_name(mkey, name, point),
                "fair_prob": round(p, 4),
                "decimal": round(dec, 3),
                "american": american,
                "book": book,
                "ev": round(ev, 4),
                "ev_per_100": round(ev * 100, 2),
                "n_books": len(all_prices),
                "width": round(width, 2),
                "all_prices": all_prices,
                "commence": ct,
            })
    return rows


def fetch_events(days=2, limit=50, markets=MARKETS):
    """Raw today/tomorrow events per sport (this does the API calls).
    Cache THIS in the app so book/market/slider changes never re-hit the API.
    Refreshes the in-season sport list first (free API call), so ended
    tournaments drop out and new seasons appear automatically."""
    discover_sports()
    window = _date_window(days)
    out = {}
    for label, key in SPORTS.items():
        evs = [e for e in get_live_odds(key, limit=limit,
                                        books=CONSENSUS_BOOKS, markets=markets)
               if e.get("commence_time", "")[:10] in window]
        if not evs and key in ESPN_FALLBACK:
            # quota exhausted (or feed empty): free ESPN/DraftKings moneylines
            from live_data import get_espn_odds
            evs = [e for e in get_espn_odds(ESPN_FALLBACK[key], days=days)
                   if e.get("commence_time", "")[:10] in window]
        out[label] = evs
    return out


def value_bets(events_by_sport, min_ev=-1.0, bet_books=None, min_books=2):
    """Compute bets from pre-fetched events. `bet_books` (e.g. ['fanduel'])
    restricts which book's price is used; None = best of FanDuel + DraftKings.
    min_books=1 additionally admits single-book de-vigged lines."""
    bb = bet_books or VALUE_BOOKS
    out = {}
    for label, events in events_by_sport.items():
        rows = [r for e in events for r in _bets_for_event(e, label, bb, min_books=min_books)
                if r["ev"] >= min_ev]
        out[label] = sorted(rows, key=lambda x: x["ev"], reverse=True)
    return out


def all_value_bets(days=2, min_ev=-1.0, bet_books=None):
    """Convenience: fetch + compute in one call (used by the CLI)."""
    return value_bets(fetch_events(days=days), min_ev=min_ev, bet_books=bet_books)


def find_value_bets(sport_label, sport_key, days=2, min_ev=-1.0, limit=50, bet_books=None):
    """Single-sport value bets (kept for back-compat)."""
    events = [e for e in get_live_odds(sport_key, limit=limit,
                                       books=CONSENSUS_BOOKS, markets=MARKETS)
              if e.get("commence_time", "")[:10] in _date_window(days)]
    return value_bets({sport_label: events}, min_ev=min_ev, bet_books=bet_books)[sport_label]


# ── Side bets (event-specific prop markets) ────────────────────────────────
# These only exist on The Odds API's per-event endpoint (1 credit per market
# per fetch), so the app loads them on demand for one selected game.

_TENNIS_SIDE = {
    "alternate_spreads": "Game handicap",
    "alternate_totals":  "Total games",
}

_SOCCER_SIDE = {
    "alternate_totals":           "Goals O/U",
    "team_totals":                "Team goals O/U",
    "totals_h1":                  "1st-half goals",
    "btts":                       "Both teams to score",
    "draw_no_bet":                "Draw no bet",
    "h2h_h1":                     "1st-half result",
    "alternate_spreads":          "Goal handicap",
    "alternate_totals_corners":   "Corners O/U",
    "player_goal_scorer_anytime": "Anytime goalscorer",
    "player_first_goal_scorer":   "First goalscorer",
    "player_shots_on_target":     "Shots on target",
    "player_assists":             "Assists O/U",
}

_US_SIDE = {
    "alternate_spreads": "Alt spreads",
    "alternate_totals":  "Alt totals",
    "team_totals":       "Team totals",
}

SIDE_MARKETS = {
    "World Cup":     dict(_SOCCER_SIDE),
    "ATP Wimbledon": dict(_TENNIS_SIDE),
    "WTA Wimbledon": dict(_TENNIS_SIDE),
    "MLB":           dict(_US_SIDE),
}


def _side_pick(name, desc, point, mkey):
    """Human pick label: 'Lamine Yamal Over 0.5', 'Over 14.5', 'Oyarzabal' (scorer)."""
    if name == "Yes" and desc and point is None:
        return desc                        # scorer-style: the player IS the pick
    parts = []
    if desc:
        parts.append(desc)
    parts.append(name)
    if point is not None:
        parts.append(f"{point:+g}" if "spread" in mkey else f"{point:g}")
    return " ".join(parts)


def side_bets_for_event(event, sport_label, bet_books=None):
    """Rows for one event's side markets.

    Two honest modes per outcome group:
    - mode='fair': the group is a mutually-exclusive set (Over/Under pair,
      Yes/No, match result) -> power de-vig + median across books -> real
      fair prob and EV, same math as the main board.
    - mode='shop': not de-viggable (e.g. anytime scorer: many players can
      score). No fair prob exists; we report the best price vs the median
      book price (pure line-shopping edge, NOT an EV claim).
    """
    if not event:
        return []
    bb = bet_books or VALUE_BOOKS
    ct = event.get("commence_time", "")
    home, away = event.get("home_team", ""), event.get("away_team", "")
    match = f"{away} vs {home}" if home and away else (away or home)
    label_map = SIDE_MARKETS.get(sport_label, {})

    # groups[(mkey, gid)] = {book_key: {(name, desc, point): (dec, am, title)}}
    groups = {}
    for bm in event.get("bookmakers", []):
        for m in bm.get("markets", []):
            mkey = m["key"]
            if mkey not in label_map:
                continue
            for o in m.get("outcomes", []):
                price = o.get("price")
                if not isinstance(price, (int, float)):
                    continue
                name, desc, point = o["name"], o.get("description"), o.get("point")
                gid = (desc, abs(point) if point is not None else None)
                okey = (name, desc, point)
                groups.setdefault((mkey, gid), {}).setdefault(bm["key"], {})[okey] = (
                    american_to_decimal(int(price)), int(price), bm["title"])

    rows = []
    for (mkey, gid), books_d in groups.items():
        # exclusive set -> de-vig; else line-shop mode
        sig = None
        per_book_fair = []
        for d in books_d.values():
            keys = set(d.keys())
            if sig is None:
                sig = keys
            if keys != sig or len(keys) < 2:
                continue
            imps = {k: 1.0 / d[k][0] for k in keys}
            if not 0.9 <= sum(imps.values()) <= 1.35:
                continue
            per_book_fair.append(_devig_power(imps))
        devig_ok = bool(per_book_fair) and sig is not None and len(sig) >= 2

        all_keys = set()
        for d in books_d.values():
            all_keys.update(d.keys())

        for okey in all_keys:
            name, desc, point = okey
            cands = [books_d[bk][okey] for bk in bb
                     if bk in books_d and okey in books_d[bk]]
            if not cands:
                continue                   # user's book(s) don't price it
            dec, american, book = max(cands, key=lambda t: t[0])
            all_prices = {d[okey][2]: d[okey][1] for d in books_d.values() if okey in d}
            all_decs = [d[okey][0] for d in books_d.values() if okey in d]
            row = {
                "sport": sport_label, "match": match,
                "date": ct[:10], "time": ct[11:16] + " UTC" if len(ct) >= 16 else "",
                "market": label_map[mkey], "market_key": mkey,
                "pick": _side_pick(name, desc, point, mkey),
                "decimal": round(dec, 3), "american": american, "book": book,
                "n_books": len(all_prices), "all_prices": all_prices,
                "commence": ct,
            }
            if devig_ok and okey in sig:
                p = statistics.median(f[okey] for f in per_book_fair)
                ev = p * dec - 1
                row.update({"mode": "fair", "fair_prob": round(p, 4),
                            "ev": round(ev, 4), "ev_per_100": round(ev * 100, 2)})
            else:
                med = statistics.median(all_decs)
                row.update({"mode": "shop", "fair_prob": None, "ev": None,
                            "shop_edge": round((dec / med - 1) * 100, 2)})
            rows.append(row)

    # fair rows first (sorted by EV), then shop rows (sorted by shop edge)
    fair = sorted([r for r in rows if r["mode"] == "fair"],
                  key=lambda x: x["ev"], reverse=True)
    shop = sorted([r for r in rows if r["mode"] == "shop"],
                  key=lambda x: x["shop_edge"], reverse=True)
    return fair + shop


# ── Closing-line snapshots (for CLV grading) ───────────────────────────────

def snapshot_closing(events_by_sport):
    """Record the latest pre-kickoff best FD/DK price for every outcome.
    The last snapshot before a game starts approximates its closing line."""
    try:
        with open(CLOSING_FILE) as f:
            store = json.load(f)
    except Exception:
        store = {}
    now = datetime.datetime.now(datetime.timezone.utc)
    for label, events in events_by_sport.items():
        for e in events:
            ct = e.get("commence_time", "")
            try:
                start = datetime.datetime.fromisoformat(ct.replace("Z", "+00:00"))
                if now >= start:
                    continue               # game already started: freeze the line
            except Exception:
                pass
            for r in _bets_for_event(e, label, VALUE_BOOKS):
                key = f"{label}|{r['match']}|{r['market']}|{r['pick']}"
                store[key] = {"decimal": r["decimal"], "american": r["american"],
                              "ts": now.isoformat(timespec="seconds"),
                              "commence": ct}
    try:
        with open(CLOSING_FILE, "w") as f:
            json.dump(store, f)
    except Exception:
        pass
    return store


def _clv_daemon_loop(check_every=300):
    """Background CLV guard: if a game kicks off within 30 minutes and nobody
    has refreshed the odds recently, take one snapshot so the closing line
    gets captured. Quota-aware: skips entirely when credits run low."""
    import time, os
    from live_data import get_quota
    while True:
        time.sleep(check_every)
        try:
            with open(CLOSING_FILE) as f:
                store = json.load(f)
            now = datetime.datetime.now(datetime.timezone.utc)
            soon = False
            for entry in store.values():
                try:
                    start = datetime.datetime.fromisoformat(
                        entry["commence"].replace("Z", "+00:00"))
                    if datetime.timedelta(0) <= start - now <= datetime.timedelta(minutes=30):
                        soon = True
                        break
                except Exception:
                    continue
            if not soon:
                continue
            # someone (a visitor) already snapshotted recently -> skip
            if time.time() - os.path.getmtime(CLOSING_FILE) < 25 * 60:
                continue
            q = get_quota()
            if q and q.get("remaining", 0) < 40:
                continue                   # protect the last credits
            snapshot_closing(fetch_events(days=2))
        except Exception:
            continue


def start_clv_daemon():
    """Start the CLV snapshot guard once per process (daemon thread)."""
    import threading
    t = threading.Thread(target=_clv_daemon_loop, daemon=True, name="clv-daemon")
    t.start()
    return t


def closing_price(sport, match, market, pick):
    """The stored closing line for a logged bet, or None. Only 'closed' once
    the game has started (before that it's still the current line)."""
    try:
        with open(CLOSING_FILE) as f:
            store = json.load(f)
    except Exception:
        return None
    entry = store.get(f"{sport}|{match}|{market}|{pick}")
    if not entry:
        return None
    try:
        start = datetime.datetime.fromisoformat(entry["commence"].replace("Z", "+00:00"))
        if datetime.datetime.now(datetime.timezone.utc) < start:
            return None                    # not closed yet
    except Exception:
        pass
    return entry


# ── Staking & parlays ──────────────────────────────────────────────────────

def kelly_stake(fair_prob, decimal_odds, bankroll=1000.0, fraction=0.5):
    b = decimal_odds - 1
    if b <= 0:
        return 0.0
    q = 1 - fair_prob
    k = (b * fair_prob - q) / b
    return max(0.0, bankroll * k * fraction)


def _american_from_decimal(dec):
    if dec >= 2:
        return f"+{int(round((dec - 1) * 100))}"
    return f"-{int(round(100 / (dec - 1)))}"


def _assemble(legs, label, note):
    cp, cd = 1.0, 1.0
    for l in legs:
        cp *= l["fair_prob"]
        cd *= l["decimal"]
    ev = cp * cd - 1
    return {
        "label": label,
        "note": note,
        "legs": legs,
        "combined_prob": cp,
        "combined_decimal": cd,
        "combined_american": _american_from_decimal(cd),
        "ev": ev,
        "ev_per_100": ev * 100,
        "payout_per_100": cd * 100,
    }


def build_parlay(value_bets, max_legs=3, min_prob=0.0, min_ev=0.0,
                 rank="prob", label="Parlay", note=""):
    """Pick legs (one per match, so legs stay independent), ranked by
    probability or EV, then combine. None if fewer than 2 qualify."""
    pool = [b for b in value_bets if b["ev"] >= min_ev and b["fair_prob"] >= min_prob]
    key = (lambda x: x["fair_prob"]) if rank == "prob" else (lambda x: x["ev"])
    seen, legs = set(), []
    for b in sorted(pool, key=key, reverse=True):
        if b["match"] in seen:
            continue
        seen.add(b["match"])
        legs.append(b)
        if len(legs) >= max_legs:
            break
    if len(legs) < 2:
        return None
    return _assemble(legs, label, note)


def build_target_payout_parlay(flat, lo=1.5, hi=2.0, min_prob=0.0, min_ev=-0.03,
                               label="1.5–2x Payout"):
    """Stack the most likely legs (one per match) until the combined payout
    lands inside [lo, hi]. Skips any leg that would overshoot. A single leg
    is allowed if it alone lands in the window."""
    pool = [b for b in flat if b["ev"] >= min_ev and b["fair_prob"] >= min_prob]
    # Best EV first: fewest legs = least compounded vig. A single pick that
    # pays 1.5-2x on its own beats a long chain of heavy favorites.
    pool.sort(key=lambda x: (-x["ev"], -x["fair_prob"]))
    seen, legs, cd = set(), [], 1.0
    for b in pool:
        if b["match"] in seen:
            continue
        nd = cd * b["decimal"]
        if nd > hi:
            continue                       # this leg would overshoot the window
        seen.add(b["match"])
        legs.append(b)
        cd = nd
        if cd >= lo:
            break
    if not legs or not (lo <= cd <= hi):
        return None
    return _assemble(legs, label,
                     note=f"Best-EV path to a {lo:g}–{hi:g}x payout — fewest legs, least vig")


def build_mixed_parlay(flat, min_prob=0.35, min_ev=-0.06, label="Mixed Legs",
                       max_legs=3):
    """Market-diverse parlay: first fill one leg per DIFFERENT market type
    (ML, Spread, Total), preferring different sports; beyond that, top up by
    always adding to the LEAST-represented market so the mix stays balanced.
    +EV legs first, then highest probability. min_ev is looser than other
    builders because spreads/totals carry slightly more vig — the card's EV
    readout stays honest either way."""
    pool = [b for b in flat if b["ev"] >= min_ev and b["fair_prob"] >= min_prob]
    pool.sort(key=lambda x: (-(x["ev"] > 0), -x["fair_prob"]))
    seen_match, seen_market, seen_sport, legs = set(), set(), set(), []
    for strict in (True, False):
        for b in pool:
            mkt = b.get("market", "ML")
            if b["match"] in seen_match or mkt in seen_market:
                continue
            if strict and b["sport"] in seen_sport:
                continue
            legs.append(b)
            seen_match.add(b["match"])
            seen_market.add(mkt)
            seen_sport.add(b["sport"])
            if len(seen_market) >= 3 or len(legs) >= max_legs:
                break
        if len(seen_market) >= 3 or len(legs) >= max_legs:
            break
    # top up toward max_legs, always feeding the least-used market
    while len(legs) < max_legs:
        counts = {m: sum(1 for l in legs if l.get("market", "ML") == m)
                  for m in ("ML", "Spread", "Total")}
        placed = False
        for mkt in sorted(counts, key=lambda m: counts[m]):
            cand = next((b for b in pool if b["match"] not in seen_match
                         and b.get("market", "ML") == mkt), None)
            if cand:
                legs.append(cand)
                seen_match.add(cand["match"])
                placed = True
                break
        if not placed:
            break
    if len(legs) < 2:
        return None
    return _assemble(legs, label,
                     note="One leg each from different markets (ML · Spread · Total)")


def build_parlay_suite(all_bets, min_leg_prob=0.0, mixed_max_legs=3):
    """Curated parlays spanning the risk spectrum. `min_leg_prob` (0-1) is the
    floor each leg's fair probability must clear."""
    flat = [b for sport in all_bets.values() for b in sport]
    pos = [b for b in flat if b["ev"] > 0]
    out = []
    likely = build_parlay(flat, max_legs=3, rank="prob",
                          min_prob=max(0.55, min_leg_prob), min_ev=-0.03,
                          label="Most Likely to Hit",
                          note="Highest-probability favorites · best available price")
    modest = build_target_payout_parlay(flat, lo=1.5, hi=2.0, min_prob=min_leg_prob)
    mixed = build_mixed_parlay(flat, min_prob=max(0.35, min_leg_prob),
                               max_legs=mixed_max_legs)
    balanced = build_parlay(pos, max_legs=3, rank="prob", min_prob=min_leg_prob,
                           label="Balanced +EV",
                           note="Three +EV picks with the best shot to hit")
    value = build_parlay(pos, max_legs=4, rank="ev", min_prob=min_leg_prob,
                        label="Max Value +EV",
                        note="Four highest-EV picks — long shot, huge payout")
    sigs = set()
    for p in (likely, modest, mixed, balanced, value):
        if not p:
            continue
        sig = tuple(sorted((l["pick"], l["match"]) for l in p["legs"]))
        if sig in sigs:
            continue
        sigs.add(sig)
        out.append(p)
    return out


if __name__ == "__main__":
    print("=== VALUE BETS — Today + Tomorrow (bet at FanDuel / DraftKings) ===")
    print("    fair % = median power-devig across up to 8 US books\n")
    data = all_value_bets(days=2)
    for sport, bets in data.items():
        pos = [b for b in bets if b["ev"] > 0]
        print(f"{SPORT_TAGS.get(sport,'')} {sport}: {len(bets)} priced, {len(pos)} +EV")
        for b in pos[:5]:
            stake = kelly_stake(b["fair_prob"], b["decimal"])
            print(f"   BET [{b['market']:<6}] {b['pick']:<24} {b['american']:>+6} @ {b['book']:<11} "
                  f"fair {b['fair_prob']*100:4.1f}% ({b['n_books']} books, width {b['width']:.1f}%)  "
                  f"EV {b['ev_per_100']:+5.1f}/$100  stake ${stake:5.2f}")
        print()

    print("=== BEST PARLAYS ===\n")
    for p in build_parlay_suite(data):
        print(f"{p['label']}: {len(p['legs'])} legs | hit {p['combined_prob']*100:.1f}% | "
              f"odds {p['combined_american']} ({p['combined_decimal']:.2f}x) | "
              f"EV {p['ev_per_100']:+.1f}/$100")
        for l in p["legs"]:
            print(f"    - [{l['market']:<6}] {l['pick']:<24} {l['american']:>+6} @ {l['book']:<11} ({l['sport']})")
        print()
