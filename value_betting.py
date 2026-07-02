"""
Value betting engine — finds +EV bets and builds parlays.

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

# Books used to BUILD the fair consensus (more books = sharper truth estimate).
# <=10 bookmakers costs the same API credits as 2, so this is free accuracy.
CONSENSUS_BOOKS = ["fanduel", "draftkings", "betmgm", "williamhill_us",
                   "betrivers", "bovada", "pointsbetus", "unibet_us"]

# Books the user can actually bet at (default: both).
VALUE_BOOKS = ["fanduel", "draftkings"]

MARKETS = "h2h,spreads,totals"
MARKET_LABELS = {"h2h": "ML", "spreads": "Spread", "totals": "Total"}

CLOSING_FILE = "closing_lines.json"

SPORTS = {
    "MLB":           "baseball_mlb",
    "World Cup":     "soccer_fifa_world_cup",
    "ATP Wimbledon": "tennis_atp_wimbledon",
    "WTA Wimbledon": "tennis_wta_wimbledon",
}

SPORT_TAGS = {
    "MLB": "⚾", "World Cup": "🌍", "ATP Wimbledon": "🎾", "WTA Wimbledon": "🎾",
}


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


def _bets_for_event(e, label, bet_books):
    """Bet rows for one event across all markets. Fair prob = consensus of all
    books; the price/EV use only `bet_books` (where the user will bet)."""
    lines = _book_lines(e)
    ct = e.get("commence_time", "")
    home, away = e.get("home_team", ""), e.get("away_team", "")
    match = f"{away} vs {home}" if home and away else (away or home)
    rows = []
    for (mkey, group), books_d in lines.items():
        if len(books_d) < 2:
            continue                       # need >=2 books for a consensus
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
    Cache THIS in the app so book/market/slider changes never re-hit the API."""
    window = _date_window(days)
    return {label: [e for e in get_live_odds(key, limit=limit,
                                             books=CONSENSUS_BOOKS, markets=markets)
                    if e.get("commence_time", "")[:10] in window]
            for label, key in SPORTS.items()}


def value_bets(events_by_sport, min_ev=-1.0, bet_books=None):
    """Compute bets from pre-fetched events. `bet_books` (e.g. ['fanduel'])
    restricts which book's price is used; None = best of FanDuel + DraftKings."""
    bb = bet_books or VALUE_BOOKS
    out = {}
    for label, events in events_by_sport.items():
        rows = [r for e in events for r in _bets_for_event(e, label, bb)
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


def build_parlay_suite(all_bets, min_leg_prob=0.0):
    """Curated parlays spanning the risk spectrum. `min_leg_prob` (0-1) is the
    floor each leg's fair probability must clear."""
    flat = [b for sport in all_bets.values() for b in sport]
    pos = [b for b in flat if b["ev"] > 0]
    out = []
    likely = build_parlay(flat, max_legs=3, rank="prob",
                          min_prob=max(0.55, min_leg_prob), min_ev=-0.03,
                          label="Most Likely to Hit",
                          note="Highest-probability favorites · best available price")
    balanced = build_parlay(pos, max_legs=3, rank="prob", min_prob=min_leg_prob,
                           label="Balanced +EV",
                           note="Three +EV picks with the best shot to hit")
    value = build_parlay(pos, max_legs=4, rank="ev", min_prob=min_leg_prob,
                        label="Max Value +EV",
                        note="Four highest-EV picks — long shot, huge payout")
    sigs = set()
    for p in (likely, balanced, value):
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
