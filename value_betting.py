"""
Value betting engine — finds +EV bets and builds parlays.

Method (no ML model needed for these sports): de-vigging + line shopping.
  1. For each game, take FanDuel and DraftKings odds.
  2. Remove each book's vig to get its "fair" probability per outcome.
  3. Average the two books -> consensus fair probability.
  4. Take the BEST price available across the two books for each outcome.
  5. EV per $1 = fair_prob * best_decimal - 1.
     If a book is offering a better price than the fair value implies, EV > 0.

Restricted to FanDuel + DraftKings. Today + tomorrow only.
Sports: MLB, FIFA World Cup, ATP Wimbledon, WTA Wimbledon.
"""

import datetime
from live_data import get_live_odds
from sports_betting import american_to_decimal

VALUE_BOOKS = ["fanduel", "draftkings"]

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


def _book_lines(event, books=VALUE_BOOKS):
    """{book_key: {outcome_name: (decimal, american, book_title)}} for h2h."""
    out = {}
    for bm in event.get("bookmakers", []):
        if bm["key"] not in books:
            continue
        for m in bm.get("markets", []):
            if m["key"] != "h2h":
                continue
            d = {}
            for o in m["outcomes"]:
                price = o["price"]
                if not isinstance(price, (int, float)):
                    continue
                d[o["name"]] = (american_to_decimal(int(price)), int(price), bm["title"])
            if d:
                out[bm["key"]] = d
    return out


def _fair_probs(book_lines):
    """De-vig each book that priced the full market, then average -> consensus fair prob.
    Always uses ALL books available (FanDuel + DraftKings) for the best truth estimate,
    regardless of which book the user ends up betting at."""
    names = set()
    for d in book_lines.values():
        names.update(d.keys())
    if not names:
        return {}
    fair_books = []
    for d in book_lines.values():
        if not all(n in d for n in names):
            continue  # skip incomplete markets (e.g. one book missing the draw)
        imp = {n: 1.0 / d[n][0] for n in names}
        over = sum(imp.values())
        if over <= 0:
            continue
        fair_books.append({n: imp[n] / over for n in names})
    if not fair_books:
        return {}
    return {n: sum(f[n] for f in fair_books) / len(fair_books) for n in names}


def _bets_for_event(e, label, bet_books):
    """Build bet rows for one event. Fair prob = both-book consensus; the price/EV
    use only `bet_books` (the books the user will actually bet at)."""
    lines = _book_lines(e)
    if len(lines) < 2:
        return []                       # need both books for a fair consensus
    fair = _fair_probs(lines)
    if not fair:
        return []
    ct = e.get("commence_time", "")
    home, away = e.get("home_team", ""), e.get("away_team", "")
    match = f"{away} vs {home}" if home and away else (away or home)
    rows = []
    for name, p in fair.items():
        # best price for this outcome among the chosen books only
        cands = [lines[bk][name] for bk in bet_books if bk in lines and name in lines[bk]]
        if not cands:
            continue
        dec, american, book = max(cands, key=lambda t: t[0])
        ev = p * dec - 1
        rows.append({
            "sport": label,
            "match": match,
            "home": home, "away": away,
            "date": ct[:10],
            "time": ct[11:16] + " UTC" if len(ct) >= 16 else "",
            "pick": name,
            "fair_prob": round(p, 4),
            "decimal": round(dec, 3),
            "american": american,
            "book": book,
            "ev": round(ev, 4),
            "ev_per_100": round(ev * 100, 2),
        })
    return rows


def fetch_events(days=2, limit=50):
    """Raw today/tomorrow events per sport (book-agnostic — this does the API calls).
    Cache THIS in the app so switching books/sliders never re-hits the API."""
    window = _date_window(days)
    return {label: [e for e in get_live_odds(key, limit=limit, books=VALUE_BOOKS)
                    if e.get("commence_time", "")[:10] in window]
            for label, key in SPORTS.items()}


def value_bets(events_by_sport, min_ev=-1.0, bet_books=None):
    """Compute bets from pre-fetched events. `bet_books` (e.g. ['fanduel']) restricts
    which book's price is used; None = best of FanDuel + DraftKings. Cheap (no API)."""
    bb = bet_books or VALUE_BOOKS
    out = {}
    for label, events in events_by_sport.items():
        rows = [r for e in events for r in _bets_for_event(e, label, bb) if r["ev"] >= min_ev]
        out[label] = sorted(rows, key=lambda x: x["ev"], reverse=True)
    return out


def all_value_bets(days=2, min_ev=-1.0, bet_books=None):
    """Convenience: fetch + compute in one call (used by the CLI)."""
    return value_bets(fetch_events(days=days), min_ev=min_ev, bet_books=bet_books)


def find_value_bets(sport_label, sport_key, days=2, min_ev=-1.0, limit=50, bet_books=None):
    """Single-sport value bets (kept for back-compat)."""
    events = [e for e in get_live_odds(sport_key, limit=limit, books=VALUE_BOOKS)
              if e.get("commence_time", "")[:10] in _date_window(days)]
    return value_bets({sport_label: events}, min_ev=min_ev, bet_books=bet_books)[sport_label]


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
    """Pick +EV legs (one per match), ranked by probability or EV, then combine.
    Returns None if fewer than 2 qualifying legs exist."""
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
    """Curated parlays spanning the risk spectrum:
       - Most Likely to Hit: highest-probability favorites (near-fair price)
       - Balanced +EV: +EV picks with a real shot
       - Max Value +EV: highest-EV long shots, big payout

    `min_leg_prob` (0-1) is the floor each leg's fair probability must clear —
    raise it to force more likely-to-hit legs into every parlay.
    """
    flat = [b for sport in all_bets.values() for b in sport]
    pos = [b for b in flat if b["ev"] > 0]          # +EV legs only
    out = []
    # Favorites most likely to hit (allow near-fair pricing, not just +EV)
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
    print("=== VALUE BETS — Today + Tomorrow (FanDuel / DraftKings) ===\n")
    data = all_value_bets(days=2)
    for sport, bets in data.items():
        pos = [b for b in bets if b["ev"] > 0]
        print(f"{SPORT_TAGS.get(sport,'')} {sport}: {len(bets)} priced, {len(pos)} +EV")
        for b in pos[:5]:
            stake = kelly_stake(b["fair_prob"], b["decimal"])
            print(f"   BET {b['pick']:<22} {b['american']:>+5} @ {b['book']:<11} "
                  f"fair {b['fair_prob']*100:4.1f}%  EV {b['ev_per_100']:+5.1f}/$100  "
                  f"stake ${stake:5.2f}  [{b['date']} {b['time']}]")
        print()

    print("=== BEST PARLAYS ===\n")
    for p in build_parlay_suite(data):
        print(f"{p['label']}: {len(p['legs'])} legs | "
              f"hit {p['combined_prob']*100:.1f}% | odds {p['combined_american']} "
              f"({p['combined_decimal']:.2f}x) | EV {p['ev_per_100']:+.1f}/$100")
        for l in p["legs"]:
            print(f"    - {l['pick']:<22} {l['american']:>+5} @ {l['book']:<11} ({l['sport']})")
        print()
