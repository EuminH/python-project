"""
Live data feeds:
- The Odds API: live odds from FanDuel, DraftKings, BetMGM etc.
- ESPN unofficial API: live scores, standings, team stats
"""

import json
import urllib.request
import os
import datetime
from dotenv import load_dotenv

load_dotenv()
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")
ODDS_BASE = "https://api.the-odds-api.com/v4"
ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports"

# Optional premium source (paid): https://developer.oddsjam.com — endpoint
# contract taken from the oddsjam-api wrapper. Dormant without a key.
ODDSJAM_API_KEY = os.getenv("ODDSJAM_API_KEY", "")
ODDSJAM_BASE = "https://api-external.oddsjam.com/api"

QUOTA_FILE = "quota.json"


def _get(url, timeout=10):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            # The Odds API reports remaining monthly credits in headers —
            # persist the latest reading so the app can show a quota meter.
            rem = r.headers.get("x-requests-remaining")
            if rem is not None:
                try:
                    with open(QUOTA_FILE, "w") as f:
                        json.dump({"remaining": float(rem),
                                   "used": float(r.headers.get("x-requests-used", 0) or 0),
                                   "ts": datetime.datetime.now().isoformat(timespec="seconds")}, f)
                except Exception:
                    pass
            return json.loads(r.read())
    except Exception as e:
        print(f"  [!] Request failed: {e}")
        return None


def get_quota():
    """Latest known The Odds API credit usage, or None if never recorded."""
    try:
        with open(QUOTA_FILE) as f:
            return json.load(f)
    except Exception:
        return None


def get_sports_list():
    """All sports The Odds API knows about, with their active flags.
    This endpoint is FREE — it does not consume quota credits."""
    return _get(f"{ODDS_BASE}/sports/?apiKey={ODDS_API_KEY}") or []


# ── OddsJam (optional premium source) ──────────────────────────────────────

ODDSJAM_LEAGUES = {
    "baseball_mlb":          ("baseball", "mlb"),
    "basketball_wnba":       ("basketball", "wnba"),
    "americanfootball_nfl":  ("football", "nfl"),
    "basketball_nba":        ("basketball", "nba"),
    "icehockey_nhl":         ("hockey", "nhl"),
    "soccer_epl":            ("soccer", "epl"),
    "tennis_atp":            ("tennis", "atp"),
    "tennis_wta":            ("tennis", "wta"),
}

_OJ_BOOK_KEYS = {
    "draftkings": "draftkings", "fanduel": "fanduel", "betmgm": "betmgm",
    "caesars": "williamhill_us", "williamhill": "williamhill_us",
    "betrivers": "betrivers", "bovada": "bovada", "pointsbet": "pointsbetus",
    "unibet": "unibet_us", "espnbet": "espnbet",
}


def _oj_market(market_name):
    m = (market_name or "").lower()
    if m == "moneyline":
        return "h2h"
    if "spread" in m or "run line" in m or "puck line" in m or "game handicap" in m:
        return "spreads"
    if m.startswith("total"):
        return "totals"
    return None


def _oj_outcome(mkey, name):
    """OddsJam outcome name -> (name, point). 'Over 32.5' / 'Lions -3.5' / team."""
    if mkey == "h2h":
        return name, None
    parts = name.rsplit(" ", 1)
    if len(parts) == 2:
        try:
            return parts[0], float(parts[1].replace("+", ""))
        except ValueError:
            pass
    return name, None


def oddsjam_to_events(rows, limit=50):
    """Convert OddsJam's flat odds rows into The-Odds-API-shaped events."""
    games = {}
    for r in rows:
        g = r.get("game") or {}
        gid = g.get("id")
        mkey = _oj_market(r.get("market_name"))
        price = r.get("price")
        book = (r.get("sports_book") or {}).get("name", "")
        if gid is None or not mkey or not isinstance(price, (int, float)) or not book:
            continue
        if r.get("is_live"):
            continue
        try:
            start = datetime.datetime.fromisoformat(str(g.get("start_date", "")))
            commence = start.astimezone(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        except Exception:
            continue
        ev = games.setdefault(gid, {
            "id": str(gid),
            "home_team": g.get("home_team", ""), "away_team": g.get("away_team", ""),
            "commence_time": commence, "_any_books": True, "_books": {},
        })
        bkey = _OJ_BOOK_KEYS.get(book.lower().replace(" ", ""), book.lower().replace(" ", "_"))
        name, point = _oj_outcome(mkey, r.get("name", ""))
        mkts = ev["_books"].setdefault(bkey, {"title": book, "markets": {}})
        out = {"name": name, "price": int(price)}
        if point is not None:
            out["point"] = point
        mkts["markets"].setdefault(mkey, []).append(out)

    events = []
    for ev in list(games.values())[:limit]:
        books = ev.pop("_books")
        ev["bookmakers"] = [{"key": bk, "title": d["title"],
                             "markets": [{"key": mk, "outcomes": outs}
                                         for mk, outs in d["markets"].items()]}
                            for bk, d in books.items()]
        events.append(ev)
    return events


def get_oddsjam_odds(sport_key, limit=50):
    """Odds from OddsJam for one of our sport keys. Returns [] without a key,
    so this source is completely dormant unless ODDSJAM_API_KEY is set."""
    if not ODDSJAM_API_KEY:
        return []
    lookup = sport_key if sport_key in ODDSJAM_LEAGUES else (
        "tennis_atp" if sport_key.startswith("tennis_atp") else
        "tennis_wta" if sport_key.startswith("tennis_wta") else None)
    m = ODDSJAM_LEAGUES.get(lookup) if lookup else None
    if not m:
        return []
    sport, league = m
    rows = _get(f"{ODDSJAM_BASE}/v1/odds?key={ODDSJAM_API_KEY}&sport={sport}&league={league}")
    if not isinstance(rows, list):
        return []
    return oddsjam_to_events(rows, limit=limit)


def get_tennis_rankings():
    """{normalized player name: world rank} for ATP + WTA singles, via ESPN
    (free, no key). Names are casefolded and accent-stripped for matching."""
    import unicodedata

    def norm(name):
        return unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().casefold().strip()

    out = {}
    for tour in ("atp", "wta"):
        data = _get(f"{ESPN_BASE}/tennis/{tour}/rankings")
        if not data:
            continue
        for rl in data.get("rankings", [])[:1]:
            for r in rl.get("ranks", []):
                nm = r.get("athlete", {}).get("displayName", "")
                rk = r.get("current")
                if nm and isinstance(rk, int):
                    out[norm(nm)] = rk
    return out


def get_espn_odds(sport="mlb", days=2, limit=50):
    """FREE fallback odds from ESPN's scoreboard (DraftKings lines syndicated
    via ESPN — no key, no quota). Moneyline only; events are shaped like The
    Odds API's so the value engine can consume them unchanged."""
    s, l = ESPN_SPORTS.get(sport, (None, None))
    if not s:
        return []
    events, today = [], datetime.date.today()
    for i in range(days):
        day = (today + datetime.timedelta(days=i)).strftime("%Y%m%d")
        data = _get(f"{ESPN_BASE}/{s}/{l}/scoreboard?dates={day}")
        if not data:
            continue
        for ev in data.get("events", []):
            comp = ev.get("competitions", [{}])[0]
            if comp.get("status", {}).get("type", {}).get("completed"):
                continue
            oddsl = [o for o in comp.get("odds", []) if o]
            if not oddsl:
                continue
            o = oddsl[0]
            comps = comp.get("competitors", [])
            home = next((c for c in comps if c.get("homeAway") == "home"), {})
            away = next((c for c in comps if c.get("homeAway") == "away"), {})
            hname = home.get("team", {}).get("displayName", "")
            aname = away.get("team", {}).get("displayName", "")
            ml = o.get("moneyline") or {}

            def _price(side):
                try:
                    return int(str((ml.get(side) or {}).get("close", {}).get("odds", "")))
                except Exception:
                    return None

            hp, ap = _price("home"), _price("away")
            if hp is None or ap is None or not (hname and aname):
                continue
            outcomes = [{"name": hname, "price": hp}, {"name": aname, "price": ap}]
            dp = _price("draw")
            if dp is not None:
                outcomes.append({"name": "Draw", "price": dp})
            book = (o.get("provider") or {}).get("displayName", "ESPN BET")
            events.append({
                "id": str(ev.get("id", "")),
                "home_team": hname, "away_team": aname,
                "commence_time": ev.get("date", ""),
                "_espn_fallback": True,
                "bookmakers": [{
                    "key": "draftkings" if "DraftKings" in book else "espnbet",
                    "title": f"{book} · ESPN",
                    "markets": [{"key": "h2h", "outcomes": outcomes}],
                }],
            })
            if len(events) >= limit:
                return events
    return events


# ── The Odds API ───────────────────────────────────────────────────────────

SPORT_KEYS = {
    "nfl":      "americanfootball_nfl",
    "nba":      "basketball_nba",
    "mlb":      "baseball_mlb",
    "nhl":      "icehockey_nhl",
    "epl":      "soccer_epl",
    "ncaaf":    "americanfootball_ncaaf",
    "ncaab":    "basketball_ncaab",
    "worldcup": "soccer_fifa_world_cup",
    "atp":      "tennis_atp_wimbledon",
    "wta":      "tennis_wta_wimbledon",
}

BOOKS = ["fanduel", "draftkings", "betmgm", "caesars", "pointsbet"]


def get_live_odds(sport="nba", markets="h2h", limit=20, books=None):
    """Fetch live odds for a sport. `books` restricts to specific sportsbooks
    (e.g. ['fanduel','draftkings']); defaults to all major US books."""
    key = SPORT_KEYS.get(sport, sport)
    book_str = ",".join(books if books else BOOKS)
    url = (f"{ODDS_BASE}/sports/{key}/odds/"
           f"?apiKey={ODDS_API_KEY}&regions=us&markets={markets}"
           f"&oddsFormat=american&bookmakers={book_str}")
    data = _get(url)
    if not data:
        return []
    return data[:limit]


def get_event_odds(sport, event_id, markets, books=None):
    """Odds for ONE event's additional/prop markets (per-event endpoint).
    Costs ~1 API credit per market requested — fetch on demand, not in bulk."""
    key = SPORT_KEYS.get(sport, sport)
    book_str = ",".join(books) if books else ",".join(BOOKS)
    url = (f"{ODDS_BASE}/sports/{key}/events/{event_id}/odds"
           f"?apiKey={ODDS_API_KEY}&regions=us&markets={markets}"
           f"&oddsFormat=american&bookmakers={book_str}")
    return _get(url)


def print_live_odds(sport="nba"):
    events = get_live_odds(sport)
    if not events:
        print(f"  No odds available for {sport.upper()} right now.")
        return

    print(f"\n=== Live {sport.upper()} Odds (FanDuel / DraftKings / BetMGM) ===\n")
    for event in events:
        home = event.get("home_team", "")
        away = event.get("away_team", "")
        commence = event.get("commence_time", "")[:10]
        print(f"  {away} @ {home}  [{commence}]")
        print(f"  {'Book':<15} {'Away':>8} {'Home':>8}")
        print(f"  {'-'*33}")
        for book in event.get("bookmakers", []):
            markets = {m["key"]: m for m in book.get("markets", [])}
            h2h = markets.get("h2h", {})
            outcomes = {o["name"]: o["price"] for o in h2h.get("outcomes", [])}
            away_o = outcomes.get(away, "N/A")
            home_o = outcomes.get(home, "N/A")
            away_str = f"{away_o:+d}" if isinstance(away_o, int) else str(away_o)
            home_str = f"{home_o:+d}" if isinstance(home_o, int) else str(home_o)
            print(f"  {book['title']:<15} {away_str:>8} {home_str:>8}")
        print()


def get_best_lines(sport="nba"):
    """Find the best available line for each team across all books."""
    events = get_live_odds(sport)
    results = []
    for event in events:
        home = event.get("home_team", "")
        away = event.get("away_team", "")
        best_home = best_away = None
        best_home_book = best_away_book = ""
        for book in event.get("bookmakers", []):
            for market in book.get("markets", []):
                if market["key"] != "h2h":
                    continue
                for o in market["outcomes"]:
                    if o["name"] == home:
                        if best_home is None or o["price"] > best_home:
                            best_home = o["price"]
                            best_home_book = book["title"]
                    elif o["name"] == away:
                        if best_away is None or o["price"] > best_away:
                            best_away = o["price"]
                            best_away_book = book["title"]
        results.append({
            "matchup": f"{away} @ {home}",
            "best_home": best_home, "best_home_book": best_home_book,
            "best_away": best_away, "best_away_book": best_away_book,
        })
    return results


# ── ESPN API ───────────────────────────────────────────────────────────────

ESPN_SPORTS = {
    "nba":  ("basketball", "nba"),
    "wnba": ("basketball", "wnba"),
    "mlb":  ("baseball", "mlb"),
    "nfl":  ("football", "nfl"),
    "nhl":  ("hockey", "nhl"),
    "epl":  ("soccer", "eng.1"),
    "worldcup": ("soccer", "fifa.world"),
    "ncaaf":("football", "college-football"),
    "ncaab":("basketball", "mens-college-basketball"),
}


def get_espn_scores(sport="nba"):
    """Get live/recent scores from ESPN."""
    s, l = ESPN_SPORTS.get(sport, ("basketball", "nba"))
    url = f"{ESPN_BASE}/{s}/{l}/scoreboard"
    data = _get(url)
    if not data:
        return []
    games = []
    for event in data.get("events", []):
        comp = event.get("competitions", [{}])[0]
        competitors = comp.get("competitors", [])
        status = comp.get("status", {}).get("type", {})
        game = {
            "id": event.get("id"),
            "name": event.get("name"),
            "status": status.get("description", ""),
            "completed": status.get("completed", False),
            "teams": [],
        }
        for c in competitors:
            game["teams"].append({
                "name": c.get("team", {}).get("displayName", ""),
                "abbr": c.get("team", {}).get("abbreviation", ""),
                "score": c.get("score", "0"),
                "home": c.get("homeAway", "") == "home",
                "winner": c.get("winner", False),
            })
        games.append(game)
    return games


def print_espn_scores(sport="nba"):
    games = get_espn_scores(sport)
    if not games:
        print(f"  No {sport.upper()} games right now.")
        return
    print(f"\n=== ESPN Live {sport.upper()} Scores ===\n")
    for g in games:
        teams = g["teams"]
        away = next((t for t in teams if not t["home"]), {})
        home = next((t for t in teams if t["home"]), {})
        status = g["status"]
        winner_sym = lambda t: " ✓" if t.get("winner") else ""
        print(f"  {away.get('name','')}{winner_sym(away)} {away.get('score','')} "
              f"@ {home.get('name','')}{winner_sym(home)} {home.get('score','')}  [{status}]")


def get_espn_standings(sport="nba"):
    """Get current standings from ESPN."""
    s, l = ESPN_SPORTS.get(sport, ("basketball", "nba"))
    url = f"{ESPN_BASE}/{s}/{l}/standings"
    data = _get(url)
    if not data:
        return []
    standings = []
    for group in data.get("children", [data]):
        for entry in group.get("standings", {}).get("entries", []):
            team = entry.get("team", {}).get("displayName", "")
            stats = {s["name"]: s["displayValue"]
                     for s in entry.get("stats", [])}
            standings.append({"team": team, "stats": stats})
    return standings


def get_espn_team_stats(sport="nba"):
    """Fetch team-level stats to feed the ML model."""
    s, l = ESPN_SPORTS.get(sport, ("basketball", "nba"))
    url = f"{ESPN_BASE}/{s}/{l}/teams"
    data = _get(url)
    if not data:
        return []
    teams = []
    for t in data.get("sports", [{}])[0].get("leagues", [{}])[0].get("teams", []):
        team = t.get("team", {})
        teams.append({
            "id":   team.get("id"),
            "name": team.get("displayName"),
            "abbr": team.get("abbreviation"),
            "wins": team.get("record", {}).get("items", [{}])[0]
                        .get("stats", [{}])[0].get("value", 0)
                    if team.get("record") else 0,
        })
    return teams


def get_live_features(sport="nba"):
    """
    Pull ESPN standings + scores and return a feature dict
    that can be merged into the ML model pipeline.
    """
    standings = get_espn_standings(sport)
    features = {}
    for entry in standings:
        team = entry["team"]
        stats = entry["stats"]
        features[team] = {
            "wins":           float(stats.get("wins", 0) or 0),
            "losses":         float(stats.get("losses", 0) or 0),
            "win_pct":        float(stats.get("winPercent", 0) or 0),
            "points_for":     float(stats.get("pointsFor", 0) or 0),
            "points_against": float(stats.get("pointsAgainst", 0) or 0),
            "streak":         stats.get("streak", ""),
        }
    return features


# ── Quick demo ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sport = sys.argv[1] if len(sys.argv) > 1 else "nba"

    print(f"\n{'='*50}")
    print(f"  Live data feed — {sport.upper()}")
    print(f"{'='*50}")

    print_espn_scores(sport)
    print_live_odds(sport)

    print(f"\n=== Best available lines ===")
    best = get_best_lines(sport)
    if best:
        print(f"  {'Matchup':<35} {'Best Away':>12} {'Book':<15} {'Best Home':>10} {'Book'}")
        print(f"  {'-'*85}")
        for b in best:
            away_str = f"{b['best_away']:+d}" if b['best_away'] else "N/A"
            home_str = f"{b['best_home']:+d}" if b['best_home'] else "N/A"
            print(f"  {b['matchup']:<35} {away_str:>12} {b['best_away_book']:<15} "
                  f"{home_str:>10} {b['best_home_book']}")

    print(f"\n=== ESPN team features (for ML model) ===")
    features = get_live_features(sport)
    for team, f in list(features.items())[:5]:
        print(f"  {team:<30} W:{f['wins']:.0f} L:{f['losses']:.0f} "
              f"Win%:{f['win_pct']:.3f} PF:{f['points_for']:.0f} PA:{f['points_against']:.0f}")
