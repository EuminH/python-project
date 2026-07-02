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
    "mlb":  ("baseball", "mlb"),
    "nfl":  ("football", "nfl"),
    "nhl":  ("hockey", "nhl"),
    "epl":  ("soccer", "eng.1"),
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
