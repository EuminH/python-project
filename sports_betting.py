"""
Sports Betting Tool
- Odds comparison (via The Odds API)
- Bet tracker (log, view, P&L)
- Odds calculator (payout, implied probability, arbitrage)
"""

import json
import os
import datetime
import urllib.request
import urllib.parse

BETS_FILE = "bets.json"
ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "")
ODDS_API_BASE = "https://api.the-odds-api.com/v4"


# ── Utilities ──────────────────────────────────────────────────────────────

def american_to_decimal(american: int) -> float:
    if american > 0:
        return american / 100 + 1
    return 100 / abs(american) + 1


def decimal_to_implied_prob(decimal: float) -> float:
    return 1 / decimal


def american_implied_prob(american: int) -> float:
    return decimal_to_implied_prob(american_to_decimal(american))


def payout(stake: float, american: int) -> float:
    return stake * american_to_decimal(american)


def profit(stake: float, american: int) -> float:
    return payout(stake, american) - stake


# ── Odds Calculator ────────────────────────────────────────────────────────

def calculator_menu():
    print("\n=== Odds Calculator ===")
    try:
        american = int(input("Enter American odds (e.g. -110 or +150): "))
        stake = float(input("Enter stake ($): "))
    except ValueError:
        print("Invalid input.")
        return

    dec = american_to_decimal(american)
    impl = american_implied_prob(american) * 100
    pay = payout(stake, american)
    pro = profit(stake, american)

    print(f"\n  Decimal odds:       {dec:.4f}")
    print(f"  Implied prob:       {impl:.2f}%")
    print(f"  Payout (inc stake): ${pay:.2f}")
    print(f"  Profit:             ${pro:.2f}")


def kelly_menu():
    print("\n=== Kelly Criterion Stake Calculator ===")
    print("  Calculates optimal bet size based on your edge (from the +EV math).")
    try:
        american = int(input("Odds (American, e.g. +150): "))
        model_prob = float(input("Your estimated win probability (0-1, e.g. 0.55): "))
        bankroll = float(input("Current bankroll ($): "))
        fraction = float(input("Kelly fraction (0.5 = half-Kelly recommended): ") or "0.5")
    except ValueError:
        print("Invalid input.")
        return

    dec = american_to_decimal(american)
    b = dec - 1
    q = 1 - model_prob
    imp_prob = american_implied_prob(american)
    edge = model_prob - imp_prob
    kelly = (b * model_prob - q) / b
    frac_kelly = max(0.0, kelly * fraction)
    stake = bankroll * frac_kelly

    print(f"\n  Implied prob (bookmaker): {imp_prob*100:.2f}%")
    print(f"  Your estimated prob:      {model_prob*100:.2f}%")
    print(f"  Edge:                     {edge*100:+.2f}%")
    if edge <= 0:
        print("  ✗ No edge — do NOT bet.")
        return
    print(f"  Full Kelly:               {kelly*100:.2f}% of bankroll")
    print(f"  Half-Kelly stake:         ${stake:.2f} (recommended)")
    print(f"  Potential profit:         ${profit(stake, american):.2f}")


def parlay_menu():
    print("\n=== Parlay Calculator (+EV Math) ===")
    print("  Enter each leg. A parlay is +EV only if EVERY leg is +EV.")
    legs = []
    while True:
        raw = input(f"\nLeg {len(legs)+1} odds (American, blank to finish): ").strip()
        if not raw:
            break
        try:
            american = int(raw)
            prob = float(input(f"  Your estimated win prob for this leg (0-1): "))
            legs.append((american, prob))
        except ValueError:
            print("  Invalid, skipping.")

    if len(legs) < 2:
        print("Need at least 2 legs.")
        return

    print(f"\n  {'Leg':<5} {'Odds':>8} {'Imp Prob':>10} {'Your Prob':>10} {'Edge':>8} {'EV/$1':>8}")
    print(f"  {'-'*55}")
    all_ev_positive = True
    combined_prob = 1.0
    combined_odds = 1.0
    for i, (american, prob) in enumerate(legs):
        dec = american_to_decimal(american)
        imp = american_implied_prob(american)
        edge = prob - imp
        ev = prob * (dec - 1) - (1 - prob)
        combined_prob *= prob
        combined_odds *= dec
        if ev <= 0:
            all_ev_positive = False
        print(f"  {i+1:<5} {american:>+8} {imp*100:>9.2f}% {prob*100:>9.2f}% {edge*100:>+7.2f}% {ev:>+8.4f}")

    combined_ev = combined_prob * (combined_odds - 1) - (1 - combined_prob)
    print(f"\n  Combined probability: {combined_prob*100:.3f}%")
    print(f"  Combined odds:        {combined_odds:.2f}x (American: +{int((combined_odds-1)*100)})")
    print(f"  Parlay EV per $1:     {combined_ev:+.4f}")
    if all_ev_positive and combined_ev > 0:
        print("  ✓ All legs +EV — parlay is mathematically justified.")
    elif combined_ev > 0:
        print("  ⚠ Parlay is +EV overall but some legs are -EV individually.")
    else:
        print("  ✗ Parlay is -EV — not recommended.")


def arbitrage_menu():
    print("\n=== Arbitrage Checker ===")
    print("Enter odds for each side (American). Leave blank to stop.")
    odds_list = []
    while True:
        raw = input(f"  Odds for selection {len(odds_list)+1}: ").strip()
        if not raw:
            break
        try:
            odds_list.append(int(raw))
        except ValueError:
            print("  Invalid, skipping.")

    if len(odds_list) < 2:
        print("Need at least 2 selections.")
        return

    implied_sum = sum(american_implied_prob(o) for o in odds_list)
    margin = (implied_sum - 1) * 100

    print(f"\n  Implied probability sum: {implied_sum*100:.2f}%")
    if implied_sum < 1:
        print(f"  *** ARBITRAGE OPPORTUNITY! Guaranteed profit margin: {abs(margin):.2f}% ***")
        try:
            total = float(input("  Total stake to split ($): "))
            for i, o in enumerate(odds_list):
                stake_i = total * (american_implied_prob(o) / implied_sum)
                print(f"    Selection {i+1} (odds {o:+d}): stake ${stake_i:.2f}, return ${payout(stake_i, o):.2f}")
        except ValueError:
            pass
    else:
        print(f"  No arbitrage. Book margin: {margin:.2f}%")


# ── Bet Tracker ────────────────────────────────────────────────────────────

def load_bets():
    if not os.path.exists(BETS_FILE):
        return []
    with open(BETS_FILE) as f:
        return json.load(f)


def save_bets(bets):
    with open(BETS_FILE, "w") as f:
        json.dump(bets, f, indent=2)


def log_bet():
    print("\n=== Log a Bet ===")
    try:
        sport = input("Sport: ").strip()
        event = input("Event (e.g. Chiefs vs Eagles): ").strip()
        pick = input("Your pick: ").strip()
        american = int(input("Odds (American, e.g. -110): "))
        stake = float(input("Stake ($): "))
        book = input("Sportsbook: ").strip()
    except ValueError:
        print("Invalid input.")
        return

    bet = {
        "id": int(datetime.datetime.now().timestamp()),
        "date": datetime.date.today().isoformat(),
        "sport": sport,
        "event": event,
        "pick": pick,
        "odds": american,
        "stake": stake,
        "book": book,
        "result": "pending",
        "payout": 0.0,
    }
    bets = load_bets()
    bets.append(bet)
    save_bets(bets)
    print(f"  Bet logged (id: {bet['id']}). Potential profit: ${profit(stake, american):.2f}")


def settle_bet():
    bets = load_bets()
    pending = [b for b in bets if b["result"] == "pending"]
    if not pending:
        print("\nNo pending bets.")
        return

    print("\n=== Settle a Bet ===")
    for i, b in enumerate(pending):
        print(f"  [{i}] {b['date']} | {b['event']} | {b['pick']} | {b['odds']:+d} | ${b['stake']:.2f}")
    try:
        idx = int(input("Select bet number: "))
        result = input("Result (win/loss/push): ").strip().lower()
    except ValueError:
        print("Invalid input.")
        return

    if result not in ("win", "loss", "push"):
        print("Invalid result.")
        return

    bet = pending[idx]
    if result == "win":
        bet["payout"] = payout(bet["stake"], bet["odds"])
    elif result == "push":
        bet["payout"] = bet["stake"]
    else:
        bet["payout"] = 0.0
    bet["result"] = result

    for b in bets:
        if b["id"] == bet["id"]:
            b.update(bet)
    save_bets(bets)
    print(f"  Settled as {result}. Payout: ${bet['payout']:.2f}")


def view_bets():
    bets = load_bets()
    if not bets:
        print("\nNo bets recorded.")
        return

    print(f"\n{'Date':<12}{'Event':<28}{'Pick':<18}{'Odds':>6}{'Stake':>8}{'Result':<10}{'P&L':>8}")
    print("-" * 92)

    total_staked = total_pnl = 0.0
    for b in bets:
        pnl = b["payout"] - b["stake"] if b["result"] != "pending" else 0.0
        total_staked += b["stake"]
        if b["result"] != "pending":
            total_pnl += pnl
        pnl_str = f"${pnl:+.2f}" if b["result"] != "pending" else "-"
        print(f"{b['date']:<12}{b['event']:<28}{b['pick']:<18}{b['odds']:>+6}${b['stake']:>7.2f}  {b['result']:<10}{pnl_str:>8}")

    settled = [b for b in bets if b["result"] != "pending"]
    wins = sum(1 for b in settled if b["result"] == "win")
    print("-" * 92)
    print(f"  Total staked: ${total_staked:.2f}  |  P&L: ${total_pnl:+.2f}  |  "
          f"Record: {wins}-{len(settled)-wins} ({len([b for b in bets if b['result']=='pending'])} pending)")


# ── Odds Comparison ────────────────────────────────────────────────────────

def fetch_odds(sport="americanfootball_nfl"):
    if not ODDS_API_KEY:
        print("\n  [!] Set ODDS_API_KEY environment variable to fetch live odds.")
        print("      Get a free key at https://the-odds-api.com")
        return []

    url = (f"{ODDS_API_BASE}/sports/{sport}/odds/"
           f"?apiKey={ODDS_API_KEY}&regions=us&markets=h2h&oddsFormat=american")
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            return json.loads(resp.read())
    except Exception as e:
        print(f"  Error fetching odds: {e}")
        return []


def odds_comparison():
    print("\n=== Live Odds Comparison ===")
    sports = {
        "1": ("americanfootball_nfl", "NFL"),
        "2": ("basketball_nba", "NBA"),
        "3": ("baseball_mlb", "MLB"),
        "4": ("icehockey_nhl", "NHL"),
        "5": ("soccer_epl", "EPL Soccer"),
    }
    for k, (_, name) in sports.items():
        print(f"  [{k}] {name}")
    choice = input("Select sport: ").strip()
    sport_key, sport_name = sports.get(choice, ("americanfootball_nfl", "NFL"))

    print(f"\n  Fetching {sport_name} odds...")
    events = fetch_odds(sport_key)
    if not events:
        return

    for event in events[:5]:
        home = event.get("home_team", "")
        away = event.get("away_team", "")
        print(f"\n  {away} @ {home}")
        print(f"  {'Book':<20} {'Away':>8} {'Home':>8}")
        print(f"  {'-'*38}")
        for book in event.get("bookmakers", []):
            markets = {m["key"]: m for m in book.get("markets", [])}
            h2h = markets.get("h2h", {})
            outcomes = {o["name"]: o["price"] for o in h2h.get("outcomes", [])}
            away_odds = outcomes.get(away, "N/A")
            home_odds = outcomes.get(home, "N/A")
            away_str = f"{away_odds:+d}" if isinstance(away_odds, int) else str(away_odds)
            home_str = f"{home_odds:+d}" if isinstance(home_odds, int) else str(home_odds)
            print(f"  {book['title']:<20} {away_str:>8} {home_str:>8}")


# ── Main Menu ──────────────────────────────────────────────────────────────

def main():
    menu = {
        "1": ("Odds Calculator", calculator_menu),
        "2": ("Kelly Criterion Stake", kelly_menu),
        "3": ("Parlay Calculator (+EV)", parlay_menu),
        "4": ("Arbitrage Checker", arbitrage_menu),
        "5": ("Log a Bet", log_bet),
        "6": ("Settle a Bet", settle_bet),
        "7": ("View Bets & P&L", view_bets),
        "8": ("Live Odds Comparison", odds_comparison),
        "q": ("Quit", None),
    }

    print("\n╔══════════════════════════════╗")
    print("║   Sports Betting Tool v1.0   ║")
    print("╚══════════════════════════════╝")

    while True:
        print("\n--- Menu ---")
        for k, (label, _) in menu.items():
            print(f"  [{k}] {label}")
        choice = input("Select: ").strip().lower()
        if choice == "q":
            break
        action = menu.get(choice)
        if action:
            action[1]()
        else:
            print("Invalid choice.")


if __name__ == "__main__":
    main()
