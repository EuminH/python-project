import pandas as pd


def divider(title):
    print(f"\n{'='*50}")
    print(f"  {title}")
    print(f"{'='*50}\n")


# ── NBA 2024/25 ────────────────────────────────────────────────────────────

def analyze_nba():
    divider("NBA 2021/22 — Top Players by RAPTOR Rating")
    try:
        df = pd.read_csv('nba.csv')
        df = df[df['season'] == 2022].copy()
        df = df[df['season_type'] == 'RS']
        df = df[df['mp'] >= 500]
        df = df.sort_values('raptor_total', ascending=False)

        print("Top 15 players by total RAPTOR (overall impact):")
        cols = ['player_name', 'team', 'mp', 'raptor_offense', 'raptor_defense', 'raptor_total']
        print(df[cols].head(15).to_string(index=False))

        print("\nTop 5 offensive players:")
        print(df.nlargest(5, 'raptor_offense')[['player_name', 'team', 'raptor_offense']].to_string(index=False))

        print("\nTop 5 defensive players:")
        print(df.nlargest(5, 'raptor_defense')[['player_name', 'team', 'raptor_defense']].to_string(index=False))

        print("\nBest teams by avg RAPTOR:")
        team_raptor = df.groupby('team')['raptor_total'].mean().sort_values(ascending=False).head(10)
        print(team_raptor.round(2).to_string())
    except Exception as e:
        print(f"NBA error: {e}")


# ── MLB 2024 ───────────────────────────────────────────────────────────────

# Retrosheet game log columns (subset we care about)
GL_COLS = {0: 'date', 3: 'away_team', 6: 'home_team', 9: 'away_score', 10: 'home_score'}

def analyze_mlb():
    divider("MLB 2024 Season")
    try:
        raw = pd.read_csv('mlb_raw.txt', header=None, usecols=list(GL_COLS.keys()))
        raw = raw.rename(columns=GL_COLS)
        raw['date'] = raw['date'].astype(str)
        raw = raw[raw['date'].str.startswith('2024')]

        teams = pd.concat([raw['home_team'], raw['away_team']]).unique()
        records = []
        for team in teams:
            home = raw[raw['home_team'] == team]
            away = raw[raw['away_team'] == team]
            hw = (home['home_score'] > home['away_score']).sum()
            hl = (home['home_score'] < home['away_score']).sum()
            aw = (away['away_score'] > away['home_score']).sum()
            al = (away['away_score'] < away['home_score']).sum()
            w, l = hw + aw, hl + al
            rs = home['home_score'].sum() + away['away_score'].sum()
            ra = home['away_score'].sum() + away['home_score'].sum()
            g = w + l
            records.append({'Team': team, 'W': w, 'L': l,
                            'RS': int(rs), 'RA': int(ra), 'Diff': int(rs - ra),
                            'Win%': round(w / g * 100, 1) if g else 0})

        table = pd.DataFrame(records).sort_values('W', ascending=False).reset_index(drop=True)
        table.index += 1
        print(table.to_string())
    except Exception as e:
        print(f"MLB error: {e}")


# ── World Cup (from local CSV) ─────────────────────────────────────────────

def analyze_worldcup():
    divider("FIFA World Cup (All-time)")
    try:
        df = pd.read_csv('worldcup.csv')
        df = df[df['tournament'] == 'FIFA World Cup'].copy()
        df = df.dropna(subset=['home_team', 'away_team', 'home_score', 'away_score'])
        total = len(df)

        df['result'] = 'D'
        df.loc[df['home_score'] > df['away_score'], 'result'] = 'H'
        df.loc[df['home_score'] < df['away_score'], 'result'] = 'A'

        hw = (df['result'] == 'H').sum()
        d = (df['result'] == 'D').sum()
        aw = (df['result'] == 'A').sum()
        print(f"Matches: {total}")
        print(f"Home/neutral wins: {hw} ({hw/total*100:.1f}%)")
        print(f"Draws:             {d} ({d/total*100:.1f}%)")
        print(f"Away wins:         {aw} ({aw/total*100:.1f}%)")

        goals = df.groupby('home_team')['home_score'].sum().add(
            df.groupby('away_team')['away_score'].sum(), fill_value=0
        ).sort_values(ascending=False)
        print("\nTop 10 scoring nations (all-time):")
        print(goals.head(10).to_string())

        # Most wins
        wins = pd.concat([
            df[df['result'] == 'H']['home_team'],
            df[df['result'] == 'A']['away_team']
        ]).value_counts().head(10)
        print("\nMost World Cup match wins (all-time):")
        print(wins.to_string())
    except FileNotFoundError:
        print("worldcup.csv not found.")


# ── Wimbledon Tennis (from local CSV) ─────────────────────────────────────

def analyze_wimbledon(filepath, label):
    divider(f"Wimbledon {label} (Local CSV)")
    try:
        df = pd.read_csv(filepath)
        df = df.dropna(subset=['Winner', 'Loser'])
        print(f"Matches: {len(df)}\n")

        # Most wins
        wins = df['Winner'].value_counts().head(10)
        print("Most match wins:")
        print(wins.to_string())

        # Upset analysis using odds
        odds_w = 'B365W' if 'B365W' in df.columns else None
        odds_l = 'B365L' if 'B365L' in df.columns else None
        if odds_w and odds_l:
            df2 = df.dropna(subset=[odds_w, odds_l])
            upsets = df2[df2[odds_w] > df2[odds_l]]
            print(f"\nUpsets (underdog won): {len(upsets)} of {len(df2)} ({len(upsets)/len(df2)*100:.1f}%)")
            print("\nBiggest upsets:")
            top = upsets.nlargest(5, odds_w)[['Winner', 'Loser', odds_w, odds_l]]
            print(top.to_string(index=False))
    except FileNotFoundError:
        print(f"{filepath} not found.")
        if "mens" in filepath:
            print("Download from: https://www.tennis-data.co.uk/2024/wimbledon.csv")
        else:
            print("Download from: https://www.tennis-data.co.uk/2024w/wimbledon.csv")


# ── Main ───────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    analyze_nba()
    analyze_mlb()
    analyze_worldcup()
    analyze_wimbledon('wimbledon_mens.csv', "Men's 2024")
    analyze_wimbledon('wimbledon_womens.csv', "Women's 2024")
