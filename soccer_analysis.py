import pandas as pd

df = pd.read_csv('soccer_data.csv')

# Clean up
df = df.dropna(subset=['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR'])

print("=== 2024/25 Premier League Analysis ===\n")

# ── Results summary ────────────────────────────────────────────────────────
total = len(df)
home_wins = (df['FTR'] == 'H').sum()
draws = (df['FTR'] == 'D').sum()
away_wins = (df['FTR'] == 'A').sum()

print(f"Matches played: {total}")
print(f"Home wins: {home_wins} ({home_wins/total*100:.1f}%)")
print(f"Draws:     {draws} ({draws/total*100:.1f}%)")
print(f"Away wins: {away_wins} ({away_wins/total*100:.1f}%)\n")

# ── Top scorers (teams) ────────────────────────────────────────────────────
goals = df.groupby('HomeTeam')['FTHG'].sum() + df.groupby('AwayTeam')['FTAG'].sum()
goals = goals.sort_values(ascending=False)
print("=== Goals Scored (Top 10) ===")
print(goals.head(10).to_string())

# ── League table ───────────────────────────────────────────────────────────
teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
table = []

for team in teams:
    home = df[df['HomeTeam'] == team]
    away = df[df['AwayTeam'] == team]

    hw = (home['FTR'] == 'H').sum()
    hd = (home['FTR'] == 'D').sum()
    hl = (home['FTR'] == 'A').sum()
    aw = (away['FTR'] == 'A').sum()
    ad = (away['FTR'] == 'D').sum()
    al = (away['FTR'] == 'H').sum()

    w, d, l = hw + aw, hd + ad, hl + al
    gf = home['FTHG'].sum() + away['FTAG'].sum()
    ga = home['FTAG'].sum() + away['FTHG'].sum()
    pts = w * 3 + d

    table.append({'Team': team, 'W': w, 'D': d, 'L': l, 'GF': int(gf), 'GA': int(ga), 'GD': int(gf - ga), 'Pts': pts})

table_df = pd.DataFrame(table).sort_values('Pts', ascending=False).reset_index(drop=True)
table_df.index += 1
print("\n=== League Table ===")
print(table_df.to_string())

# ── Betting value analysis ─────────────────────────────────────────────────
odds_cols = ['B365H', 'B365D', 'B365A']
if all(c in df.columns for c in odds_cols):
    print("\n=== Betting Value (Bet365) ===")
    df2 = df.dropna(subset=odds_cols)

    # Implied probabilities
    df2 = df2.copy()
    df2['imp_H'] = 1 / df2['B365H']
    df2['imp_D'] = 1 / df2['B365D']
    df2['imp_A'] = 1 / df2['B365A']
    df2['margin'] = (df2['imp_H'] + df2['imp_D'] + df2['imp_A'] - 1) * 100

    print(f"Avg book margin: {df2['margin'].mean():.2f}%")

    # ROI if you always bet home / draw / away
    for outcome, col, result in [('Home', 'B365H', 'H'), ('Draw', 'B365D', 'D'), ('Away', 'B365A', 'A')]:
        stake = len(df2)
        returns = df2[df2['FTR'] == result][col].sum()
        roi = (returns - stake) / stake * 100
        print(f"  Bet always {outcome}: ROI = {roi:+.1f}%")
else:
    print("\n[!] Bet365 odds columns not found in CSV.")
