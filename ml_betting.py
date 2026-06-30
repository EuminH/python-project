"""
ML Betting Model — Premier League multi-season
- 6 seasons of data (2019-2025)
- Enhanced features: rolling form, H2H record, home/away streaks, goal difference
- Gradient Boosting with hyperparameter tuning
- Kelly Criterion & dynamic staking
- Calibration tracking
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.calibration import CalibratedClassifierCV


# ── Kelly Criterion ────────────────────────────────────────────────────────

def kelly_fraction(prob, odds_decimal, fraction=0.5):
    b = odds_decimal - 1
    q = 1 - prob
    kelly = (b * prob - q) / b
    return max(0.0, kelly * fraction)


def dynamic_stake(bankroll, prob, odds_decimal, fraction=0.5):
    return bankroll * kelly_fraction(prob, odds_decimal, fraction)


def vig(odds_list):
    implied_sum = sum(1 / o for o in odds_list)
    return (implied_sum - 1) / implied_sum * 100


def parlay_ev(legs):
    combined_prob = combined_odds = 1.0
    for prob, odds in legs:
        combined_prob *= prob
        combined_odds *= odds
    ev = combined_prob * (combined_odds - 1) - (1 - combined_prob)
    return combined_prob, combined_odds, ev


# ── Data loading ───────────────────────────────────────────────────────────

XG_NAME_MAP = {
    'Arsenal': 'Arsenal FC', 'Chelsea': 'Chelsea FC', 'Liverpool': 'Liverpool FC',
    'Manchester City': 'Manchester City FC', 'Manchester United': 'Manchester United FC',
    'Tottenham Hotspur': 'Tottenham Hotspur FC', 'Newcastle United': 'Newcastle United FC',
    'Aston Villa': 'Aston Villa FC', 'West Ham United': 'West Ham United FC',
    'Brighton & Hove Albion': 'Brighton & Hove Albion FC', 'Brentford': 'Brentford FC',
    'Fulham': 'Fulham FC', 'Crystal Palace': 'Crystal Palace FC',
    'Wolverhampton Wanderers': 'Wolverhampton Wanderers FC', 'Everton': 'Everton FC',
    'AFC Bournemouth': 'AFC Bournemouth', 'Nottingham Forest': 'Nottingham Forest FC',
    'Leicester City': 'Leicester City FC', 'Ipswich Town': 'Ipswich Town FC',
    'Southampton': 'Southampton FC', 'Swansea City': 'Swansea City AFC',
    'Stoke City': 'Stoke City FC', 'Burnley': 'Burnley FC', 'Watford': 'Watford FC',
    'Huddersfield Town': 'Huddersfield Town AFC', 'Cardiff City': 'Cardiff City FC',
    'Sheffield United': 'Sheffield United FC', 'Norwich City': 'Norwich City FC',
    'Leeds United': 'Leeds United FC',
}


def load_data():
    """Load multi-season data. Odds come from soccer_data.csv (2024/25 only)."""
    try:
        multi = pd.read_csv('epl_multi.csv')
        multi['Date'] = pd.to_datetime(multi['Date'])
        multi['B365H'] = np.nan
        multi['B365D'] = np.nan
        multi['B365A'] = np.nan
        multi['home_xg'] = np.nan
        multi['away_xg'] = np.nan
        try:
            odds_df = pd.read_csv('soccer_data.csv')
            odds_df['Date'] = pd.to_datetime(odds_df['Date'], dayfirst=True)
            name_map = {
                'Man United': 'Manchester United FC', 'Man City': 'Manchester City FC',
                'Arsenal': 'Arsenal FC', 'Chelsea': 'Chelsea FC', 'Liverpool': 'Liverpool FC',
                'Tottenham': 'Tottenham Hotspur FC', 'Newcastle': 'Newcastle United FC',
                'Aston Villa': 'Aston Villa FC', 'West Ham': 'West Ham United FC',
                'Brighton': 'Brighton & Hove Albion FC', 'Brentford': 'Brentford FC',
                'Fulham': 'Fulham FC', 'Crystal Palace': 'Crystal Palace FC',
                'Wolves': 'Wolverhampton Wanderers FC', 'Everton': 'Everton FC',
                'Bournemouth': 'AFC Bournemouth', 'Nottm Forest': "Nottingham Forest FC",
                'Leicester': 'Leicester City FC', 'Ipswich': 'Ipswich Town FC',
                'Southampton': 'Southampton FC',
            }
            odds_df['HomeTeam'] = odds_df['HomeTeam'].map(name_map).fillna(odds_df['HomeTeam'])
            odds_df['AwayTeam'] = odds_df['AwayTeam'].map(name_map).fillna(odds_df['AwayTeam'])
            odds_cols = [c for c in ['Date','HomeTeam','AwayTeam','B365H','B365D','B365A'] if c in odds_df.columns]
            merged = multi.merge(odds_df[odds_cols], on=['Date','HomeTeam','AwayTeam'], how='left', suffixes=('','_odds'))
            for col in ['B365H','B365D','B365A']:
                if col+'_odds' in merged.columns:
                    merged[col] = merged[col+'_odds']
                    merged.drop(columns=[col+'_odds'], inplace=True)
            multi = merged
        except Exception:
            pass
        # Merge xG data
        try:
            xg_df = pd.read_csv('xg_data.csv')
            xg_df['Date'] = pd.to_datetime(xg_df['date'])
            xg_df['HomeTeam'] = xg_df['home_team'].map(XG_NAME_MAP).fillna(xg_df['home_team'])
            xg_df['AwayTeam'] = xg_df['away_team'].map(XG_NAME_MAP).fillna(xg_df['away_team'])
            xg_cols = xg_df[['Date','HomeTeam','AwayTeam','home_xg','away_xg']]
            merged = multi.merge(xg_cols, on=['Date','HomeTeam','AwayTeam'], how='left', suffixes=('','_xg'))
            for col in ['home_xg','away_xg']:
                if col+'_xg' in merged.columns:
                    merged[col] = merged[col+'_xg']
                    merged.drop(columns=[col+'_xg'], inplace=True)
            multi = merged
        except Exception:
            pass
        return multi.sort_values('Date').reset_index(drop=True)
    except FileNotFoundError:
        df = pd.read_csv('soccer_data.csv')
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
        df['home_xg'] = np.nan
        df['away_xg'] = np.nan
        return df.sort_values('Date').reset_index(drop=True)


# ── Feature engineering ────────────────────────────────────────────────────

def build_features(df, window=5):
    records = []
    home_history = {}
    away_history = {}
    all_history  = {}
    h2h_history  = {}
    xg_home_history = {}  # team → list of home xG values
    xg_away_history = {}  # team → list of away xG values

    def avg(lst, n=window):
        recent = lst[-n:] if len(lst) >= n else lst
        if not recent:
            return 0.0, 0.0, 0.0
        return (np.mean([x[0] for x in recent]),
                np.mean([x[1] for x in recent]),
                np.mean([x[2] for x in recent]))

    def avg1(lst, n=window):
        recent = lst[-n:] if len(lst) >= n else lst
        return float(np.mean(recent)) if recent else 0.0

    for _, row in df.iterrows():
        ht, at = row['HomeTeam'], row['AwayTeam']
        h2h_key = (ht, at)

        h_gs, h_gc, h_pts = avg(home_history.get(ht, []))
        a_gs, a_gc, a_pts = avg(away_history.get(at, []))
        oh_gs, oh_gc, oh_pts = avg(all_history.get(ht, []))
        oa_gs, oa_gc, oa_pts = avg(all_history.get(at, []))

        h2h = h2h_history.get(h2h_key, [])[-5:]
        h2h_hw = h2h.count('H') / len(h2h) if h2h else 0.33
        h2h_aw = h2h.count('A') / len(h2h) if h2h else 0.33

        home_xg_avg = avg1(xg_home_history.get(ht, []))
        away_xg_avg = avg1(xg_away_history.get(at, []))

        has_odds = 'B365H' in row and not pd.isna(row.get('B365H', np.nan))
        if has_odds:
            imp_h = 1 / row['B365H']
            imp_d = 1 / row['B365D']
            imp_a = 1 / row['B365A']
            margin = imp_h + imp_d + imp_a
            imp_h /= margin; imp_d /= margin; imp_a /= margin
        else:
            imp_h = imp_d = imp_a = 0.333

        records.append({
            'home_form_goals':    h_gs,
            'home_form_conceded': h_gc,
            'home_form_pts':      h_pts,
            'away_form_goals':    a_gs,
            'away_form_conceded': a_gc,
            'away_form_pts':      a_pts,
            'home_overall_pts':   oh_pts,
            'away_overall_pts':   oa_pts,
            'home_gd_trend':      h_gs - h_gc,
            'away_gd_trend':      a_gs - a_gc,
            'h2h_home_win_rate':  h2h_hw,
            'h2h_away_win_rate':  h2h_aw,
            'imp_home':           imp_h,
            'imp_draw':           imp_d,
            'imp_away':           imp_a,
            'home_xg_avg':        home_xg_avg,
            'away_xg_avg':        away_xg_avg,
            'xg_diff':            home_xg_avg - away_xg_avg,
            'result':             row['FTR'],
            'b365_home':          row.get('B365H', np.nan),
            'b365_draw':          row.get('B365D', np.nan),
            'b365_away':          row.get('B365A', np.nan),
        })

        h_result = {'H': 3, 'D': 1, 'A': 0}[row['FTR']]
        a_result = {'A': 3, 'D': 1, 'H': 0}[row['FTR']]
        home_history.setdefault(ht, []).append((row['FTHG'], row['FTAG'], h_result))
        away_history.setdefault(at, []).append((row['FTAG'], row['FTHG'], a_result))
        all_history.setdefault(ht, []).append((row['FTHG'], row['FTAG'], h_result))
        all_history.setdefault(at, []).append((row['FTAG'], row['FTHG'], a_result))
        h2h_history.setdefault(h2h_key, []).append(row['FTR'])

        # Track xG history (only when StatsBomb data is available)
        hxg = row.get('home_xg', np.nan)
        axg = row.get('away_xg', np.nan)
        if not pd.isna(hxg):
            xg_home_history.setdefault(ht, []).append(float(hxg))
            xg_away_history.setdefault(at, []).append(float(axg))

    return pd.DataFrame(records)


FEATURE_COLS = [
    'home_form_goals', 'home_form_conceded', 'home_form_pts',
    'away_form_goals', 'away_form_conceded', 'away_form_pts',
    'home_overall_pts', 'away_overall_pts',
    'home_gd_trend', 'away_gd_trend',
    'h2h_home_win_rate', 'h2h_away_win_rate',
    'imp_home', 'imp_draw', 'imp_away',
    'home_xg_avg', 'away_xg_avg', 'xg_diff',
]


# ── Model tuning ───────────────────────────────────────────────────────────

def tune_model(df_feat):
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [2, 3, 4],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.8, 1.0],
    }
    gs = GridSearchCV(GradientBoostingClassifier(random_state=42),
                      param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    gs.fit(df_feat[FEATURE_COLS], df_feat['result'])
    return gs.best_estimator_, gs.best_params_, gs.best_score_


# ── Backtest ───────────────────────────────────────────────────────────────

def backtest(df_feat, model, flat_stake=10.0, bankroll=1000.0, threshold=0.05):
    # Train on all multi-season data except last season (which has odds)
    df_odds = df_feat[df_feat['b365_home'].notna()].copy().reset_index(drop=True)
    df_train = df_feat[df_feat['b365_home'].isna()].copy()

    if len(df_odds) == 0:
        # Fallback: use 60/40 split on whatever we have
        split = int(len(df_feat) * 0.6)
        df_train = df_feat.iloc[:split]
        df_odds = df_feat.iloc[split:].copy().reset_index(drop=True)
        df_odds['b365_home'] = np.nan

    split = int(len(df_odds) * 0.6)
    train_odds = df_odds.iloc[:split]
    test = df_odds.iloc[split:]

    # Train on multi-season + first 60% of odds data
    full_train = pd.concat([df_train, train_odds], ignore_index=True)
    model.fit(full_train[FEATURE_COLS], full_train['result'])

    # Calibrate on odds training portion
    cal_model = CalibratedClassifierCV(model, cv=3, method='isotonic')
    cal_model.fit(train_odds[FEATURE_COLS], train_odds['result'])

    probs = cal_model.predict_proba(test[FEATURE_COLS])
    classes = list(cal_model.classes_)

    flat_staked = flat_return = 0
    kelly_staked = kelly_return = 0
    kelly_bank = bankroll
    bets = []
    kelly_curve = [bankroll]

    for i, (_, row) in enumerate(test.iterrows()):
        p_h = probs[i][classes.index('H')]
        p_d = probs[i][classes.index('D')]
        p_a = probs[i][classes.index('A')]

        for outcome, model_prob, imp_prob, odds in [
            ('H', p_h, row['imp_home'], row['b365_home']),
            ('D', p_d, row['imp_draw'], row['b365_draw']),
            ('A', p_a, row['imp_away'], row['b365_away']),
        ]:
            if pd.isna(odds):
                continue
            edge = model_prob - imp_prob
            if edge > threshold:
                win = row['result'] == outcome
                flat_staked += flat_stake
                flat_return += flat_stake * odds if win else 0

                k_stake = dynamic_stake(kelly_bank, model_prob, odds, fraction=0.5)
                kelly_staked += k_stake
                ret = k_stake * odds if win else 0
                kelly_return += ret
                kelly_bank += ret - k_stake

                bets.append({
                    'outcome': outcome,
                    'edge': round(edge, 3),
                    'model_prob': round(model_prob, 3),
                    'odds': odds,
                    'win': win,
                    'kelly_stake': round(k_stake, 2),
                    'kelly_return': round(ret, 2),
                })
                kelly_curve.append(round(kelly_bank, 2))

    flat_roi = (flat_return - flat_staked) / flat_staked * 100 if flat_staked else 0
    kelly_roi = (kelly_return - kelly_staked) / kelly_staked * 100 if kelly_staked else 0
    return (bets, flat_staked, flat_return, flat_roi,
            kelly_staked, kelly_return, kelly_roi,
            kelly_curve, cal_model, classes, FEATURE_COLS, test, probs)


ODDS_TO_ML = {
    'Manchester United': 'Manchester United FC',
    'Manchester City': 'Manchester City FC',
    'Arsenal': 'Arsenal FC',
    'Chelsea': 'Chelsea FC',
    'Liverpool': 'Liverpool FC',
    'Tottenham Hotspur': 'Tottenham Hotspur FC',
    'Newcastle United': 'Newcastle United FC',
    'Aston Villa': 'Aston Villa FC',
    'West Ham United': 'West Ham United FC',
    'Brighton & Hove Albion': 'Brighton & Hove Albion FC',
    'Brentford': 'Brentford FC',
    'Fulham': 'Fulham FC',
    'Crystal Palace': 'Crystal Palace FC',
    'Wolverhampton Wanderers': 'Wolverhampton Wanderers FC',
    'Everton': 'Everton FC',
    'Bournemouth': 'AFC Bournemouth',
    'Nottingham Forest': 'Nottingham Forest FC',
    'Leicester City': 'Leicester City FC',
    'Ipswich Town': 'Ipswich Town FC',
    'Southampton': 'Southampton FC',
}


def _build_form_state(df, window=5):
    """Replay all historical matches to build current form state for each team."""
    home_hist, away_hist, all_hist, h2h_hist = {}, {}, {}, {}
    xg_home_hist, xg_away_hist = {}, {}

    for _, row in df.iterrows():
        ht, at = row['HomeTeam'], row['AwayTeam']
        h2h_key = (ht, at)
        ftr = row['FTR']
        hr = {'H': 3, 'D': 1, 'A': 0}[ftr]
        ar = {'A': 3, 'D': 1, 'H': 0}[ftr]
        home_hist.setdefault(ht, []).append((row['FTHG'], row['FTAG'], hr))
        away_hist.setdefault(at, []).append((row['FTAG'], row['FTHG'], ar))
        all_hist.setdefault(ht, []).append((row['FTHG'], row['FTAG'], hr))
        all_hist.setdefault(at, []).append((row['FTAG'], row['FTHG'], ar))
        h2h_hist.setdefault(h2h_key, []).append(ftr)
        hxg = row.get('home_xg', np.nan)
        axg = row.get('away_xg', np.nan)
        if not pd.isna(hxg):
            xg_home_hist.setdefault(ht, []).append(float(hxg))
            xg_away_hist.setdefault(at, []).append(float(axg))

    def avg(lst):
        r = lst[-window:] if len(lst) >= window else lst
        return (np.mean([x[0] for x in r]), np.mean([x[1] for x in r]), np.mean([x[2] for x in r])) if r else (0.0, 0.0, 0.0)

    def avg1(lst):
        r = lst[-window:] if len(lst) >= window else lst
        return float(np.mean(r)) if r else 0.0

    state = {}
    teams = set(list(home_hist.keys()) + list(away_hist.keys()))
    for t in teams:
        hgs, hgc, hpts = avg(home_hist.get(t, []))
        ags, agc, apts = avg(away_hist.get(t, []))
        ogs, ogc, opts = avg(all_hist.get(t, []))
        state[t] = {
            'home': (hgs, hgc, hpts),
            'away': (ags, agc, apts),
            'overall_pts': opts,
            'xg_home': avg1(xg_home_hist.get(t, [])),
            'xg_away': avg1(xg_away_hist.get(t, [])),
        }
    return state, h2h_hist


def predict_upcoming(trained_model, classes, df, upcoming_odds):
    """Generate ML predictions for upcoming matches from The Odds API."""
    from sports_betting import american_to_decimal
    state, h2h_hist = _build_form_state(df)

    results = []
    for event in upcoming_odds:
        home_raw = event.get('home_team', '')
        away_raw = event.get('away_team', '')
        ht = ODDS_TO_ML.get(home_raw, home_raw)
        at = ODDS_TO_ML.get(away_raw, away_raw)

        best_h = best_a = None
        best_book = ''
        for book in event.get('bookmakers', []):
            for market in book.get('markets', []):
                if market['key'] != 'h2h':
                    continue
                for o in market['outcomes']:
                    price = o['price']
                    dec = american_to_decimal(price) if isinstance(price, int) else float(price)
                    if o['name'] == home_raw:
                        if best_h is None or dec > best_h:
                            best_h = dec; best_book = book['title']
                    elif o['name'] == away_raw:
                        if best_a is None or dec > best_a:
                            best_a = dec

        if not best_h or not best_a:
            continue

        imp_h = 1 / best_h; imp_a = 1 / best_a
        imp_d = max(0.05, 1 - imp_h - imp_a)
        margin = imp_h + imp_d + imp_a
        imp_h /= margin; imp_d /= margin; imp_a /= margin
        odds_d = 1 / imp_d

        hs = state.get(ht, {'home': (0,0,0), 'away': (0,0,0), 'overall_pts': 0, 'xg_home': 0, 'xg_away': 0})
        as_ = state.get(at, {'home': (0,0,0), 'away': (0,0,0), 'overall_pts': 0, 'xg_home': 0, 'xg_away': 0})

        hgs, hgc, hpts = hs['home']
        ags, agc, apts = as_['away']
        hxg = hs['xg_home']
        axg = as_['xg_away']

        h2h = h2h_hist.get((ht, at), [])[-5:]
        h2h_hw = h2h.count('H') / len(h2h) if h2h else 0.33
        h2h_aw = h2h.count('A') / len(h2h) if h2h else 0.33

        feat = [[
            hgs, hgc, hpts,
            ags, agc, apts,
            hs['overall_pts'], as_['overall_pts'],
            hgs - hgc, ags - agc,
            h2h_hw, h2h_aw,
            imp_h, imp_d, imp_a,
            hxg, axg, hxg - axg,
        ]]

        probs = trained_model.predict_proba(feat)[0]
        idx = {c: i for i, c in enumerate(classes)}
        p_h = probs[idx.get('H', 0)]
        p_d = probs[idx.get('D', 1)]
        p_a = probs[idx.get('A', 2)]

        best_outcome = max([('H', p_h, imp_h, best_h), ('D', p_d, imp_d, odds_d), ('A', p_a, imp_a, best_a)],
                           key=lambda x: x[1] - x[2])

        results.append({
            'home': home_raw, 'away': away_raw,
            'date': event.get('commence_time', '')[:10],
            'time': event.get('commence_time', '')[11:16] + ' UTC',
            'book': best_book,
            'p_home': round(p_h, 3), 'p_draw': round(p_d, 3), 'p_away': round(p_a, 3),
            'imp_home': round(imp_h, 3), 'imp_draw': round(imp_d, 3), 'imp_away': round(imp_a, 3),
            'edge_home': round(p_h - imp_h, 3),
            'edge_draw': round(p_d - imp_d, 3),
            'edge_away': round(p_a - imp_a, 3),
            'odds_h': round(best_h, 2), 'odds_d': round(odds_d, 2), 'odds_a': round(best_a, 2),
            'best_outcome': best_outcome[0],
            'best_edge': round(best_outcome[1] - best_outcome[2], 3),
            'best_prob': round(best_outcome[1], 3),
            'best_odds': round(best_outcome[3], 2),
        })

    return sorted(results, key=lambda x: x['best_edge'], reverse=True)


def calibration_report(bets):
    if not bets:
        return
    df = pd.DataFrame(bets)
    df['prob_bucket'] = pd.cut(df['model_prob'], bins=[0,.3,.4,.5,.6,.7,1.0],
                                labels=['0-30%','30-40%','40-50%','50-60%','60-70%','70-100%'])
    cal = df.groupby('prob_bucket', observed=True)['win'].agg(['mean','count'])
    cal.columns = ['Actual Win Rate','Count']
    print("\n--- Calibration ---")
    print(cal.to_string())


def parlay_section(bets):
    top = [b for b in bets if b['edge'] > 0.1][:4]
    if len(top) < 2:
        return
    cp, co, ev = parlay_ev([(b['model_prob'], b['odds']) for b in top])
    print(f"\n--- Parlay ({len(top)} legs): prob {cp*100:.2f}% odds {co:.2f}x EV {ev:+.4f} {'✓+EV' if ev>0 else '✗-EV'}")


def main():
    print("=== ML Betting Model — Premier League Multi-Season ===\n")
    df = load_data()
    seasons = df['Season'].unique() if 'Season' in df.columns else ['2024/25']
    print(f"Loaded {len(df)} matches | Seasons: {', '.join(map(str, seasons))}\n")

    df_feat = build_features(df)

    odds_rows = df_feat.dropna(subset=['b365_home'])
    if len(odds_rows):
        avg_v = np.mean([vig([r['b365_home'], r['b365_draw'], r['b365_away']])
                         for _, r in odds_rows.head(100).iterrows()])
        print(f"Avg book vig: {avg_v:.2f}%\n")

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Random Forest':       RandomForestClassifier(n_estimators=200, random_state=42),
        'Gradient Boosting':   GradientBoostingClassifier(n_estimators=200, random_state=42),
    }
    print("--- Model accuracy (5-fold CV) ---")
    for name, m in models.items():
        s = cross_val_score(m, df_feat[FEATURE_COLS], df_feat['result'], cv=5)
        print(f"  {name:<25} {s.mean()*100:.1f}% ± {s.std()*100:.1f}%")

    print("\n--- Tuning Gradient Boosting (grid search) ---")
    best_model, best_params, best_score = tune_model(df_feat)
    print(f"  Best params: {best_params}")
    print(f"  Tuned CV accuracy: {best_score*100:.1f}%")

    print("\n--- Backtest ---")
    (bets, flat_staked, flat_return, flat_roi,
     kelly_staked, kelly_return, kelly_roi,
     kelly_curve, model, classes, fcols, test, probs) = backtest(df_feat, best_model)

    wins = sum(1 for b in bets if b['win'])
    if bets:
        print(f"  Value bets: {len(bets)} | Win rate: {wins/len(bets)*100:.1f}%")
    print(f"  Flat ROI:   {flat_roi:+.1f}%  (${flat_return-flat_staked:+.0f})")
    print(f"  Kelly ROI:  {kelly_roi:+.1f}%  (bankroll: ${kelly_curve[-1]:.0f})")

    calibration_report(bets)
    parlay_section(bets)


if __name__ == '__main__':
    main()
