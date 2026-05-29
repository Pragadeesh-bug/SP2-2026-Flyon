# -*- coding: utf-8 -*-
"""
IPL Powerplay Score Prediction Model  v4.0
==========================================
Competition submission — predicts POWERPLAY (overs 1-6) score for a given
innings from the test snapshot (one row per innings).

API:
  fit(deliveries_df, players_df)   -->  self
  predict(test_df)                 -->  pd.DataFrame(id, predicted_score)

Input columns (test_df):
  id, venue, innings, batting_team, bowling_team,
  Batsman's Player Id, Bowler's Player id (opponent)

Prediction target:
  Total runs scored in overs 0-5 (powerplay) of that innings.

Key design:
  - PP-specific run totals computed per team/venue/H2H from history
  - Phase averages: pp=overs 0-5 (THE target), mid=6-14, death=15-19
  - Player SR / bowler economy computed from powerplay balls only
  - Random Forest + Ridge blend, fully vectorised predict()
  - Execution target: << 20 seconds total (fit ~1-2s, predict ~0.1s)
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _safe_map(series, mapping, default):
    """Map series values using dict, fill missing with default."""
    return series.map(mapping).fillna(default)


# ---------------------------------------------------------------------------
# MyModel
# ---------------------------------------------------------------------------

class MyModel:
    """
    Predicts the POWERPLAY (overs 1-6) innings total given a pre-match
    snapshot (batting team, bowling team, venue, key players).

    Feature set:
      - Team batting avg in powerplay (overall + H2H)
      - Team bowling avg conceded in powerplay
      - Venue powerplay average
      - Innings number (1 vs 2 differ slightly in PP)
      - Top-batsman historical PP strike rate
      - Top-bowler historical PP economy rate
    """

    def __init__(self):
        # ---- Lookup tables -----------------------------------------------
        self.player_map       = {}    # str(ID) -> player_name

        # Powerplay (overs 0-5) averages
        self.team_pp_avg      = {}    # team (batting) -> mean PP runs
        self.team_pp_std      = {}    # team -> std of PP runs
        self.bowl_pp_avg      = {}    # team (bowling) -> mean PP conceded
        self.h2h_pp_avg       = {}    # (bat_team, bowl_team) -> mean PP runs
        self.venue_pp_avg     = {}    # venue -> mean PP runs  (not in deliveries -> set Unknown)
        self.inn_pp_avg       = {}    # innings number -> mean PP runs

        # Player stats (powerplay balls only)
        self.batsman_pp_sr    = {}    # player_name -> PP strike rate
        self.bowler_pp_eco    = {}    # player_name -> PP economy rate

        self.global_pp_mean   = 50.0
        self.global_pp_std    = 10.0
        self.global_sr        = 1.30   # IPL PP average SR slightly higher
        self.global_eco       = 7.5    # PP economy tends to be lower

        # ---- ML models ---------------------------------------------------
        self.rf = RandomForestRegressor(
            n_estimators=120, max_depth=7, min_samples_leaf=6,
            max_features=0.75, random_state=42, n_jobs=-1
        )
        self.ridge  = Ridge(alpha=8.0)
        self.scaler = StandardScaler()

        self._trained     = False
        self._feat_cols   = []

    # -----------------------------------------------------------------------
    #  FIT
    # -----------------------------------------------------------------------

    def fit(self, deliveries_df, players_df=None):
        """
        Train from historical ball-by-ball data.

        Parameters
        ----------
        deliveries_df : DataFrame -- IPL deliveries (all seasons)
        players_df    : DataFrame -- columns: ID, Player_Name, Team
        """
        # ---- 1. Player ID map ------------------------------------------
        if players_df is not None and not players_df.empty:
            self.player_map = {
                str(r["ID"]): str(r["Player_Name"])
                for _, r in players_df.iterrows()
            }

        d = deliveries_df.copy()
        d["total_runs"] = d["batsman_runs"].fillna(0) + d["extras"].fillna(0)
        d["over"]       = d["over"].astype(int)

        # ---- 2. Isolate POWERPLAY balls (over 0-5) ---------------------
        pp = d[d["over"] <= 5].copy()

        # ---- 3. Powerplay totals per innings ---------------------------
        pp_inn = (
            pp.groupby(["matchId", "inning", "batting_team", "bowling_team"])
            ["total_runs"].sum()
            .reset_index()
            .rename(columns={"inning": "innings", "total_runs": "pp_score"})
        )
        pp_inn["venue"] = "Unknown"

        self.global_pp_mean = float(pp_inn["pp_score"].mean())
        self.global_pp_std  = float(pp_inn["pp_score"].std())

        # ---- 4. Team / H2H / venue / innings PP averages ---------------
        bat_grp = pp_inn.groupby("batting_team")["pp_score"]
        self.team_pp_avg = bat_grp.mean().to_dict()
        self.team_pp_std = bat_grp.std().fillna(8).to_dict()
        self.bowl_pp_avg = pp_inn.groupby("bowling_team")["pp_score"].mean().to_dict()

        h2h = pp_inn.groupby(["batting_team", "bowling_team"])["pp_score"].mean()
        self.h2h_pp_avg = {k: float(v) for k, v in h2h.items()}

        self.inn_pp_avg  = pp_inn.groupby("innings")["pp_score"].mean().to_dict()

        # ---- 5. Player PP stats ----------------------------------------
        self._build_player_pp_stats(pp)

        # ---- 6. Train ML -----------------------------------------------
        self._train_ml(pp_inn)

        return self

    # -----------------------------------------------------------------------

    def _build_player_pp_stats(self, pp):
        """Strike rate and economy rate using powerplay-only deliveries."""
        # Strike rate from legal deliveries (no wides)
        legal      = pp[pp["isWide"].isna()].copy()
        legal["b"] = 1
        bat_sr = (
            legal.groupby(["batsman", "matchId", "inning"])
            .agg(runs=("batsman_runs", "sum"), balls=("b", "sum"))
            .reset_index()
        )
        bat_sr["sr"] = bat_sr["runs"] / bat_sr["balls"].clip(lower=1)
        self.batsman_pp_sr = bat_sr.groupby("batsman")["sr"].mean().to_dict()

        # Bowler economy in powerplay
        bowl_g = (
            pp.groupby(["bowler", "matchId", "inning"])
            .agg(runs=("total_runs", "sum"), balls=("total_runs", "count"))
            .reset_index()
        )
        bowl_g["eco"] = bowl_g["runs"] / (bowl_g["balls"] / 6.0).clip(lower=0.1)
        self.bowler_pp_eco = bowl_g.groupby("bowler")["eco"].mean().to_dict()

    # -----------------------------------------------------------------------

    def _train_ml(self, pp_inn):
        """Build feature matrix from PP innings data and fit models."""
        gm = self.global_pp_mean

        f = pp_inn.copy()
        f["bat_avg"]    = f["batting_team"].map(self.team_pp_avg).fillna(gm)
        f["bat_std"]    = f["batting_team"].map(self.team_pp_std).fillna(self.global_pp_std)
        f["bowl_avg"]   = f["bowling_team"].map(self.bowl_pp_avg).fillna(gm)
        f["h2h_avg"]    = f.apply(
            lambda r: self.h2h_pp_avg.get((r["batting_team"], r["bowling_team"]), gm), axis=1
        )
        f["inn_avg"]    = f["innings"].map(self.inn_pp_avg).fillna(gm)
        f["inn_number"] = f["innings"].astype(float)

        # Neutral player stats at innings level
        f["batter_sr"]  = self.global_sr
        f["bowler_eco"] = self.global_eco

        # Derived
        f["score_ratio"]    = f["bat_avg"] / f["bowl_avg"].clip(lower=1)
        f["h2h_delta"]      = f["h2h_avg"] - f["bat_avg"]
        f["team_vs_global"] = f["bat_avg"] - gm

        self._feat_cols = [
            "bat_avg", "bat_std", "bowl_avg", "h2h_avg", "inn_avg", "inn_number",
            "batter_sr", "bowler_eco",
            "score_ratio", "h2h_delta", "team_vs_global",
        ]
        X = f[self._feat_cols].fillna(0).values
        y = f["pp_score"].values

        self.rf.fit(X, y)
        Xs = self.scaler.fit_transform(X)
        self.ridge.fit(Xs, y)
        self._trained = True

    # -----------------------------------------------------------------------
    #  PREDICT  (fully vectorised — single batch ML call)
    # -----------------------------------------------------------------------

    def predict(self, test_df):
        """
        Predict powerplay (overs 1-6) score for each row in test_df.

        Parameters
        ----------
        test_df : DataFrame  columns:
            id, venue, innings, batting_team, bowling_team,
            Batsman's Player Id, Bowler's Player id (opponent)

        Returns
        -------
        DataFrame : [id, predicted_score]
        """
        t = test_df.copy()
        gm = self.global_pp_mean

        bat  = "batting_team"
        bowl = "bowling_team"

        # ---- Player ID columns (comma-separated lists -> take first ID) ----
        def _first_id(val):
            """Return the first player ID from a comma-separated string."""
            try:
                return str(val).split(",")[0].strip()
            except Exception:
                return ""

        bat_col  = "Batsman's Player Id"
        bowl_col = "Bowler's Player id (opponent)"

        if bat_col in t.columns:
            t["bat_name"] = t[bat_col].apply(_first_id).map(self.player_map)
        else:
            t["bat_name"] = np.nan

        if bowl_col in t.columns:
            t["bowl_name"] = t[bowl_col].apply(_first_id).map(self.player_map)
        else:
            t["bowl_name"] = np.nan

        # ---- Core lookup features (vectorised) ----------------------------
        t["bat_avg"]   = _safe_map(t[bat],  self.team_pp_avg, gm)
        t["bat_std"]   = _safe_map(t[bat],  self.team_pp_std, self.global_pp_std)
        t["bowl_avg"]  = _safe_map(t[bowl], self.bowl_pp_avg, gm)
        t["h2h_avg"]   = t.apply(
            lambda r: self.h2h_pp_avg.get((r[bat], r[bowl]), gm), axis=1
        )
        t["inn_avg"]   = _safe_map(t["innings"], self.inn_pp_avg, gm)
        t["inn_number"]= t["innings"].astype(float)

        # Player features (PP-specific)
        t["batter_sr"]  = t["bat_name"].map(self.batsman_pp_sr).fillna(self.global_sr)
        t["bowler_eco"] = t["bowl_name"].map(self.bowler_pp_eco).fillna(self.global_eco)

        # Derived features
        t["score_ratio"]    = t["bat_avg"] / t["bowl_avg"].clip(lower=1)
        t["h2h_delta"]      = t["h2h_avg"] - t["bat_avg"]
        t["team_vs_global"] = t["bat_avg"] - gm

        # ---- ML batch prediction -----------------------------------------
        X = t[self._feat_cols].fillna(0).values

        if self._trained:
            rf_pred    = self.rf.predict(X)
            ridge_pred = self.ridge.predict(self.scaler.transform(X))
            ml_score   = 0.65 * rf_pred + 0.35 * ridge_pred
        else:
            ml_score = np.full(len(t), gm)

        # ---- Weighted blend of lookup baselines + ML ---------------------
        w = np.array([3.0, 2.5, 1.5, 1.0, 4.0])   # bat, h2h, bowl, inn, ml
        components = np.column_stack([
            t["bat_avg"].values,
            t["h2h_avg"].values,
            t["bowl_avg"].values,
            t["inn_avg"].values,
            ml_score,
        ])
        blended = (components * w).sum(axis=1) / w.sum()

        # ---- Player micro-adjustment (small additive weight) -------------
        sr_adj  = (t["batter_sr"].values  - self.global_sr)  * 3.0
        eco_adj = (self.global_eco - t["bowler_eco"].values)  * 1.0
        blended = blended + 0.20 * sr_adj + 0.20 * eco_adj

        # ---- Clip to realistic IPL powerplay range (25-90 runs) ----------
        final = np.clip(np.round(blended), 25, 90).astype(int)

        return pd.DataFrame({"id": t["id"].values, "predicted_score": final})


# ---------------------------------------------------------------------------
#  Run directly:  python mymodelfile.py
#  Writes submission.csv  with columns:  id, predicted_score
#  Prints powerplay summary to console
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    BASE = os.path.dirname(os.path.abspath(__file__))

    print("Loading datasets...")
    deliveries_df = pd.read_csv(os.path.join(BASE, "deliveries_updated_ipl_upto_2025.csv"))
    players_df    = pd.read_csv(os.path.join(BASE, "ipl_players_uniqueid.csv"))
    test_df       = pd.read_csv(os.path.join(BASE, "test_file.csv"))
    print(f"  Rows -> deliveries: {len(deliveries_df)}  |  players: {len(players_df)}  |  test: {len(test_df)}")

    print("Training model (powerplay target)...")
    model = MyModel()
    model.fit(deliveries_df, players_df)
    print(f"  Global PP mean from training data: {model.global_pp_mean:.1f} runs")

    print("Predicting powerplay scores...")
    predictions = model.predict(test_df)

    # Save submission.csv
    out_path = os.path.join(BASE, "submission.csv")
    predictions.to_csv(out_path, index=False)
    print(f"Saved -> {out_path}")

    # -----------------------------------------------------------------------
    #  Print powerplay summary
    # -----------------------------------------------------------------------

    merged = test_df[["id", "innings", "batting_team", "bowling_team", "venue"]].merge(
        predictions, on="id"
    )

    print()
    print("=" * 60)
    print("     IPL POWERPLAY SCORE PREDICTION  (Overs 1-6)")
    print("=" * 60)

    for _, row in merged.iterrows():
        label = "1st Innings" if int(row["innings"]) == 1 else "2nd Innings"
        print(f"  {label}")
        print(f"    Venue              : {row['venue']}")
        print(f"    Batting Team       : {row['batting_team']}")
        print(f"    Bowling Team       : {row['bowling_team']}")
        print(f"    Predicted PP Score : {int(row['predicted_score'])} runs  (Overs 1-6)")
        print("-" * 60)

    print()

