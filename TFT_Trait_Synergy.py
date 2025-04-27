
# TFT Trait Synergy Analysis 


from __future__ import annotations

import itertools
import math
import random
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_distances


# 1. Data Loading helpers ------------------------------------------------------

DATA_DIR = Path("/Users/chenwen/Downloads")
RAW1 = DATA_DIR / "unprocessed_challenger_match_data.csv"
RAW2 = DATA_DIR / "unprocessed_gm_match_data.csv"

_TRAIT_COL_CACHE: List[str] | None = None  # filled lazily


def load_data() -> pd.DataFrame:
    """Read the two raw CSVs, drop dups, and return a merged DataFrame."""
    df = pd.concat([
        pd.read_csv(RAW1),
        pd.read_csv(RAW2),
    ], ignore_index=True).drop_duplicates()
    return df


def trait_columns(df: pd.DataFrame) -> List[str]:
    """Return column names that hold trait names (contain 'traits' & 'name')."""
    global _TRAIT_COL_CACHE
    if _TRAIT_COL_CACHE is None:
        _TRAIT_COL_CACHE = [c for c in df.columns if "traits" in c and "name" in c]
    return _TRAIT_COL_CACHE



# 2. Core analytic helpers -----------------------------------------------------

def _rows_with_traits(df: pd.DataFrame) -> List[Set[str]]:
    """Return list of trait‑sets per player‑match row."""
    tcols = trait_columns(df)
    trait_df = df[tcols].fillna("")
    return [set(row.dropna().astype(str)) for _, row in trait_df.iterrows()]


# Memoised for speed 
_TRAIT_SETS_TOP4: List[Set[str]] | None = None
_TRAIT_SETS_BOT4: List[Set[str]] | None = None

def top_bot_trait_sets(df: pd.DataFrame) -> Tuple[List[Set[str]], List[Set[str]]]:
    """Split into top4 vs bottom4 and return their trait‑sets lists."""
    global _TRAIT_SETS_TOP4, _TRAIT_SETS_BOT4
    if _TRAIT_SETS_TOP4 is None or _TRAIT_SETS_BOT4 is None:
        top = df[df["placement"] <= 4]
        bot = df[df["placement"] >= 5]
        _TRAIT_SETS_TOP4 = _rows_with_traits(top)
        _TRAIT_SETS_BOT4 = _rows_with_traits(bot)
    return _TRAIT_SETS_TOP4, _TRAIT_SETS_BOT4



# 3. Win indicators ------------------------------------------------------------


def trait_win_table(df: pd.DataFrame) -> pd.DataFrame:
    """Return DataFrame with counts and win‑rates for each trait."""
    top, bot = top_bot_trait_sets(df)
    trait_all = Counter(itertools.chain.from_iterable(top + bot))
    trait_top = Counter(itertools.chain.from_iterable(top))
    trait_bot = Counter(itertools.chain.from_iterable(bot))

    rows = []
    for trait, total in trait_all.items():
        wins = trait_top[trait]
        losses = trait_bot[trait]
        win_rate = wins / total
        lift = (wins / len(top)) / (losses / len(bot)) if losses else math.inf
        rows.append((trait, total, wins, losses, win_rate, lift))

    return pd.DataFrame(rows, columns=[
        "trait", "total", "wins", "losses", "win_rate", "top4_lift"])


def unit_win_table(df: pd.DataFrame) -> pd.DataFrame:
    """Similar table for units. Unit cols usually contain 'unit' & 'name'."""
    unit_cols = [c for c in df.columns if "units" in c and "character_id" in c]
    top = df[df["placement"] <= 4]
    bot = df[df["placement"] >= 5]
    def collect(sub: pd.DataFrame):
        return Counter(itertools.chain.from_iterable(
            sub[unit_cols].fillna("").values.flatten()))
    top_c, bot_c = collect(top), collect(bot)
    all_c = top_c + bot_c
    rows = []
    for unit, total in all_c.items():
        wins = top_c[unit]
        losses = bot_c[unit]
        win_rate = wins / total
        lift = (wins / len(top)) / (losses / len(bot)) if losses else math.inf
        rows.append((unit, total, wins, losses, win_rate, lift))
    return pd.DataFrame(rows, columns=[
        "unit", "total", "wins", "losses", "win_rate", "top4_lift"])


# 4. Synergy discovery ---------------------------------------------------------


def trait_co_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Return symmetric co‑occurrence counts for traits in top‑4 boards."""
    top_sets, _ = top_bot_trait_sets(df)
    traits = sorted({t for s in top_sets for t in s})
    idx = {t: i for i, t in enumerate(traits)}
    mat = np.zeros((len(traits), len(traits)), dtype=int)
    for s in top_sets:
        for a, b in itertools.combinations(s, 2):
            i, j = idx[a], idx[b]
            mat[i, j] += 1
            mat[j, i] += 1
    return pd.DataFrame(mat, index=traits, columns=traits)


def top_synergy_pairs(df: pd.DataFrame, top_n: int = 20) -> List[Tuple[str, str, int, float]]:
    """Return top_n trait pairs by *lift* (obs/expected)."""
    cm = trait_co_matrix(df)
    totals = cm.sum(axis=1)
    total_games = len(top_bot_trait_sets(df)[0])
    pairs = []
    for i, j in itertools.combinations(range(len(cm)), 2):
        obs = cm.iat[i, j]
        if obs < 5:
            continue  # too rare
        exp = (totals[i] * totals[j]) / total_games
        lift = obs / exp if exp else math.inf
        pairs.append((cm.index[i], cm.columns[j], obs, lift))
    pairs.sort(key=lambda x: x[3], reverse=True)
    return pairs[:top_n]


# 5. Interactive query functions ----------------------------------------------

def _all_trait_sets(df: pd.DataFrame) -> List[Set[str]]:
    return _rows_with_traits(df)


class TraitHelper:
    """Namespace object exposing the four user‑level queries."""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.trait_sets = _all_trait_sets(df)
        self.traits = sorted({t for s in self.trait_sets for t in s})
        self.vector_matrix = self._build_matrix()

    # ---------------- Similarity --------------------------------------------
    def _build_matrix(self) -> np.ndarray:
        trait_idx = {t: i for i, t in enumerate(self.traits)}
        mat = np.zeros((len(self.trait_sets), len(self.traits)), dtype=int)
        for r, s in enumerate(self.trait_sets):
            for t in s:
                mat[r, trait_idx[t]] = 1
        return mat

    def vectorise(self, combo: Set[str]) -> np.ndarray:
        v = np.zeros(len(self.traits), dtype=int)
        for t in combo:
            if t in self.traits:
                v[self.traits.index(t)] = 1
        return v.reshape(1, -1)

    def find_most_similar(self, combo: Set[str]) -> Tuple[Set[str], float]:
        """Return existing trait set with smallest cosine distance to *combo*."""
        vec = self.vectorise(combo)
        dists = cosine_distances(vec, self.vector_matrix)[0]
        idx = np.argmin(dists)
        return self.trait_sets[idx], dists[idx]

    # ---------------- Graph helpers -----------------------------------------
    def _build_graph(self) -> Dict[str, Set[str]]:
        graph = defaultdict(set)
        top_sets, _ = top_bot_trait_sets(self.df)
        for s in top_sets:
            for a, b in itertools.combinations(s, 2):
                graph[a].add(b)
                graph[b].add(a)
        return graph

    def shortest_trait_path(self, start: str, goal: str) -> List[str]:
        """BFS through co‑occurrence graph (unweighted)."""
        graph = self._build_graph()
        if start not in graph or goal not in graph:
            return []
        queue = deque([[start]])
        visited = {start}
        while queue:
            path = queue.popleft()
            node = path[-1]
            if node == goal:
                return path
            for nbr in graph[node]:
                if nbr not in visited:
                    visited.add(nbr)
                    queue.append(path + [nbr])
        return []

    # ---------------- Stats --------------------------------------------------
    def most_connected_trait(self) -> Tuple[str, int]:
        g = self._build_graph()
        trait, adj = max(g.items(), key=lambda kv: len(kv[1]))
        return trait, len(adj)

    def trait_stats(self, trait: str) -> Dict[str, float | int]:
        table = trait_win_table(self.df).set_index("trait")
        if trait not in table.index:
            return {}
        row = table.loc[trait]
        return {
            "total": int(row.total),
            "wins": int(row.wins),
            "losses": int(row.losses),
            "win_rate": round(row.win_rate, 3),
            "top4_lift": round(row.top4_lift, 3),
        }


# 6. Output ---------------------------------------------


def _demo():
    df = load_data()
    helper = TraitHelper(df)

    print("Top 10 traits by top‑4 lift:")
    print(trait_win_table(df).sort_values("top4_lift", ascending=False).head(10))

    print("\nMost synergistic pairs:")
    for a, b, obs, lift in top_synergy_pairs(df, 10):
        print(f"{a} + {b}: lift={lift:.2f} (obs={obs})")

    sample_combo = {"Set8_Ace", "Set8_Sureshot"}
    best_match, dist = helper.find_most_similar(sample_combo)
    print(f"\nMost similar to {sample_combo}: {best_match} (cos‑dist {dist:.3f})")

    path = helper.shortest_trait_path("Set8_Ace", "Set8_AnimaSquad")
    print(f"\nShortest trait path Ace → AnimaSquad: {path}")

    mc_trait, degree = helper.most_connected_trait()
    print(f"\nMost connected trait: {mc_trait} (co‑occurs with {degree} others)")

    print("\nStats for Set8_Ace:")
    print(helper.trait_stats("Set8_Ace"))


if __name__ == "__main__":
    _demo()
