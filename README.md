# SI507-Final-Project

# 1 What’s inside?
This project mines high-level Teamfight Tactics (Set 8 / Patch 13.4-13.5) match data from Philippine Challenger & Grand-Master players, then  asks two practical questions. First, which individual traits and units are most closely associated with success, defined here as finishing in the top four (“wins”) rather than the bottom four (“losses”)? Second, do especially strong trait pairs emerge automatically from match data or is winning usually the result of blending many small packages? The project uses a single script (TFT_Trait_Synergy.py).

# 2 Data Source
Raw CSVs:
unprocessed_challenger_match_data.csv
unprocessed_gm_match_data.csv

These were produced by an earlier study (rndmagtanong’s project “ph_tft”, link: https://github.com/rndmagtanong/ph_tft/tree/master) that hits the Riot Developer API, pulls the last 50 ranked matches for every PH Challenger/GM+ player (March 16 2023 snapshot), then flattens & stores one row per player-match. 
Trait names follow Riot’s internal identifiers; see the full Set 13 list at https://lolchess.gg/synergies/set13.

# 3 Setup
(1) clone / download the repo
(2) – create & activate Python 3.10+ virtual env
(3) install requirements
pip install pandas numpy scikit-learn matplotlib


# 4 Running the analysis
python "TFT Investigation.py"
Top 10 traits by top‑4 lift:
              trait  total  wins  losses  win_rate  top4_lift
16   Set8_Corrupted   2371  1563     808  0.659216   1.934406
27  Set8_Forecaster   1228   792     436  0.644951   1.816514
23     Set8_Arsenal   1640  1048     592  0.639024   1.770270
26    Set8_Civilian   2088  1181     907  0.565613   1.302095
4     Set8_Renegade   2216  1253     963  0.565433   1.301142
22    Set8_ExoPrime   2299  1294    1005  0.562853   1.287562
7       Set8_Threat   4699  2592    2107  0.551607   1.230185
24     Set8_Deadeye   2606  1428    1178  0.547966   1.212224
18         Set8_Ace   2986  1635    1351  0.547555   1.210215
0    Set8_Channeler   3221  1708    1513  0.530270   1.128883

Most synergistic pairs:
/Users/chenwen/Downloads/TFT Investigation.py:142: FutureWarning: Series.__getitem__ treating keys as positions is deprecated. In a future version, integer keys will always be treated as labels (consistent with DataFrame behavior). To access a value by position, use `ser.iloc[pos]`
  exp = (totals[i] * totals[j]) / total_games
Set8_Duelist + Set8_Recon: lift=0.02 (obs=866)
Set8_AnimaSquad + Set8_Recon: lift=0.02 (obs=764)
Set8_Civilian + Set8_Forecaster: lift=0.02 (obs=792)
Set8_Heart + Set8_Supers: lift=0.02 (obs=613)
Set8_Arsenal + Set8_Deadeye: lift=0.02 (obs=1048)
Set8_AnimaSquad + Set8_Duelist: lift=0.02 (obs=883)
Set8_Admin + Set8_Hacker: lift=0.01 (obs=788)
Set8_Admin + Set8_Heart: lift=0.01 (obs=822)
Set8_GenAE + Set8_Hacker: lift=0.01 (obs=993)
Set8_Duelist + Set8_Supers: lift=0.01 (obs=437)

Most similar to {'Set8_Sureshot', 'Set8_Ace'}: {'', 'Set8_Threat', 'Set8_Ace'} (cos‑dist 0.423)

Shortest trait path Ace → AnimaSquad: ['Set8_Ace', 'Set8_AnimaSquad']

Most connected trait: Set8_Channeler (co‑occurs with 27 others)

Stats for Set8_Ace:
{'total': 2986, 'wins': 1635, 'losses': 1351, 'win_rate': np.float64(0.548), 'top4_lift': np.float64(1.21)}

  
# 5 How it works 
Load & merge the two raw CSVs, drop redundant cosmetic columns.

Trait/Unit Lift

For every trait and unit, compute
top4_lift = (win_rate ÷ overall_top4_rate)
to highlight over-performers.

Pairwise synergy search

Build a co-occurrence matrix for traits in top-4 boards.

Calculate pairwise lift vs expectation under independence and list the top 10.

# 6 Interactive helpers:

find_most_similar: cosine distance between the user’s trait set and every board in data.

shortest_trait_path: BFS over a trait-graph where edge weight = inverse co-occurrence frequency.

most_connected_trait: counts how many different traits each trait appears with in top-4 boards.

trait_stats: quick per-trait counts, win-rate & lift.

