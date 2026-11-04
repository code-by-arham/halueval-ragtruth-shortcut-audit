"""Stratified random CovidQA subsample used for MiniCheck (100 PASS + 100 FAIL, seed 11).

CovidQA passages average ~2,600 words (~9 MiniCheck chunks per claim), so scoring all
1,000 rows takes ~5 h on 2 CPU threads. The subsample is drawn once, independently of
any model score: for each label, 100 rows are chosen without replacement with
numpy.random.default_rng(11) and kept in their original file order.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--records", type=Path, default=Path("halubench/data/halubench_records.csv"))
ap.add_argument("--output", type=Path, default=Path("halubench/data/halubench_covidqa_subsample200.csv"))
ap.add_argument("--per-label", type=int, default=100)
ap.add_argument("--seed", type=int, default=11)
a = ap.parse_args()

r = pd.read_csv(a.records, keep_default_na=False)
c = r[r.source_ds == "covidQA"]
rng = np.random.default_rng(a.seed)
idx = np.concatenate([np.sort(rng.choice(c.index[c.label == lab].to_numpy(), a.per_label, replace=False)) for lab in (0, 1)])
s = c.loc[idx].reset_index(drop=True)
s.to_csv(a.output, index=False)
print(len(s), "rows,", s.question_id.nunique(), "passages,", s.label.value_counts().to_dict())
