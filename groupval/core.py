from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm


@dataclass
class GroupReport:
    per_group: pd.DataFrame
    gap_labels: float
    gap_gold: float
    bias: float
    interval: tuple
    pooled_accuracy: float
    variance_by_group: dict


def validate(labels, gold, group, cluster=None, reference=None, draws=2000, level=0.90, seed=0):
    d = pd.DataFrame({"c": np.asarray(labels, float), "y": np.asarray(gold, float), "g": np.asarray(group)})
    d["k"] = np.arange(len(d)) if cluster is None else np.asarray(cluster)
    d = d.dropna(subset=["c", "y"])
    groups = sorted(d.g.unique())
    if len(groups) != 2:
        raise ValueError("validate compares exactly two groups")
    a = groups[0] if reference is None else reference
    b = [g for g in groups if g != a][0]
    rows = []
    for g in (a, b):
        s = d[d.g == g]
        rows.append(dict(group=g, n=len(s), clusters=s.k.nunique(), gold_share=s.y.mean(), label_share=s.c.mean(),
                         sensitivity=s[s.y == 1].c.mean(),
                         specificity=1 - s[s.y == 0].c.mean() if (s.y == 0).any() else np.nan,
                         error=(s.c != s.y).mean()))
    per = pd.DataFrame(rows).set_index("group")
    gap_labels = per.label_share[a] - per.label_share[b]
    gap_gold = per.gold_share[a] - per.gold_share[b]
    rng = np.random.default_rng(seed)
    blocks = {g: [x for _, x in d[d.g == g].groupby("k")] for g in (a, b)}
    stats = []
    for _ in range(draws):
        m = {g: pd.concat([blocks[g][i] for i in rng.integers(0, len(blocks[g]), len(blocks[g]))]) for g in (a, b)}
        stats.append((m[a].c.mean() - m[b].c.mean()) - (m[a].y.mean() - m[b].y.mean()))
    lo, hi = np.percentile(stats, [50 * (1 - level), 100 - 50 * (1 - level)])
    var = {g: float(np.var(d[d.g == g].c - d[d.g == g].y)) for g in (a, b)}
    return GroupReport(per, gap_labels, gap_gold, gap_labels - gap_gold, (lo, hi), float((d.c == d.y).mean()), var)


def plan_gold(variance_a, variance_b, halfwidth, level=0.90, design_effect=1.0):
    z = norm.ppf(0.5 + level / 2)
    return int(np.ceil(design_effect * z ** 2 * (variance_a + variance_b) / halfwidth ** 2))


def allocate_gold(variance_a, variance_b, halfwidth, level=0.90, design_effect=1.0):
    z = norm.ppf(0.5 + level / 2)
    sa, sb = np.sqrt(variance_a), np.sqrt(variance_b)
    total = design_effect * z ** 2 * (sa + sb) ** 2 / halfwidth ** 2
    if sa + sb == 0:
        return 0, 0
    return int(np.ceil(total * sa / (sa + sb))), int(np.ceil(total * sb / (sa + sb)))
