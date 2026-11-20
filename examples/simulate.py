import numpy as np

from groupval import plan_gold, validate

rng = np.random.default_rng(1)
n = 2000
group = np.where(rng.integers(0, 2, n) == 0, "D", "R")
gold = (rng.random(n) < 0.95).astype(float)
miss = np.where(group == "R", 0.18, 0.10)
labels = np.where((gold == 1) & (rng.random(n) < miss), 0.0, gold)

report = validate(labels, gold, group, reference="D")
print(report.per_group.round(3))
print(f"pooled accuracy {report.pooled_accuracy:.3f}")
print(f"bias in the D minus R gap {100 * report.bias:.1f} points, 90% interval {100 * report.interval[0]:.1f} to {100 * report.interval[1]:.1f}")
v = report.variance_by_group
print("gold labels per group for +/- 3 points:", plan_gold(v["D"], v["R"], 0.03))
