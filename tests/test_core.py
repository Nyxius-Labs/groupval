import numpy as np

from groupval import plan_gold, validate


def test_no_bias_when_labels_equal_gold():
    rng = np.random.default_rng(0)
    gold = (rng.random(500) < 0.5).astype(float)
    group = np.repeat(["a", "b"], 250)
    r = validate(gold, gold, group, draws=200)
    assert r.bias == 0
    assert r.pooled_accuracy == 1


def test_detects_group_differential_misses():
    rng = np.random.default_rng(1)
    n = 4000
    group = np.where(rng.integers(0, 2, n) == 0, "D", "R")
    gold = np.ones(n)
    labels = np.where((group == "R") & (rng.random(n) < 0.2), 0.0, 1.0)
    r = validate(labels, gold, group, reference="D", draws=300)
    assert 0.15 < r.bias < 0.25
    assert r.interval[0] > 0


def test_plan_gold_scales_with_precision():
    assert plan_gold(0.1, 0.1, 0.02) > plan_gold(0.1, 0.1, 0.04)


def test_allocate_gold():
    from groupval import allocate_gold, plan_gold
    a, b = allocate_gold(0.04, 0.16, 0.03)
    assert b > a and abs(b / a - 2) < 0.05
    e = allocate_gold(0.09, 0.09, 0.03)
    assert e[0] == e[1] and abs(e[0] - plan_gold(0.09, 0.09, 0.03)) <= 1
    assert sum(allocate_gold(0.04, 0.16, 0.03)) <= 2 * plan_gold(0.04, 0.16, 0.03)
