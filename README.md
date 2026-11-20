# groupval

Validate language model labels within each group before comparing groups.

Language model labels can be accurate overall and still err more on one group's texts, which biases any comparison between groups. On the congressional tweets of Gilardi, Alizadeh and Kubli (2023), coders with accuracy near 0.87 overstate the Democrat minus Republican gap in political content by more than four times, because they miss more Republican political tweets. `groupval` reports sensitivity, specificity and error for each group, the bias in the between-group gap with an interval clustered by author, and the number of expert labels each group needs to estimate that bias to a chosen precision.

## Install

```
pip install groupval
```

## Use

```python
from groupval import validate, plan_gold

r = validate(labels, gold, group, cluster=author, reference="D")
r.per_group
r.bias, r.interval
plan_gold(r.variance_by_group["D"], r.variance_by_group["R"], halfwidth=0.03)
```

`labels` and `gold` are 0/1 arrays on the expert-labeled items, `group` holds the two groups being compared and `cluster` the author or account. `python examples/simulate.py` runs a small simulated case.

## Reference

Crainic, J., and Yee, B. Accurate Overall, Biased Between Parties: Validating Language Model Labels Within Each Group. Extended abstract, 2026.

## License

MIT
