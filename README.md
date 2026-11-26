# groupval

Validate language model labels within each group before comparing groups.

Language model labels can be accurate overall and still err more on one group's texts, which biases any comparison between groups. On the congressional tweets of Gilardi, Alizadeh and Kubli (2023), Qwen 2.5 7B and Llama 3.2 3B, with pooled accuracies of 87 and 83 percent, report a Democrat minus Republican gap in political content of about 9 points, about four times the expert gap of 2.2 points, because they miss more Republican political tweets. `groupval` reports sensitivity, specificity and error for each group, the bias in the between-group gap with an interval clustered by author, and the number of expert labels each group needs to estimate that bias to a chosen precision.

## Install

```
pip install groupval
```

## Use

```python
from groupval import allocate_gold, plan_gold, validate

r = validate(labels, gold, group, cluster=author, reference="D")
r.per_group
r.bias, r.interval
plan_gold(r.variance_by_group["D"], r.variance_by_group["R"], halfwidth=0.03)
allocate_gold(r.variance_by_group["D"], r.variance_by_group["R"], halfwidth=0.03)
```

`labels` and `gold` are 0/1 arrays on the expert-labeled items, `group` holds the two groups being compared and `cluster` the author or account. `plan_gold` gives the expert labels per group under an equal split. `allocate_gold` gives the split that minimizes the variance of the group gap for the same precision, with each group's share proportional to the standard deviation of the model's error in that group. `python examples/simulate.py` runs a small simulated case.

## Reference

Crainic, J., and Yee, B. Validating Language Model Labels Within Each Group: Party Gaps in Congressional Tweets. Extended abstract, 2026.

## License

MIT
