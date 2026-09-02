# `falsification/`

Sequential falsification with e-values, for validating claims about sparse
autoencoders. Follows POPPER (Huang et al., 2025); see `../PROJECT.md` for the
research framing.

```
evalues.py       p-to-e calibration, sequential aggregation, implication check
permutation.py   the four p-value sources, each scoped to a claim type
worked_example.py  applies the framework to data already in this repo
tests/           correctness tests, including empirical Type-I control
```

## Quick start

```python
from falsification.evalues import FalsificationTest, SequentialFalsifier
from falsification.permutation import seed_permutation_test

f = SequentialFalsifier(
    main_hypothesis="vSAE learns a better-organised feature space than the SAE",
    alpha=0.1,
)

res = seed_permutation_test(vsae_scores_per_seed, sae_scores_per_seed)
decision = f.add(FalsificationTest(
    name="feature-independence, size-matched",
    null_hypothesis="vSAE is no better than SAE",
    alt_hypothesis="vSAE is better than SAE",
    p_value=res["p_value"],
    unit_of_analysis=res["unit_of_analysis"],
    n_units=res["n_units"],
    confounders_controlled=("live feature count", "L0"),
))
print(f.report())
```

## The two rules that matter

**Match the test to the claim's unit of analysis.** A claim about an *architecture*
needs a permutation test over *training seeds*. Token-level tests describe two
specific checkpoints; with a million eval tokens they report p ≈ 1e-300 for
differences that are pure seed noise. `paired_token_test` raises unless you
explicitly acknowledge the narrower scope.

| function | unit | supports a claim about |
|---|---|---|
| `seed_permutation_test` | training run | an architecture |
| `monotone_trend_test` | sweep condition | a dose-response (1 run/condition) |
| `subsample_null_test` | random sub-dictionary | a score, controlling dictionary size |
| `paired_token_test` | eval token | two specific checkpoints |

**Declare confounders.** A test whose sub-null is not implied by the main null
carries no evidence about the main hypothesis, however small its p-value. Record
them on `FalsificationTest`; `SequentialFalsifier` excludes such tests from the
product rather than down-weighting them, because the assumption is binary. This is
the check that catches the preprint's SCR/TPP error — run `worked_example.py`.

## Reading the output

`E` is the aggregated evidence, `1/α` the threshold. Because `E` is a
supermartingale under the null, you may stop whenever you like — that is the whole
point, and it is what Fisher's combined test cannot do
(`test_fisher_combination_would_fail_where_e_values_hold` demonstrates the
inflation).

Always read `p_floor` alongside `p_value`. It is the smallest p-value the design
could produce, and if it is above what you need, the experiment was underpowered
before it ran. `seeds_required(alpha, kappa, n_tests)` inverts this into a run budget.

## Caveats

- E-values require *conditionally valid* p-values: choose the next test without
  looking at the data it will be computed on.
- Type-I control is over the randomness the p-values model. Seed-permutation tests
  control error over training seeds; they say nothing about bias shared by every
  run (a buggy trainer, a mislabelled hook point).
- Non-rejection is not evidence of a null. For equivalence claims (E1 in
  `PROJECT.md`) use an equivalence test with a pre-specified margin.
