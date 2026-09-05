# What this project is, in plain terms

## The problem we're circling

Interpretability researchers routinely claim things like "architecture A learns
better features than architecture B." In practice these claims get checked
informally: train both, run a benchmark suite, eyeball the numbers, write a
sentence. We wrote a preprint that did exactly this, comparing a "variational"
sparse autoencoder (a vSAE — an autoencoder that's supposed to learn a
probability distribution over each feature's activation, sample from it, and
use a KL-divergence penalty to keep that distribution well-behaved) against a
standard sparse autoencoder. We found two problems with our own process after
the fact, and they're common enough elsewhere that they're worth writing up on
their own:

- **We had no rule for combining evidence.** We ran several different
  benchmarks, and in one section of the paper we concluded a hypothesis was
  confirmed, and in another section we concluded the same hypothesis was
  rejected. Nothing in our method could have caught that contradiction, because
  we never wrote down a rule for what "the evidence, combined, says."
- **We drew conclusions the data didn't license.** The vSAE scored better on a
  couple of benchmarks that reward being able to selectively knock out
  individual features. But those benchmarks are easier to win when fewer of
  your features are "alive" in the first place — and our vSAE only had 18% of
  its features alive, versus 90% for the baseline. So the benchmark win was
  plausibly just a side effect of having a sparser, smaller effective
  dictionary, not evidence of genuinely better-organized features. We hadn't
  checked for that before writing it up as a finding.

So the project split into two related pieces of work: fixing our statistical
process, and using that fixed process to re-examine our own preprint's claims
line by line. The second piece turned out to be much more consequential than
we expected, because checking the code against the claims surfaced real bugs,
not just statistical sloppiness.

## What we built: a falsification framework

We adapted an approach from a recent paper (POPPER) that combines two ideas:
first, before you even run a test, you check whether a "yes" on that test would
actually be *implied* by the hypothesis you care about (this is what caught the
selectivity-benchmark problem above — selectivity scores are not implied by
"better organized features" once live-feature count differs between the two
models). Second, you accumulate evidence across multiple tests using
"e-values," a form of statistical evidence that can be combined sequentially
without inflating your false-positive rate, even if you decide to run more
tests after seeing earlier results.

The unit of randomness in an SAE comparison is the training seed — you have to
train several independent copies of each model with different random seeds and
compare the *distributions* of results, not compare two single runs. This
sounds simple but has a nasty consequence: with permutation tests (the
statistically honest way to compare small numbers of seeds), there's a hard
floor on how significant a result can ever look, no matter how large the true
effect is, if you only have a handful of seeds per group. Six seeds per group
literally cannot produce a "5-sigma" result, full stop, regardless of effect
size. We ran simulations to nail down exactly how many seeds we'd need for
different effect sizes, and settled on 13 seeds per arm as enough to reach
5-sigma confidence on the comparisons we cared about. We also had to run
"pipeline negative controls" — split identical, same-config runs into two
arbitrary fake groups and make sure the framework doesn't falsely find a
difference between them — to make sure the whole measurement apparatus is
trustworthy before trusting any real result out of it.

This framework itself (the statistical machinery — permutation tests,
e-values, the implication checker) is one of the two deliverables of this
project, independent of anything we conclude about vSAEs specifically.

## What we found when we pointed it at our own preprint

We designed four planned experiments before looking at any seeded results,
specifically to stress-test the preprint's central claims. Running them
surfaced a chain of real implementation bugs, several of which reversed
conclusions we'd already published.

**The variational SAE, as implemented, isn't actually variational — or wasn't,
where we'd been evaluating it.** There's a flag in the code that turns sampling
on or off. Every checkpoint we'd been analyzing had sampling *off*, meaning the
model was fully deterministic. With sampling off and the noise variance fixed
at 1, the "KL penalty" that's supposed to regularize a learned probability
distribution mathematically collapses into a plain L2 penalty on the
activations. So every "vSAE" result we'd published up to that point was
actually just a description of a TopK sparse autoencoder with an extra L2
penalty — nothing variational about it at all.

**Once we compared that L2-penalized model against a plain TopK baseline
properly, the two were indistinguishable — but only after we found and matched
five separate implementation differences that had nothing to do with the
actual math.** We read the two pieces of training code line by line and found
things like: a different scale used to initialize the decoder weights, a
gradient-projection step that one implementation applied and the other didn't,
and even the initial random weight distribution (uniform vs. Gaussian) being
different between the two files. Each of these individually caused a big,
statistically clear difference in results — bigger than a lot of published
architectural claims in this field — and they partially canceled each other
out in confusing ways as we found and fixed them one at a time. Once literally
every difference in the code was accounted for, the two models became
statistically indistinguishable. The conclusion here is double-edged: the
underlying math really is identical (we verified this to six decimal places),
but two "identical" implementations trained completely differently until every
incidental engineering detail was matched — which is itself an important and
under-reported finding, since none of those details appear in either paper's
equations.

**Turning sampling actually on revealed something real and important: sampling
noise is genuinely destructive to this kind of model, and it's not really
about the strength of the KL penalty.** We tested strengths of the KL penalty
across four orders of magnitude and found that none of them got a truly
sampling-enabled model anywhere close to the baseline's reconstruction
quality. Then we ran the decisive control: sampling on, KL penalty entirely
off. That recovered only about 5% of the performance gap. So roughly 95% of
the damage was coming from the act of sampling itself, not from the
regularization term. That's a more interesting and more architectural finding
than the one we'd originally set out to test.

**We initially misread why sampling was so damaging, because of a bug that
corrupted how we were reading the model's internal state.** There's a save-time
rescaling step in the training code that correctly adjusts most of the model's
bias parameters when it saves a checkpoint, but it was incorrectly applying
that same rescaling to the parameter controlling the noise variance — a
parameter for which that kind of rescaling isn't even mathematically valid.
The effect was that every saved checkpoint's internal noise-variance parameter
*looked* like it had collapsed all the way down to the numerical clamp we'd
put in place, which we initially reported as "the model learns to turn its own
noise off, so sampling is harmless at evaluation time." Once we found and fixed
the bug and re-read the checkpoints correctly, both halves of that reversed:
the noise parameter had settled at a moderate, non-collapsed value the whole
time, and re-measuring showed that leaving sampling on at evaluation time
costs a real, non-trivial amount of reconstruction quality. This bug is now
fixed for future training runs, but every checkpoint already saved before the
fix needs a small correction applied by hand before its internals can be
trusted, since the raw saved numbers are wrong.

**We also found that initializing the noise level very low (rather than
letting it start at a typical starting value) recovers most — about 84% — of
the performance gap caused by sampling**, which fits the story above: the
damage is worst early in training, when the "real" signal in each feature is
still small relative to the injected noise, so a smaller starting noise level
gives the model a much better chance to establish good features before noise
can scramble that process. A real, smaller gap remains even with the best
initialization we tried, and we tracked that remaining gap to instability in
which features get selected as "top-k" from one forward pass to the next —
under sampling noise, the model's choice of active features churns
significantly even at the end of training, whereas with the better
initialization it stabilizes early and stays stable. This supports a broader
hypothesis we're interested in: that sampling-based regularization and
"hard," discrete, winner-take-all feature selection (like top-k) are
fundamentally in tension with each other, independent of anything about sparse
autoencoders specifically — sampling noise near a hard decision boundary
naturally causes the boundary to flip back and forth. We haven't yet tested
whether that tension shows up less with "softer" selection mechanisms, which
is the natural next experiment.

**A separate confound we caught but haven't yet fixed**: the headline
comparison in the original preprint gave the baseline model a standard
dead-feature revival mechanism and didn't give the same mechanism to the vSAE.
That mechanism exists specifically to prevent features from going permanently
inactive during training, so that comparison was partly measuring "has a
dead-feature fix" versus "doesn't," not "vSAE" versus "baseline." It doesn't
affect the other comparisons we've run, which held that setting fixed on both
sides.

**One planned experiment turned out to be blocked, not just undone**: we
wanted to check whether the vSAE's apparent advantage on a couple of specific
benchmarks was really just because it had a much smaller number of "alive"
features (fewer live features makes certain kinds of surgical benchmarks
easier to win almost automatically, regardless of feature quality). Answering
that properly requires the actual trained weights from the original preprint's
larger-scale runs on a different, bigger language model — and it turns out
those particular trained model files were never saved anywhere we can find on
this machine. Only the derived analysis outputs survive. So this check either
needs a substantially larger new training run at the original scale, or
finding the original weights somewhere else entirely.

## Where this leaves the paper's original claims

The honest updated picture: the specific numeric comparisons in the published
preprint don't hold up once implementation bugs and mismatches are corrected —
several of its measurements were literally reading corrupted numbers out of
checkpoints, or comparing two things that differed in ways the equations never
described. But underneath all of that, a real and more interesting scientific
result emerged: injecting sampling noise into a model that also does hard,
discrete top-k feature selection is genuinely destructive, largely independent
of how strong the accompanying regularization penalty is, and the mechanism
appears to be that noise makes the discrete selection unstable. That's a
sharper and more general claim than anything in the original draft, and it
survived being checked against the code — which several of the original claims
did not.

## What's left to do

The most pressing item is mechanical: several officially-reported numbers for
the sampling-enabled models were computed from checkpoints affected by the
noise-variance bug, using the framework's standard evaluation pipeline rather
than the quick corrected estimate we used to spot the problem. Those need to
be properly recomputed on the existing checkpoints — no new training required,
just re-reading them correctly.

Beyond that, the interesting open scientific question is whether the
noise-versus-hard-selection tension we found is specific to top-k selection or
is a general property of combining sampling with any kind of discrete,
winner-take-all feature choice. Testing that means running the same
sampling-on-vs-off comparison against a couple of other sparse autoencoder
variants that use different, less rigidly discrete selection mechanisms. That
would turn a finding about one architecture into a finding about a whole class
of architectures — arguably the most valuable open thread here. There's also
some pure desk work worth doing with no new training at all: writing up how
few training seeds are typical in published SAE papers and what statistical
ceiling that implies for their claims, which turns our sample-size argument
from a methodological aside into a measured, citable fact about the field.
