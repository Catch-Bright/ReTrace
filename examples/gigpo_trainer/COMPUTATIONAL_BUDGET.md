# Computational budget

Enable with `algorithm.gigpo.enable_time_breakdown=True`. The ALFWorld 1.5B
WiGPO script already enables it; pass the same override to other launch scripts.
All `time_breakdown/*` metrics are **wall-clock seconds per training iteration**,
not per-token times or summed GPU-seconds.

## Shared components

The model-stage entries are exact aliases of the existing GiGPO trainer timers.
Their scopes are not narrowed to GPU kernels or expanded to absorb WiGPO work.

| Metric | Existing timer / boundary |
| --- | --- |
| `Rollout` | `timing_s/gen`: the entire `multi_turn_loop`, including environment interactions, trajectory collection, and worker calls |
| `OldProb` | `timing_s/old_log_prob`: old-policy scoring plus the existing entropy/probability diagnostics and batch union |
| `RefProb` | `timing_s/ref`: reference scoring and batch union; zero if disabled |
| `Update` | `timing_s/update_actor`: the complete actor worker call, all minibatches and PPO epochs, backward, optimizer, communication and configured offloading; zero if skipped |
| `AE` | Exactly `core_gigpo.episode_norm_reward(...)`, including token broadcast, in both GiGPO and WiGPO |

GRPO also records `AE` around its own episode advantage routine. GiGPO additionally
records `Grouping` and `A_step` around its original grouping and step advantage
functions. The paper's Figure 6 defines the component categories but does not
specify every instrumentation boundary; matching these repository scopes does
not establish identical instrumentation to the authors' measurement run.

## WiGPO components

`inter-step` accumulates disjoint intervals for:

- Constructing stage metadata from rollout milestone flags (or static anchors).
- Computing the configured shaped reward, including `reverse_decay_from_step_split`.
  When shaping is disabled, computing the discounted returns actually used instead.
- Anchor-state grouping and intersection with stage IDs.
- The original state-group advantage only when `step_advantage_w != 0`.
- Plain subgroup normalization when action aggregation is disabled.

Online milestone detection inside environment workers remains in `Rollout`.
It must not also be added to `inter-step`: doing so would overlap the unchanged
GiGPO rollout timer. Thus `inter-step` measures the post-rollout stage/reward work,
not every possible milestone-related operation in the system.

`intra-step` accumulates disjoint intervals for:

- Canonical action keys and think/action token-boundary detection.
- Reducing existing OldProb action-token log probabilities to a probability score.
- Aggregating action rewards, calculating subgroup baselines, probability ratios,
  clipping, fallback advantages and reweighted think advantages.
- Broadcasting and assigning think/action advantages to response tokens.

There is no additional model forward in intra-step. OldProb scores each sampled
prompt + think + action sequence before the actor update. Its action-token scores
already condition on that sequence's think and preceding action tokens. WiGPO
uses `exp(mean(action_log_probs) / T)`, a length-normalized probability score,
not the unnormalized probability of the entire action string. The action mask
currently covers the opening action tag through the valid response end.

Profiling disables verbose grouping/mask summaries; rho diagnostic masks, counts
and percentiles are computed outside the intra-step interval and remain logged.
Discarded discounted returns and zero-weight advantage branches are skipped,
rather than executed and hidden from the budget. Grouping needed by active
subgroups is retained even when the original step advantage has zero weight.
Final addition of the episode/step/subgroup terms is accounted for in `Other`.

## Totals and reporting

- `MeasuredTotal`: sum of the seven components above (plus `Grouping`/`A_step`
  when profiling the GiGPO baseline).
- `TrainTotal`: independently measured from immediately before rollout through
  actor-update metric collection, before validation, checkpointing and generation
  dumps. It includes reward processing, batch balancing, diagnostics and other
  driver work between rollout and update.
- `Other = TrainTotal - MeasuredTotal`: remaining work, not silently charged to
  inter-step or intra-step. It is not clamped, so accounting errors stay visible.

For WiGPO, `TrainTotal = Rollout + OldProb + RefProb + Update + AE + inter-step +
intra-step + Other`. Do not add `timing_s/adv` or `timing_s/step` to these numbers:
they are enclosing timers. `TrainTotal` excludes data loading and initial batch
construction, end-of-iteration metric computation/logger writes, evaluation and
checkpoint saving; it is a training-core measurement, not total job duration.
With asynchronous reward computation, model-stage timers include any overlapping
work naturally; do not add remote reward worker runtimes again.

Compare the same hardware, model, rollout group size, PPO epochs, microbatch
settings, gradient checkpointing and offloading. Report actual trajectory steps
and prompt/response token counts too. Update trains on all collected steps and
can exceed rollout even with one PPO epoch. Exclude warmup iterations and report
the mean and variation over multiple iterations; five iterations are insufficient
for a stable long-horizon timing estimate. Existing validation can stay enabled
because it does not enter `TrainTotal`, though it can affect subsequent cache state.
