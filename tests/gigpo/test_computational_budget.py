"""Budget boundaries must not change credit assignment or count diagnostics."""

import numpy as np
import pytest
import torch

from gigpo import budget, core_gigpo, core_wigpo


@pytest.fixture
def inputs():
    mask = torch.tensor([[1., 1., 1., 0.]] * 8)
    return dict(
        token_level_rewards=torch.tensor([[0., 0., float(i % 3), 0.] for i in range(8)]),
        step_rewards=torch.tensor([1., 3., 0., 2., 5., 7., -1., 4.]),
        response_mask=mask,
        anchor_obs=np.array(['x'] * 6 + ['y'] * 2, dtype=object),
        segment_ids=np.array([0] * 4 + [1] * 4),
        index=np.array(['task'] * 8, dtype=object),
        traj_index=np.array([f'traj{i}' for i in range(8)], dtype=object),
        action_keys=np.array(['a', 'a', 'b', 'b', 'a', 'a', core_wigpo.INVALID_ACTION_KEY, 'b'], dtype=object),
        action_token_mask=torch.tensor([[0., 0., 1., 0.]] * 8),
        located=np.array([True] * 7 + [False]),
        old_log_probs=-torch.arange(1., 33.).reshape(8, 4) / 10,
    )


def legacy_advantages(inputs, mode, step_weight, subgroup_weight):
    """Original three-term computation, including terms with zero weight."""
    x = inputs
    remove_std = mode == 'mean_norm'
    episode = core_gigpo.episode_norm_reward(
        x['token_level_rewards'], x['response_mask'], x['index'], x['traj_index'],
        remove_std=remove_std,
    )
    groups = core_gigpo.build_step_group(x['anchor_obs'], x['index'])
    step = core_gigpo.step_norm_reward(x['step_rewards'], x['response_mask'], groups, remove_std=remove_std)
    subgroups = core_wigpo.build_subgroup_uids(groups, x['segment_ids'])
    if x['action_keys'] is None:
        sub = core_gigpo.step_norm_reward(x['step_rewards'], x['response_mask'], subgroups, remove_std=remove_std)
    elif x['old_log_probs'] is None:
        rewards = core_wigpo.aggregate_action_group_rewards(x['step_rewards'], subgroups, x['action_keys'], summarize=False)
        sub = core_gigpo.step_norm_reward(rewards, x['response_mask'], subgroups, remove_std=remove_std, std_rewards=x['step_rewards'])
    else:
        probs = core_wigpo.compute_action_probs(x['old_log_probs'], x['action_token_mask'], x['response_mask'])
        think, act, _ = core_wigpo.compute_prob_weighted_think_action_advantages(
            x['step_rewards'], subgroups, x['action_keys'], probs, x['located'],
            remove_std=remove_std, summarize=False,
        )
        sub = (act[:, None] * x['action_token_mask'] + think[:, None] * (1 - x['action_token_mask'])) * x['response_mask']
    return episode + step_weight * step + subgroup_weight * sub


@pytest.mark.parametrize('mode', ['mean_norm', 'mean_std_norm'])
@pytest.mark.parametrize('branch', ['plain', 'aggregate', 'probability'])
@pytest.mark.parametrize('weights', [(0., 1.), (0.5, 1.), (1., 0.), (0., 0.)])
def test_profiling_and_skipping_unused_terms_preserve_advantages(inputs, mode, branch, weights):
    if branch == 'plain':
        inputs['action_keys'] = None
    if branch != 'probability':
        inputs['old_log_probs'] = None
    expected = legacy_advantages(inputs, mode, *weights)
    timings = {}
    actual, returns, _ = core_wigpo.compute_wigpo_outcome_advantage(
        **inputs, mode=mode, step_advantage_w=weights[0], subgroup_advantage_w=weights[1],
        enable_time_breakdown=True, time_breakdown=timings,
    )
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(returns, expected)
    assert timings['AE'] >= 0
    if weights == (0., 0.):
        assert set(timings) == {'AE'}
    elif branch == 'plain' or weights[1] == 0:
        assert 'intra-step' not in timings
    else:
        assert timings['intra-step'] >= 0


def test_zero_step_weight_does_not_evaluate_discarded_step_advantage(inputs, monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail('zero-weight GiGPO step advantage was evaluated')
    monkeypatch.setattr(core_gigpo, 'step_norm_reward', fail)
    core_wigpo.compute_wigpo_outcome_advantage(**inputs, step_advantage_w=0., enable_time_breakdown=True)


def test_diagnostic_work_is_outside_intra_timer(inputs, monkeypatch):
    clock = [0.]
    monkeypatch.setattr(core_wigpo, 'perf_counter', lambda: clock[0])
    percentile = np.percentile

    def costly_diagnostic(*args, **kwargs):
        clock[0] += 100.
        return percentile(*args, **kwargs)

    monkeypatch.setattr(np, 'percentile', costly_diagnostic)
    timings = {}
    _, _, stats = core_wigpo.compute_prob_weighted_think_action_advantages(
        inputs['step_rewards'], inputs['segment_ids'], inputs['action_keys'],
        torch.linspace(.1, .8, 8), inputs['located'], summarize=False, time_breakdown=timings,
    )
    assert stats['n_split'] > 0
    assert clock[0] == 100.
    assert timings['intra-step'] == 0.


def test_common_timers_are_exact_aliases_and_totals_do_not_double_count():
    raw = dict(gen=10., old_log_prob=2., ref=3., update_actor=20., adv=999.,
               step=2000., testing=1000., save_checkpoint=500.)
    algorithm = {'AE': .1, 'inter-step': .2, 'intra-step': .3}
    result = budget.budget_metrics(raw, algorithm, train_total=40.)
    for name, original in [('Rollout', 'gen'), ('OldProb', 'old_log_prob'), ('RefProb', 'ref'), ('Update', 'update_actor')]:
        assert result[f'time_breakdown/{name}'] == raw[original]
    assert result['time_breakdown/MeasuredTotal'] == pytest.approx(35.6)
    assert result['time_breakdown/Other'] == pytest.approx(4.4)
    assert result['time_breakdown/TrainTotal'] == 40.


def test_gigpo_ae_uses_same_function_boundary(inputs, monkeypatch):
    clock = [0.]
    monkeypatch.setattr(budget, 'perf_counter', lambda: clock[0])
    original = core_gigpo.episode_norm_reward

    def episode(*args, **kwargs):
        clock[0] += 7.
        return original(*args, **kwargs)

    monkeypatch.setattr(core_gigpo, 'episode_norm_reward', episode)
    gigpo_times, wigpo_times = {}, {}
    core_gigpo.compute_gigpo_outcome_advantage(
        **{k: inputs[k] for k in ('token_level_rewards', 'step_rewards', 'response_mask', 'anchor_obs', 'index', 'traj_index')},
        time_breakdown=gigpo_times,
    )
    core_wigpo.compute_wigpo_outcome_advantage(
        **inputs, step_advantage_w=0., subgroup_advantage_w=0., time_breakdown=wigpo_times,
    )
    assert gigpo_times['AE'] == wigpo_times['AE'] == 7.


def test_trainer_counts_action_preparation_and_skips_disabled_intra(inputs, monkeypatch):
    from verl import DataProto
    from verl.trainer.ppo.ray_trainer import compute_advantage

    clock = [0.]
    monkeypatch.setattr(budget, 'perf_counter', lambda: clock[0])
    monkeypatch.setattr(core_wigpo, 'perf_counter', lambda: clock[0])
    original = core_wigpo.canonical_action_key

    def canonical(*args, **kwargs):
        clock[0] += 2.
        return original(*args, **kwargs)

    def action_mask(*args, **kwargs):
        assert kwargs['summarize'] is False
        clock[0] += 3.
        return inputs['action_token_mask'], inputs['located']

    monkeypatch.setattr(core_wigpo, 'canonical_action_key', canonical)
    monkeypatch.setattr(core_wigpo, 'build_action_token_mask', action_mask)
    data = DataProto.from_dict(
        tensors={**{k: inputs[k] for k in ('token_level_rewards', 'step_rewards', 'response_mask', 'old_log_probs')},
                 'responses': torch.zeros((8, 4), dtype=torch.long)},
        non_tensors={'uid': inputs['index'], 'traj_uid': inputs['traj_index'],
                     'anchor_obs': inputs['anchor_obs'], 'segment_ids': inputs['segment_ids'],
                     'actions': inputs['action_keys']},
    )
    timings = {}
    compute_advantage(data, 'wigpo', action_group_agg=True, use_prob_reweight=True,
                      tokenizer=object(), step_advantage_w=0., enable_time_breakdown=True,
                      time_breakdown=timings)
    assert timings['intra-step'] == 19.  # Eight canonicalizations plus mask construction.
    timings = {}
    clock[0] = 0.
    compute_advantage(data, 'wigpo', action_group_agg=True, use_prob_reweight=True,
                      tokenizer=object(), subgroup_advantage_w=0., step_advantage_w=0.,
                      enable_time_breakdown=True, time_breakdown=timings)
    assert clock[0] == 0.
    assert set(timings) == {'AE'}
