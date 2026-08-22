set -x
ENGINE=${1:-vllm}
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export ALFWORLD_DATA=/root/.cache/alfworld
export HF_HOME=/var/gfs/wangyuhao/hf_cache
export HF_DATASETS_CACHE=/var/gfs/wangyuhao/hf_cache/datasets
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

num_cpus_per_env_worker=0.1 # The CPU resource allocated for each environment worker.

train_data_size=16
val_data_size=128
group_size=8
mode="mean_std_norm" # "mean_norm" or "mean_std_norm"
# Within each subgroup (obs cluster n segment), average step rewards over identical
# canonical actions so the same action from the same state/stage gets the same credit
# (removes reward differences caused by *later* action choices). Invalid actions keep
# their own reward: "both invalid" does not mean "the same action".
# NOTE: with mode=mean_std_norm, averaging shrinks the within-subgroup std, so the
# normalization denominator shrinks too and the subgroup advantage scale grows.
# Tune subgroup_advantage_w if the gradient magnitude changes noticeably.
action_group_agg=True

# Probability-weighted think/action advantage separation:
# When enabled, think and action tokens get different advantages based on how strongly
# the think supports the action. Steps with same action in same subgroup get:
#   - Same action advantage (action tokens learn "is this action good?")
#   - Different think advantages scaled by p_i/p_bar (think tokens learn "how much do I support it?")
# where p_i is the model's probability for the action after this step's think.
# Requires action_group_agg=True.
use_prob_reweight=False
# rho_i clip bounds for the prob-weighted split. Empty (or null) = collect mode:
# no clipping is applied and the raw p_i/p_bar ratio distribution is logged to
# wandb (gigpo/rho_*) so you can pick bounds from data for the real run.
prob_reweight_clip_min=0.5  # min value for rho_i scaling factor
prob_reweight_clip_max=1.5  # max value for rho_i scaling factor
# Temperature T in p_i = exp(sum_j log pi_old(y_j) / (T * |a_i|)). Controls how far
# rho_i = p_i/p_bar spreads from 1: T>1 flattens toward 1 (weaker reweight), T<1
# sharpens (stronger). 1.0 = untempered.
prob_reweight_temperature=1.0
# reverse_decay_from_step_split: Each step gets reward from its OWN milestone + every LATER
# completed milestone, with discount measured from THIS step i: gamma^(e_s - i) + sum_j gamma^(e_j - i).
# Each milestone contributes r_max/M. Completing all M gives max reward = r_max.
subtask_reward_mode="reverse_decay_from_step_split"
model_path=/var/gfs/wangyuhao/model/qwen2.5_1_5b_instruct

# WiGPO: segment-refined GiGPO for ALFWorld.
#
# Two milestone-detection paths are supported (chosen via env.alfworld.use_milestone_detector):
#   - True  (below) -> facts-based state machine. Each env worker requests the
#                      TextWorld facts (holds / inReceptacle / isHot / ...) and
#                      detects per-task milestones (M differs by task type) with
#                      no external milestone file. Segments are derived from the
#                      per-step milestone_achieved flags in the trainer.
#   - False         -> legacy static action-anchor matching against
#                      alfworld_milestones.json (kept for repro experiments).
#
# The milestone_path below is only consumed by the legacy path; it is harmless
# (and kept) when the facts-based detector is enabled.
milestone_path=$(python3 -c "import os, gigpo.core_wigpo as m; print(os.path.join(os.path.dirname(os.path.abspath(m.__file__)), 'alfworld_milestones.json'))")
if [ ! -f "$milestone_path" ]; then
    echo "ERROR: could not resolve milestone_path (got: '$milestone_path')" >&2
    exit 1
fi
echo "Using milestone_path=$milestone_path"

# We only use data preparation to indicate the modality and the data size.
python3 -m examples.data_preprocess.prepare \
    --mode 'text' \
    --train_data_size $train_data_size \
    --val_data_size $val_data_size

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=wigpo \
    data.train_files=$HOME/data/verl-agent/text/train.parquet \
    data.val_files=$HOME/data/verl-agent/text/test.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=2048 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$model_path \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    algorithm.gamma=0.95 \
    algorithm.gigpo.step_advantage_w=0 \
    algorithm.gigpo.subgroup_advantage_w=1.0 \
    algorithm.gigpo.action_group_agg=$action_group_agg \
    algorithm.gigpo.use_prob_reweight=$use_prob_reweight \
    algorithm.gigpo.prob_reweight_clip_min=${prob_reweight_clip_min:-null} \
    algorithm.gigpo.prob_reweight_clip_max=${prob_reweight_clip_max:-null} \
    algorithm.gigpo.prob_reweight_temperature=$prob_reweight_temperature \
    algorithm.gigpo.milestone_threshold=0.95 \
    algorithm.gigpo.milestone_path=$milestone_path \
    algorithm.gigpo.mode=$mode \
    algorithm.gigpo.use_subtask_shaped_reward=True \
    algorithm.gigpo.subtask_reward_mode=$subtask_reward_mode \
    env.env_name=alfworld/AlfredTWEnv \
    env.seed=0 \
    env.max_steps=50 \
    env.alfworld.use_milestone_detector=True \
    env.rollout.n=$group_size \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_agent_alfworld' \
    trainer.experiment_name='wigpo_qwen2.5_1.5b-rw4' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    trainer.total_epochs=150 \
    trainer.val_before_train=True $@
