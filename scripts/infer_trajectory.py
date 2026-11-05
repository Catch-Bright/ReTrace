"""Roll out a trained (merged HF) policy in ALFWorld / WebShop and dump full trajectories.

Mirrors the validation rollout in `TrajectoryCollector.vanilla_multi_turn_loop`
(same env managers, same history_length, same chat-template prompt), but uses vLLM
directly so no trainer / FSDP is needed. For every step it records the full prompt,
the raw response (<think>...</think><action>...</action>), the parsed action, reward,
validity and done flag, and writes:

    <out_dir>/trajectories.jsonl      one line per episode (all steps)
    <out_dir>/summary.json            success rate etc.
    <out_dir>/traj_<k>_<status>.md    human-readable transcripts for appendix

Usage (see bottom of the file for a full example):
    python scripts/infer_trajectory.py --env alfworld --model_path /path/to/merged_hf \
        --out_dir outputs/traj_alfworld
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import ray
from omegaconf import OmegaConf
from transformers import AutoTokenizer

ENV_DEFAULTS = {
    # env_name, training seed, max_steps, max_prompt_length (from the run_*_wigpo_7b.sh scripts)
    "alfworld": dict(env_name="alfworld/AlfredTWEnv", seed=2, max_steps=50, max_prompt_length=2048),
    "webshop": dict(env_name="Webshop", seed=1, max_steps=15, max_prompt_length=4096),
}
MAX_RESPONSE_LENGTH = 512


def build_config(args, d):
    cfg = OmegaConf.load(os.path.join(os.path.dirname(__file__), "..", "verl", "trainer", "config", "ppo_trainer.yaml"))
    cfg.env.env_name = d["env_name"]
    cfg.env.seed = args.seed if args.seed is not None else d["seed"]
    cfg.env.max_steps = args.max_steps or d["max_steps"]
    cfg.env.rollout.n = 1
    cfg.env.resources_per_worker.num_cpus = 0.1
    cfg.data.train_batch_size = 1  # make_envs also builds train envs; keep them minimal
    cfg.data.val_batch_size = args.num_episodes
    cfg.data.max_prompt_length = d["max_prompt_length"]
    cfg.data.max_response_length = MAX_RESPONSE_LENGTH
    if args.env == "alfworld":
        cfg.env.alfworld.use_milestone_detector = True  # same as training; gives milestone info
    return cfg


def split_response(text):
    think = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    action = re.search(r"<action>(.*?)</action>", text, re.DOTALL)
    return (think.group(1).strip() if think else None), (action.group(1).strip() if action else None)


def to_py(x):
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


ALF_TASK_TYPES = ["pick_and_place", "pick_two_obj_and_place", "look_at_obj_in_light",
                  "pick_heat_then_place_in_recep", "pick_cool_then_place_in_recep", "pick_clean_then_place_in_recep"]


def task_type_of(gamefile):
    for t in ALF_TASK_TYPES:
        if gamefile and t in gamefile:
            return t
    return None


def render_markdown(ep):
    """Full transcript incl. the exact prompt the model saw (for debugging)."""
    lines = [f"# Episode {ep['episode']}  ({'SUCCESS' if ep['won'] else 'FAILURE'}, {len(ep['steps'])} steps, "
             f"return={ep['return']}, task_type={ep['task_type']})", ""]
    for s in ep["steps"]:
        lines += [f"## Step {s['step'] + 1}", "", "**Prompt (observation given to the agent)**", "", "```text",
                  s["prompt"].strip(), "```", "", "**Agent response**", "", "```text", s["response"].strip(), "```", "",
                  f"- parsed action: `{s['action']}`  (valid={s['is_action_valid']})",
                  f"- reward: {s['reward']}  done: {s['done']}", ""]
    return "\n".join(lines)


def task_first(text):
    """Step-1 ALFWorld text: move 'Your task is to: ...' in front of the room description (paper layout)."""
    m = re.search(r"Your task is to:[^\n]*", text)
    if not m or text.find("You are in the middle of a room") < 0:
        return text
    task = m.group(0).strip()
    rest = (text[:m.start()] + text[m.end():]).strip()
    head, _, scene = rest.partition("You are in the middle of a room")
    return f"{head.strip()}{task} You are in the middle of a room{scene}"


def render_appendix(ep):
    """Paper-appendix format: raw environment text and the agent's response, step by step."""
    clean = lambda s: re.sub(r"\n{2,}", "\n", s.strip())
    lines = []
    for s in ep["steps"]:
        n = s["step"] + 1
        env_text = clean(s["env_text"])
        if n == 1:
            env_text = clean(task_first(env_text))
        lines += [f"Environment (Step {n})", "", env_text, "", f"Agent (Step {n})", "", s["response"].strip(), ""]
    last = ep["steps"][-1]
    lines += [f"Environment (Step {len(ep['steps']) + 1})", "", clean(last["next_env_text"]), ""]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", required=True, choices=list(ENV_DEFAULTS))
    p.add_argument("--model_path", required=True, help="merged HF model dir (see model_merger.py)")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--num_episodes", type=int, default=16, help="number of parallel val episodes")
    p.add_argument("--num_show", type=int, default=3, help="how many markdown transcripts to write per status")
    p.add_argument("--task_type", default=None, choices=ALF_TASK_TYPES,
                   help="ALFWorld only: restrict transcripts to one task type")
    p.add_argument("--min_steps", type=int, default=4, help="prefer transcripts with at least this many steps")
    p.add_argument("--seed", type=int, default=None, help="env seed (default: the training seed)")
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--temperature", type=float, default=0.4, help="0.4 = val_kwargs.temperature in the scripts")
    p.add_argument("--tp", type=int, default=1, help="vLLM tensor parallel size")
    p.add_argument("--gpu_mem", type=float, default=0.5)
    p.add_argument("--ray_num_cpus", type=int, default=64)
    args = p.parse_args()

    os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
    if args.env == "alfworld":
        os.environ.setdefault("ALFWORLD_DATA", "/root/.cache/alfworld")
    os.makedirs(args.out_dir, exist_ok=True)

    d = ENV_DEFAULTS[args.env]
    cfg = build_config(args, d)

    if args.task_type:
        if args.env != "alfworld":
            raise SystemExit("--task_type only applies to --env alfworld")
        # Make the ALFWorld env load only this task type (ids as in config_tw.yaml), so every episode is of it.
        from agent_system.environments.env_package.alfworld import envs as alf_envs
        type_id = {"pick_and_place": 1, "look_at_obj_in_light": 2, "pick_clean_then_place_in_recep": 3,
                   "pick_heat_then_place_in_recep": 4, "pick_cool_then_place_in_recep": 5,
                   "pick_two_obj_and_place": 6}[args.task_type]
        orig_load = alf_envs.load_config_file

        def load_only_type(path):
            c = orig_load(path)
            c["env"]["task_types"] = [type_id]
            return c

        alf_envs.load_config_file = load_only_type

    # Bound Ray's CPU pool (all host CPUs can stall GCS inside a cgroup), then build envs.
    ray.init(num_cpus=args.ray_num_cpus)
    from agent_system.environments.env_manager import make_envs
    train_envs, envs = make_envs(cfg)  # we only use the val envs

    from vllm import LLM, SamplingParams
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    llm = LLM(model=args.model_path, tensor_parallel_size=args.tp, dtype="bfloat16",
              gpu_memory_utilization=args.gpu_mem, enable_chunked_prefill=False,
              max_model_len=d["max_prompt_length"] + MAX_RESPONSE_LENGTH)
    sampling = SamplingParams(temperature=args.temperature, top_p=1.0, top_k=-1, max_tokens=MAX_RESPONSE_LENGTH)

    n = args.num_episodes
    obs, infos = envs.reset(kwargs=None)
    steps = [[] for _ in range(n)]
    gamefiles = [None] * n
    is_done = np.zeros(n, dtype=bool)
    returns = np.zeros(n, dtype=np.float32)
    won = np.zeros(n, dtype=bool)

    for t in range(cfg.env.max_steps):
        # same prompt construction as TrajectoryCollector.preprocess_single_sample
        prompts = [tokenizer.apply_chat_template([{"role": "user", "content": o}], add_generation_prompt=True,
                                                 tokenize=False) for o in obs["text"]]
        outs = llm.generate(prompts, sampling)
        responses = [o.outputs[0].text for o in outs]

        # The projection functions rewrite the list in place (full response -> parsed action), so pass a copy
        # and keep `responses` as the raw model output.
        next_obs, rewards, dones, infos = envs.step(list(responses))
        rewards = np.asarray(rewards).reshape(n)
        dones = np.asarray(dones).reshape(n)

        for i in range(n):
            if is_done[i]:
                continue  # env already finished; its extra steps are not part of the episode
            think, action_tag = split_response(responses[i])
            info = infos[i]
            gamefile = info.get("extra.gamefile")
            if gamefile:
                gamefiles[i] = gamefile
            steps[i].append(dict(
                step=t, prompt=obs["text"][i], response=responses[i], think=think, action_tag=action_tag,
                # raw environment text (no template / history / admissible-action list), as shown in the paper appendix
                env_text=obs["anchor"][i], next_env_text=next_obs["anchor"][i],
                action=to_py(info.get("action")), is_action_valid=bool(info.get("is_action_valid", True)),
                reward=float(rewards[i]), done=bool(dones[i]),
                milestone_phase=to_py(info.get("milestone_phase")),
                milestone_achieved=to_py(info.get("milestone_achieved")),
            ))
            returns[i] += rewards[i]
            if dones[i]:
                won[i] = bool(info.get("won", False))
        is_done |= dones.astype(bool)
        obs = next_obs
        print(f"[step {t + 1}] done {int(is_done.sum())}/{n}", flush=True)
        if is_done.all():
            break

    episodes = [dict(episode=i, won=bool(won[i]), task_type=task_type_of(gamefiles[i]), gamefile=gamefiles[i],
                     **{"return": float(returns[i])}, steps=steps[i]) for i in range(n)]
    with open(os.path.join(args.out_dir, "trajectories.jsonl"), "w") as f:
        for ep in episodes:
            f.write(json.dumps(ep, ensure_ascii=False) + "\n")

    per_type = {}
    for ep in episodes:
        per_type.setdefault(ep["task_type"], []).append(ep["won"])
    summary = dict(env=args.env, model_path=args.model_path, num_episodes=n, success_rate=float(won.mean()),
                   mean_steps=float(np.mean([len(s) for s in steps])), seed=cfg.env.seed,
                   per_task_type={k: dict(n=len(v), success_rate=float(np.mean(v))) for k, v in per_type.items()})
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

    # Appendix candidates: genuinely won (env's own `won` flag), every action valid, matching the requested task
    # type, and long enough to show multi-step reasoning. Shorter ones first so they fit on a page.
    def candidates(flag):
        c = [e for e in episodes if e["won"] == flag and (args.task_type is None or e["task_type"] == args.task_type)]
        if flag:
            c = [e for e in c if all(s["is_action_valid"] for s in e["steps"])]
        long_enough = [e for e in c if len(e["steps"]) >= args.min_steps]
        if c and not long_enough:
            print(f"WARNING: no {'success' if flag else 'failure'} episode has >= {args.min_steps} steps; "
                  f"falling back to shorter ones", flush=True)
        return sorted(long_enough or c, key=lambda e: len(e["steps"]))[:args.num_show]

    for status, flag in (("success", True), ("failure", False)):
        for k, ep in enumerate(candidates(flag)):
            stem = f"traj_{k}_{status}_{ep['task_type']}_ep{ep['episode']}"
            with open(os.path.join(args.out_dir, stem + ".md"), "w") as f:
                f.write(render_markdown(ep))
            with open(os.path.join(args.out_dir, stem + ".appendix.txt"), "w") as f:
                f.write(render_appendix(ep))

    envs.close()
    train_envs.close()


if __name__ == "__main__":
    main()

# Full example (run on a GPU node):
#   python scripts/model_merger.py merge --backend fsdp \
#       --local_dir checkpoints/verl_agent_alfworld/wigpo_qwen2.5_7b-rw4-actgrewight-seed2/global_step_150/actor \
#       --target_dir /var/gfs/BaseLLM/wangyuhao/merged/alfworld_wigpo_7b
#   python scripts/infer_trajectory.py --env alfworld \
#       --model_path /var/gfs/BaseLLM/wangyuhao/merged/alfworld_wigpo_7b \
#       --out_dir outputs/traj_alfworld --num_episodes 16
