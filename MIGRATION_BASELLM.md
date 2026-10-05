# 迁移后的启动方式

个人目录：`/var/gfs/BaseLLM/wangyuhao`。
项目目录：`/var/gfs/BaseLLM/wangyuhao/long_horizon/verl-agent`。

## WebShop

在实际训练节点执行。准备脚本在需要时解压原有环境包，并修正环境内的旧项目安装路径。

```bash
cd /var/gfs/BaseLLM/wangyuhao/long_horizon/verl-agent
bash scripts/prepare_local_overlay.sh webshop
source /root/envs/webshop-uv-py31020/bin/activate
export JAVA_HOME=/root/software/jdk11
export PATH="${JAVA_HOME}/bin:${PATH}"
export WANDB_MODE=offline
export WANDB_DIR="$PWD"
export WANDB_SILENT=true
bash examples/gigpo_trainer/run_webshop_wigpo.sh
# 或运行 7B（按需选择一个）：
# bash examples/gigpo_trainer/run_webshop_wigpo_7b.sh
```

## ALFWorld

使用单独的终端或先退出其他环境。

```bash
cd /var/gfs/BaseLLM/wangyuhao/long_horizon/verl-agent
bash scripts/prepare_local_overlay.sh alfworld
source /root/envs/alfworld-uv-py312/bin/activate
export WANDB_MODE=offline
export WANDB_DIR="$PWD"
export WANDB_SILENT=true
bash examples/gigpo_trainer/run_alfworld_wigpo.sh
# 或运行 7B（按需选择一个）：
# bash examples/gigpo_trainer/run_alfworld_wigpo_7b.sh
```

## Sokoban

环境压缩包永久保存在 `/var/gfs/BaseLLM/wangyuhao/envs/sokoban-local-overlay-py31020.tar.gz`，准备脚本会按需解压到 `/root/envs/sokoban-uv-py31020` 和 `/root/software/python-3.10.20-standalone`。在另一台挂载相同共享盘的训练节点执行：

```bash
cd /var/gfs/BaseLLM/wangyuhao/long_horizon/verl-agent
bash scripts/prepare_local_overlay.sh sokoban
source /root/envs/sokoban-uv-py31020/bin/activate
export HF_HOME=/var/gfs/BaseLLM/wangyuhao/hf_cache
export WANDB_MODE=offline
export WANDB_DIR="$PWD"
export WANDB_SILENT=true
bash examples/gigpo_trainer/run_sokoban.sh
```

脚本默认使用 `Qwen/Qwen2.5-VL-3B-Instruct` 和 2 张 GPU；模型权重需要在训练节点可下载，或预先放入共享的 `HF_HOME` 缓存。环境包不包含模型权重。Python 包版本记录在共享目录的 `sokoban-py31020.freeze.txt`。

## 注意

- 不再需要单独执行原来的 tar 解压命令；准备脚本会按需解压。如果手动重新解压，再运行一次准备脚本以修正安装记录。
- 原有环境包保持原样，仍解压到 `/root/envs`、`/root/software` 和（ALFWorld）`/root/.cache/alfworld`。
- 原来的个人目录暂时保留作为回退。在实际训练节点验证成功前，不应删除。
- 历史 WandB run 的配置和日志保留原始内容，其中旧路径属于历史记录；运行目录可以搬迁，无需修改 run ID 或重新训练。
- WebShop 和 ALFWorld 继续使用原有的两个本地 overlay 环境；Sokoban 使用单独打包的 Python 3.10 环境。
- 迁移状态位于本次操作节点 `/tmp/wangyuhao-migration-state.json`，最终 rsync 统计位于 `/tmp/wangyuhao-copy-final.log`。分片日志位于 `/tmp/wangyuhao-copy-shards/`。
