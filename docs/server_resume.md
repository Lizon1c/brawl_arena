# 服务器续跑指南 (v5.3, 2026-09-06)

本机 (RTX 4070 Laptop 8GB) 训练在 it~1050 暂停, 权重与对手快照已入库:

- `checkpoints/v53_it1050.pt` — 学习中学生 (= 暂停时的 `runs/student_rl/rl_latest.pt`)
- `checkpoints/v53_frozen_src.pt` — 当前 selfplay 冷冻对手 (= `runs/student_rl/frozen_src.pt`)

## 恢复训练

```bash
python -u student_rl.py --iters 100000 \
  --init checkpoints/v53_it1050.pt \
  --backbone cnn --snapshot-prob 0.15 --mb-size 256
```

服务器显存充足 (84GB), `--mb-size` 可提到 512/1024 试吞吐; CNN 细流在 RL 中冻结,
大头是 4 层 RoPE transformer (总参数 3.92M)。

## 评估协议 (与本机链条可比)

```bash
# 主指标: 战斗基本功 (duel, 双策略)
python -u scripts/eval_duel.py checkpoints/v53_it1050.pt 10
# 副指标: 战略层 (宝石 50 局, 固定种子)
python -u student_rl.py --eval <ckpt> --n 50 --seed0 6000 --backbone cnn
```

历史评估链见 `docs/eval_history.log`。参考点: 宝石 28% (it~1000),
历史最佳 38% (it770); duel sampled 命中 5.7-6.3/局, 0 击杀。

## 当前配比 (v5.3)

9 环境: vs_bots = gem×2 + duo×2 (duo=2v1, 两个学习中学生同局, 逐学生奖励分解);
selfplay×4 = gem/ball/ko/duel (冷冻对手采样行动); spectate×1 = gem。
棘轮晋升: selfplay 聚合 ≥0.60 + ≥60 局 + 间隔 ≥50 iter (已 7 次晋升, 最近 it952)。

## 已知问题 / 待办

- `maps_custom/showdown_recovered.txt` 未过校验 (草丛块 25>16 + 8 格不对称), 启动时跳过。
- 宝石胜率在 16-38% 振荡, 中枢 ~26-28%, 12 个评估点无趋势突破; duel 命中稳但
  击杀转化为 0。候选干预 (未拍板): selfplay duel 槽换第 3 个 gem vs_bots 槽 (预案B);
  对手池采样 (AlphaStar league); shaping 退火; argmax 部署侧 entropy 退火。
- 双系统架构 (共享 stem + ViT-tiny 粗流 + 逆动力学执行器 + 惊讶通道) 方案在
  `docs/dual_system_plan.md`, 尚未实施。
