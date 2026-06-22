# SportSwarm-CPP

**Data-Informed Multi-Robot Dynamic Coverage Path Planning for Ball Retrieval on Rectangular Sports Fields**

面向足球/网球训练场多机器人捡球的完整仿真框架，对应 proposal `PROPOSAL_SportSwarm-CPP.md` 中的 **GB-CPP V1 → SportSwarm 完整版** 技术路线。

## 已实现功能（对照 Proposal）

| Proposal 模块 | 实现 | 文件 |
|---------------|------|------|
| 双场地（足球 7v7 + 网球场） | ✅ | `configs/football_full.yaml`, `tennis_court.yaml` |
| BLF 热力图（高斯 / StatsBomb / hybrid） | ✅ | `src/statsbomb_blf.py`, `scripts/blf_from_statsbomb.py` |
| 战术风格 BLF（进攻/防守） | ✅ | `blf.tactical_style` |
| 球事件：uniform / BLF / semi-Markov 流 | ✅ | `src/ball_events.py` |
| GB-greedy（Tennibot baseline） | ✅ | `single_greedy` |
| Uniform CPP（boustrophedon 网格） | ✅ | `uniform_cpp` |
| BLF-weighted CPP | ✅ | `blf_weighted_cpp` |
| MDCPP-style Voronoi | ✅ | `voronoi_assignment` |
| BLF-informed 部署（Lloyd 优化） | ✅ | `blf_informed_deployment`, `src/deployment.py` |
| GMM 宏观重分配（SwarmPRM-lite） | ✅ | `gmm_swarm`, `src/gmm.py` |
| CVaR 避人（SportSwarm-full） | ✅ | `sportswarm_full`, `src/cvar_planner.py` |
| N ∈ {1,2,4,8} + Pareto 曲线 | ✅ | `plot_pareto`, `metrics.pareto_front` |
| 10-seed batch 实验 | ✅ | `scripts/run_batch.py` |
| 5 组消融实验 | ✅ | `scripts/run_ablation.py`, `src/ablation.py` |

## 环境

```bash
cd ~/Desktop/SportSwarm_CPP_Demo
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 scripts/generate_sample_statsbomb.py   # 生成示例 StatsBomb 事件
```

## 运行

```bash
# 完整足球实验（8 种方法 × N=1,2,4,8）
python3 main.py configs/football_full.yaml

# 轻量 baseline demo
python3 main.py configs/default.yaml

# 网球场 benchmark
python3 main.py configs/tennis_court.yaml

# StatsBomb → BLF 热力图
python3 scripts/blf_from_statsbomb.py --style offensive

# 10-seed 批量实验
python3 scripts/run_batch.py

# 5 组消融
python3 scripts/run_ablation.py
```

## 输出

- `outputs/figures/*.png` — 路径图、对比图、N scaling、Pareto、BLF 增益
- `outputs/results/metrics.csv` — 单次运行指标
- `outputs/results/batch_results.csv` — 多 seed 批量结果
- `outputs/results/ablation_results.csv` — 消融实验

## 方法对比

| 方法 | 策略 |
|------|------|
| `single_greedy` | 单机最近球贪心（Tennibot-style） |
| `multi_greedy` | 多机各自抢最近未分配球 |
| `uniform_cpp` | Boustrophedon 网格覆盖 +  opportunistic 捡球 |
| `blf_weighted_cpp` | 高 BLF 区优先覆盖顺序 |
| `voronoi_assignment` | Voronoi 分区 + 最近球（MDCPP-style） |
| `blf_informed_deployment` | Lloyd 部署 + multi greedy |
| `gmm_swarm` | BLF 部署 + GMM 在线重分配 |
| `sportswarm_full` | BLF + GMM + CVaR 避人 |

## 指标

- `time_to_clear_s` — 清场时间
- `total_distance_m` — 总路径
- `clearance_rate` — 捡球完成比例
- `total_cost` — N × 单机成本
- `cost_efficiency` — clearance / cost

## 目录结构

```text
SportSwarm_CPP_Demo/
  configs/
    default.yaml           # 轻量 4 方法 demo
    football_full.yaml     # 完整足球实验
    tennis_court.yaml      # 网球场 benchmark
  data/statsbomb_sample/   # 示例 StatsBomb 出界事件
  scripts/
    blf_from_statsbomb.py
    generate_sample_statsbomb.py
    run_batch.py
    run_ablation.py
  src/
    statsbomb_blf.py       # StatsBomb → BLF
    ball_events.py         # 球事件流
    deployment.py          # Lloyd 部署优化
    gmm.py                 # GMM 重分配
    cvar_planner.py        # CVaR 避人
    coverage.py            # CPP 覆盖路径
    ablation.py            # 消融配置
    batch_experiments.py   # 批量实验
    ...
```

## 下一步（Proposal V4+，尚未实现）

- Gazebo + TurtleBot 实体 demo
- 接入真实 StatsBomb Open Data 完整赛季
- 世界杯 2022 case study overlay 图
- SwarmPRM 完整 PRM 微观路径（当前为 2D 几何仿真）

## 引用

见 proposal §13：SwarmPRM (IROS'24)、MDCPP 2025、JTEC 2018 网球 CPP、StatsBomb/SALT 2025 等。
