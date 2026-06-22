# SportSwarm-CPP 演示讲稿（约 3 分钟）

## 30 秒版

我想把刘畅组 SwarmPRM 的 GMM/CVaR 思路用到体育捡球：用 Ball-Landing Field 描述球落点先验，在 7 人制足球训练场上对比单机 greedy、多机 greedy、Voronoi 分区和 BLF-informed 部署，看清场时间和总路径如何随机器人数量变化。

## 展开版

1. **痛点**：训练时球频繁出界，人工捡球打断节奏。
2. **现状**：Tennibot 等是单机 + 最近球贪心；学术几乎没人做「多机 + 数据先验 + 部署优化」。
3. **BLF**：2D 热力图，红色区域 = 球更常落下的边线/底线位置（V1 用高斯模拟，后续接 StatsBomb）。
4. **方法对比**：打开 `comparison.png`，讲 N=1 vs N=2 vs N=4。
5. **BLF-informed**：机器人初始位置放在高发区，减少长距离折返。
6. **诚实边界**：V1 无 CVaR 避人、无 GMM 在线重分配、无 Gazebo；下一步对接 SwarmPRM。

## 教授可能追问

- **BLF 和 NR-CCP 的 risk field 有何类比？** 都是二维非均匀先验；NR-CCP 是压实 risk，这里是落点概率。
- **为什么用 GMM？** 落点往往呈簇状，多个 modal 可对应多个 robot 责任区（SwarmPRM 宏观规划）。
- **baseline 公平吗？** 所有方法共享同一组球位置（固定 seed）。
