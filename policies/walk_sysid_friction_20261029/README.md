# Walk：旧 SysID＋摩擦 DR 部署包

包含共享摩擦与地面独立摩擦两组各两个不同训练 seed 的候选。每组原先训练 8 个 seeds，
从每个 seed 的 20k／25k／29999 checkpoint 中筛选。目录 rank 保留原 Newton 筛选顺序。
本次未重新训练，旧版 policies/mimic/walk、mimic_dr/walk 与 mimic_dr_contact_dr/walk 保持不变。

| 目录 | 原 Newton clean / 完整 DR | MuJoCo 地面 μ=1 | MuJoCo 地面 U(0.5,1.5) | MuJoCo 全部测试 |
|---|---|---|---|---|
| `dr_shared/rank1_s49_i25000` | 64/64；64/64 | 64/64 | 32/32 | 241/241 |
| `dr_shared/rank2_s47_i29999` | 64/64；64/64 | 64/64 | 32/32 | 241/241 |
| `dr_ground/rank1_s49_i29999` | 64/64；64/64 | 64/64 | 32/32 | 241/241 |
| `dr_ground/rank2_s45_i25000` | 64/64；64/64 | 64/64 | 32/32 | 239/241 |

共享组两份与地面独立组第一名均 241/241；地面独立组第二名 239/241，其余两次摔倒发生在固定摩擦测试。
964 次中无数值异常。地面独立组第二名实际关节位置最大超限约 26.76°，其他三份约 4.52–5.48°；
上述均包含扰动试验，不能把不摔倒直接当作硬件稳定保证。优先使用各组 rank1，第二名保留作对照。

每包包括 `policy.pt`、`contract.json`、`motion.npz/json`、原始训练 checkpoint、
`training/params/env.yaml`、`agent.yaml`、SysID 参数、原始训练命令、原筛选及本次 MuJoCo 验证结果。

在现有部署命令中使用对应目录，例如：

```text
--tracking policies/walk_sysid_friction_20261029/dr_shared/rank1_s49_i25000
```

这是有限时长的 mimic walk 动作，不是可通过速度命令无限行走的 locomotion policy。
参考动作 400 帧、50Hz，最后一帧 7.98 秒；保留原始起止姿态与过渡，没有改成新的直立起步版本。

训练使用官方 Isaac 6.2 G1 29DOF 无手资产、旧 PhysX 辨识的 26 关节参数，
Newton MJWarp 400Hz 显式 SysIdDCMotor、50Hz policy、4096 环境、30000 次更新。
动作变化惩罚 -0.01，脚滑奖励权重 0；摩擦范围均为 0.5–1.5。各组精确采样规则见原始 env.yaml。
共享与独立摩擦是训练方案；本次 MuJoCo 验证统一机器人摩擦 1、地面 priority 1。

原 Newton 筛选使用 shared-dr-v1 跟踪终止规则与完整 DR；MuJoCo 采用独立原生 mj_step、
现有部署观测和动作接口，完整 7.98 秒动作＋2 秒末帧保持。初始纠偏步行不算失败，
torso 高度低于 0.35m 或倾角大于 75°持续 0.2 秒算摔倒；数值异常另列。
每份 241 次测试：参考初态 1 次、μ=1 初态扰动 64 次、其他 9 个固定摩擦各 16 次、
地面随机摩擦 32 次。MuJoCo 随机评估只有地面摩擦和初态扰动，不含其他运行时 DR。

[完整 MuJoCo 报告与四段左右对比视频](native_mujoco_validation/report.html)。
各视频使用固定参考初态、地面 μ=1，左 reference、右 policy，下方显示位置误差随时间变化。
误差经过任务的平移／朝向对齐；脚滑和实际关节超限另存 JSON，不作为本次不摔倒判据。

导出已逐一比较冻结训练观测／动作／目标角与现有 G1TrackingPolicyRunner；
每份还用 256 个随机输入核对 eager 与 TorchScript 一致性。没有修改部署 loop 或保护条件。
硬件端使用真实 encoder，不要再次叠加本仿真中的 SysID bias 或延时。
模型无 Dex3；本次没有验证双 Dex3、真机、go/go2 切换或长期站立。

复现命令见 [REPRODUCE.txt](REPRODUCE.txt)，文件摘要见 `SHA256SUMS`。
