# CR7 固定摩擦与摩擦 DR 部署候选（2026-10-29）

共 15 个独立部署包：固定地面摩擦 0.5–1.3 每档一个最佳 checkpoint，共享摩擦 DR 和地面单独 DR 各三个。
所有包均是 **旧 PhysX SysID 参数 + Newton 训练**，没有混入新 Newton SysID 的 checkpoint。
起步时走几步、短暂跟踪偏差不直接判失败；以下排名采用全程只按摔倒终止的已完成评估。

优先顺序：`fixed/mu_0.7`，然后 `fixed/mu_0.8`。这两包在独立种子 2031/2032 的 clean 和 DR 测试中均为 64/64。
`fixed/mu_1.3` 为另一候选，独立验证分别为 61/64 和 60/64。

## 文件索引

| 相对本目录的部署路径 | 训练 seed | checkpoint 迭代 | clean 全程不摔 | DR 全程不摔 |
|---|---:|---:|---:|---:|
| `fixed/mu_0.5` | 42 | 29999 | 14/64 (21.88%) | 11/64 (17.19%) |
| `fixed/mu_0.6` | 44 | 25000 | 44/64 (68.75%) | 32/64 (50.00%) |
| `fixed/mu_0.7` | 44 | 29999 | 64/64 (100.00%) | 64/64 (100.00%) |
| `fixed/mu_0.8` | 43 | 29999 | 63/64 (98.44%) | 64/64 (100.00%) |
| `fixed/mu_0.9` | 44 | 25000 | 43/64 (67.19%) | 51/64 (79.69%) |
| `fixed/mu_1.0` | 43 | 20000 | 56/64 (87.50%) | 37/64 (57.81%) |
| `fixed/mu_1.1` | 43 | 25000 | 38/64 (59.38%) | 43/64 (67.19%) |
| `fixed/mu_1.2` | 42 | 25000 | 15/64 (23.44%) | 12/64 (18.75%) |
| `fixed/mu_1.3` | 43 | 25000 | 63/64 (98.44%) | 64/64 (100.00%) |
| `dr_shared/rank1_s45_i20000` | 45 | 20000 | 41/64 (64.06%) | 38/64 (59.38%) |
| `dr_shared/rank2_s43_i25000` | 43 | 25000 | 46/64 (71.88%) | 30/64 (46.88%) |
| `dr_shared/rank3_s47_i29999` | 47 | 29999 | 37/64 (57.81%) | 29/64 (45.31%) |
| `dr_ground/rank1_s48_i25000` | 48 | 25000 | 37/64 (57.81%) | 21/64 (32.81%) |
| `dr_ground/rank2_s42_i29999` | 42 | 29999 | 16/64 (25.00%) | 9/64 (14.06%) |
| `dr_ground/rank3_s48_i20000` | 48 | 20000 | 17/64 (26.56%) | 6/64 (9.38%) |

DR 按 checkpoint 排名前三，没有要求三个不同 seed。地面 DR 第 1、3 名来自 seed 48 的不同训练阶段；全部原始模型文件互不相同。
只纳入四批评估都数值有效的候选；异常仿真没有伪记作 0% 成功率。完整筛选范围/候选来源见 `selection.json`。

## 现有 g1_deploy 的使用方式

在原来的 `scripts/run_policy_loop.py` 启动命令中，把 tracking 参数指向整个包目录，例如：

```text
--tracking policies/cr7_friction_20261029/fixed/mu_0.7
```

其他候选只需替换此路径，例如 `policies/cr7_friction_20261029/dr_shared/rank1_s45_i20000`。
沿用已经确认的机器人网络、IMU 安装角、等待阶段和站立 policy 参数。本次没有修改 loop 或其保护阈值。
特别是 loop 原有的 `--tracking_max_tilt_error` 默认仍为 0.8 rad；它与这里离线 fall-only 评分不是同一个判断。
每包的 `contract.json` 已包含 29 关节映射、Kp/Kd、默认姿态、action scale 和 50 Hz 控制周期。
传给 `--tracking` 的是目录；`model_*.pt` 是训练 checkpoint，`policy.pt` 才是导出的 actor。

每个候选包包含：

- `policy.pt`、`contract.json`、`motion.npz`、`motion.json`：现有 tracking runner 的部署文件。
- `model_*.pt`：对应的原始训练 checkpoint。
- `training/params/env.yaml`、`agent.yaml`：保存的完整训练配置。
- `training/experiment.json`：训练任务、命令、seed、版本、摩擦设置。
- `training/sysid_parameters.json`：对应的五项关节参数；其余 training JSON 记录来源和训练汇总。
- `selection.json`、`evaluations/`：选模分数、脚滑/实际超限统计、原始评估及已有的独立验证。
- `validation.json`、`manifest.json`：离线部署接口检查及文件 SHA256。

## 三组训练设置

| 项目 | fixed | dr_shared | dr_ground |
|---|---|---|---|
| 引擎 | Newton MJWarp | Newton MJWarp | Newton MJWarp |
| 物理步长 / PD | 5 ms，原生隐式 PD | 2.5 ms，显式 SysID PD | 2.5 ms，显式 SysID PD |
| policy | 50 Hz | 50 Hz | 50 Hz |
| 摩擦 | 地面固定指定 μ；机器人 1 | 每环境一个 U(0.5,1.5)，机器人全部碰撞形状共享；地面 0 | 地面每环境 U(0.5,1.5)；机器人 1；地面 priority=1 |
| SysID | 旧五参数全用，延迟向上取整到 5 ms | 同旧五参数，2.5 ms 延迟精度 | 同左 |
| action rate 惩罚 | -0.01 | -0.01 | -0.01 |
| foot slip 惩罚 | 0 | 0 | 0 |

共有设置：官方 Isaac 6.2 `g1_29dof.usd`、无 Dex3/灵巧手、29 个身体关节；4096 环境，训练预算 30000 次更新。
这些候选从 20000、25000、29999 中选择，不保证最终迭代总是最好。参考动作均为同一份 297 帧 / 50 Hz 的 grounded CR7（完整片段 5.92 s）。
训练保留 torso 质量 log-uniform ×[0.8,1.25]、观测噪声、初始状态扰动、10–15 s 间隔 XY 速度推扰 ±0.5 m/s。
训练没有新增 COM/随机编码器 DR；固定的 SysID encoder bias 与随机编码器 DR 是两回事。
原始奖励、终止项及所有细节以各包原封不动的 env.yaml 与匹配源码为准；本次只重新选择/导出，没有重新训练或改训练终止项。

## 评分和验证范围

摔倒定义：torso 高度低于 0.35 m，或相对竖直倾角超过 75°，连续 0.2 s。
关闭原 anchor/末端跟踪误差及 500°/s 终止。小幅起步调整不触发这些摔倒条件。
每列分数来自 seed 2026/2027 各 32 个 frame-0 完整起点试验，总计 64；中间帧起点只作补充，不混入主分数。
排名先比较 clean/DR 两列中较低的全程成功率，再比较两列平均值，最后比较存活期间的身体误差。

clean 统一地面 μ=1，仍保留初始状态扰动；不是在每包自己的训练 μ 上测。
DR 评估包含质量、COM、编码器、观测噪声与 1–3 s 六维速度推扰；fixed 在地面 U(0.5,1.5) 下测，
两组旧 explicit400 DR 候选均沿用原先的“脚部共享 U(0.5,1.5)、地面 0”测试协议。
因此 fixed 和 DR 两大类的数值不是严格仅改变摩擦的对照，详见逐候选 evaluations。

已逐包验证 256 组输入上的 CPU eager / TorchScript 输出完全一致；再对照匹配的冻结 Newton 环境，
在参考帧 0、74、148、222、295 加关节扰动，比较部署 obs、action、encoder 目标。
全部通过；PD 增益逐关节与保存的训练 env.yaml 核对。验证结果见各包 validation.json。
硬件应直接输入 encoder 反馈、输出 encoder 目标；**不要再次添加 SysID bias 或软件延时**。
这些项已经描述真实硬件的特性，重复施加会改变策略的输入输出关系。

导出阶段完成离线接口验证；后续原生 MuJoCo 验证见下节，尚无真机验证。部分评估仍记录实际关节超限，
没有通过裁剪 action/目标来隐藏；例如 μ=0.7 的筛选最大实际超限为 clean 9.46° / DR 13.44°。
`index.csv` 给出每包身体误差、接触滑速和实际超限峰值，完整记录在 evaluations。

## 新 Newton SysID

状态与查询时间见 `new_sysid_status.json`。这批仍在训练，最新完成不等于已经筛选验证。
之前只完成评估的新 400 Hz μ=0.5 两个最终 checkpoint，全程不摔仍为 0/64；不能据此断言新参数所有摩擦档都不行。
本目录先提供用户指定的旧实验候选，不将新完成但未评估模型标为最佳。

复现入口见 `REPRODUCE.txt`；保存的两份训练源码压缩包与导出脚本在 `training_sources/`。

## 后续原生 MuJoCo 验证（2026-10-29）

已完成独立 `mujoco.mj_step` 闭环测试：15个候选 × 241条 = 3615条，均覆盖完整5.92秒动作及2秒收尾。
模型为同一官方无手G1 USD转换的物理模型，保留旧SysID；使用当前部署runner，没有用旧Newton记录替代物理运行。

μ=0.7 与 μ=0.8 均 241/241 通过，包括μ=1的64次初态扰动、九个其他固定μ各16次，以及地面U(0.5,1.5)的32次。
这里随机DR仅改变地面摩擦，未加入之前完整DR评估里的质量/COM/编码器/推扰/观测噪声。
共享DR第一名在μ=1下49/64通过、地面DR第一名36/64；其nominal起跳双脚离地峰值仅约8cm，参考为48.5cm。
μ=0.7/0.8该峰值约41.6/40.3cm。起跳幅度只是额外诊断，没有改变只按摔倒计分的要求。

两优先候选在241条中的最大实际关节超限分别10.05°/11.33°，未通过裁剪动作隐去。
未测试真机、双Dex3附加质量或go/go2切换过程。

- [完整报告与五段左右对比视频](native_mujoco_validation/report.html)
- [起跳高度对照](native_mujoco_validation/jump_quality.json)
- [审计与复现命令](native_mujoco_validation/REPRODUCE.txt)
- `native_mujoco_validation/trial_results.tar.gz`：所有原始评分与nominal物理轨迹。
- 各候选的 `evaluations/native_mujoco/summary.json`：对应结果。
