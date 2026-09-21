# 周日真机测试候选：2026-09-20

记录时间：2026-09-18。以下路径均相对于 `g1_deploy/`。
“候选”表示本机已有 `policy.pt + contract.json`，不代表已经通过真机验证。
新六模型已完成 CPU 导出一致性检查和部署加载检查，尚无独立 Mac 步态或真机测试结果。
文件已在 Linux checkout；本记录不代表已 push 或同步到 Mac。

## 主要候选及区别

| 部署目录（均在 policies/ 下） | 方法 / 输入 | 训练区别 | 周日要回答的问题 |
|---|---|---|---|
| `ysud` | 旧深度蒸馏；本体 + 深度接口，既有实验表明视觉贡献很小 | YSU 全关节动作尺度 0.5；旧粗糙地形训练；最终 `model_3999.pt` | 已有真机基线：用户反馈总体能走，但容易走歪、起步蹦；当天复测用于比较 |
| `blind_s42`、`blind_s43`、`blind_s44` | **直接 PPO，不是蒸馏**；仅本体五帧历史，共 690 维 | plane 平地，无楼梯、相机、height scanner / scan；动作尺度 0.5；DR29 本体随机化和推扰；各 `model_5999.pt` | 完全无视觉能否稳定走路，起步、低速和偏航是否改善 |
| `ymsdm_s42`、`ymsdm_s43` | 旧 YMS 视觉蒸馏；690 维本体 + 3×38×64 深度 | 旧 mesh 蒸馏 sweep；YMS 按关节动作尺度；各 `model_7999.pt` | 与新视觉组对比：走歪、起步、地形通过表现 |
| `tracking_depth_s42`、`tracking_depth_s43`、`tracking_depth_s44` | **新视觉蒸馏学生**；690 维本体 + 3×38×64 深度 | 原六类 rough 地形，mesh；新教师横向速度误差平方系数 4、线速度奖励权重 2；学生加入每次 reset 相机俯仰 ±5° DR，保留深度噪声、本体 DR 和推扰；YMS 动作尺度；各 `model_7999.pt` | 更重视 tracking 后是否更直；视觉是否有实际收益；相机角度变化时是否更稳定 |
| `student_yms` | 旧**盲蒸馏**；仅本体，无相机 | 从旧 YMS 教师蒸馏；YMS 动作尺度；`model_3999.pt` | 可选：比较“旧粗糙任务盲蒸馏”和“新平地直接 PPO” |

同组 s42 / s43 / s44 是相同配置、不同随机种子；新视觉组每个 seed 都有对应的新教师。
旧 `ymsdm` sweep 使用旧 YMS s43 教师，因此跨组比较不是只改了一个变量的消融实验。
`tracking_depth_s43` 仅因训练末期指标相对较好而先试，尚不能断定真机最好。

YSU 的 0.5 与 YMS 按关节尺度区别明显，例如 YMS 髋 pitch 约 0.11、膝 0.1738、踝 0.625。
必须加载整个策略目录，让 runner 读取配套 contract，不能混用其他模型的动作尺度或 contract。

## 名称核对及备选

本机**没有 `policies/ymsdm_v2/`**；不能把口头说的 “ymsdm_v2” 自动对应到 `ymsd_v2`。
目前能确认的旧视觉版本如下，优先级低于上面的主要候选：

| 目录 | 可确认的来源 / 区别 |
|---|---|
| `ymsd` | `g1-depth-sym/ymsd/model_3999.pt`，早期 YMS 深度学生 |
| `ymsd_v2` | `g1-depth-noise/ymsd/s43/model_7999.pt`，depth-noise sweep |
| `ymsd_v3` | `g1-ymsd-var/ymsd/s44/model_7999.pt`，var sweep 的基线 arm，seed 44；不是 ymsdm s43 |
| `student_cw` | CW 教师的盲蒸馏，尺度 0.5；不同教师基线 |
| `mjd`、`mjd_v2`、`mjd15` | MuJoCo-aligned 系列深度学生，尺度 0.5；分别来自 depth-sym/mjd、depth-noise/mjd、depth-sym/mjd15 |
| `yhkd`、`yhkd_v2` | YHK 深度学生；YMS 尺度中髋 pitch / 膝改为 0.5；分别来自 depth-sym 和 depth-noise |
| `depth_student_cl`、`depth_student_cl_prefix_camera`、`depth_student_w100` | 更早的 CL / W100 深度学生；两个 CL 目录记录同一原始 checkpoint，不能当成两个独立训练结果 |

上述备选已有部署文件，但本次没有重新验证其真机效果。周日时间有限时，先不扩展到这些旧分支。
`teacher_*`、`*_teacher` 以及 `ckpt/split_20260917/teacher_s*/` 是特权教师，
依赖真机当前没有提供的观测，不列入直接上机清单。

## 建议测试顺序与记录

1. `ysud`：复测当天硬件状态与已知基线。
2. `blind_s42`：测新纯本体方案；再以相同条件测 s43 / s44。
3. `ymsdm_s43`：测旧 mesh 视觉方案，建立视觉组基线。
4. `tracking_depth_s43`：测新视觉方案；再以相同条件测 s42 / s44。
5. 有时间再补 `ymsdm_s42`、`student_yms`，或按具体问题选择旧版本。

时间紧张时先完成 `ysud → blind_s42 → ymsdm_s43 → tracking_depth_s43` 四个代表模型。
先在 Mac 排除明显的加载、相机和步态问题，命令见 [Mac 测试说明](SPLIT_20260917.md)。
真机采用仓库已有的上机流程；Mac 示例的 `lo0`、domain 72 和 `--skip_release_mode`
是仿真设置，不直接照搬到真机。

每个模型先保持相同初始姿态、相机安装、速度与航向设置，以 0.5 m/s、vy=0 比较直行；
再根据表现比较低速与更高速度。每组尽量重复 3 次，并记录：

| 模型 / seed | 机器人型号 | 速度 / 时长 | 起步蹦跳 | 实际前进距离 | 横向偏移 / 航向偏差 | 骨盆倾斜 | 绊脚 / 中止原因 | 视频 / 日志 |
|---|---|---|---|---|---|---|---|---|
| 待测 | | | | | | | | |

新横向 tracking 惩罚更强，不等于已经证明能侧走；本轮先用 vy=0 检查走歪。
当前 runner 用航向误差生成转向指令，因此直行表现包含部署端航向反馈，比较时保持设置一致。

视觉消融与相机角度扫描先在 Mac 做：同一地形正常深度 vs `--blind frozen`，
相机绝对向下俯仰角 42.6° / 47.6° / 52.6°。冻结图像也是输入改变，
应结合障碍通过表现判断视觉收益；不能仅凭平地不摔下结论。
训练碰撞体：新盲走 plane，新视觉教师 / 学生 mesh；Mac 碰撞体由 XML 决定。

## 溯源

- 新模型工作流：`g1-split-walk-1`，2026-09-18 北京时间 03:41 完成，六组均 COMPLETED / exit 0。
- 新模型部署清单与哈希：[split_20260917_manifest.json](split_20260917_manifest.json)。
- 新策略目录均含原始 `checkpoint.pt`、导出的 `policy.pt`、`contract.json` 和训练配置。
- 旧模型的原始 checkpoint 路径记录在各自 `contract.json` 的 `checkpoint` 字段。
- 用户报告：`ysud` 已上真机；修好相机后的旧视觉模型在仿真中能看出视觉参与，但尚无真机结论。
- 本文记录测试候选和计划；周日结果另行填写，不能把训练成功率当作真机通过率。
