# G1 的 LEAPP 真机部署

更新日期：2026-10-14。本文记录 Mac 上已完成的 LEAPP 接入及本次真机测试。

## 已验证的策略和环境

- 策略：`height35-student-s43-i16000`，带深度视觉的 PPO student。
- 导出目录：`/Users/hehu/Downloads/mac-handoff/exported-policy`。
- 入口：`scripts/run_policy_loop.py --depth ... --leapp ...`。
- 推理实现：官方 `leapp.InferenceManager`，加载 YAML、ONNX 和初始状态 safetensors。
- 本机环境：Conda `deploy`，Python 3.10.20，PyTorch 2.13.0，LEAPP 0.6.0，ONNX Runtime 1.23.2。
- 通信：Unitree SDK2 Python 1.0.1、CycloneDDS **0.10.2**；DDS domain 0。
- Mac 网口：`en6`，IP `192.168.123.222`；运动计算机 `.161`，Orin/PC2 `.164`。
- 本次真机记录报告 `mode_machine=5`，即 29-DoF。硬件、网卡、路由状态会变，日期不能忽略。

原来的 `.pt` 部署路径保留：不传 `--leapp` 就仍然使用 TorchScript。新参数不是任意 ONNX
模型的通用入口，而是带完整输入输出描述的官方 LEAPP bundle 入口。

## 真机测试证据

用户在 2026-10-14 现场确认该命令已工作。对应本地记录：

`recordings/20261014_174723_148168_Isaac-Velocity-Rough-G1-29Dof-AirTime100-Yms-DepthDistill`

已读取该记录的 metadata、事件和逐步数据：

- 实际参数含 `--real`、`--align`、`--leapp`，视觉模式 `blind=off`。
- 987 个策略周期，约 19.74 秒名义控制时间；控制频率 50 Hz。
- 事件顺序：setup → ramp → align → policy_start → interrupt → policy_stop → damp。
- 操作者 Ctrl+C 结束；`policy_stop.error=null`，`clean_shutdown=true`，录制无丢弃。
- 987 个关节目标均为有限值；日志中没有 fall 事件。
- 命令前进速度范围 0–0.60 m/s，横移为 0；这是指令范围，不是实测行走速度。
- 深度帧龄：中位数 7.85 ms，P99 15.73 ms，最大 16.83 ms。
- 实际循环中 LEAPP 推理：中位数 0.515 ms，P99 3.834 ms，最大 4.419 ms。

上述证据覆盖本次短时真机运行，不代表所有地形、速度或其他 checkpoint 都已验证。

接入前还完成了 200 步新旧实现对比：覆盖 reset、IMU 安装偏移、非对称深度图及无效像素，
电机目标最大差 `6.079673767089844e-06 rad`，PD 参数差为 0；相关 17 项测试通过。

## 1. 恢复机器人网段路由

在 Mac 终端执行。公司 VPN 可能将 `192.168.123.x` 导向其他接口：

```bash
sudo route -n add -net 192.168.123.0/24 -interface en6
route -n get 192.168.123.161
```

结果应包含 `interface: en6`。若第一条提示 `File exists`，使用：

```bash
sudo route -n change -net 192.168.123.0/24 -interface en6
```

本次 Mac 的地址已经是 `192.168.123.222`，需要恢复的是路由。重启或 VPN 重连后可能需要重做。

## 2. 启动深度发布器

Mac 终端 A：

```bash
ssh -t unitree@192.168.123.164 \
  '/usr/bin/python3 ~/depth_publisher.py --host 192.168.123.222'
```

如果已经在 Orin 的 SSH shell 内：

```bash
/usr/bin/python3 ~/depth_publisher.py --host 192.168.123.222
```

保持该进程运行。已有发布器就复用，不要启动第二个。PC2 必须用已安装 RealSense 依赖的
`/usr/bin/python3`。发布器裁剪、降采样后经 UDP 5601 发出 64×38 深度图；默认 90 Hz，必要时回退 60 Hz。

## 3. 启动带深度的 LEAPP 策略

机器人按既有流程进入调试模式并支撑好。Mac 终端 B：

```bash
conda activate deploy
cd /Users/hehu/workspace/g1_deploy

python scripts/run_policy_loop.py --real --align --teleop --no_strafe \
  --depth policies/height35-student-s43-i16000 \
  --leapp /Users/hehu/Downloads/mac-handoff/exported-policy/Isaac-Velocity-Rough-G1-29Dof-AirTime100-Yms-DepthDistill.yaml \
  --leapp_depth_rotation 180 \
  --policy_physics g1_29dof \
  --vx 0.0 --vy 0.0 --duration inf \
  --ramp_s 3.0 --domain_id 0 --interface en6 \
  --depth_echo 5602
```

- `--depth` 在此提供匹配的 `contract.json`；不会加载该目录的 `policy.pt`。
- `--leapp` 指向 YAML，相关 ONNX 和 safetensors 保持在同一目录。
- 先完成模型加载、预热和深度检查，再检查控制权并提示操作。
- 输入 `go` 回车：3 秒 ramp 到初始姿势；随后保持该姿势。
- 调整好后输入 `go2` 回车：默认倒计时 3 秒，然后策略接管。
- 策略运行后键盘直接按键控制，不用回车；`?` 显示当前按键帮助。
- `--duration inf` 不会像 `--duration 20` 那样在 20 秒后自动结束。
- `Ctrl+C` 退出并执行阻尼；不要用 Ctrl+Z 挂起控制进程。

LEAPP 模式不支持 `--action_limit`，因为图对外输出已经是关节目标，不是 raw action。
`--clamp_targets` 仍是可选的关节目标限位，默认关闭，与既有部署一致。
没有恢复此前删除的“快摔时保持某姿势”逻辑；保留原来的跌倒退出和阻尼。

## 4. 查看或录制深度

Mac 终端 C：

```bash
conda activate deploy
cd /Users/hehu/workspace/g1_deploy
python scripts/view_depth.py --port 5602 --window --gray --hz 20
```

要保存可播放的灰度视频，在末尾加 `--record`，Ctrl+C 停止并保存。
5602 的 echo 从 `go2` 后策略实际运行时开始。控制进程占用 5601，预览应使用 5602。

## 数据边界

```text
Orin RealSense → 深度 UDP ───────────┐
G1 rt/lowstate → 关节和 IMU ────────┼→ G1LeappPolicyRunner
键盘 → 速度和航向目标 ──────────────┘        ↓
                                官方 InferenceManager
                             YAML + ONNX + feedback state
                                           ↓
                              关节目标 / Kp / Kd
                                           ↓
                               29 电机映射 → rt/lowcmd
```

`g1_deploy/leapp.py` 只适配输入输出：WXYZ 转 XYZW、IMU 安装偏移、按名称确定关节映射、
14 个未由该电机接口读取的手指按默认姿势和零速度填充，以及相机输入约定。

图内已经包含 5 帧本体历史、3 帧深度历史、上一 action、观测处理、归一化和
`default + scale × action`。不能在图外重复执行这些计算。

本次 Newton 导出图内有 180° 图像翻转，因此适配器默认先旋转输入 180°，使网络收到的
方向与既有 G1 相机路径一致。`--leapp_depth_rotation 0` 只适用于本来就符合图输入方向的图像。
真实相机的非正值/非有限值仍按既有部署解释为无回波并填 3 m，之后的裁剪、缩放、历史由图执行。
这与直接把零深度送入官方图不同：官方图保留零为近距离。

## 离线检查与官方仿真入口

只检查 Mac CPU 推理，不连接机器人：

```bash
conda activate deploy
python /Users/hehu/workspace/leapp_mac_check_20261014/check_inference.py
```

该脚本调用官方 `InferenceManager`，执行 1,000 步合成输入检查及 64 步 reset 重放。
它是检查脚本，不是 Isaac Lab 仿真。

本地 IsaacLab 分支为 `/Users/hehu/workspace/IsaacLab_leapp`，提交 `09108ccf2`，已含交接补丁。
该版本官方仿真入口是 `scripts/reinforcement_learning/leapp/deploy.py`，参数为 `--leapp_model`；
新版文档的 `isaaclab leapp deploy --pipeline ...` 不适用于这个旧入口。
Linux 上的已验证命令保存在 `/Users/hehu/Downloads/mac-handoff/official-commands.txt`。
Mac 当前 `deploy` 环境只验证过推理/真机控制，未配置完整 Isaac Lab 仿真环境。

官方 runtime 和独立的 `G1LeappPolicyRunner` 均可脱离 `run_policy_loop.py` 调用；新的真机入口
仍须提供状态/相机输入、调度、控制权与退出处理、电机输出。

## 常见故障

- `Address already in use`：检查 Mac 上 `lsof -nP -iUDP:5601`，确认旧进程身份和状态后再处理。
  被 Ctrl+Z 挂起的旧控制进程仍会占用端口，不要直接恢复它让旧命令继续执行。
- 相机 `Device or resource busy`：Orin 上用 `sudo fuser -v /dev/video*` 确认占用者，优先复用已有发布器。
- DDS `ddsi_udp_conn_write ... failed`：核对网线、en6 链路和路由。此前有一次是运动中网线松动。
- 升级 Python 依赖时保留 CycloneDDS 0.10.2；其他版本的仿真通信通过不能证明真机类型匹配。

## 其他策略不能只靠文件扩展名判断兼容

2026-10-14 检查 `policies/Junnosuke`：5 个导出文件都是 TorchScript 盲 MLP，
`937 → 512 → 256 → 128 → 29`，ELU，无模型内历史缓存，normalizer 为 Identity。
README 指定 10 帧历史加 7 维命令；现有 G1 Runner 则是 690 输入、43 输出。
缺少 ACTIVE_JOINT 顺序、7 维命令语义、默认姿势、动作缩放、PD 和控制周期的完整训练配置。
这些模型可在 Mac 上执行推理，但截至本文更新时尚不能直接用现有真机命令测试。
