# sway-strong-s42-student-s43-i12000

用户选定的真机测试候选：strong teacher seed 42（6k）蒸馏出的 student seed 43（12k）。
`policy.pt` 和 `contract.json` 与已录制的 MuJoCo 候选逐字节一致。
与 `height35-student-s43-i16000` 的运行契约相同，仅 checkpoint 来源路径不同。

输入为 690 维状态历史及 `3 × 38 × 64` 深度历史，43 维动作映射至 29 个电机，50 Hz 控制。
继续使用 height35 的相机发布端及 `g1_29dof` 配置；参数和校验值见 `manifest.json`。

## 真机启动

在已有 `torch`、`numpy`、`unitree_sdk2py` 的部署环境中运行。
相机发布端沿用现有 PC2 配置，将深度帧发送到运行控制程序的电脑。
将下面的 `eth0` 换成连接机器人的实际网卡。

```bash
conda activate env_isaaclab
cd ~/workspace/g1_deploy
export PYTHONPATH="$PWD"
python scripts/run_policy_loop.py --real \
  --depth policies/sway-strong-s42-student-s43-i12000 \
  --policy_physics g1_29dof --teleop --no_strafe \
  --vx 0.5 --vy 0.0 --duration inf --ramp_s 3.0 --align \
  --domain_id 0 --interface eth0
```

按程序提示进入控制；`--align` 阶段输入 `go2` 后接入策略。
W/S 调整前进速度，Q/E 调整朝向，R 将目标朝向设为当前朝向，X 退出。
空格仅将速度指令归零；该模型仍有停车漂移，不能把空格当作急停。
真机保留默认控制权检查及深度帧时效检查，默认记录日志到 `recordings/`。

## 已验证范围

- 导出数值等价；部署端加载、关节映射与控制参数检查记录在 `validation.json`。
- 原 MuJoCo 动力学下已有 20 段连续正侧视频，与原 teacher 和 height35 基线对照。
- 0.5 m/s 平地指令下实际均速约 0.605 m/s，20 cm 上楼约 10.52 秒到顶且全段未倒。
- 同次协议末段停车漂移约 1.28 m；30 cm 慢速上楼、35 cm 上下楼有失败记录。

这是用户选定的测试包，尚无本模型真机验证，整体部署验收未完成。
先在有保护的平地范围验证控制、停车与转向，保留实际测试日志供复核。
