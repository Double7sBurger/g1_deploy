"use strict";

(() => {
  const $ = id => document.getElementById(id);
  const {
    TraceChart,
    number,
    colors
  } = window.DeploynoseCharts;
  const views = {
    overview: ["部署总览", "DEPLOYMENT OVERVIEW"],
    joints: ["关节与振动", "JOINTS & VIBRATION"],
    imu: ["姿态与偏航", "ORIENTATION & HEADING"],
    timing: ["控制时序", "CONTROL TIMING"],
    policy: ["Policy 信号", "POLICY SIGNALS"],
    events: ["事件与配置", "EVENTS & CONFIGURATION"],
  };
  let dataset = null,
    currentView = "overview",
    charts = [],
    loadVersion = 0,
    drawPending = false;
  let motorNames = [],
    actionNames = [],
    derivatives = new Map();
  const session = {
    range: [0, 1],
    fullRange: [0, 1],
    events: [],
    cursor: null,
    setRange(start, end) {
      if (!Number.isFinite(start) || !Number.isFinite(end) || start >= end) return;
      start = Math.max(this.fullRange[0], start);
      end = Math.min(this.fullRange[1], end);
      if (end - start < Math.min(0.001, (this.fullRange[1] - this.fullRange[0]) / 100)) return;
      this.range = [start, end];
      this.cursor = null;
      $("range-start").value = start.toFixed(4);
      $("range-end").value = end.toFixed(4);
      if (!drawPending) {
        drawPending = true;
        requestAnimationFrame(() => {
          charts.forEach(chart => chart.draw());
          drawPending = false;
        });
      }
    },
    reset() {
      this.setRange(...this.fullRange);
    },
    hover(value) {
      this.cursor = value;
      charts.forEach(chart => chart.drawOverlay());
    },
  };

  function showMessage(text) {
    $("message").textContent = text;
    $("message").hidden = !text;
  }

  function array(name) {
    return dataset?.arrays[name];
  }

  function values(name) {
    return array(name)?.data;
  }

  function trace(key, column, name, timeKey = "t", scale = 1, extra = {}) {
    const a = array(key),
      x = values(timeKey);
    if (!a || !x) return null;
    const stride = a.shape.slice(1).reduce((n, dimension) => n * dimension, 1);
    if (column >= stride) return null;
    return {
      x,
      y: a.data,
      stride,
      column,
      name,
      scale,
      ...extra
    };
  }

  function panel(title, unit, series, options = {}) {
    charts.push(new TraceChart($("charts"), {
      title,
      unit,
      series,
      ...options
    }, session));
  }

  function notice(text, info = false) {
    const element = document.createElement("div");
    element.className = "notice" + (info ? " info" : "");
    element.textContent = text;
    $("notices").append(element);
  }

  function metric(label, value, unit, foot) {
    const card = document.createElement("article");
    card.className = "metric";
    const name = document.createElement("div");
    name.className = "metric-label";
    name.textContent = label;
    const main = document.createElement("div");
    main.className = "metric-value";
    main.textContent = value;
    const suffix = document.createElement("small");
    suffix.textContent = unit;
    main.append(suffix);
    const note = document.createElement("div");
    note.className = "metric-foot";
    note.textContent = foot;
    card.append(name, main, note);
    $("metrics").append(card);
  }

  function fillSelect(id, names, none = false) {
    const select = $(id);
    select.replaceChildren();
    if (none) {
      const option = document.createElement("option");
      option.value = "-1";
      option.textContent = "不对比";
      select.append(option);
    }
    names.forEach((name, index) => {
      const option = document.createElement("option");
      option.value = String(index);
      option.textContent = `${String(index).padStart(2, "0")}  ${name}`;
      select.append(option);
    });
  }

  function validClock(clock) {
    for (let i = 0; i < clock.length; i++)
      if (!Number.isFinite(clock[i]) || (i && clock[i] < clock[i - 1])) return false;
    return true;
  }

  async function readBundle(file) {
    if (!window.DecompressionStream) throw new Error("当前浏览器不支持离线解压，请使用新版 Chrome、Edge 或 Firefox。");
    const reader = file.stream().pipeThrough(new DecompressionStream("gzip")).getReader();
    const chunks = [];
    let size = 0;
    try {
      while (true) {
        const {
          value,
          done
        } = await reader.read();
        if (done) break;
        size += value.byteLength;
        if (size > 512 * 1024 * 1024) throw new Error("记录展开后超过 512 MiB，请导入较短的测试记录。");
        chunks.push(value);
      }
    } catch (error) {
      await reader.cancel().catch(() => {});
      if (error.message.includes("512 MiB")) throw error;
      throw new Error("解压失败，请确认这是分析工具生成的完整 .deploynose 文件。");
    }
    const bytes = new Uint8Array(size);
    let position = 0;
    for (const chunk of chunks) {
      bytes.set(chunk, position);
      position += chunk.byteLength;
    }
    if (size < 16 || new TextDecoder().decode(bytes.subarray(0, 8)) !== "DNOSE001") throw new Error(
      "不是 deploynose 记录文件。");
    const header = new DataView(bytes.buffer),
      length = header.getUint32(8, true);
    if (length > 8 * 1024 * 1024 || length + 16 > size || header.getUint32(12, true) !== 0) throw new Error(
      "文件头损坏或版本不受支持。");
    const manifest = JSON.parse(new TextDecoder("utf-8", {
      fatal: true
    }).decode(bytes.subarray(16, 16 + length)));
    if (manifest.format !== "deploynose" || manifest.version !== 1) throw new Error("文件版本不受支持，请更新 deploynose。");
    const base = 16 + Math.ceil(length / 8) * 8;
    if (!Number.isSafeInteger(manifest.payload_bytes) || base + manifest.payload_bytes !== size) throw new Error(
      "文件内容不完整。");
    const types = {
      "<f4": Float32Array,
      "<f8": Float64Array,
      "<u4": Uint32Array,
      "<i4": Int32Array,
      "|u1": Uint8Array
    };
    const arrays = Object.create(null);
    const regions = [];
    if (!manifest.arrays || typeof manifest.arrays !== "object") throw new Error("文件缺少曲线数据。");
    for (const [name, spec] of Object.entries(manifest.arrays)) {
      const Type = types[spec.dtype];
      if (!Type || !Array.isArray(spec.shape) || spec.shape.length < 1 || spec.shape.length > 3 ||
        spec.shape.some(d => !Number.isSafeInteger(d) || d < 0)) throw new Error(`无效数据维度：${name}`);
      const count = spec.shape.reduce((a, b) => a * b, 1);
      if (!Number.isSafeInteger(count) || !Number.isSafeInteger(spec.offset) || spec.offset < 0 || spec.offset %
        8 ||
        spec.nbytes !== count * Type.BYTES_PER_ELEMENT || spec.offset + spec.nbytes > manifest.payload_bytes)
      throw new Error(`数据边界错误：${name}`);
      arrays[name] = {
        shape: spec.shape,
        data: new Type(bytes.buffer, base + spec.offset, count)
      };
      if (spec.nbytes) regions.push([spec.offset, spec.offset + spec.nbytes]);
    }
    regions.sort((a, b) => a[0] - b[0]);
    for (let i = 1; i < regions.length; i++)
      if (regions[i][0] < regions[i - 1][1]) throw new Error("文件包含重叠的数据区段。");
    if (!arrays.t || arrays.t.shape.length !== 1 || arrays.t.data.length < 2 || !arrays.q || arrays.q.shape
      .length !== 2) throw new Error("记录缺少有效的机器人状态序列。");
    for (const key of ["t", "target_t", "control_t", "policy_t", "write_t", "depth_t", "spectrum_frequency"]) {
      if (arrays[key] && (arrays[key].shape.length !== 1 || !validClock(arrays[key].data))) throw new Error(
        `时间或频率轴无效：${key}`);
    }
    const groups = {
      t: ["q", "dq", "ddq", "tau_est", "temperature", "vol", "motorstate", "orientation_rpy", "gyroscope",
        "accelerometer", "receipt_interval_ms", "robot_interval_ms"
      ],
      target_t: ["target_q", "target_dq", "target_kp", "target_kd", "target_tau"],
      control_t: ["command", "heading_target", "yaw", "heading_error", "state_age_ms", "control_interval_ms"],
      policy_t: ["action_raw", "observation", "inference_ms"],
      write_t: ["write_ms", "write_success"],
      depth_t: ["depth_age_ms"],
      spectrum_frequency: ["spectrum_density"],
    };
    for (const [clock, signals] of Object.entries(groups))
      for (const key of signals) {
        if (arrays[key] && (!arrays[clock] || arrays[key].shape[0] !== arrays[clock].data.length)) throw new Error(
          `信号与时间轴不匹配：${key}`);
      }
    if (arrays.q.shape[1] < 1 || arrays.q.shape[1] > 256) throw new Error("关节通道数量无效。");
    if (!manifest.metadata || !manifest.summary || !Array.isArray(manifest.summary.events)) throw new Error(
      "文件缺少诊断元数据。");
    for (const key of ["motor_names", "policy_joint_names"])
      if (manifest.metadata[key] && (!Array.isArray(manifest.metadata[key]) || manifest.metadata[key].some(name =>
          typeof name !== "string"))) throw new Error("关节名称格式无效。");
    return {
      ...manifest,
      arrays,
      filename: file.name
    };
  }

  async function loadFile(file) {
    if (!file) return;
    const version = ++loadVersion;
    $("loading").hidden = false;
    showMessage("");
    try {
      const result = await readBundle(file);
      if (version === loadVersion) install(result);
    } catch (error) {
      if (version === loadVersion) showMessage("无法打开文件：" + (error.message || "文件损坏。请重新运行分析工具导出 .deploynose 文件。"));
    } finally {
      if (version === loadVersion) $("loading").hidden = true;
    }
  }

  function install(result) {
    dataset = result;
    derivatives = new Map();
    const {
      metadata: m,
      summary: s
    } = dataset;
    const n = array("q").shape[1];
    motorNames = Array.from({
      length: n
    }, (_, i) => m.motor_names?.[i] || `motor_${i}（未映射槽位）`);
    const actions = array("action_raw")?.shape[1] || 0;
    actionNames = Array.from({
      length: actions
    }, (_, i) => m.policy_joint_names?.[i] || `action_${i}`);
    fillSelect("joint-select", motorNames);
    fillSelect("compare-select", motorNames, true);
    fillSelect("action-select", actionNames);
    $("joint-select").value = String(Math.min(3, n - 1));
    $("observation-select").value = "0";
    $("observation-select").max = String(Math.max(0, (array("observation")?.shape[1] || 1) - 1));
    session.events = s.events.filter(event => Number.isFinite(event.time_s) && typeof event.name === "string");
    const times = Object.keys(dataset.arrays).filter(key => key === "t" || key.endsWith("_t")).map(values).filter(a =>
      a.length);
    session.fullRange = [Math.min(...times.map(a => a[0])), Math.max(...times.map(a => a[a.length - 1]))];
    if (session.fullRange[0] === session.fullRange[1]) session.fullRange[1] += 1;
    session.range = [...session.fullRange];
    session.cursor = null;
    $("range-start").value = session.range[0].toFixed(4);
    $("range-end").value = session.range[1].toFixed(4);
    const source = String(m.arguments?.depth || m.arguments?.policy || m.policy_path || "未命名 policy").split(/[\\/]/)
      .filter(Boolean).pop();
    $("side-policy").textContent = source;
    $("side-date").textContent = m.wall_time_ns ? new Date(m.wall_time_ns / 1e6).toLocaleString("zh-CN", {
      hour12: false
    }) : "时间未记录";
    const mode = m.demo ? "演示 · 合成数据" : m.arguments?.sim === "sync" ? "仿真 · 同步模式" : m.arguments?.real ? "real 部署模式" :
      "外部 DDS 数据源";
    $("side-mode").textContent = mode;
    $("side-samples").textContent = values("t").length.toLocaleString();
    $("side-joints").textContent = `${m.motor_names?.length || n} 关节 / ${n} 槽位`;
    $("side-hash").textContent = m.policy_sha256 ? String(m.policy_sha256).slice(0, 20) + "…" : "未提供";
    $("run-title").textContent = source + " / " + dataset.filename;
    $("run-info").hidden = false;
    $("empty-state").hidden = true;
    $("workspace").hidden = false;
    const dropped = Object.values(s.dropped || {}).reduce((sum, count) => sum + (Number(count) || 0), 0);
    const incomplete = !s.clean_shutdown || !!s.writer_error || dropped > 0;
    $("integrity-badge").className = "badge" + (incomplete ? " warn" : "");
    $("integrity-badge").textContent = m.demo ? "演示数据" : incomplete ? "记录存在缺口" : "记录已正常关闭";
    $("notices").replaceChildren();
    if (m.demo) notice("当前为合成演示，用于体验操作；曲线不来自真实机器人。", true);
    if (incomplete) notice(
      `记录完整性：${s.clean_shutdown ? "正常关闭" : "未正常关闭"}；队列丢弃 ${dropped} 条。${s.writer_error ? "写盘错误：" + s.writer_error : "请结合时间轴和原始日志判断缺失区间。"}`
      );
    if (s.robot_tick_gaps || s.duplicate_or_reset_ticks) notice(
      `机器人时间戳：${s.robot_tick_gaps || 0} 个间隔缺口，${s.duplicate_or_reset_ticks || 0} 个重复或回退。频谱已排除不连续窗口。`);
    if (s.nonfinite_state_values) notice(`状态数据包含 ${s.nonfinite_state_values} 个非有限值；曲线保留缺口，不把缺失值替换成零。`);
    if (m.arguments?.sim === "sync" && !m.demo) notice("同步仿真：宿主机运行速度可快于仿真时间。这里的控制间隔是宿主机耗时，不代表真机控制频率。频谱使用机器人 tick。",
      true);
    notice("当前数据没有机身实测线速度或位置；可以检查关节速度和偏航响应，不能据此判断真实行走速度与轨迹。", true);
    $("metrics").replaceChildren();
    metric("状态采样跨度", number(s.recorded_seconds), "s", "首条至末条状态 · 宿主机");
    metric("状态样本", values("t").length.toLocaleString(), "", `${n} 个协议电机槽位`);
    metric("控制周期 P99", number(s.control_interval_ms?.p99), "ms", "完整 policy 阶段 · 宿主机");
    metric("偏航误差 P95", number(s.heading_error_deg?.p95), "°", "IMU 航向 · 完整 policy 阶段");
    showMessage("");
    renderView();
  }

  function selectedJoints() {
    const indices = [Number($("joint-select").value), Number($("compare-select").value)];
    return [...new Set(indices)].filter(i => Number.isInteger(i) && i >= 0 && i < motorNames.length);
  }

  function jointVelocityRate(index) {
    if (derivatives.has(index)) return derivatives.get(index);
    if (!array("dq") || !array("robot_interval_ms")) return null;
    const rate = new Float32Array(values("t").length).fill(NaN),
      dt = values("robot_interval_ms");
    const positive = Array.from(dt).filter(v => v > 0 && Number.isFinite(v)).sort((a, b) => a - b);
    const typical = positive[Math.floor(positive.length / 2)];
    const dq = trace("dq", index, "");
    if (!dq) return null;
    for (let i = 1; i < rate.length; i++)
      if (dt[i] > 0 && dt[i] <= typical * 1.5) rate[i] = (dq.y[i * dq.stride + index] - dq.y[(i - 1) * dq.stride +
        index]) / (dt[i] * 0.001);
    derivatives.set(index, rate);
    return rate;
  }

  function imuCharts() {
    const axes = ["x", "y", "z"];
    const heading = values("heading_target");
    const wrappedTarget = heading && values("control_t") ? {
      x: values("control_t"),
      y: Float64Array.from(heading, value => Math.atan2(Math.sin(value), Math.cos(value))),
      name: "目标航向",
      scale: 180 / Math.PI,
      step: true,
    } : null;
    panel("机身姿态", "°", ["roll", "pitch", "yaw"].map((name, i) => trace("orientation_rpy", i, name, "t", 180 / Math
      .PI)), {
      subtitle: "由 IMU 四元数计算；角度跨越 ±180° 时会跳变"
    });
    panel("航向跟踪", "°", [wrappedTarget, trace("yaw", 0, "IMU 航向", "control_t", 180 / Math.PI)], {
      subtitle: "目标和 IMU 航向均包裹至 ±180°；IMU 并非外部定位真值"
    });
    panel("偏航误差", "°", [trace("heading_error", 0, "包裹后的目标 − IMU", "control_t", 180 / Math.PI)], {
      reference: 0
    });
    panel("角速度", "rad/s", axes.map((name, i) => trace("gyroscope", i, name)));
    panel("IMU 加速度", "m/s²", axes.map((name, i) => trace("accelerometer", i, name)), {
      subtitle: "传感器坐标系原始输出，未去除重力；未积分为线速度"
    });
    panel("转向指令与响应", "rad/s", [trace("command", 2, "yaw rate 指令", "control_t", 1, {
      step: true
    }), trace("gyroscope", 2, "IMU 机身 z 轴角速度")], {
      subtitle: "机身倾斜时，z 轴角速度与世界航向变化率并不相同"
    });
  }

  function timingCharts() {
    const reference = dataset.metadata.arguments?.sim === "sync" ? undefined : dataset.summary.expected_control_ms;
    panel("控制循环周期", "ms", [trace("control_interval_ms", 0, "相邻控制步间隔", "control_t")], {
      reference,
      subtitle: "宿主机控制步起始时间之差"
    });
    panel("Policy 推理耗时", "ms", [trace("inference_ms", 0, "模型调用耗时", "policy_t")]);
    panel("输入状态年龄", "ms", [trace("state_age_ms", 0, "采集输入时距状态回调的时间", "control_t")], {
      subtitle: "仅表示本机已知年龄，不包含机器人到本机的传输延迟"
    });
    panel("DDS 写入耗时", "ms", [trace("write_ms", 0, "本地 Write 调用", "write_t")]);
    panel("深度帧年龄", "ms", [trace("depth_age_ms", 0, "部署端使用的帧年龄", "depth_t")], {
      reference: Number.isFinite(dataset.metadata.arguments?.depth_max_age) ? dataset.metadata.arguments
        .depth_max_age * 1000 : undefined
    });
    panel("状态接收间隔", "ms", [trace("receipt_interval_ms", 0, "本机回调间隔"), trace("robot_interval_ms", 0, "机器人 tick 间隔")], {
      subtitle: "两种时钟分开保留，不推断网络单程延迟"
    });
    panel("指令写入结果", "0 / 1", [trace("write_success", 0, "0 失败 / 1 成功", "write_t", 1, {
      step: true
    })], {
      subtitle: "本地 DDS 写入结果；不是电机执行确认"
    });
  }

  function renderView() {
    if (!dataset) return;
    charts.forEach(chart => chart.destroy());
    charts = [];
    $("charts").replaceChildren();
    $("details").replaceChildren();
    $("breadcrumb").textContent = views[currentView][0];
    $("page-title").textContent = views[currentView][0];
    $("page-eyebrow").textContent = views[currentView][1];
    document.querySelectorAll("nav button").forEach(button => {
      button.classList.toggle("active", button.dataset.view === currentView);
      button.setAttribute("aria-current", button.dataset.view === currentView ? "page" : "false");
    });
    $("toolbar").hidden = currentView === "events";
    $("selectors").hidden = !["joints", "policy"].includes(currentView);
    for (const id of ["joint-label", "compare-label"]) $(id).hidden = currentView !== "joints";
    for (const id of ["action-label", "observation-label"]) $(id).hidden = currentView !== "policy";
    if (currentView === "overview") {
      panel("机身姿态", "°", [trace("orientation_rpy", 0, "roll", "t", 180 / Math.PI), trace("orientation_rpy", 1,
        "pitch", "t", 180 / Math.PI)]);
      panel("偏航误差", "°", [trace("heading_error", 0, "目标 − IMU 航向", "control_t", 180 / Math.PI)], {
        reference: 0
      });
      panel("速度指令", "m/s", [trace("command", 0, "vx 指令", "control_t", 1, {
        step: true
      }), trace("command", 1, "vy 指令", "control_t", 1, {
        step: true
      })], {
        subtitle: "操作指令，不是机器人实测行走速度"
      });
      panel("控制周期与状态年龄", "ms", [trace("control_interval_ms", 0, "控制周期", "control_t"), trace("state_age_ms", 0, "状态年龄",
        "control_t")]);
      eventTable(6);
    } else if (currentView === "joints") {
      const joints = selectedJoints();
      const series = (key, scale = 1) => joints.map((joint, i) => trace(key, joint, motorNames[joint], "t", scale, {
        color: colors[i * 2]
      }));
      panel("关节位置与目标", "rad", joints.flatMap((joint, i) => [trace("q", joint, motorNames[joint] + " / 反馈", "t", 1, {
        color: colors[i * 2]
      }), trace("target_q", joint, motorNames[joint] + " / 目标", "target_t", 1, {
        step: true,
        color: colors[i * 2 + 1]
      })]), {
        wide: true,
        subtitle: "目标对应 DDS 写入尝试；反馈与目标各自使用原始时间戳"
      });
      panel("关节速度", "rad/s", series("dq"));
      panel("关节速度变化率", "rad/s²", joints.map((joint, i) => {
        const y = jointVelocityRate(joint);
        return y ? {
          x: values("t"),
          y,
          name: motorNames[joint],
          color: colors[i * 2]
        } : null;
      }), {
        subtitle: "由 dq 与机器人 tick 差分；不跨越重复或缺失 tick"
      });
      panel("估计力矩", "N·m", series("tau_est"), {
        subtitle: "电机 tau_est，不是外部力矩传感器真值"
      });
      const channels = array("temperature")?.shape[2] || 1;
      panel("电机温度", "°C", joints.flatMap(joint => Array.from({
        length: channels
      }, (_, channel) => trace("temperature", joint * channels + channel,
        `${motorNames[joint]} / 温度 ${channel + 1}`))), {
        subtitle: "保留 SDK 两个温度通道，不猜测传感器位置"
      });
      panel("关节速度频谱", "(rad/s)²/Hz", joints.map((joint, i) => trace("spectrum_density", joint, motorNames[joint],
        "spectrum_frequency", 1, {
          color: colors[i * 2]
        })), {
        spectral: true,
        logY: true,
        wide: true,
        subtitle: "完整 policy 阶段的 Welch 频谱，不随时间选区重算；高频峰值不等同于故障",
        empty: "没有足够的连续、等间隔有效样本用于频谱分析"
      });
      if (joints.some(i => i >= (dataset.metadata.motor_names?.length || 29))) detailNote("未映射槽位",
        "协议包含额外电机槽位，未映射的零值不能说明对应硬件正常或存在。");
      if (joints.some(i => i === 13 || i === 14) && values("mode_machine")?.includes(6)) detailNote("27-DoF 机型",
        "这份记录出现 mode_machine=6；该机型没有 waist roll / pitch，相关槽位可能持续为零。");
    } else if (currentView === "imu") imuCharts();
    else if (currentView === "timing") timingCharts();
    else if (currentView === "policy") {
      const action = Number($("action-select").value),
        input = Number($("observation-select").value);
      panel("Policy 原始 action", "无量纲", [trace("action_raw", action, actionNames[action] || "action", "policy_t")], {
        wide: true,
        subtitle: "模型原始输出，位于 action clipping、缩放和关节映射之前"
      });
      panel("实际模型输入", "原始输入单位", [Number.isInteger(input) && input >= 0 ? trace("observation", input,
        `input_0[${input}]`, "policy_t") : null], {
        wide: true,
        subtitle: "按模型 input_0 展平后的索引查看；保留原始观测历史排列"
      });
      panel("模型推理耗时", "ms", [trace("inference_ms", 0, "推理", "policy_t")]);
      panel("深度帧年龄", "ms", [trace("depth_age_ms", 0, "深度输入年龄", "depth_t")]);
      detailNote("Policy 输入范围", "本文件包含曲线分析所需的数值输入。完整深度图堆叠与原始 DDS 消息仍保存在部署目录的 recording.sqlite3 中。");
    } else {
      eventTable();
      for (const [title, value] of [
          ["数据完整性与统计", dataset.summary],
          ["部署配置与 policy 标识", dataset.metadata]
        ]) {
        const section = document.createElement("section");
        section.className = "detail-card";
        const heading = document.createElement("h3");
        heading.textContent = title;
        const pre = document.createElement("pre");
        pre.textContent = JSON.stringify(value, null, 2);
        section.append(heading, pre);
        $("details").append(section);
      }
    }
  }

  function detailNote(title, text) {
    const card = document.createElement("section");
    card.className = "detail-card";
    const heading = document.createElement("h3");
    heading.textContent = title;
    const content = document.createElement("p");
    content.textContent = text;
    card.append(heading, content);
    $("details").append(card);
  }

  function eventTable(limit = Infinity) {
    const section = document.createElement("section");
    section.className = "detail-card";
    const heading = document.createElement("h3");
    heading.textContent = "记录事件";
    const table = document.createElement("table");
    table.className = "events-table";
    const row = document.createElement("tr");
    for (const name of ["时间 / s", "事件", "详情"]) {
      const th = document.createElement("th");
      th.textContent = name;
      row.append(th);
    }
    const head = document.createElement("thead");
    head.append(row);
    table.append(head);
    const body = document.createElement("tbody");
    for (const event of session.events.slice(0, limit)) {
      const tr = document.createElement("tr"),
        time = document.createElement("td"),
        name = document.createElement("td"),
        info = document.createElement("td");
      const button = document.createElement("button");
      button.textContent = event.time_s.toFixed(3);
      button.title = "查看事件附近的曲线";
      button.addEventListener("click", () => {
        currentView = "overview";
        renderView();
        const center = Math.max(session.fullRange[0], Math.min(session.fullRange[1], event.time_s));
        session.setRange(center - 1, center + 1);
        window.scrollTo({
          top: 0,
          behavior: "smooth"
        });
      });
      time.append(button);
      name.textContent = event.name;
      const {
        time_s,
        name: eventName,
        ...extra
      } = event;
      info.textContent = Object.keys(extra).length ? JSON.stringify(extra) : "—";
      tr.append(time, name, info);
      body.append(tr);
    }
    table.append(body);
    section.append(heading, table);
    if (!session.events.length) {
      const p = document.createElement("p");
      p.textContent = "该文件没有事件记录。";
      section.append(p);
    }
    $("details").append(section);
  }

  function demo() {
    ++loadVersion;
    $("loading").hidden = true;
    const arrays = Object.create(null),
      count = 2400,
      dt = 0.01,
      steps = count / 2,
      joints = 29;

    function make(key, rows, columns, fn) {
      const data = new Float64Array(rows * columns);
      for (let i = 0; i < rows; i++)
        for (let j = 0; j < columns; j++) data[i * columns + j] = fn(i, j);
      arrays[key] = {
        shape: columns === 1 ? [rows] : [rows, columns],
        data
      };
      return data;
    }
    const t = make("t", count, 1, i => i * dt);
    const bump = time => Math.exp(-(((time - 13.5) / 0.6) ** 2));
    make("q", count, joints, (i, j) => .2 * Math.sin(t[i] * 6 + j * .7) + .025 * bump(t[i]) * Math.sin(t[i] * 125));
    make("dq", count, joints, (i, j) => 1.2 * Math.cos(t[i] * 6 + j * .7) + 3 * bump(t[i]) * Math.cos(t[i] * 125));
    make("tau_est", count, joints, (i, j) => 5 * Math.sin(t[i] * 6 + j * .7) + 3 * bump(t[i]));
    make("temperature", count, joints * 2, (i, j) => 32 + j / 10 + i / 1000);
    arrays.temperature.shape = [count, joints, 2];
    make("orientation_rpy", count, 3, (i, j) => .04 * Math.sin(t[i] * (j + 1)) + (j === 2 ? .12 * bump(t[i]) : 0));
    make("gyroscope", count, 3, (i, j) => .04 * (j + 1) * Math.cos(t[i] * (j + 1)));
    make("accelerometer", count, 3, (i, j) => (j === 2 ? 9.81 : 0) + .5 * Math.sin(t[i] * 6 + j));
    make("robot_interval_ms", count, 1, i => i ? 10 : NaN);
    make("receipt_interval_ms", count, 1, i => i ? 10 + Math.sin(i) * .3 : NaN);
    for (const key of ["control_t", "policy_t", "target_t", "depth_t", "write_t"]) make(key, steps, 1, i => i * .02);
    make("target_q", steps, joints, (i, j) => .2 * Math.sin(i * .12 + j * .7 + .08));
    make("command", steps, 3, (i, j) => j === 0 ? (i < 150 ? .1 : .4) : 0);
    make("heading_error", steps, 1, i => -.04 * Math.sin(i * .06) - .12 * bump(i * .02));
    make("heading_target", steps, 1, () => 0);
    make("yaw", steps, 1, i => -arrays.heading_error.data[i]);
    make("control_interval_ms", steps, 1, i => i ? 20 + .2 * Math.sin(i) + (i === 680 ? 18 : 0) : NaN);
    make("state_age_ms", steps, 1, i => 1.2 + .3 * Math.sin(i * .7));
    make("inference_ms", steps, 1, i => .6 + .08 * Math.sin(i));
    make("write_ms", steps, 1, i => .04 + .005 * Math.sin(i));
    make("write_success", steps, 1, () => 1);
    make("depth_age_ms", steps, 1, i => 20 + 8 * Math.sin(i * .06));
    make("action_raw", steps, joints, (i, j) => 2 * Math.sin(i * .12 + j * .7));
    make("observation", steps, 6, (i, j) => Math.sin(i * .12 + j));
    make("spectrum_frequency", 129, 1, i => i * 50 / 128);
    make("spectrum_density", 129, joints, (i, j) => 1e-6 + .3 * Math.exp(-(((i * 50 / 128 - 1) / .5) ** 2)) + .006 *
      Math.exp(-(((i * 50 / 128 - 20) / 1.5) ** 2)) * (1 + j / 20));
    const names = ["left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee", "left_ankle_pitch",
      "left_ankle_roll", "right_hip_pitch", "right_hip_roll", "right_hip_yaw", "right_knee", "right_ankle_pitch",
      "right_ankle_roll", "waist_yaw", "waist_roll", "waist_pitch", "left_shoulder_pitch", "left_shoulder_roll",
      "left_shoulder_yaw", "left_elbow", "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
      "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow", "right_wrist_roll",
      "right_wrist_pitch", "right_wrist_yaw"
    ];
    const percentile = (key, fraction, scale = 1) => {
      const data = Array.from(arrays[key].data).filter(Number.isFinite).map(value => Math.abs(value * scale)).sort((
        a, b) => a - b);
      const index = (data.length - 1) * fraction,
        lo = Math.floor(index),
        hi = Math.ceil(index);
      return data[lo] + (data[hi] - data[lo]) * (index - lo);
    };
    install({
      filename: "演示记录 · 非真机数据",
      arrays,
      metadata: {
        demo: true,
        arguments: {
          depth: "demo_depth_policy",
          depth_max_age: .2
        },
        motor_names: names,
        policy_joint_names: names,
        control_dt: .02
      },
      summary: {
        samples: count,
        recorded_seconds: t[count - 1],
        clean_shutdown: true,
        dropped: {},
        expected_control_ms: 20,
        control_interval_ms: {
          p99: percentile("control_interval_ms", .99)
        },
        heading_error_deg: {
          p95: percentile("heading_error", .95, 180 / Math.PI)
        },
        events: [{
          time_s: 0,
          name: "policy_start"
        }, {
          time_s: 3,
          name: "command_change",
          vx: .4
        }, {
          time_s: 13.5,
          name: "demo_marker",
          note: "合成振动片段"
        }, {
          time_s: 23.98,
          name: "policy_stop"
        }]
      }
    });
  }

  $("open-file").addEventListener("click", () => $("file-input").click());
  $("drop-zone").addEventListener("click", () => $("file-input").click());
  $("drop-zone").addEventListener("keydown", event => {
    if (["Enter", " "].includes(event.key)) {
      event.preventDefault();
      $("file-input").click();
    }
  });
  $("file-input").addEventListener("change", event => {
    loadFile(event.target.files[0]);
    event.target.value = "";
  });
  $("demo-button").addEventListener("click", demo);
  document.querySelector(".brand").addEventListener("click", event => {
    event.preventDefault();
    currentView = "overview";
    if (dataset) renderView();
    window.scrollTo(0, 0);
  });
  document.querySelectorAll("nav button").forEach(button => button.addEventListener("click", () => {
    if (!dataset) {
      showMessage("先导入 .deploynose 文件，或点击“查看演示数据”。");
      return;
    }
    currentView = button.dataset.view;
    renderView();
  }));
  for (const id of ["joint-select", "compare-select", "action-select", "observation-select"]) $(id).addEventListener(
    "change", renderView);
  $("reset-range").addEventListener("click", () => session.reset());
  $("apply-range").addEventListener("click", () => {
    const start = Number($("range-start").value),
      end = Number($("range-end").value);
    if (start >= end || start >= session.fullRange[1] || end <= session.fullRange[0]) showMessage(
      "请输入与记录范围重叠的有效时间窗口，结束时间应大于开始时间。");
    else {
      showMessage("");
      session.setRange(start, end);
    }
  });
  let dragDepth = 0;
  window.addEventListener("dragenter", event => {
    if (event.dataTransfer?.types.includes("Files")) {
      event.preventDefault();
      dragDepth++;
      $("drop-overlay").hidden = false;
    }
  });
  window.addEventListener("dragover", event => {
    event.preventDefault();
    if (event.dataTransfer) event.dataTransfer.dropEffect = "copy";
  });
  window.addEventListener("dragleave", () => {
    dragDepth = Math.max(0, dragDepth - 1);
    if (!dragDepth) $("drop-overlay").hidden = true;
  });
  window.addEventListener("drop", event => {
    event.preventDefault();
    dragDepth = 0;
    $("drop-overlay").hidden = true;
    if (event.dataTransfer.files.length !== 1) showMessage("请一次拖入一个 .deploynose 文件。");
    else loadFile(event.dataTransfer.files[0]);
  });
  window.addEventListener("keydown", event => {
    if (["INPUT", "SELECT", "TEXTAREA"].includes(event.target.tagName) || event.ctrlKey || event.metaKey || event
      .altKey) return;
    if (event.key.toLowerCase() === "o") {
      event.preventDefault();
      $("file-input").click();
    }
    if (event.key === "Escape" && dataset) session.reset();
  });
})();
