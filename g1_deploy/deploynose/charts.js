"use strict";

// Each trace keeps its own sampling clock. Rendering reduces points per pixel,
// retaining extrema so short spikes remain visible when a long run is zoomed out.
window.DeploynoseCharts = (() => {
  const colors = ["#268b71", "#d78a42", "#637eb5", "#b77498", "#819c4b", "#557a85"];
  const number = value => {
    if (!Number.isFinite(value)) return "—";
    if (value !== 0 && (Math.abs(value) >= 10000 || Math.abs(value) < 0.001)) return value.toExponential(2);
    return Number(value.toPrecision(5)).toString();
  };

  function lowerBound(values, target) {
    let lo = 0,
      hi = values.length;
    while (lo < hi) {
      const mid = (lo + hi) >>> 1;
      if (values[mid] < target) lo = mid + 1;
      else hi = mid;
    }
    return lo;
  }

  function valueAt(series, index) {
    return series.y[index * (series.stride || 1) + (series.column || 0)] * (series.scale ?? 1);
  }

  function download(blob, filename) {
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  class TraceChart {
    constructor(parent, specification, session) {
      this.spec = specification;
      this.session = session;
      this.series = specification.series.filter(s => s && s.x.length && s.y.length);
      this.spectral = !!specification.spectral;
      this.card = document.createElement("article");
      this.card.className = "chart-card" + (specification.wide ? " wide" : "");
      const header = document.createElement("div");
      header.className = "chart-header";
      const caption = document.createElement("div");
      const title = document.createElement("h3");
      title.textContent = specification.title;
      const subtitle = document.createElement("p");
      subtitle.textContent = specification.subtitle || specification.unit;
      caption.append(title, subtitle);
      const actions = document.createElement("div");
      actions.className = "chart-actions";
      for (const [name, handler] of [
          ["CSV", () => this.exportCSV()],
          ["PNG", () => this.exportPNG()]
        ]) {
        const button = document.createElement("button");
        button.textContent = name;
        button.setAttribute("aria-label", `${specification.title} 导出 ${name}`);
        button.addEventListener("click", handler);
        actions.append(button);
      }
      header.append(caption, actions);
      this.card.append(header);
      parent.append(this.card);
      if (!this.series.length) {
        const empty = document.createElement("div");
        empty.className = "chart-empty";
        empty.textContent = specification.empty || "这份记录没有该信号";
        this.card.append(empty);
        actions.hidden = true;
        return;
      }
      const legend = document.createElement("div");
      legend.className = "legend";
      this.series.forEach((s, i) => {
        s.color ||= colors[i % colors.length];
        const label = document.createElement("span"),
          swatch = document.createElement("i");
        swatch.style.backgroundColor = s.color;
        label.append(swatch, document.createTextNode(s.name));
        legend.append(label);
      });
      this.card.append(legend);
      this.plot = document.createElement("div");
      this.plot.className = "plot";
      this.canvas = document.createElement("canvas");
      this.canvas.setAttribute("aria-label", specification.title + "，可导出 CSV 查看精确数值");
      this.overlay = document.createElement("canvas");
      this.overlay.className = "overlay";
      this.overlay.tabIndex = 0;
      this.overlay.setAttribute("aria-label", specification.title + "：拖动选区，双击重置，也可使用上方时间输入框");
      this.tip = document.createElement("div");
      this.tip.className = "chart-tooltip";
      this.tip.hidden = true;
      this.plot.append(this.canvas, this.overlay, this.tip);
      this.card.append(this.plot);
      this.stats = document.createElement("table");
      this.stats.className = "stats";
      this.card.append(this.stats);
      this.resize = new ResizeObserver(() => this.draw());
      this.resize.observe(this.plot);
      this.overlay.addEventListener("pointermove", event => this.pointerMove(event));
      this.overlay.addEventListener("pointerleave", () => {
        if (this.drag == null) {
          this.tip.hidden = true;
          this.session.hover(null);
        }
      });
      this.overlay.addEventListener("pointerdown", event => {
        if (event.button || this.spectral) return;
        this.drag = this.pixel(event);
        this.dragEnd = this.drag;
        this.overlay.setPointerCapture(event.pointerId);
      });
      this.overlay.addEventListener("pointerup", event => {
        if (this.drag == null) return;
        const a = this.time(this.drag),
          b = this.time(this.pixel(event));
        if (Math.abs(this.pixel(event) - this.drag) > 5) this.session.setRange(Math.min(a, b), Math.max(a,
        b));
        this.drag = null;
        this.overlay.releasePointerCapture(event.pointerId);
        this.drawOverlay();
      });
      this.overlay.addEventListener("pointercancel", () => {
        this.drag = null;
        this.drawOverlay();
      });
      this.overlay.addEventListener("dblclick", () => {
        if (!this.spectral) this.session.reset();
      });
      this.overlay.addEventListener("wheel", event => {
        if (this.spectral) return;
        event.preventDefault();
        const center = this.time(this.pixel(event)),
          factor = Math.exp(Math.sign(event.deltaY) * 0.2);
        this.session.setRange(center - (center - this.xmin) * factor, center + (this.xmax - center) * factor);
      }, {
        passive: false
      });
    }
    destroy() {
      this.resize?.disconnect();
    }
    pixel(event) {
      return Math.max(this.left, Math.min(this.width - this.right, event.clientX - this.overlay
        .getBoundingClientRect().left));
    }
    time(pixel) {
      return this.xmin + (pixel - this.left) / this.innerWidth * (this.xmax - this.xmin);
    }
    x(value) {
      return this.left + (value - this.xmin) / (this.xmax - this.xmin) * this.innerWidth;
    }
    y(value) {
      if (this.spec.logY) value = Math.log10(value);
      return this.top + (this.ymax - value) / (this.ymax - this.ymin) * this.innerHeight;
    }
    draw() {
      if (!this.canvas) return;
      this.width = this.plot.clientWidth;
      this.height = this.plot.clientHeight;
      if (!this.width) return;
      const ratio = window.devicePixelRatio || 1;
      for (const canvas of [this.canvas, this.overlay]) {
        canvas.width = Math.round(this.width * ratio);
        canvas.height = Math.round(this.height * ratio);
        canvas.getContext("2d").setTransform(ratio, 0, 0, ratio, 0, 0);
      }
      this.left = 58;
      this.right = 16;
      this.top = 17;
      this.bottom = 32;
      this.innerWidth = this.width - this.left - this.right;
      this.innerHeight = this.height - this.top - this.bottom;
      if (this.spectral) {
        this.xmin = 0;
        this.xmax = Math.max(...this.series.map(s => s.x[s.x.length - 1])) || 1;
      } else [this.xmin, this.xmax] = this.session.range;
      let low = Infinity,
        high = -Infinity;
      for (const series of this.series) {
        const start = Math.max(0, lowerBound(series.x, this.xmin) - 1);
        const end = Math.min(series.x.length, lowerBound(series.x, this.xmax) + 1);
        for (let i = start; i < end; i++) {
          let value = valueAt(series, i);
          if (!Number.isFinite(value) || (this.spec.logY && value <= 0)) continue;
          if (this.spec.logY) value = Math.log10(value);
          low = Math.min(low, value);
          high = Math.max(high, value);
        }
      }
      if (Number.isFinite(this.spec.reference)) {
        low = Math.min(low, this.spec.reference);
        high = Math.max(high, this.spec.reference);
      }
      const hasValues = Number.isFinite(low);
      if (!hasValues) {
        low = -1;
        high = 1;
      }
      const padding = (high - low || Math.max(Math.abs(low) * 0.1, 0.1)) * 0.12;
      this.ymin = low - padding;
      this.ymax = high + padding;
      const ctx = this.canvas.getContext("2d");
      ctx.clearRect(0, 0, this.width, this.height);
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, this.width, this.height);
      ctx.font = '10px "SFMono-Regular", Consolas, monospace';
      ctx.lineWidth = 1;
      for (let i = 0; i <= 4; i++) {
        const y = this.top + this.innerHeight * i / 4;
        ctx.strokeStyle = "#edf1ef";
        ctx.beginPath();
        ctx.moveTo(this.left, y);
        ctx.lineTo(this.width - this.right, y);
        ctx.stroke();
        const value = this.ymax - (this.ymax - this.ymin) * i / 4;
        ctx.fillStyle = "#8d9d94";
        ctx.textAlign = "right";
        ctx.fillText(this.spec.logY ? (10 ** value).toExponential(0) : number(value), this.left - 9, y + 3);
      }
      const tickCount = this.width < 420 ? 3 : 5;
      for (let i = 0; i <= tickCount; i++) {
        const x = this.left + this.innerWidth * i / tickCount;
        ctx.textAlign = i === 0 ? "left" : i === tickCount ? "right" : "center";
        ctx.fillText(number(this.time(x)), x, this.height - 11);
      }
      ctx.save();
      ctx.beginPath();
      ctx.rect(this.left, this.top, this.innerWidth, this.innerHeight);
      ctx.clip();
      if (!this.spectral) {
        for (const event of this.session.events) {
          if (event.time_s < this.xmin || event.time_s > this.xmax) continue;
          ctx.strokeStyle = "#cfdad4";
          ctx.setLineDash([3, 4]);
          ctx.beginPath();
          ctx.moveTo(this.x(event.time_s), this.top);
          ctx.lineTo(this.x(event.time_s), this.top + this.innerHeight);
          ctx.stroke();
        }
      }
      if (Number.isFinite(this.spec.reference)) {
        ctx.strokeStyle = "#a7b2aa";
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.moveTo(this.left, this.y(this.spec.reference));
        ctx.lineTo(this.width - this.right, this.y(this.spec.reference));
        ctx.stroke();
      }
      ctx.setLineDash([]);
      for (const series of this.series) this.drawSeries(ctx, series);
      ctx.setLineDash([]);
      this.updateStats();
      ctx.restore();
      if (!hasValues) {
        ctx.fillStyle = "#91a198";
        ctx.textAlign = "center";
        ctx.fillText("此时间窗口没有有效样本", this.width / 2, this.height / 2);
      }
      this.drawOverlay();
    }
    // Min, max, peak-to-peak, mean and RMS over whatever the chart is currently showing, so
    // zooming into one stride gives that stride's numbers rather than the whole run's.
    summarize(series) {
      const start = Math.max(0, lowerBound(series.x, this.xmin));
      const end = Math.min(series.x.length, lowerBound(series.x, this.xmax) + 1);
      let min = Infinity, max = -Infinity, sum = 0, sumSq = 0, n = 0;
      for (let i = start; i < end; i++) {
        const value = valueAt(series, i);
        if (!Number.isFinite(value)) continue;
        if (value < min) min = value;
        if (value > max) max = value;
        sum += value;
        sumSq += value * value;
        n++;
      }
      if (!n) return null;
      const mean = sum / n;
      return {
        n,
        min,
        max,
        p2p: max - min,
        mean,
        rms: Math.sqrt(Math.max(sumSq / n - mean * mean, 0))
      };
    }

    updateStats() {
      if (!this.stats) return;
      const unit = this.spec.unit ? ` [${this.spec.unit}]` : "";
      const columns = [
        ["最小", "窗口内的最小值"],
        ["最大", "窗口内的最大值"],
        ["峰峰", "最大 − 最小。只由两个采样点决定，一个毛刺就能把它顶高"],
        ["均值", "窗口内的算术平均，告诉你信号偏在哪一侧"],
        ["RMS", "绕均值的标准差。每个采样都参与，衡量平均摆动幅度；和峰峰差得多说明是偶发尖峰"],
      ];
      const rows = [];
      const header = document.createElement("tr");
      const first = document.createElement("th");
      first.textContent = "曲线" + unit;
      header.append(first);
      for (const [name, help] of columns) {
        const cell = document.createElement("th");
        cell.textContent = name;
        cell.title = help;
        header.append(cell);
      }
      // Keeps the column count equal to the data rows, which carry a trailing ratio cell.
      header.append(document.createElement("th"));
      rows.push(header);
      let previous = null;
      for (const series of this.series) {
        const s = this.summarize(series);
        if (!s) continue;
        const row = document.createElement("tr");
        const label = document.createElement("td");
        const swatch = document.createElement("i");
        swatch.style.backgroundColor = series.color;
        if (series.dash) swatch.classList.add("dashed");
        label.append(swatch, document.createTextNode(series.name));
        row.append(label);
        for (const value of [s.min, s.max, s.p2p, s.mean, s.rms]) {
          const cell = document.createElement("td");
          cell.textContent = number(value);
          row.append(cell);
        }
        if (series.dash && previous) {
          const ratio = document.createElement("td");
          ratio.className = "ratio";
          ratio.textContent = `峰峰 ${number(previous.p2p / s.p2p)}× · RMS ${number(previous.rms / s.rms)}×`;
          ratio.title = "主记录相对这条对比曲线的倍数";
          row.append(ratio);
        } else {
          row.append(document.createElement("td"));
          previous = s;
        }
        rows.push(row);
      }
      this.stats.replaceChildren(...rows);
      this.stats.hidden = rows.length < 2;
    }

    drawSeries(ctx, series) {
      const start = Math.max(0, lowerBound(series.x, this.xmin) - 1);
      const end = Math.min(series.x.length, lowerBound(series.x, this.xmax) + 1);
      let bucket = [],
        pixel = null,
        previousY = null;
      ctx.beginPath();
      ctx.strokeStyle = series.color;
      ctx.lineWidth = series.dash ? 1.1 : 1.4;
      ctx.setLineDash(series.dash ? [5, 4] : []);
      const flush = () => {
        if (!bucket.length) return;
        let min = bucket[0],
          max = bucket[0];
        for (const point of bucket) {
          if (point[2] < min[2]) min = point;
          if (point[2] > max[2]) max = point;
        }
        const points = [bucket[0], min, max, bucket[bucket.length - 1]].sort((a, b) => a[0] - b[0]);
        for (const [, x, y] of points) {
          if (previousY == null) ctx.moveTo(x, y);
          else if (series.step) {
            ctx.lineTo(x, previousY);
            ctx.lineTo(x, y);
          } else ctx.lineTo(x, y);
          previousY = y;
        }
        bucket = [];
      };
      for (let i = start; i < end; i++) {
        const value = valueAt(series, i),
          x = this.x(series.x[i]);
        if (!Number.isFinite(value) || (this.spec.logY && value <= 0)) {
          flush();
          previousY = null;
          continue;
        }
        const nextPixel = Math.floor(x);
        if (nextPixel !== pixel) {
          flush();
          pixel = nextPixel;
        }
        bucket.push([i, x, this.y(value)]);
      }
      flush();
      ctx.stroke();
    }
    pointerMove(event) {
      if (!this.canvas) return;
      const pixel = this.pixel(event),
        t = this.time(pixel);
      if (this.drag != null) this.dragEnd = pixel;
      if (this.spectral) {
        this.localCursor = t;
        this.drawOverlay();
      } else this.session.hover(t);
      const lines = [`${this.spectral ? "频率" : "时间"}  ${number(t)} ${this.spectral ? "Hz" : "s"}`];
      for (const series of this.series) {
        if (t < series.x[0] || t > series.x[series.x.length - 1]) continue;
        let index = Math.min(lowerBound(series.x, t), series.x.length - 1);
        if (index > 0 && (series.step ? series.x[index] > t : t - series.x[index - 1] < series.x[index] - t))
          index--;
        lines.push(`${series.name}  ${number(valueAt(series, index))} ${this.spec.unit}`);
        lines.push(`  采样 ${number(series.x[index])} ${this.spectral ? "Hz" : "s"}`);
      }
      this.tip.textContent = lines.join("\n");
      this.tip.hidden = false;
      this.tip.style.top = "8px";
      this.tip.style.left = Math.max(0, Math.min(pixel + 12, this.width - this.tip.offsetWidth - 4)) + "px";
    }
    drawOverlay() {
      if (!this.overlay) return;
      const ctx = this.overlay.getContext("2d");
      ctx.clearRect(0, 0, this.width, this.height);
      const cursor = this.spectral ? this.localCursor : this.session.cursor;
      if (cursor != null && cursor >= this.xmin && cursor <= this.xmax) {
        ctx.strokeStyle = "#6a8c7c";
        ctx.setLineDash([3, 3]);
        ctx.beginPath();
        ctx.moveTo(this.x(cursor), this.top);
        ctx.lineTo(this.x(cursor), this.height - this.bottom);
        ctx.stroke();
      }
      if (this.drag != null) {
        ctx.fillStyle = "#268b7124";
        ctx.fillRect(Math.min(this.drag, this.dragEnd), this.top, Math.abs(this.dragEnd - this.drag), this
          .innerHeight);
      }
    }
    exportCSV() {
      const escape = text => '"' + String(text).replaceAll('"', '""') + '"';
      const rows = [
        ["series", this.spectral ? "frequency_hz" : "host_time_s", "value", "unit"].join(",")
      ];
      for (const series of this.series) {
        const start = lowerBound(series.x, this.xmin),
          end = Math.min(series.x.length, lowerBound(series.x, this.xmax) + 1);
        for (let i = start; i < end && series.x[i] <= this.xmax; i++) {
          const value = valueAt(series, i);
          rows.push([escape(series.name), series.x[i], Number.isFinite(value) ? value : "", escape(this.spec
            .unit)].join(","));
        }
      }
      download(new Blob(["\ufeff", rows.join("\n")], {
        type: "text/csv;charset=utf-8"
      }), this.spec.title + ".csv");
    }
    exportPNG() {
      const canvas = document.createElement("canvas");
      canvas.width = this.canvas.width;
      canvas.height = this.canvas.height + 85;
      const ctx = canvas.getContext("2d");
      ctx.fillStyle = "white";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = "#193a31";
      ctx.font = "18px sans-serif";
      ctx.fillText("deploynose / " + this.spec.title, 20, 28);
      ctx.font = "12px sans-serif";
      ctx.fillText(this.series.map(s => s.name).join(" · ") + " [" + this.spec.unit + "]", 20, 53);
      ctx.drawImage(this.canvas, 0, 72);
      canvas.toBlob(blob => download(blob, this.spec.title + ".png"));
    }
  }
  return {
    TraceChart,
    lowerBound,
    valueAt,
    colors,
    number
  };
})();
