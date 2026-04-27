(function () {
  const Dashboard = window.FlexLlamaDashboard || {};
  window.FlexLlamaDashboard = Dashboard;

  const config = Dashboard.config;
  const state = Dashboard.state;
  const utils = Dashboard.utils;

  function startGpuMetricsRefresh() {
    if (state.gpuMetricsInterval) {
      clearInterval(state.gpuMetricsInterval);
    }

    state.gpuMetricsInterval = setInterval(
      fetchGpuMetrics,
      config.GPU_METRICS_REFRESH_INTERVAL,
    );
    console.log(
      `GPU metrics auto-refresh started: every ${config.GPU_METRICS_REFRESH_INTERVAL}ms`,
    );
  }

  function stopGpuMetricsRefresh() {
    if (state.gpuMetricsInterval) {
      clearInterval(state.gpuMetricsInterval);
      state.gpuMetricsInterval = null;
    }
  }

  function isGpuMetricsRefreshRunning() {
    return Boolean(state.gpuMetricsInterval);
  }

  async function fetchGpuMetrics() {
    if (Date.now() < state.gpuMetricsRateLimitedUntil) {
      return;
    }

    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(function () {
        controller.abort();
      }, config.REQUEST_TIMEOUT);

      const response = await fetch(config.GPU_METRICS_URL, {
        signal: controller.signal,
        headers: {
          Accept: "application/json",
          "Cache-Control": "no-cache",
        },
      });

      clearTimeout(timeoutId);

      if (response.status === 429) {
        const retryAfterHeader = response.headers.get("Retry-After");
        const retryAfterSeconds =
          Math.max(1, parseInt(retryAfterHeader || "5", 10) || 5);
        state.gpuMetricsRateLimitedUntil = Date.now() + retryAfterSeconds * 1000;

        if (!state.gpuMetricsState || state.gpuMetricsState.status !== "available") {
          updateGpuMetricsPanel({ status: "unavailable", reason: "rate_limited" });
        }
        return;
      }

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();
      state.gpuMetricsRateLimitedUntil = 0;
      state.gpuMetricsState = data;
      state.lastGpuMetricsTime = new Date();
      updateGpuMetricsPanel(data);
    } catch (error) {
      console.error("Failed to fetch GPU metrics:", error);
      updateGpuMetricsPanel({
        status: "unavailable",
        reason: "fetch_error",
        collection_error: error.message,
      });
    }
  }

  function updateGpuMetricsPanel(data) {
    const container = document.getElementById("gpuMetricsContainer");
    if (!container) return;

    container.innerHTML = "";

    if (data.status === "available" && data.gpus && data.gpus.length > 0) {
      const associations = data.runner_associations || {};
      const gpuRunnerMap = buildGpuRunnerMap(associations);

      data.gpus.forEach(function (gpu) {
        const card = createGpuMetricCard(gpu, data.gpu_history || {}, gpuRunnerMap);
        container.appendChild(card);
      });
      return;
    }

    container.innerHTML = createGpuMetricsUnavailable(data);
  }

  function buildGpuRunnerMap(associations) {
    const map = {};
    Object.entries(associations).forEach(function (entry) {
      const runnerName = entry[0];
      const info = entry[1];
      let gpuKeys = [];

      if (Array.isArray(info.gpu_ids)) {
        gpuKeys = info.gpu_ids;
      } else if (Array.isArray(info.gpu_indices)) {
        gpuKeys = info.gpu_indices.map(function (idx) {
          return String(idx);
        });
      }

      gpuKeys.forEach(function (gpuKey) {
        if (!map[gpuKey]) {
          map[gpuKey] = [];
        }
        map[gpuKey].push(runnerName);
      });
    });

    return map;
  }

  function createGpuMetricsUnavailable(data) {
    const reason = data.reason || "unknown";
    const reasonText = {
      unsupported_platform:
        "GPU metrics are not supported on this operating system.",
      tool_not_found:
        "No supported GPU telemetry tool found. Install nvidia-smi and/or amd-smi.",
      command_failed:
        "GPU telemetry command failed. Check driver/tool installation and GPU visibility.",
      command_timeout: "amd-smi command timed out.",
      parse_error: "Failed to parse amd-smi output.",
      no_visible_gpus:
        "No visible GPUs detected. Ensure GPU devices are accessible.",
      disabled_in_config:
        "GPU metrics collection is disabled in configuration.",
      fetch_error: "Could not reach the GPU metrics endpoint.",
      rate_limited:
        "GPU metrics request rate limited. The dashboard will retry shortly.",
    };

    let message = reasonText[reason] || "GPU metrics are currently unavailable.";
    if (data.collection_error) {
      message += ` (${utils.sanitizeHTML(data.collection_error)})`;
    }

    return (
      '<div class="gpu-metrics-unavailable">' +
      '<div class="unavailable-icon">&#9888;</div>' +
      '<div class="unavailable-text">GPU metrics unavailable</div>' +
      `<div class="unavailable-reason">${message}</div>` +
      "</div>"
    );
  }

  function createGpuMetricCard(gpu, history, gpuRunnerMap) {
    const idx = gpu.index;
    const gpuKey = gpu.id ? String(gpu.id) : String(idx);
    const name = gpu.name || `GPU ${idx}`;
    const vendor = gpu.vendor || "unknown";
    const runners = gpuRunnerMap[gpuKey] || gpuRunnerMap[String(idx)] || [];

    const card = document.createElement("div");
    card.className = "gpu-metric-card";

    const memUsed =
      gpu.memory_used_mb !== null && gpu.memory_used_mb !== undefined
        ? utils.formatMb(gpu.memory_used_mb)
        : "--";
    const memTotal =
      gpu.memory_total_mb !== null && gpu.memory_total_mb !== undefined
        ? utils.formatMb(gpu.memory_total_mb)
        : "--";
    const memPercent =
      gpu.memory_used_mb != null &&
      gpu.memory_total_mb != null &&
      gpu.memory_total_mb > 0
        ? Math.round((gpu.memory_used_mb / gpu.memory_total_mb) * 100)
        : null;
    const util =
      gpu.utilization_gpu_percent !== null &&
      gpu.utilization_gpu_percent !== undefined
        ? `${Math.round(gpu.utilization_gpu_percent)}%`
        : "--";
    const temp =
      gpu.temperature_c !== null && gpu.temperature_c !== undefined
        ? `${Math.round(gpu.temperature_c)}°C`
        : "--";

    const historySet = history[gpuKey] || {};
    const memHistory = historySet.memory_used_mb || [];
    const utilHistory = historySet.utilization_gpu_percent || [];
    const tempHistory = historySet.temperature_c || [];

    card.innerHTML =
      '<div class="gpu-card-header">' +
      `<h3 class="gpu-name">${utils.sanitizeHTML(name)}</h3>` +
      `<span class="gpu-index">${utils.sanitizeHTML(vendor.toUpperCase())} GPU ${idx}</span>` +
      "</div>" +
      '<div class="gpu-metrics-row">' +
      `<div class="gpu-metric-chip${memPercent !== null && memPercent > 90 ? " metric-warning" : ""}">` +
      '<span class="metric-label">VRAM</span>' +
      `<span class="metric-value">${memUsed} / ${memTotal}</span>` +
      `${memPercent !== null ? `<span class="metric-percent">${memPercent}%</span>` : ""}` +
      "</div>" +
      '<div class="gpu-metric-chip">' +
      '<span class="metric-label">Util</span>' +
      `<span class="metric-value">${util}</span>` +
      "</div>" +
      `<div class="gpu-metric-chip${gpu.temperature_c !== null && gpu.temperature_c > 85 ? " metric-warning" : ""}">` +
      '<span class="metric-label">Temp</span>' +
      `<span class="metric-value">${temp}</span>` +
      "</div>" +
      "</div>" +
      '<div class="gpu-sparklines">' +
      '<div class="sparkline-group">' +
      '<span class="sparkline-label">VRAM</span>' +
      `<div class="sparkline-container">${renderSparkline(memHistory, "#64ffda")}</div>` +
      "</div>" +
      '<div class="sparkline-group">' +
      '<span class="sparkline-label">Util</span>' +
      `<div class="sparkline-container">${renderSparkline(utilHistory, "#ffb74d")}</div>` +
      "</div>" +
      '<div class="sparkline-group">' +
      '<span class="sparkline-label">Temp</span>' +
      `<div class="sparkline-container">${renderSparkline(tempHistory, "#e57373")}</div>` +
      "</div>" +
      "</div>" +
      (runners.length > 0
        ? '<div class="gpu-runners"><span class="gpu-runners-label">Runners:</span> ' +
          runners
            .map(function (runnerName) {
              return utils.sanitizeHTML(runnerName);
            })
            .join(", ") +
          "</div>"
        : "") +
      (gpu.power_w !== null && gpu.power_w !== undefined
        ? `<div class="gpu-extra">Power: ${gpu.power_w.toFixed(1)}W</div>`
        : "");

    return card;
  }

  function renderSparkline(values, color) {
    const emptySparkline =
      '<svg class="sparkline" viewBox="0 0 120 24"><line x1="0" y1="12" x2="120" y2="12" stroke="' +
      color +
      '" stroke-opacity="0.2" stroke-width="1"/></svg>';

    if (!values || values.length < 2) {
      return emptySparkline;
    }

    const filtered = values.filter(function (value) {
      return value !== null && value !== undefined;
    });
    if (filtered.length < 2) {
      return emptySparkline;
    }

    const min = Math.min.apply(null, filtered);
    const max = Math.max.apply(null, filtered);
    let range = max - min;
    if (range === 0) {
      range = 1;
    }

    const width = 120;
    const height = 24;
    const padding = 2;
    const step = (width - padding * 2) / (filtered.length - 1);

    const points = filtered.map(function (value, index) {
      const x = padding + index * step;
      const y =
        height -
        padding -
        ((value - min) / range) * (height - padding * 2);
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    });

    const pathD = `M${points.join(" L")}`;
    return (
      `<svg class="sparkline" viewBox="0 0 ${width} ${height}">` +
      `<path d="${pathD}" fill="none" stroke="${color}" stroke-width="1.5" stroke-linejoin="round" stroke-linecap="round"/>` +
      "</svg>"
    );
  }

  Dashboard.gpu = {
    startGpuMetricsRefresh,
    stopGpuMetricsRefresh,
    isGpuMetricsRefreshRunning,
    fetchGpuMetrics,
  };
})();
