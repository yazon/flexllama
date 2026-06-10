(function () {
  const Dashboard = window.FlexLlamaDashboard || {};
  window.FlexLlamaDashboard = Dashboard;

  const config = Dashboard.config;
  const state = Dashboard.state;
  const utils = Dashboard.utils;

  // Module-scoped Chart.js instances keyed by model alias. Charts are created
  // ONCE per model and updated in place; the canvas is never innerHTML-wiped,
  // which would destroy Chart's canvas and leak the instance.
  const charts = {};

  const GEN_COLOR = "#64ffda";
  const PROMPT_COLOR = "#ffb74d";

  function startThroughputMetricsRefresh() {
    if (state.throughputMetricsInterval) {
      clearInterval(state.throughputMetricsInterval);
    }

    state.throughputMetricsInterval = setInterval(
      fetchThroughputMetrics,
      config.THROUGHPUT_METRICS_REFRESH_INTERVAL
    );
    console.log(
      `Throughput metrics auto-refresh started: every ${config.THROUGHPUT_METRICS_REFRESH_INTERVAL}ms`
    );
  }

  function stopThroughputMetricsRefresh() {
    if (state.throughputMetricsInterval) {
      clearInterval(state.throughputMetricsInterval);
      state.throughputMetricsInterval = null;
    }
  }

  function isThroughputMetricsRefreshRunning() {
    return Boolean(state.throughputMetricsInterval);
  }

  async function fetchThroughputMetrics() {
    if (Date.now() < state.throughputMetricsRateLimitedUntil) {
      return;
    }

    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(function () {
        controller.abort();
      }, config.REQUEST_TIMEOUT);

      const response = await fetch(config.THROUGHPUT_METRICS_URL, {
        signal: controller.signal,
        headers: {
          Accept: "application/json",
          "Cache-Control": "no-cache",
        },
      });

      clearTimeout(timeoutId);

      if (response.status === 429) {
        const retryAfterHeader = response.headers.get("Retry-After");
        const retryAfterSeconds = Math.max(
          1,
          parseInt(retryAfterHeader || "5", 10) || 5
        );
        state.throughputMetricsRateLimitedUntil = Date.now() + retryAfterSeconds * 1000;

        if (
          !state.throughputMetricsState ||
          state.throughputMetricsState.status !== "available"
        ) {
          updateThroughputPanel({
            status: "unavailable",
            reason: "rate_limited",
          });
        }
        return;
      }

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();
      state.throughputMetricsRateLimitedUntil = 0;
      state.throughputMetricsState = data;
      updateThroughputPanel(data);
    } catch (error) {
      console.error("Failed to fetch throughput metrics:", error);
      updateThroughputPanel({
        status: "unavailable",
        reason: "fetch_error",
        collection_error: error.message,
      });
    }
  }

  function updateThroughputPanel(data) {
    const container = document.getElementById("throughputContainer");
    if (!container) return;

    const models = Array.isArray(data.models) ? data.models : [];

    if (data.status !== "available" || models.length === 0) {
      destroyAllCharts();
      container.innerHTML = createThroughputUnavailable(data);
      return;
    }

    // Remove any leftover empty-state node before rendering cards.
    const emptyState = container.querySelector(".throughput-unavailable");
    if (emptyState) {
      emptyState.remove();
    }

    const history = data.throughput_history || {};
    const seen = {};

    models.forEach(function (model) {
      seen[model.alias] = true;
      renderModelChart(container, model, history[model.alias] || {});
    });

    // Destroy charts for models that disappeared from the snapshot.
    Object.keys(charts).forEach(function (alias) {
      if (!seen[alias]) {
        destroyChart(alias);
      }
    });
  }

  function renderModelChart(container, model, historySet) {
    const alias = model.alias;
    const cardId = `throughput-card-${utils.slugify(alias)}`;
    let card = document.getElementById(cardId);

    if (!card) {
      card = document.createElement("div");
      card.className = "throughput-card";
      card.id = cardId;
      card.innerHTML =
        '<div class="throughput-card-header">' +
        `<h3 class="throughput-name">${utils.sanitizeHTML(alias)}</h3>` +
        '<div class="throughput-stats">' +
        '<span class="throughput-chip throughput-avg">Avg/1m <strong>--</strong></span>' +
        '<span class="throughput-chip throughput-peak">Peak <strong>--</strong></span>' +
        "</div>" +
        "</div>" +
        '<div class="throughput-chart-wrapper"><canvas></canvas></div>';
      container.appendChild(card);
    }

    updateStatChip(card, ".throughput-avg strong", model.avg_gen_tps_1m);
    updateStatChip(card, ".throughput-peak strong", model.peak_gen_tps);

    const genHistory = historySet.generation_tokens_per_second || [];
    const promptHistory = historySet.prompt_tokens_per_second || [];
    const labels = genHistory.map(function (_, index) {
      return String(index + 1);
    });

    let chart = charts[alias];
    if (!chart) {
      if (!window.Chart) {
        return;
      }
      const canvas = card.querySelector("canvas");
      const ctx = canvas.getContext("2d");
      chart = new window.Chart(ctx, {
        type: "line",
        data: {
          labels: labels,
          datasets: [
            buildDataset("Generation tok/s", genHistory, GEN_COLOR),
            buildDataset("Prompt tok/s", promptHistory, PROMPT_COLOR),
          ],
        },
        options: buildChartOptions(),
      });
      charts[alias] = chart;
      return;
    }

    chart.data.labels = labels;
    chart.data.datasets[0].data = genHistory;
    chart.data.datasets[1].data = promptHistory;
    chart.update("none");
  }

  function buildDataset(label, data, color) {
    return {
      label: label,
      data: data,
      borderColor: color,
      backgroundColor: color,
      borderWidth: 1.5,
      pointRadius: 0,
      tension: 0.3,
      spanGaps: true,
    };
  }

  function buildChartOptions() {
    return {
      animation: false,
      responsive: true,
      maintainAspectRatio: false,
      interaction: { intersect: false, mode: "index" },
      scales: {
        x: { display: false },
        y: {
          beginAtZero: true,
          ticks: { color: "#78909c", font: { size: 10 } },
          grid: { color: "rgba(255, 255, 255, 0.05)" },
        },
      },
      plugins: {
        legend: {
          labels: { color: "#b0bec5", boxWidth: 12, font: { size: 11 } },
        },
        tooltip: { enabled: true },
      },
    };
  }

  function updateStatChip(card, selector, value) {
    const node = card.querySelector(selector);
    if (!node) return;
    if (value === null || value === undefined) {
      node.textContent = "--";
    } else {
      node.textContent = `${Number(value).toFixed(1)} t/s`;
    }
  }

  function destroyChart(alias) {
    const chart = charts[alias];
    if (chart) {
      try {
        chart.destroy();
      } catch (error) {
        console.error("Failed to destroy throughput chart:", error);
      }
      delete charts[alias];
    }
    const card = document.getElementById(`throughput-card-${utils.slugify(alias)}`);
    if (card) {
      card.remove();
    }
  }

  function destroyAllCharts() {
    Object.keys(charts).forEach(function (alias) {
      destroyChart(alias);
    });
  }

  function createThroughputUnavailable(data) {
    const reason = data.reason || "unknown";
    const reasonText = {
      disabled: "Token throughput tracking is disabled in configuration.",
      fetch_error: "Could not reach the throughput metrics endpoint.",
      rate_limited:
        "Throughput metrics request rate limited. The dashboard will retry shortly.",
      internal_error: "The throughput metrics endpoint returned an error.",
    };

    let message =
      data.status === "disabled"
        ? reasonText.disabled
        : reasonText[reason] ||
          "No token throughput recorded yet. Charts appear after the first completion.";
    if (data.collection_error) {
      message += ` (${utils.sanitizeHTML(data.collection_error)})`;
    }

    return (
      '<div class="throughput-unavailable">' +
      '<div class="unavailable-icon">&#9888;</div>' +
      '<div class="unavailable-text">No token throughput data</div>' +
      `<div class="unavailable-reason">${message}</div>` +
      "</div>"
    );
  }

  Dashboard.throughput = {
    startThroughputMetricsRefresh,
    stopThroughputMetricsRefresh,
    isThroughputMetricsRefreshRunning,
    fetchThroughputMetrics,
  };
})();
