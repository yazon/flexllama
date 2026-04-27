(function () {
  const Dashboard = window.FlexLlamaDashboard || {};
  window.FlexLlamaDashboard = Dashboard;

  Dashboard.config = {
    API_BASE_URL: window.FLEXLLAMA_CONFIG.HEALTH_ENDPOINT || "/health",
    GPU_METRICS_URL:
      window.FLEXLLAMA_CONFIG.GPU_METRICS_ENDPOINT || "/v1/metrics/gpus",
    REFRESH_INTERVAL: 2000,
    GPU_METRICS_REFRESH_INTERVAL: 2000,
    REQUEST_TIMEOUT: 5000,
  };

  Dashboard.state = Dashboard.state || {
    refreshInterval: null,
    gpuMetricsInterval: null,
    lastUpdateTime: null,
    lastGpuMetricsTime: null,
    operationStates: {},
    gpuMetricsState: null,
    gpuMetricsRateLimitedUntil: 0,
  };

  Dashboard.constants = {
    STATUS_MAP: {
      ok: { label: "Ready", icon: "✓", color: "ok" },
      loading: { label: "Loading", icon: "⟳", color: "loading" },
      error: { label: "Error", icon: "✗", color: "error" },
      not_running: { label: "Not Running", icon: "⏸", color: "error" },
      not_loaded: { label: "Not Loaded", icon: "○", color: "error" },
      start: { label: "Start", icon: "▶" },
    },
  };

  function sanitizeHTML(text) {
    const element = document.createElement("div");
    element.innerText = String(text);
    return element.innerHTML;
  }

  function slugify(text) {
    return text
      .toString()
      .trim()
      .toLowerCase()
      .replace(/\s+/g, "-")
      .replace(/[^\w\-]+/g, "")
      .replace(/\-\-+/g, "-")
      .replace(/^-+/, "")
      .replace(/-+$/, "");
  }

  function formatAutoUnloadStatus(timeoutSeconds, countdownSeconds) {
    if (timeoutSeconds === 0) {
      return "Disabled";
    }

    let status = `${timeoutSeconds}s`;
    if (countdownSeconds !== null && countdownSeconds !== undefined) {
      if (countdownSeconds <= 0) {
        status +=
          ' <span class="countdown-warning">(Unloading now...)</span>';
      } else {
        status += ` <span class="countdown-timer">(Unloading in ${countdownSeconds}s)</span>`;
      }
    }

    return status;
  }

  function formatMb(mb) {
    if (mb === null || mb === undefined) return "--";
    if (mb >= 1024) return `${(mb / 1024).toFixed(1)} GB`;
    return `${Math.round(mb)} MB`;
  }

  function getTimeDifference(date) {
    const now = new Date();
    const diff = Math.floor((now - date) / 1000);

    if (diff < 60) return `${diff}s ago`;
    if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
    if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
    return `${Math.floor(diff / 86400)}d ago`;
  }

  Dashboard.utils = {
    sanitizeHTML,
    slugify,
    formatAutoUnloadStatus,
    formatMb,
    getTimeDifference,
  };

  function updateLastUpdatedTime() {
    Dashboard.state.lastUpdateTime = new Date();
    const lastUpdatedElement = document.getElementById("lastUpdated");
    if (lastUpdatedElement) {
      lastUpdatedElement.textContent =
        `Last updated: ${Dashboard.state.lastUpdateTime.toLocaleTimeString()}`;
    }
  }

  function showError(message) {
    const errorToast = document.getElementById("errorToast");
    const errorMessage = document.getElementById("errorMessage");
    if (!errorToast || !errorMessage) return;

    errorMessage.textContent = message;
    errorToast.classList.remove("hidden");

    setTimeout(function () {
      hideError();
    }, 5000);
  }

  function hideError() {
    const errorToast = document.getElementById("errorToast");
    if (errorToast) {
      errorToast.classList.add("hidden");
    }
  }

  Dashboard.ui = {
    updateLastUpdatedTime,
    showError,
    hideError,
  };

  window.hideError = hideError;

  document.addEventListener("DOMContentLoaded", function () {
    console.log("FlexLLama Dashboard initialized");

    if (Dashboard.runners) {
      Dashboard.runners.startAutoRefresh();
      Dashboard.runners.fetchHealthData();
    }

    if (Dashboard.gpu) {
      Dashboard.gpu.startGpuMetricsRefresh();
      Dashboard.gpu.fetchGpuMetrics();
    }
  });

  document.addEventListener("visibilitychange", function () {
    if (document.hidden) {
      if (Dashboard.runners) {
        Dashboard.runners.stopAutoRefresh();
      }
      if (Dashboard.gpu) {
        Dashboard.gpu.stopGpuMetricsRefresh();
      }
      return;
    }

    if (Dashboard.runners) {
      Dashboard.runners.startAutoRefresh();
      Dashboard.runners.fetchHealthData();
    }
    if (Dashboard.gpu) {
      Dashboard.gpu.startGpuMetricsRefresh();
      Dashboard.gpu.fetchGpuMetrics();
    }
  });

  window.addEventListener("focus", function () {
    if (Dashboard.runners && !Dashboard.runners.isAutoRefreshRunning()) {
      Dashboard.runners.startAutoRefresh();
      Dashboard.runners.fetchHealthData();
    }

    if (Dashboard.gpu && !Dashboard.gpu.isGpuMetricsRefreshRunning()) {
      Dashboard.gpu.startGpuMetricsRefresh();
      Dashboard.gpu.fetchGpuMetrics();
    }
  });

  window.addEventListener("beforeunload", function () {
    if (Dashboard.runners) {
      Dashboard.runners.stopAutoRefresh();
    }
    if (Dashboard.gpu) {
      Dashboard.gpu.stopGpuMetricsRefresh();
    }
  });

  window.llama_dashboard = {
    fetchHealthData: function () {
      return Dashboard.runners && Dashboard.runners.fetchHealthData();
    },
    startAutoRefresh: function () {
      return Dashboard.runners && Dashboard.runners.startAutoRefresh();
    },
    stopAutoRefresh: function () {
      return Dashboard.runners && Dashboard.runners.stopAutoRefresh();
    },
    showError,
    hideError,
    startRunner: function (runnerName) {
      return Dashboard.runners && Dashboard.runners.startRunner(runnerName);
    },
    stopRunner: function (runnerName) {
      return Dashboard.runners && Dashboard.runners.stopRunner(runnerName);
    },
    restartRunner: function (runnerName) {
      return Dashboard.runners && Dashboard.runners.restartRunner(runnerName);
    },
    fetchGpuMetrics: function () {
      return Dashboard.gpu && Dashboard.gpu.fetchGpuMetrics();
    },
    config: Dashboard.config,
  };
})();
