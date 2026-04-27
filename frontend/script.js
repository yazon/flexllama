// Configuration
const CONFIG = {
  API_BASE_URL: window.FLEXLLAMA_CONFIG.HEALTH_ENDPOINT || "/health",
  GPU_METRICS_URL: window.FLEXLLAMA_CONFIG.GPU_METRICS_ENDPOINT || "/v1/metrics/gpus",
  REFRESH_INTERVAL: 2000,
  GPU_METRICS_REFRESH_INTERVAL: 2000,
  REQUEST_TIMEOUT: 5000,
};

// Global state
let refreshInterval = null;
let gpuMetricsInterval = null;
let lastUpdateTime = null;
let lastGpuMetricsTime = null;
let operationStates = {};
let gpuMetricsState = null;
let gpuMetricsRateLimitedUntil = 0;

// Status mapping
const STATUS_MAP = {
  ok: { label: "Ready", icon: "✓", color: "ok" },
  loading: { label: "Loading", icon: "⟳", color: "loading" },
  error: { label: "Error", icon: "✗", color: "error" },
  not_running: { label: "Not Running", icon: "⏸", color: "error" },
  not_loaded: { label: "Not Loaded", icon: "○", color: "error" },
  start: { label: "Start", icon: "▶" },
};

// Sanitize HTML to prevent XSS
function sanitizeHTML(text) {
  const element = document.createElement("div");
  element.innerText = text;
  return element.innerHTML;
}

// Helper to generate safe, DOM-friendly IDs from runner names
function slugify(text) {
  return text
    .toString()
    .trim()
    .toLowerCase()
    .replace(/\s+/g, "-") // Replace spaces with –
    .replace(/[^\w\-]+/g, "") // Remove all non-word chars
    .replace(/\-\-+/g, "-") // Collapse multiple – to single
    .replace(/^-+/, "") // Trim – from start
    .replace(/-+$/, ""); // Trim – from end
}

// Format auto-unload status with countdown
function formatAutoUnloadStatus(timeoutSeconds, countdownSeconds) {
  if (timeoutSeconds === 0) {
    return "Disabled";
  }

  let status = `${timeoutSeconds}s`;

  if (countdownSeconds !== null && countdownSeconds !== undefined) {
    if (countdownSeconds <= 0) {
      status += ' <span class="countdown-warning">(Unloading now...)</span>';
    } else {
      status += ` <span class="countdown-timer">(Unloading in ${countdownSeconds}s)</span>`;
    }
  }

  return status;
}

// Initialize the dashboard
document.addEventListener("DOMContentLoaded", function () {
  console.log("🦙 FlexLLama Dashboard initialized");
  startAutoRefresh();
  startGpuMetricsRefresh();
  fetchHealthData();
  fetchGpuMetrics();
});

// Start auto-refresh
function startAutoRefresh() {
  if (refreshInterval) {
    clearInterval(refreshInterval);
  }

  refreshInterval = setInterval(fetchHealthData, CONFIG.REFRESH_INTERVAL);
  console.log(`Auto-refresh started: every ${CONFIG.REFRESH_INTERVAL}ms`);
}

// Stop auto-refresh
function stopAutoRefresh() {
  if (refreshInterval) {
    clearInterval(refreshInterval);
    refreshInterval = null;
  }
}

// Fetch health data from API
async function fetchHealthData() {
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(
      () => controller.abort(),
      CONFIG.REQUEST_TIMEOUT,
    );

    const response = await fetch(CONFIG.API_BASE_URL, {
      signal: controller.signal,
      headers: {
        Accept: "application/json",
        "Cache-Control": "no-cache",
      },
    });

    clearTimeout(timeoutId);

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    const data = await response.json();
    if (
      !data ||
      typeof data !== "object" ||
      !Object.prototype.hasOwnProperty.call(data, "active_runners")
    ) {
      throw new Error("Unexpected /health response payload");
    }

    updateDashboard(data);
    updateLastUpdatedTime();
    hideError();
  } catch (error) {
    console.error("Failed to fetch health data:", error);
    showError(`Failed to fetch data: ${error.message}`);
  }
}

// Update the dashboard with new data
function updateDashboard(data) {
  console.log("Updating dashboard with data:", data);

  const active_runners = data.active_runners || {};
  const runner_current_models = data.runner_current_models || {};
  const runner_info = data.runner_info || {};
  const model_health = data.model_health || {};

  // Update runners section
  updateRunnersSection(
    active_runners,
    runner_current_models,
    runner_info,
    model_health,
  );

  // Update models overview section
  updateModelsSection(model_health, runner_current_models);
}

// Update runners section
function updateRunnersSection(
  activeRunners = {},
  runnerModels = {},
  runnerInfo = {},
  modelHealth = {},
) {
  const container = document.getElementById("runnersContainer");
  container.innerHTML = "";

  // Group models by runner for display
  const runnerData = {};

  // Initialize runner data with host/port info
  Object.keys(activeRunners).forEach((runnerName) => {
    const info = runnerInfo[runnerName] || {};
    runnerData[runnerName] = {
      isActive: activeRunners[runnerName],
      currentModel: runnerModels[runnerName],
      host: info.host || "unknown",
      port: info.port || "unknown",
      autoUnloadTimeout: info.auto_unload_timeout_seconds || 0,
      autoUnloadCountdown: info.auto_unload_countdown_seconds,
      models: [],
    };
  });

  // Add model health information
  Object.entries(modelHealth).forEach(([modelAlias, health]) => {
    // Find which runner this model belongs to by checking if it's currently loaded
    let assignedRunner = null;
    Object.entries(runnerModels).forEach(([runnerName, currentModel]) => {
      if (currentModel === modelAlias) {
        assignedRunner = runnerName;
      }
    });

    if (assignedRunner && runnerData[assignedRunner]) {
      runnerData[assignedRunner].models.push({
        alias: modelAlias,
        health: health,
        isLoaded: true,
      });
    }
  });

  // Create runner cards
  Object.entries(runnerData).forEach(([runnerName, data]) => {
    const runnerCard = createRunnerCard(runnerName, data);
    container.appendChild(runnerCard);
  });

  // Show empty state if no runners
  if (Object.keys(runnerData).length === 0) {
    container.innerHTML = `
            <div class="empty-state">
                <div class="icon">🏃</div>
                <p>No runners configured</p>
            </div>
        `;
  }

  // Attach event listeners to control buttons
  attachControlListeners();
}

// Create a runner card element
function createRunnerCard(runnerName, data) {
  const {
    isActive,
    currentModel,
    host,
    port,
    autoUnloadTimeout,
    autoUnloadCountdown,
    models,
  } = data;

  const card = document.createElement("div");
  card.className = `runner-card ${isActive ? "active" : "inactive"}`;

  const runnerSlug = slugify(runnerName);

  card.innerHTML = `
        <div class="runner-header">
            <h3 class="runner-title">${runnerName}</h3>
            <div class="runner-status">
                <div class="status-indicator ${isActive ? "active" : "inactive"}"></div>
                <span>${isActive ? "Active" : "Inactive"}</span>
            </div>
        </div>
        
        <div class="runner-info">
            <div><strong>Current Model:</strong> ${currentModel || "None"}</div>
            <div><strong>Status:</strong> ${isActive ? "Running" : "Stopped"}</div>
            <div><strong>Host:</strong> ${host}</div>
            <div><strong>Port:</strong> ${port}</div>
            <div><strong>Auto-unload:</strong> ${formatAutoUnloadStatus(autoUnloadTimeout, autoUnloadCountdown)}</div>
            ${isActive ? `<div><strong>URL:</strong> http://${host}:${port}</div>` : ""}
        </div>
        
        <div class="models-list">
            ${
              models.length > 0
                ? models.map((model) => createModelItem(model)).join("")
                : '<div class="empty-state"><p>No models currently loaded</p></div>'
            }
        </div>
        
        ${createControlPanel(runnerName, data, runnerSlug)}
    `;

  return card;
}

// Create a model item element
function createModelItem(model) {
  const { alias, health, isLoaded } = model;
  const status = health.status || "unloaded";
  const statusInfo = STATUS_MAP[status] || STATUS_MAP["error"];

  return `
        <div class="model-item status-${statusInfo.color}">
            <div class="status-gauge">
                <div class="gauge-circle status-${statusInfo.color}">
                    <div class="gauge-inner">${statusInfo.icon}</div>
                </div>
            </div>
            <div class="model-info">
                <div class="model-name" title="${sanitizeHTML(alias)}">${sanitizeHTML(alias)}</div>
                <div class="model-status status-${statusInfo.color}">${statusInfo.label}</div>
                <div class="model-message">${sanitizeHTML(health.message || "No message")}</div>
                <div class="model-details">
                    ${isLoaded ? "Currently loaded" : "Available but not loaded"}
                </div>
            </div>
        </div>
    `;
}

// Create control panel for a runner
function createControlPanel(runnerName, data, runnerSlug) {
  const { isActive } = data;
  const operationState = operationStates[runnerName];

  // Determine button states based on runner status and operation state
  const isOperating = operationState && operationState.inProgress;
  const operationType = operationState ? operationState.type : null;

  // Button enable/disable logic
  const canStart = !isActive && !isOperating;
  const canStop = isActive && !isOperating;
  const canRestart = isActive && !isOperating;

  // Define icons for control buttons
  const loadingIcon = STATUS_MAP.loading.icon;
  const startIcon = STATUS_MAP.start.icon;
  const stopIcon = STATUS_MAP.not_running.icon;
  const restartIcon = "↻";

  return `
        <div class="runner-controls" data-runner="${sanitizeHTML(runnerName)}" data-runner-slug="${runnerSlug}">
            <div class="control-buttons">
                <button class="control-button btn-start ${isOperating && operationType === "start" ? "loading" : ""}" 
                        data-action="start" 
                        ${!canStart ? "disabled" : ""}>
                    <span class="btn-icon">${isOperating && operationType === "start" ? loadingIcon : startIcon}</span>
                    <span class="btn-text">${isOperating && operationType === "start" ? "Starting..." : "Start"}</span>
                </button>
                <button class="control-button btn-stop ${isOperating && operationType === "stop" ? "loading" : ""}" 
                        data-action="stop" 
                        ${!canStop ? "disabled" : ""}>
                    <span class="btn-icon">${isOperating && operationType === "stop" ? loadingIcon : stopIcon}</span>
                    <span class="btn-text">${isOperating && operationType === "stop" ? "Stopping..." : "Stop"}</span>
                </button>
                <button class="control-button btn-restart ${isOperating && operationType === "restart" ? "loading" : ""}" 
                        data-action="restart" 
                        ${!canRestart ? "disabled" : ""}>
                    <span class="btn-icon">${isOperating && operationType === "restart" ? loadingIcon : restartIcon}</span>
                    <span class="btn-text">${isOperating && operationType === "restart" ? "Restarting..." : "Restart"}</span>
                </button>
            </div>
            <div class="control-status ${operationState ? operationState.statusClass : ""}" id="status-${runnerSlug}">${operationState ? operationState.message : ""}</div>
        </div>
    `;
}

// Add event listeners to control buttons
function attachControlListeners() {
  document.querySelectorAll(".control-button").forEach((button) => {
    // Remove existing listeners to prevent duplicates
    button.removeEventListener("click", handleControlClick);
    button.addEventListener("click", handleControlClick);
  });
}

// Handle control button clicks
function handleControlClick(event) {
  const button = event.currentTarget;
  const action = button.getAttribute("data-action");
  const runnerName = button
    .closest(".runner-controls")
    .getAttribute("data-runner");

  if (button.disabled) return;

  switch (action) {
    case "start":
      startRunner(runnerName);
      break;
    case "stop":
      confirmAndStopRunner(runnerName);
      break;
    case "restart":
      confirmAndRestartRunner(runnerName);
      break;
  }
}

// Start runner
async function startRunner(runnerName) {
  try {
    setOperationState(runnerName, "start", "Starting runner...", "status-info");

    const response = await fetch(
      `/v1/runners/${encodeURIComponent(runnerName)}/start`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
      },
    );

    const data = await response.json();

    if (data.success) {
      setOperationState(
        runnerName,
        null,
        "Runner started successfully",
        "status-success",
      );
      setTimeout(() => clearOperationState(runnerName), 3000);
    } else {
      const errorMsg = data.error
        ? data.error.message
        : "Failed to start runner";
      setOperationState(runnerName, null, errorMsg, "status-error");
      setTimeout(() => clearOperationState(runnerName), 5000);
    }
  } catch (error) {
    console.error("Error starting runner:", error);
    setOperationState(
      runnerName,
      null,
      `Error: ${error.message}`,
      "status-error",
    );
    setTimeout(() => clearOperationState(runnerName), 5000);
  }
}

// Confirm and stop runner
function confirmAndStopRunner(runnerName) {
  showConfirmationModal(
    "Stop Runner",
    `Are you sure you want to stop runner "${runnerName}"? This will interrupt any ongoing operations and unload the current model.`,
    () => stopRunner(runnerName),
  );
}

// Stop runner
async function stopRunner(runnerName) {
  try {
    setOperationState(runnerName, "stop", "Stopping runner...", "status-info");

    const response = await fetch(
      `/v1/runners/${encodeURIComponent(runnerName)}/stop`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
      },
    );

    const data = await response.json();

    if (data.success) {
      setOperationState(
        runnerName,
        null,
        "Runner stopped successfully",
        "status-success",
      );
      setTimeout(() => clearOperationState(runnerName), 3000);
    } else {
      const errorMsg = data.error
        ? data.error.message
        : "Failed to stop runner";
      setOperationState(runnerName, null, errorMsg, "status-error");
      setTimeout(() => clearOperationState(runnerName), 5000);
    }
  } catch (error) {
    console.error("Error stopping runner:", error);
    setOperationState(
      runnerName,
      null,
      `Error: ${error.message}`,
      "status-error",
    );
    setTimeout(() => clearOperationState(runnerName), 5000);
  }
}

// Confirm and restart runner
function confirmAndRestartRunner(runnerName) {
  showConfirmationModal(
    "Restart Runner",
    `Are you sure you want to restart runner "${runnerName}"? This will stop the runner, then start it again with the same model.`,
    () => restartRunner(runnerName),
  );
}

// Restart runner
async function restartRunner(runnerName) {
  try {
    setOperationState(
      runnerName,
      "restart",
      "Restarting runner...",
      "status-info",
    );

    const response = await fetch(
      `/v1/runners/${encodeURIComponent(runnerName)}/restart`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
      },
    );

    const data = await response.json();

    if (data.success) {
      setOperationState(
        runnerName,
        null,
        "Runner restarted successfully",
        "status-success",
      );
      setTimeout(() => clearOperationState(runnerName), 3000);
    } else {
      const errorMsg = data.error
        ? data.error.message
        : "Failed to restart runner";
      setOperationState(runnerName, null, errorMsg, "status-error");
      setTimeout(() => clearOperationState(runnerName), 5000);
    }
  } catch (error) {
    console.error("Error restarting runner:", error);
    setOperationState(
      runnerName,
      null,
      `Error: ${error.message}`,
      "status-error",
    );
    setTimeout(() => clearOperationState(runnerName), 5000);
  }
}

// Set operation state
function setOperationState(runnerName, operationType, message, statusClass) {
  operationStates[runnerName] = {
    inProgress: operationType !== null,
    type: operationType,
    message: message,
    statusClass: statusClass,
  };

  // Update the UI immediately
  updateRunnerControlUI(runnerName);
}

// Clear operation state
function clearOperationState(runnerName) {
  delete operationStates[runnerName];
  updateRunnerControlUI(runnerName);
}

// Update runner control UI
function updateRunnerControlUI(runnerName) {
  const runnerSlug = slugify(runnerName);
  const statusElement = document.getElementById(`status-${runnerSlug}`);
  const controlPanel = document.querySelector(
    `.runner-controls[data-runner-slug="${runnerSlug}"]`,
  );

  const state = operationStates[runnerName];

  // Update status text / styles
  if (statusElement) {
    if (state) {
      statusElement.textContent = state.message;
      statusElement.className = `control-status ${state.statusClass}`;
    } else {
      statusElement.textContent = "";
      statusElement.className = "control-status";
    }
  }

  // Update button states & loading indicators immediately
  if (!controlPanel) return;

  const startBtn = controlPanel.querySelector(".btn-start");
  const stopBtn = controlPanel.querySelector(".btn-stop");
  const restartBtn = controlPanel.querySelector(".btn-restart");

  const runnerCard = controlPanel.closest(".runner-card");
  const isActive = runnerCard && runnerCard.classList.contains("active");

  const loadingIcon = STATUS_MAP.loading.icon;
  const startIcon = STATUS_MAP.start.icon;
  const stopIcon = STATUS_MAP.not_running.icon;
  const restartIcon = "↻";

  if (state && state.inProgress) {
    // Disable all buttons while an operation is in progress
    [startBtn, stopBtn, restartBtn].forEach((btn) => {
      if (!btn) return;
      btn.disabled = true;
      btn.classList.remove("loading");
    });

    // Highlight the button corresponding to the active operation
    const setLoading = (btn, text) => {
      if (!btn) return;
      btn.classList.add("loading");
      btn.querySelector(".btn-icon").textContent = loadingIcon;
      btn.querySelector(".btn-text").textContent = text;
    };

    switch (state.type) {
      case "start":
        setLoading(startBtn, "Starting...");
        break;
      case "stop":
        setLoading(stopBtn, "Stopping...");
        break;
      case "restart":
        setLoading(restartBtn, "Restarting...");
        break;
    }
  } else {
    // No operation – restore default button states based on runner activity
    if (startBtn) {
      startBtn.disabled = isActive;
      startBtn.classList.remove("loading");
      startBtn.querySelector(".btn-icon").textContent = startIcon;
      startBtn.querySelector(".btn-text").textContent = "Start";
    }
    if (stopBtn) {
      stopBtn.disabled = !isActive;
      stopBtn.classList.remove("loading");
      stopBtn.querySelector(".btn-icon").textContent = stopIcon;
      stopBtn.querySelector(".btn-text").textContent = "Stop";
    }
    if (restartBtn) {
      restartBtn.disabled = !isActive;
      restartBtn.classList.remove("loading");
      restartBtn.querySelector(".btn-icon").textContent = restartIcon;
      restartBtn.querySelector(".btn-text").textContent = "Restart";
    }
  }
}

// Show confirmation modal
function showConfirmationModal(title, message, onConfirm) {
  // Remove existing modal if any
  const existingModal = document.querySelector(".confirmation-modal");
  if (existingModal) {
    existingModal.remove();
  }

  const modal = document.createElement("div");
  modal.className = "confirmation-modal";
  modal.innerHTML = `
        <div class="confirmation-content">
            <h3 class="confirmation-title">${sanitizeHTML(title)}</h3>
            <p class="confirmation-message">${sanitizeHTML(message)}</p>
            <div class="confirmation-buttons">
                <button class="confirmation-button primary" data-action="confirm">Confirm</button>
                <button class="confirmation-button secondary" data-action="cancel">Cancel</button>
            </div>
        </div>
    `;

  // ---------- helper to close & cleanup ----------
  const closeModal = () => {
    // Remove modal element if still in DOM
    if (modal.parentNode) {
      modal.remove();
    }
    // Always remove keydown listener to avoid leaks
    document.removeEventListener("keydown", handleEsc);
  };

  // Close on ESC key
  const handleEsc = (e) => {
    if (e.key === "Escape") {
      closeModal();
    }
  };
  document.addEventListener("keydown", handleEsc);

  // ---------- button/background listeners ----------
  modal
    .querySelector('[data-action="confirm"]')
    .addEventListener("click", () => {
      closeModal();
      onConfirm();
    });

  modal
    .querySelector('[data-action="cancel"]')
    .addEventListener("click", closeModal);

  // Close on background click
  modal.addEventListener("click", (e) => {
    if (e.target === modal) {
      closeModal();
    }
  });

  document.body.appendChild(modal);
}

// Update models overview section
function updateModelsSection(modelHealth, runnerModels) {
  const container = document.getElementById("modelsContainer");
  container.innerHTML = "";

  // Get all currently loaded models
  const loadedModels = new Set(Object.values(runnerModels).filter((m) => m));

  // Create model items for all models
  const modelItems = Object.entries(modelHealth).map(([modelAlias, health]) => {
    const isLoaded = loadedModels.has(modelAlias);
    return {
      alias: modelAlias,
      health: health,
      isLoaded: isLoaded,
    };
  });

  // Sort: loaded models first, then by status, then alphabetically
  modelItems.sort((a, b) => {
    // Loaded models first
    if (a.isLoaded !== b.isLoaded) {
      return b.isLoaded - a.isLoaded;
    }

    // Then by status priority (ok > loading > error > not_loaded > not_running > unloaded)
    const statusPriority = {
      ok: 5,
      loading: 4,
      error: 3,
      not_loaded: 2,
      not_running: 1,
      unloaded: 0,
    };
    const aPriority = statusPriority[a.health.status] || 0;
    const bPriority = statusPriority[b.health.status] || 0;

    if (aPriority !== bPriority) {
      return bPriority - aPriority;
    }

    // Finally alphabetically
    return a.alias.localeCompare(b.alias);
  });

  // Create model items
  modelItems.forEach((model) => {
    const modelElement = document.createElement("div");
    modelElement.innerHTML = createModelItem(model);
    container.appendChild(modelElement.firstElementChild);
  });

  // Show empty state if no models
  if (modelItems.length === 0) {
    container.innerHTML = `
            <div class="empty-state">
                <div class="icon">🤖</div>
                <p>No models available</p>
            </div>
        `;
  }
}

// Update last updated time
function updateLastUpdatedTime() {
  lastUpdateTime = new Date();
  const timeString = lastUpdateTime.toLocaleTimeString();
  document.getElementById("lastUpdated").textContent =
    `Last updated: ${timeString}`;
}

// Show error message
function showError(message) {
  const errorToast = document.getElementById("errorToast");
  const errorMessage = document.getElementById("errorMessage");

  errorMessage.textContent = message;
  errorToast.classList.remove("hidden");

  // Auto-hide after 5 seconds
  setTimeout(() => {
    hideError();
  }, 5000);
}

// Hide error message
function hideError() {
  const errorToast = document.getElementById("errorToast");
  errorToast.classList.add("hidden");
}

// Utility function to get human-readable time difference
function getTimeDifference(date) {
  const now = new Date();
  const diff = Math.floor((now - date) / 1000);

  if (diff < 60) return `${diff}s ago`;
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  return `${Math.floor(diff / 86400)}d ago`;
}

// Handle page visibility changes (pause when hidden)
document.addEventListener("visibilitychange", function () {
  if (document.hidden) {
    console.log("Page hidden, pausing auto-refresh");
    stopAutoRefresh();
  } else {
    console.log("Page visible, resuming auto-refresh");
    startAutoRefresh();
    fetchHealthData(); // Immediate refresh when page becomes visible
  }
});

// Handle window focus/blur
window.addEventListener("focus", function () {
  if (!refreshInterval) {
    startAutoRefresh();
    fetchHealthData();
  }
});

window.addEventListener("blur", function () {
  // Keep running even when blurred for now
  // Could add option to pause when not focused
});

// Cleanup on page unload
window.addEventListener("beforeunload", function () {
  stopAutoRefresh();
});

// Start GPU metrics auto-refresh
function startGpuMetricsRefresh() {
  if (gpuMetricsInterval) {
    clearInterval(gpuMetricsInterval);
  }
  gpuMetricsInterval = setInterval(fetchGpuMetrics, CONFIG.GPU_METRICS_REFRESH_INTERVAL);
  console.log(`GPU metrics auto-refresh started: every ${CONFIG.GPU_METRICS_REFRESH_INTERVAL}ms`);
}

// Stop GPU metrics auto-refresh
function stopGpuMetricsRefresh() {
  if (gpuMetricsInterval) {
    clearInterval(gpuMetricsInterval);
    gpuMetricsInterval = null;
  }
}

// Fetch GPU metrics from API
async function fetchGpuMetrics() {
  if (Date.now() < gpuMetricsRateLimitedUntil) {
    return;
  }

  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), CONFIG.REQUEST_TIMEOUT);

    const response = await fetch(CONFIG.GPU_METRICS_URL, {
      signal: controller.signal,
      headers: {
        Accept: "application/json",
        "Cache-Control": "no-cache",
      },
    });

    clearTimeout(timeoutId);

    if (response.status === 429) {
      const retryAfterHeader = response.headers.get("Retry-After");
      const retryAfterSeconds = Math.max(1, parseInt(retryAfterHeader || "5", 10) || 5);
      gpuMetricsRateLimitedUntil = Date.now() + retryAfterSeconds * 1000;

      // Keep last successful cards visible to avoid flicker.
      if (!gpuMetricsState || gpuMetricsState.status !== "available") {
        updateGpuMetricsPanel({ status: "unavailable", reason: "rate_limited" });
      }
      return;
    }

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    const data = await response.json();
    gpuMetricsRateLimitedUntil = 0;
    gpuMetricsState = data;
    updateGpuMetricsPanel(data);
    lastGpuMetricsTime = new Date();
  } catch (error) {
    console.error("Failed to fetch GPU metrics:", error);
    updateGpuMetricsPanel({ status: "unavailable", reason: "fetch_error", collection_error: error.message });
  }
}

// Update the GPU metrics panel
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
  } else {
    container.innerHTML = createGpuMetricsUnavailable(data);
  }
}

// Build a map from GPU id to runner info
function buildGpuRunnerMap(associations) {
  const map = {};
  Object.entries(associations).forEach(function (entry) {
    var runnerName = entry[0];
    var info = entry[1];
    var gpuKeys = [];
    if (Array.isArray(info.gpu_ids)) {
      gpuKeys = info.gpu_ids;
    } else if (Array.isArray(info.gpu_indices)) {
      gpuKeys = info.gpu_indices.map(function (idx) {
        return String(idx);
      });
    }

    gpuKeys.forEach(function (gpuKey) {
      if (!map[gpuKey]) map[gpuKey] = [];
      map[gpuKey].push(runnerName);
    }
    );
  });
  return map;
}

// Create an unavailable state message
function createGpuMetricsUnavailable(data) {
  var reason = data.reason || "unknown";
  var reasonText = {
    unsupported_platform: "GPU metrics are not supported on this operating system.",
    tool_not_found: "No supported GPU telemetry tool found. Install nvidia-smi and/or amd-smi.",
    command_failed: "GPU telemetry command failed. Check driver/tool installation and GPU visibility.",
    command_timeout: "amd-smi command timed out.",
    parse_error: "Failed to parse amd-smi output.",
    no_visible_gpus: "No visible GPUs detected. Ensure GPU devices are accessible.",
    disabled_in_config: "GPU metrics collection is disabled in configuration.",
    fetch_error: "Could not reach the GPU metrics endpoint.",
    rate_limited: "GPU metrics request rate limited. The dashboard will retry shortly.",
  };

  var message = reasonText[reason] || "GPU metrics are currently unavailable.";
  if (data.collection_error) {
    message += " (" + sanitizeHTML(data.collection_error) + ")";
  }

  return '<div class="gpu-metrics-unavailable">' +
    '<div class="unavailable-icon">&#9888;</div>' +
    '<div class="unavailable-text">GPU metrics unavailable</div>' +
    '<div class="unavailable-reason">' + message + '</div>' +
    '</div>';
}

// Create a GPU metric card element
function createGpuMetricCard(gpu, history, gpuRunnerMap) {
  var idx = gpu.index;
  var gpuKey = gpu.id ? String(gpu.id) : String(idx);
  var name = gpu.name || ("GPU " + idx);
  var vendor = gpu.vendor || "unknown";
  var runners = gpuRunnerMap[gpuKey] || gpuRunnerMap[String(idx)] || [];

  var card = document.createElement("div");
  card.className = "gpu-metric-card";

  var memUsed = gpu.memory_used_mb !== null && gpu.memory_used_mb !== undefined
    ? formatMb(gpu.memory_used_mb) : "--";
  var memTotal = gpu.memory_total_mb !== null && gpu.memory_total_mb !== undefined
    ? formatMb(gpu.memory_total_mb) : "--";
  var memPercent = (gpu.memory_used_mb != null && gpu.memory_total_mb != null && gpu.memory_total_mb > 0)
    ? Math.round((gpu.memory_used_mb / gpu.memory_total_mb) * 100) : null;
  var util = gpu.utilization_gpu_percent !== null && gpu.utilization_gpu_percent !== undefined
    ? Math.round(gpu.utilization_gpu_percent) + "%" : "--";
  var temp = gpu.temperature_c !== null && gpu.temperature_c !== undefined
    ? Math.round(gpu.temperature_c) + "°C" : "--";

  var historyKey = gpuKey;
  var memHistory = (history[historyKey] || {}).memory_used_mb || [];
  var utilHistory = (history[historyKey] || {}).utilization_gpu_percent || [];
  var tempHistory = (history[historyKey] || {}).temperature_c || [];

  card.innerHTML =
    '<div class="gpu-card-header">' +
      '<h3 class="gpu-name">' + sanitizeHTML(name) + '</h3>' +
      '<span class="gpu-index">' + sanitizeHTML(vendor.toUpperCase()) + ' GPU ' + idx + '</span>' +
    '</div>' +
    '<div class="gpu-metrics-row">' +
      '<div class="gpu-metric-chip' + (memPercent !== null && memPercent > 90 ? ' metric-warning' : '') + '">' +
        '<span class="metric-label">VRAM</span>' +
        '<span class="metric-value">' + memUsed + ' / ' + memTotal + '</span>' +
        (memPercent !== null ? '<span class="metric-percent">' + memPercent + '%</span>' : '') +
      '</div>' +
      '<div class="gpu-metric-chip">' +
        '<span class="metric-label">Util</span>' +
        '<span class="metric-value">' + util + '</span>' +
      '</div>' +
      '<div class="gpu-metric-chip' + (gpu.temperature_c !== null && gpu.temperature_c > 85 ? ' metric-warning' : '') + '">' +
        '<span class="metric-label">Temp</span>' +
        '<span class="metric-value">' + temp + '</span>' +
      '</div>' +
    '</div>' +
    '<div class="gpu-sparklines">' +
      '<div class="sparkline-group">' +
        '<span class="sparkline-label">VRAM</span>' +
        '<div class="sparkline-container">' + renderSparkline(memHistory, "#64ffda") + '</div>' +
      '</div>' +
      '<div class="sparkline-group">' +
        '<span class="sparkline-label">Util</span>' +
        '<div class="sparkline-container">' + renderSparkline(utilHistory, "#ffb74d") + '</div>' +
      '</div>' +
      '<div class="sparkline-group">' +
        '<span class="sparkline-label">Temp</span>' +
        '<div class="sparkline-container">' + renderSparkline(tempHistory, "#e57373") + '</div>' +
      '</div>' +
    '</div>' +
    (runners.length > 0
      ? '<div class="gpu-runners"><span class="gpu-runners-label">Runners:</span> ' +
        runners.map(function (r) { return sanitizeHTML(r); }).join(", ") + '</div>'
      : '') +
    (gpu.power_w !== null && gpu.power_w !== undefined
      ? '<div class="gpu-extra">Power: ' + gpu.power_w.toFixed(1) + 'W</div>'
      : '');

  return card;
}

// Render an inline SVG sparkline from an array of numeric values
function renderSparkline(values, color) {
  if (!values || values.length < 2) {
    return '<svg class="sparkline" viewBox="0 0 120 24"><line x1="0" y1="12" x2="120" y2="12" stroke="' + color + '" stroke-opacity="0.2" stroke-width="1"/></svg>';
  }

  var filtered = values.filter(function (v) { return v !== null && v !== undefined; });
  if (filtered.length < 2) {
    return '<svg class="sparkline" viewBox="0 0 120 24"><line x1="0" y1="12" x2="120" y2="12" stroke="' + color + '" stroke-opacity="0.2" stroke-width="1"/></svg>';
  }

  var min = Math.min.apply(null, filtered);
  var max = Math.max.apply(null, filtered);
  var range = max - min;
  if (range === 0) range = 1;

  var w = 120;
  var h = 24;
  var pad = 2;
  var step = (w - pad * 2) / (filtered.length - 1);

  var points = filtered.map(function (v, i) {
    var x = pad + i * step;
    var y = h - pad - ((v - min) / range) * (h - pad * 2);
    return x.toFixed(1) + "," + y.toFixed(1);
  });

  var pathD = "M" + points.join(" L");

  return '<svg class="sparkline" viewBox="0 0 ' + w + ' ' + h + '">' +
    '<path d="' + pathD + '" fill="none" stroke="' + color + '" stroke-width="1.5" stroke-linejoin="round" stroke-linecap="round"/>' +
    '</svg>';
}

// Format megabytes to a human-readable string
function formatMb(mb) {
  if (mb === null || mb === undefined) return "--";
  if (mb >= 1024) return (mb / 1024).toFixed(1) + " GB";
  return Math.round(mb) + " MB";
}

// Cleanup on page unload
window.addEventListener("beforeunload", function () {
  stopAutoRefresh();
  stopGpuMetricsRefresh();
});

// Handle visibility changes for GPU metrics too
document.addEventListener("visibilitychange", function () {
  if (document.hidden) {
    stopGpuMetricsRefresh();
  } else {
    startGpuMetricsRefresh();
    fetchGpuMetrics();
  }
});

// Export functions for debugging
window.llama_dashboard = {
  fetchHealthData,
  startAutoRefresh,
  stopAutoRefresh,
  showError,
  hideError,
  startRunner,
  stopRunner,
  restartRunner,
  fetchGpuMetrics,
  config: CONFIG,
};
