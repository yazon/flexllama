(function () {
  const Dashboard = window.FlexLlamaDashboard || {};
  window.FlexLlamaDashboard = Dashboard;

  const config = Dashboard.config;
  const state = Dashboard.state;
  const ui = Dashboard.ui;
  const utils = Dashboard.utils;
  const STATUS_MAP = Dashboard.constants.STATUS_MAP;

  function startAutoRefresh() {
    if (state.refreshInterval) {
      clearInterval(state.refreshInterval);
    }

    state.refreshInterval = setInterval(fetchHealthData, config.REFRESH_INTERVAL);
    console.log(`Auto-refresh started: every ${config.REFRESH_INTERVAL}ms`);
  }

  function stopAutoRefresh() {
    if (state.refreshInterval) {
      clearInterval(state.refreshInterval);
      state.refreshInterval = null;
    }
  }

  function isAutoRefreshRunning() {
    return Boolean(state.refreshInterval);
  }

  async function fetchHealthData() {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(function () {
        controller.abort();
      }, config.REQUEST_TIMEOUT);

      const response = await fetch(config.API_BASE_URL, {
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
      ui.updateLastUpdatedTime();
      ui.hideError();
    } catch (error) {
      console.error("Failed to fetch health data:", error);
      ui.showError(`Failed to fetch data: ${error.message}`);
    }
  }

  function updateDashboard(data) {
    const activeRunners = data.active_runners || {};
    const runnerCurrentModels = data.runner_current_models || {};
    const runnerInfo = data.runner_info || {};
    const modelHealth = data.model_health || {};

    updateRunnersSection(
      activeRunners,
      runnerCurrentModels,
      runnerInfo,
      modelHealth,
    );
    updateModelsSection(modelHealth, runnerCurrentModels);
  }

  function updateRunnersSection(
    activeRunners = {},
    runnerModels = {},
    runnerInfo = {},
    modelHealth = {},
  ) {
    const container = document.getElementById("runnersContainer");
    if (!container) return;

    container.innerHTML = "";
    const runnerData = {};

    Object.keys(activeRunners).forEach(function (runnerName) {
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

    Object.entries(modelHealth).forEach(function (entry) {
      const modelAlias = entry[0];
      const health = entry[1];

      let assignedRunner = null;
      Object.entries(runnerModels).forEach(function (modelEntry) {
        const runnerName = modelEntry[0];
        const currentModel = modelEntry[1];
        if (currentModel === modelAlias) {
          assignedRunner = runnerName;
        }
      });

      if (assignedRunner && runnerData[assignedRunner]) {
        runnerData[assignedRunner].models.push({
          alias: modelAlias,
          health,
          isLoaded: true,
        });
      }
    });

    Object.entries(runnerData).forEach(function (entry) {
      const runnerName = entry[0];
      const data = entry[1];
      const runnerCard = createRunnerCard(runnerName, data);
      container.appendChild(runnerCard);
    });

    if (Object.keys(runnerData).length === 0) {
      container.innerHTML =
        '<div class="empty-state">' +
        '<div class="icon">🏃</div>' +
        "<p>No runners configured</p>" +
        "</div>";
    }

    attachControlListeners();
  }

  function createRunnerCard(runnerName, data) {
    const runnerSlug = utils.slugify(runnerName);
    const safeRunnerName = utils.sanitizeHTML(runnerName);
    const safeCurrentModel = utils.sanitizeHTML(data.currentModel || "None");
    const safeHost = utils.sanitizeHTML(data.host);
    const safePort = utils.sanitizeHTML(String(data.port));

    const card = document.createElement("div");
    card.className = `runner-card ${data.isActive ? "active" : "inactive"}`;

    card.innerHTML = `
      <div class="runner-header">
        <h3 class="runner-title">${safeRunnerName}</h3>
        <div class="runner-status">
          <div class="status-indicator ${data.isActive ? "active" : "inactive"}"></div>
          <span>${data.isActive ? "Active" : "Inactive"}</span>
        </div>
      </div>

      <div class="runner-info">
        <div><strong>Current Model:</strong> ${safeCurrentModel}</div>
        <div><strong>Status:</strong> ${data.isActive ? "Running" : "Stopped"}</div>
        <div><strong>Host:</strong> ${safeHost}</div>
        <div><strong>Port:</strong> ${safePort}</div>
        <div><strong>Auto-unload:</strong> ${utils.formatAutoUnloadStatus(data.autoUnloadTimeout, data.autoUnloadCountdown)}</div>
        ${
          data.isActive
            ? `<div><strong>URL:</strong> http://${safeHost}:${safePort}</div>`
            : ""
        }
      </div>

      <div class="models-list">
        ${
          data.models.length > 0
            ? data.models.map(function (model) {
                return createModelItem(model);
              }).join("")
            : '<div class="empty-state"><p>No models currently loaded</p></div>'
        }
      </div>

      ${createControlPanel(runnerName, data, runnerSlug)}
    `;

    return card;
  }

  function createModelItem(model) {
    const alias = model.alias;
    const health = model.health;
    const isLoaded = model.isLoaded;

    const status = health.status || "unloaded";
    const statusInfo = STATUS_MAP[status] || STATUS_MAP.error;

    return `
      <div class="model-item status-${statusInfo.color}">
        <div class="status-gauge">
          <div class="gauge-circle status-${statusInfo.color}">
            <div class="gauge-inner">${statusInfo.icon}</div>
          </div>
        </div>
        <div class="model-info">
          <div class="model-name" title="${utils.sanitizeHTML(alias)}">${utils.sanitizeHTML(alias)}</div>
          <div class="model-status status-${statusInfo.color}">${statusInfo.label}</div>
          <div class="model-message">${utils.sanitizeHTML(health.message || "No message")}</div>
          <div class="model-details">${isLoaded ? "Currently loaded" : "Available but not loaded"}</div>
        </div>
      </div>
    `;
  }

  function createControlPanel(runnerName, data, runnerSlug) {
    const operationState = state.operationStates[runnerName];
    const isOperating = operationState && operationState.inProgress;
    const operationType = operationState ? operationState.type : null;

    const canStart = !data.isActive && !isOperating;
    const canStop = data.isActive && !isOperating;
    const canRestart = data.isActive && !isOperating;

    const loadingIcon = STATUS_MAP.loading.icon;
    const startIcon = STATUS_MAP.start.icon;
    const stopIcon = STATUS_MAP.not_running.icon;
    const restartIcon = "↻";

    return `
      <div class="runner-controls" data-runner="${utils.sanitizeHTML(runnerName)}" data-runner-slug="${runnerSlug}">
        <div class="control-buttons">
          <button class="control-button btn-start ${isOperating && operationType === "start" ? "loading" : ""}" data-action="start" ${!canStart ? "disabled" : ""}>
            <span class="btn-icon">${isOperating && operationType === "start" ? loadingIcon : startIcon}</span>
            <span class="btn-text">${isOperating && operationType === "start" ? "Starting..." : "Start"}</span>
          </button>
          <button class="control-button btn-stop ${isOperating && operationType === "stop" ? "loading" : ""}" data-action="stop" ${!canStop ? "disabled" : ""}>
            <span class="btn-icon">${isOperating && operationType === "stop" ? loadingIcon : stopIcon}</span>
            <span class="btn-text">${isOperating && operationType === "stop" ? "Stopping..." : "Stop"}</span>
          </button>
          <button class="control-button btn-restart ${isOperating && operationType === "restart" ? "loading" : ""}" data-action="restart" ${!canRestart ? "disabled" : ""}>
            <span class="btn-icon">${isOperating && operationType === "restart" ? loadingIcon : restartIcon}</span>
            <span class="btn-text">${isOperating && operationType === "restart" ? "Restarting..." : "Restart"}</span>
          </button>
        </div>
        <div class="control-status ${operationState ? operationState.statusClass : ""}" id="status-${runnerSlug}">${operationState ? operationState.message : ""}</div>
      </div>
    `;
  }

  function attachControlListeners() {
    document.querySelectorAll(".control-button").forEach(function (button) {
      button.removeEventListener("click", handleControlClick);
      button.addEventListener("click", handleControlClick);
    });
  }

  function handleControlClick(event) {
    const button = event.currentTarget;
    const action = button.getAttribute("data-action");
    const controls = button.closest(".runner-controls");
    if (!controls || button.disabled) return;

    const runnerName = controls.getAttribute("data-runner");
    if (!runnerName) return;

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
      default:
        break;
    }
  }

  async function startRunner(runnerName) {
    await performRunnerAction(
      runnerName,
      "start",
      "Starting runner...",
      "Runner started successfully",
      "Failed to start runner",
    );
  }

  function confirmAndStopRunner(runnerName) {
    showConfirmationModal(
      "Stop Runner",
      `Are you sure you want to stop runner "${runnerName}"? This will interrupt any ongoing operations and unload the current model.`,
      function () {
        stopRunner(runnerName);
      },
    );
  }

  async function stopRunner(runnerName) {
    await performRunnerAction(
      runnerName,
      "stop",
      "Stopping runner...",
      "Runner stopped successfully",
      "Failed to stop runner",
    );
  }

  function confirmAndRestartRunner(runnerName) {
    showConfirmationModal(
      "Restart Runner",
      `Are you sure you want to restart runner "${runnerName}"? This will stop the runner, then start it again with the same model.`,
      function () {
        restartRunner(runnerName);
      },
    );
  }

  async function restartRunner(runnerName) {
    await performRunnerAction(
      runnerName,
      "restart",
      "Restarting runner...",
      "Runner restarted successfully",
      "Failed to restart runner",
    );
  }

  async function performRunnerAction(
    runnerName,
    action,
    inProgressMessage,
    successMessage,
    defaultErrorMessage,
  ) {
    try {
      setOperationState(runnerName, action, inProgressMessage, "status-info");

      const response = await fetch(
        `/v1/runners/${encodeURIComponent(runnerName)}/${action}`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
        },
      );
      const data = await response.json();

      if (data.success) {
        setOperationState(runnerName, null, successMessage, "status-success");
        setTimeout(function () {
          clearOperationState(runnerName);
        }, 3000);
      } else {
        const errorMsg = data.error ? data.error.message : defaultErrorMessage;
        setOperationState(runnerName, null, errorMsg, "status-error");
        setTimeout(function () {
          clearOperationState(runnerName);
        }, 5000);
      }
    } catch (error) {
      console.error(`Error during ${action}:`, error);
      setOperationState(runnerName, null, `Error: ${error.message}`, "status-error");
      setTimeout(function () {
        clearOperationState(runnerName);
      }, 5000);
    }
  }

  function setOperationState(runnerName, operationType, message, statusClass) {
    state.operationStates[runnerName] = {
      inProgress: operationType !== null,
      type: operationType,
      message,
      statusClass,
    };
    updateRunnerControlUI(runnerName);
  }

  function clearOperationState(runnerName) {
    delete state.operationStates[runnerName];
    updateRunnerControlUI(runnerName);
  }

  function updateRunnerControlUI(runnerName) {
    const runnerSlug = utils.slugify(runnerName);
    const statusElement = document.getElementById(`status-${runnerSlug}`);
    const controlPanel = document.querySelector(
      `.runner-controls[data-runner-slug="${runnerSlug}"]`,
    );
    const operationState = state.operationStates[runnerName];

    if (statusElement) {
      if (operationState) {
        statusElement.textContent = operationState.message;
        statusElement.className = `control-status ${operationState.statusClass}`;
      } else {
        statusElement.textContent = "";
        statusElement.className = "control-status";
      }
    }

    if (!controlPanel) return;

    const startButton = controlPanel.querySelector(".btn-start");
    const stopButton = controlPanel.querySelector(".btn-stop");
    const restartButton = controlPanel.querySelector(".btn-restart");
    const runnerCard = controlPanel.closest(".runner-card");
    const isActive = runnerCard && runnerCard.classList.contains("active");

    const loadingIcon = STATUS_MAP.loading.icon;
    const startIcon = STATUS_MAP.start.icon;
    const stopIcon = STATUS_MAP.not_running.icon;
    const restartIcon = "↻";

    if (operationState && operationState.inProgress) {
      [startButton, stopButton, restartButton].forEach(function (button) {
        if (!button) return;
        button.disabled = true;
        button.classList.remove("loading");
      });

      const setLoading = function (button, text) {
        if (!button) return;
        button.classList.add("loading");
        button.querySelector(".btn-icon").textContent = loadingIcon;
        button.querySelector(".btn-text").textContent = text;
      };

      switch (operationState.type) {
        case "start":
          setLoading(startButton, "Starting...");
          break;
        case "stop":
          setLoading(stopButton, "Stopping...");
          break;
        case "restart":
          setLoading(restartButton, "Restarting...");
          break;
        default:
          break;
      }
      return;
    }

    if (startButton) {
      startButton.disabled = isActive;
      startButton.classList.remove("loading");
      startButton.querySelector(".btn-icon").textContent = startIcon;
      startButton.querySelector(".btn-text").textContent = "Start";
    }
    if (stopButton) {
      stopButton.disabled = !isActive;
      stopButton.classList.remove("loading");
      stopButton.querySelector(".btn-icon").textContent = stopIcon;
      stopButton.querySelector(".btn-text").textContent = "Stop";
    }
    if (restartButton) {
      restartButton.disabled = !isActive;
      restartButton.classList.remove("loading");
      restartButton.querySelector(".btn-icon").textContent = restartIcon;
      restartButton.querySelector(".btn-text").textContent = "Restart";
    }
  }

  function showConfirmationModal(title, message, onConfirm) {
    const existingModal = document.querySelector(".confirmation-modal");
    if (existingModal) {
      existingModal.remove();
    }

    const modal = document.createElement("div");
    modal.className = "confirmation-modal";
    modal.innerHTML = `
      <div class="confirmation-content">
        <h3 class="confirmation-title">${utils.sanitizeHTML(title)}</h3>
        <p class="confirmation-message">${utils.sanitizeHTML(message)}</p>
        <div class="confirmation-buttons">
          <button class="confirmation-button primary" data-action="confirm">Confirm</button>
          <button class="confirmation-button secondary" data-action="cancel">Cancel</button>
        </div>
      </div>
    `;

    const closeModal = function () {
      if (modal.parentNode) {
        modal.remove();
      }
      document.removeEventListener("keydown", handleEsc);
    };

    const handleEsc = function (event) {
      if (event.key === "Escape") {
        closeModal();
      }
    };
    document.addEventListener("keydown", handleEsc);

    modal
      .querySelector('[data-action="confirm"]')
      .addEventListener("click", function () {
        closeModal();
        onConfirm();
      });

    modal
      .querySelector('[data-action="cancel"]')
      .addEventListener("click", closeModal);

    modal.addEventListener("click", function (event) {
      if (event.target === modal) {
        closeModal();
      }
    });

    document.body.appendChild(modal);
  }

  function updateModelsSection(modelHealth, runnerModels) {
    const container = document.getElementById("modelsContainer");
    if (!container) return;

    container.innerHTML = "";
    const loadedModels = new Set(Object.values(runnerModels).filter(Boolean));

    const modelItems = Object.entries(modelHealth).map(function (entry) {
      const modelAlias = entry[0];
      const health = entry[1];
      return {
        alias: modelAlias,
        health,
        isLoaded: loadedModels.has(modelAlias),
      };
    });

    modelItems.sort(function (a, b) {
      if (a.isLoaded !== b.isLoaded) {
        return Number(b.isLoaded) - Number(a.isLoaded);
      }

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

      return a.alias.localeCompare(b.alias);
    });

    modelItems.forEach(function (model) {
      const modelElement = document.createElement("div");
      modelElement.innerHTML = createModelItem(model);
      container.appendChild(modelElement.firstElementChild);
    });

    if (modelItems.length === 0) {
      container.innerHTML =
        '<div class="empty-state">' +
        '<div class="icon">🤖</div>' +
        "<p>No models available</p>" +
        "</div>";
    }
  }

  Dashboard.runners = {
    startAutoRefresh,
    stopAutoRefresh,
    isAutoRefreshRunning,
    fetchHealthData,
    startRunner,
    stopRunner,
    restartRunner,
  };
})();
