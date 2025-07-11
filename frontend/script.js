// Configuration
const CONFIG = {
    API_BASE_URL: window.FLEXLLAMA_CONFIG.HEALTH_ENDPOINT || '/health',
    REFRESH_INTERVAL: 2000,
    REQUEST_TIMEOUT: 5000,
};

// Global state
let refreshInterval = null;
let lastUpdateTime = null;

// Status mapping
const STATUS_MAP = {
    'ok': { label: 'Ready', icon: '✓', color: 'ok' },
    'loading': { label: 'Loading', icon: '⟳', color: 'loading' },
    'error': { label: 'Error', icon: '✗', color: 'error' },
    'not_running': { label: 'Not Running', icon: '⏸', color: 'error' },
    'not_loaded': { label: 'Not Loaded', icon: '○', color: 'error' }
};

// Sanitize HTML to prevent XSS
function sanitizeHTML(text) {
    const element = document.createElement('div');
    element.innerText = text;
    return element.innerHTML;
}

// Initialize the dashboard
document.addEventListener('DOMContentLoaded', function() {
    console.log('🦙 FlexLLama Dashboard initialized');
    startAutoRefresh();
    fetchHealthData(); // Initial load
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
        const timeoutId = setTimeout(() => controller.abort(), CONFIG.REQUEST_TIMEOUT);
        
        const response = await fetch(CONFIG.API_BASE_URL, {
            signal: controller.signal,
            headers: {
                'Accept': 'application/json',
                'Cache-Control': 'no-cache'
            }
        });
        
        clearTimeout(timeoutId);
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const data = await response.json();
        updateDashboard(data);
        updateLastUpdatedTime();
        hideError();
        
    } catch (error) {
        console.error('Failed to fetch health data:', error);
        showError(`Failed to fetch data: ${error.message}`);
    }
}

// Update the dashboard with new data
function updateDashboard(data) {
    console.log('Updating dashboard with data:', data);
    
    const { active_runners, runner_current_models, runner_info, model_health } = data;
    
    // Update runners section
    updateRunnersSection(active_runners, runner_current_models, runner_info, model_health);
    
    // Update models overview section
    updateModelsSection(model_health, runner_current_models);
}

// Update runners section
function updateRunnersSection(activeRunners, runnerModels, runnerInfo, modelHealth) {
    const container = document.getElementById('runnersContainer');
    container.innerHTML = '';
    
    // Group models by runner for display
    const runnerData = {};
    
    // Initialize runner data with host/port info
    Object.keys(activeRunners).forEach(runnerName => {
        const info = runnerInfo[runnerName] || {};
        runnerData[runnerName] = {
            isActive: activeRunners[runnerName],
            currentModel: runnerModels[runnerName],
            host: info.host || 'unknown',
            port: info.port || 'unknown',
            models: []
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
                isLoaded: true
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
}

// Create a runner card element
function createRunnerCard(runnerName, data) {
    const { isActive, currentModel, host, port, models } = data;
    
    const card = document.createElement('div');
    card.className = `runner-card ${isActive ? 'active' : 'inactive'}`;
    
    card.innerHTML = `
        <div class="runner-header">
            <h3 class="runner-title">${runnerName}</h3>
            <div class="runner-status">
                <div class="status-indicator ${isActive ? 'active' : 'inactive'}"></div>
                <span>${isActive ? 'Active' : 'Inactive'}</span>
            </div>
        </div>
        
        <div class="runner-info">
            <div><strong>Current Model:</strong> ${currentModel || 'None'}</div>
            <div><strong>Status:</strong> ${isActive ? 'Running' : 'Stopped'}</div>
            <div><strong>Host:</strong> ${host}</div>
            <div><strong>Port:</strong> ${port}</div>
            ${isActive ? `<div><strong>URL:</strong> http://${host}:${port}</div>` : ''}
        </div>
        
        <div class="models-list">
            ${models.length > 0 ? models.map(model => createModelItem(model)).join('') : 
              '<div class="empty-state"><p>No models currently loaded</p></div>'}
        </div>
    `;
    
    return card;
}

// Create a model item element
function createModelItem(model) {
    const { alias, health, isLoaded } = model;
    const status = health.status || 'unloaded';
    const statusInfo = STATUS_MAP[status] || STATUS_MAP['error'];
    
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
                <div class="model-message">${sanitizeHTML(health.message || 'No message')}</div>
                <div class="model-details">
                    ${isLoaded ? 'Currently loaded' : 'Available but not loaded'}
                </div>
            </div>
        </div>
    `;
}

// Update models overview section
function updateModelsSection(modelHealth, runnerModels) {
    const container = document.getElementById('modelsContainer');
    container.innerHTML = '';
    
    // Get all currently loaded models
    const loadedModels = new Set(Object.values(runnerModels).filter(m => m));
    
    // Create model items for all models
    const modelItems = Object.entries(modelHealth).map(([modelAlias, health]) => {
        const isLoaded = loadedModels.has(modelAlias);
        return {
            alias: modelAlias,
            health: health,
            isLoaded: isLoaded
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
            'ok': 5, 
            'loading': 4, 
            'error': 3, 
            'not_loaded': 2,
            'not_running': 1,
            'unloaded': 0 
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
    modelItems.forEach(model => {
        const modelElement = document.createElement('div');
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
    document.getElementById('lastUpdated').textContent = `Last updated: ${timeString}`;
}

// Show error message
function showError(message) {
    const errorToast = document.getElementById('errorToast');
    const errorMessage = document.getElementById('errorMessage');
    
    errorMessage.textContent = message;
    errorToast.classList.remove('hidden');
    
    // Auto-hide after 5 seconds
    setTimeout(() => {
        hideError();
    }, 5000);
}

// Hide error message
function hideError() {
    const errorToast = document.getElementById('errorToast');
    errorToast.classList.add('hidden');
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
document.addEventListener('visibilitychange', function() {
    if (document.hidden) {
        console.log('Page hidden, pausing auto-refresh');
        stopAutoRefresh();
    } else {
        console.log('Page visible, resuming auto-refresh');
        startAutoRefresh();
        fetchHealthData(); // Immediate refresh when page becomes visible
    }
});

// Handle window focus/blur
window.addEventListener('focus', function() {
    if (!refreshInterval) {
        startAutoRefresh();
        fetchHealthData();
    }
});

window.addEventListener('blur', function() {
    // Keep running even when blurred for now
    // Could add option to pause when not focused
});

// Cleanup on page unload
window.addEventListener('beforeunload', function() {
    stopAutoRefresh();
});

// Export functions for debugging
window.llama_dashboard = {
    fetchHealthData,
    startAutoRefresh,
    stopAutoRefresh,
    showError,
    hideError,
    config: CONFIG
}; 