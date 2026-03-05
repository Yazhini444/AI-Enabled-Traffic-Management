// Global variables
let socket;
let isConnected = false;

// Initialize Socket.IO connection
function initSocket() {
    socket = io();
    
    socket.on('connect', function() {
        console.log('Connected to server');
        isConnected = true;
        updateConnectionStatus(true);
    });
    
    socket.on('disconnect', function() {
        console.log('Disconnected from server');
        isConnected = false;
        updateConnectionStatus(false);
    });
    
    socket.on('stats_update', function(data) {
        updateStats(data);
    });
    
    socket.on('signal_update', function(data) {
        updateSignalStatus(data);
    });
    
    socket.on('lane_density_update', function(data) {
        updateLaneDensity(data);
    });
    
    socket.on('emergency', function(data) {
        showEmergencyAlert(data);
    });
    
    socket.on('emergency_cleared', function(data) {
        hideEmergencyAlert();
    });
    
    socket.on('vehicle_update', function(data) {
        updateVehicle(data);
    });
    
    socket.on('vehicle_removed', function(data) {
        removeVehicle(data);
    });
    
    socket.on('congestion_update', function(data) {
        updateCongestion(data);
    });
}

// Update connection status indicator
function updateConnectionStatus(connected) {
    const statusElement = document.getElementById('connection-status');
    if (statusElement) {
        if (connected) {
            statusElement.textContent = 'Connected';
            statusElement.className = 'badge bg-success';
        } else {
            statusElement.textContent = 'Disconnected';
            statusElement.className = 'badge bg-danger';
        }
    }
}

// Update system clock
function updateClock() {
    const now = new Date();
    const timeString = now.toLocaleTimeString();
    const clockElement = document.getElementById('system-clock');
    if (clockElement) {
        clockElement.textContent = timeString;
    }
}

// Update statistics
function updateStats(data) {
    // Update vehicle count
    const vehicleCountElement = document.getElementById('totalVehicles');
    if (vehicleCountElement) {
        vehicleCountElement.textContent = data.vehicle_count.toLocaleString();
    }
    
    // Update average speed
    const avgSpeedElement = document.getElementById('avgSpeed');
    if (avgSpeedElement) {
        avgSpeedElement.textContent = data.avg_speed;
    }
    
    // Update violations
    const violationsElement = document.getElementById('violations');
    if (violationsElement) {
        violationsElement.textContent = data.violations;
    }
    
    // Update emergency events
    const emergencyEventsElement = document.getElementById('emergencies');
    if (emergencyEventsElement) {
        emergencyEventsElement.textContent = data.emergency_events;
    }
}

// Update lane densities
function updateLaneDensity(data) {
    const lane = data.lane.toLowerCase();
    const counts = data.counts;
    
    // Update lane density display
    const densityElement = document.getElementById(`${lane}Density`);
    if (densityElement) {
        const total = counts.total || 0;
        densityElement.textContent = `${total} vehicles`;
        
        // Update progress bar
        const progressBar = document.getElementById(`${lane}ProgressBar`);
        if (progressBar) {
            const percentage = Math.min(100, total * 5); // Assuming max 20 vehicles
            progressBar.style.width = `${percentage}%`;
        }
    }
}

// Update signal status
function updateSignalStatus(data) {
    // Update current phase and lane
    const currentPhaseElement = document.getElementById('currentPhase');
    if (currentPhaseElement) {
        currentPhaseElement.textContent = data.phase;
    }
    
    const currentLaneElement = document.getElementById('currentLane');
    if (currentLaneElement) {
        currentLaneElement.textContent = data.lane;
    }
    
    const timeRemainingElement = document.getElementById('timeRemaining');
    if (timeRemainingElement) {
        timeRemainingElement.textContent = data.time_remaining;
    }
    
    // Update signal lights
    const lanes = ['north', 'south', 'east', 'west'];
    
    lanes.forEach(lane => {
        // Reset all lights
        const redLight = document.getElementById(`${lane}Red`);
        const yellowLight = document.getElementById(`${lane}Yellow`);
        const greenLight = document.getElementById(`${lane}Green`);
        
        if (redLight) redLight.classList.remove('active');
        if (yellowLight) yellowLight.classList.remove('active');
        if (greenLight) greenLight.classList.remove('active');
        
        // Activate the appropriate light
        if (lane.toLowerCase() === data.lane.toLowerCase()) {
            if (data.phase === 'RED') {
                if (redLight) redLight.classList.add('active');
            } else if (data.phase === 'YELLOW') {
                if (yellowLight) yellowLight.classList.add('active');
            } else if (data.phase === 'GREEN') {
                if (greenLight) greenLight.classList.add('active');
            }
        } else {
            // Other lanes are red
            if (redLight) redLight.classList.add('active');
        }
    });
    
    // Update emergency alert
    if (data.emergency_mode && data.emergency_lane) {
        showEmergencyAlert({
            lane: data.emergency_lane,
            message: `Emergency vehicle detected in ${data.emergency_lane} lane`
        });
    } else {
        hideEmergencyAlert();
    }
}

// Show emergency alert
function showEmergencyAlert(data) {
    const alertElement = document.getElementById('emergencyAlert');
    if (alertElement) {
        const laneElement = document.getElementById('emergencyLane');
        if (laneElement) {
            laneElement.textContent = data.lane;
        }
        alertElement.classList.remove('d-none');
    }
}

// Hide emergency alert
function hideEmergencyAlert() {
    const alertElement = document.getElementById('emergencyAlert');
    if (alertElement) {
        alertElement.classList.add('d-none');
    }
}

// Update vehicle (for map)
function updateVehicle(data) {
    // This will be implemented in map.js
}

// Remove vehicle (for map)
function removeVehicle(data) {
    // This will be implemented in map.js
}

// Update congestion (for map)
function updateCongestion(data) {
    // This will be implemented in map.js
}

// Format date and time
function formatDateTime(dateString) {
    const date = new Date(dateString);
    return date.toLocaleString();
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    initSocket();
    
    // Update clock every second
    setInterval(updateClock, 1000);
    updateClock();
});

// Export functions for use in other scripts
window.trafficSystem = {
    socket,
    isConnected,
    updateStats,
    updateSignalStatus,
    updateLaneDensity,
    showEmergencyAlert,
    hideEmergencyAlert,
    formatDateTime
};