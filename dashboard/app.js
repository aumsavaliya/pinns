document.addEventListener('DOMContentLoaded', () => {
    // Chart configurations
    Chart.defaults.color = '#94a3b8';
    Chart.defaults.font.family = "'Segoe UI', Roboto, Helvetica, Arial, sans-serif";

    const commonOptions = {
        responsive: true,
        maintainAspectRatio: false,
        animation: {
            duration: 0 // Disable animation for live updates to keep it smooth
        },
        interaction: {
            mode: 'index',
            intersect: false,
        },
        plugins: {
            legend: {
                display: false
            },
            tooltip: {
                backgroundColor: 'rgba(20, 25, 40, 0.8)',
                titleColor: '#e2e8f0',
                bodyColor: '#e2e8f0',
                borderColor: 'rgba(255, 255, 255, 0.1)',
                borderWidth: 1,
                padding: 10,
                displayColors: false,
            }
        },
        scales: {
            x: {
                grid: {
                    color: 'rgba(255, 255, 255, 0.05)',
                    drawBorder: false,
                },
                ticks: {
                    maxRotation: 0,
                    autoSkip: true,
                    maxTicksLimit: 10
                }
            },
            y: {
                grid: {
                    color: 'rgba(255, 255, 255, 0.05)',
                    drawBorder: false,
                }
            }
        }
    };

    // Initialize Voltage Chart
    const voltageCtx = document.getElementById('voltage-chart').getContext('2d');
    const voltageGradient = voltageCtx.createLinearGradient(0, 0, 0, 400);
    voltageGradient.addColorStop(0, 'rgba(0, 240, 255, 0.5)');
    voltageGradient.addColorStop(1, 'rgba(0, 240, 255, 0.0)');

    const voltageChart = new Chart(voltageCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Voltage (V)',
                data: [],
                borderColor: '#00f0ff',
                backgroundColor: voltageGradient,
                borderWidth: 2,
                pointRadius: 0,
                pointHoverRadius: 4,
                fill: true,
                tension: 0.4
            }]
        },
        options: {
            ...commonOptions,
            scales: {
                ...commonOptions.scales,
                y: {
                    ...commonOptions.scales.y,
                    suggestedMin: 3.0,
                    suggestedMax: 4.2
                }
            }
        }
    });

    // Initialize Temperature Chart
    const tempCtx = document.getElementById('temp-chart').getContext('2d');
    const tempGradient = tempCtx.createLinearGradient(0, 0, 0, 400);
    tempGradient.addColorStop(0, 'rgba(176, 38, 255, 0.5)');
    tempGradient.addColorStop(1, 'rgba(176, 38, 255, 0.0)');

    const tempChart = new Chart(tempCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Temperature (°C)',
                data: [],
                borderColor: '#b026ff',
                backgroundColor: tempGradient,
                borderWidth: 2,
                pointRadius: 0,
                pointHoverRadius: 4,
                fill: true,
                tension: 0.4
            }]
        },
        options: {
            ...commonOptions,
            scales: {
                ...commonOptions.scales,
                y: {
                    ...commonOptions.scales.y,
                    suggestedMin: 20,
                    suggestedMax: 60
                }
            }
        }
    });

    // Max data points to keep on chart
    const MAX_DATA_POINTS = 60;

    // DOM Elements
    const socValue = document.getElementById('soc-value');
    const socProgress = document.getElementById('soc-progress');
    const sohValue = document.getElementById('soh-value');
    const sohProgress = document.getElementById('soh-progress');

    // Fetch data and update
    async function fetchBatteryStatus() {
        try {
            const response = await fetch('http://localhost:5000/api/battery/status');
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            updateDashboard(data);
            
        } catch (error) {
            console.error('Failed to fetch battery status:', error);
            // Fallback to mock data if API is unavailable
            mockDataUpdate();
        }
    }

    // Mock data for demonstration when API is unavailable
    let timeCounter = 0;
    function mockDataUpdate() {
        const now = new Date();
        const timeString = now.toLocaleTimeString([], { hour12: false });
        
        const t = timeCounter++;
        const mockData = {
            soc: Math.max(0, 85 - (t * 0.01)),
            soh: 98.5,
            voltage: 3.7 + Math.sin(t * 0.1) * 0.2 + (Math.random() * 0.02),
            temperature: 35 + Math.cos(t * 0.05) * 5 + (Math.random() * 0.2),
            timestamp: timeString
        };
        
        updateDashboard(mockData);
    }

    function updateDashboard(data) {
        // Update DOM elements for SOC and SOH
        const soc = parseFloat(data.soc).toFixed(1);
        const soh = parseFloat(data.soh).toFixed(1);

        socValue.textContent = soc;
        socProgress.style.width = `${soc}%`;

        sohValue.textContent = soh;
        sohProgress.style.width = `${soh}%`;

        // Update Charts
        const timeLabel = data.timestamp || new Date().toLocaleTimeString([], { hour12: false });

        updateChartData(voltageChart, timeLabel, data.voltage);
        updateChartData(tempChart, timeLabel, data.temperature);
    }

    function updateChartData(chart, label, value) {
        chart.data.labels.push(label);
        chart.data.datasets[0].data.push(value);

        if (chart.data.labels.length > MAX_DATA_POINTS) {
            chart.data.labels.shift();
            chart.data.datasets[0].data.shift();
        }

        chart.update();
    }

    // Start polling every second
    setInterval(fetchBatteryStatus, 1000);
    // Initial fetch
    fetchBatteryStatus();
});
