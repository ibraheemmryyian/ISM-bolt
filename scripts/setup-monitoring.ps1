# Monitoring and Alerting Setup Script for ISM AI Platform
# Configures Prometheus, Grafana, alerting, and dashboards

param(
    [Parameter(Position=0)]
    [ValidateSet("", "install", "configure", "test", "dashboard", "help")]
    [string]$Command = "install"
)

# Configuration
$MonitoringDir = "monitoring"
$GrafanaDir = "$MonitoringDir/grafana"
$PrometheusDir = "$MonitoringDir/prometheus"
$AlertingDir = "$MonitoringDir/alerting"

# Colors for output
$Red = [System.ConsoleColor]::Red
$Green = [System.ConsoleColor]::Green
$Yellow = [System.ConsoleColor]::Yellow
$Blue = [System.ConsoleColor]::Blue

# Logging functions
function Write-MonitoringLog {
    param([string]$Message, [System.ConsoleColor]$Color = $Green)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logMessage = "[$timestamp] $Message"
    Write-Host $logMessage -ForegroundColor $Color
}

# Check prerequisites
function Test-MonitoringPrerequisites {
    Write-MonitoringLog "Checking monitoring prerequisites..." $Blue
    
    $prerequisites = @{
        "Docker" = $false
        "Docker Compose" = $false
        "Kubectl" = $false
        "Helm" = $false
    }
    
    # Check Docker
    try {
        $null = docker --version 2>$null
        $prerequisites["Docker"] = $true
        Write-MonitoringLog "✅ Docker found"
    }
    catch {
        Write-MonitoringLog "❌ Docker not found" $Red
    }
    
    # Check Docker Compose
    try {
        $null = docker-compose --version 2>$null
        $prerequisites["Docker Compose"] = $true
        Write-MonitoringLog "✅ Docker Compose found"
    }
    catch {
        Write-MonitoringLog "❌ Docker Compose not found" $Red
    }
    
    # Check kubectl
    try {
        $null = kubectl version --client 2>$null
        $prerequisites["Kubectl"] = $true
        Write-MonitoringLog "✅ kubectl found"
    }
    catch {
        Write-MonitoringLog "❌ kubectl not found" $Yellow
    }
    
    # Check Helm
    try {
        $null = helm version 2>$null
        $prerequisites["Helm"] = $true
        Write-MonitoringLog "✅ Helm found"
    }
    catch {
        Write-MonitoringLog "❌ Helm not found" $Yellow
    }
    
    return $prerequisites
}

# Create monitoring directory structure
function New-MonitoringStructure {
    Write-MonitoringLog "Creating monitoring directory structure..." $Blue
    
    $directories = @(
        $MonitoringDir,
        $GrafanaDir,
        "$GrafanaDir/dashboards",
        "$GrafanaDir/datasources",
        "$GrafanaDir/provisioning",
        $PrometheusDir,
        $AlertingDir,
        "$AlertingDir/rules",
        "$AlertingDir/templates"
    )
    
    foreach ($dir in $directories) {
        if (!(Test-Path $dir)) {
            New-Item -ItemType Directory -Path $dir -Force | Out-Null
            Write-MonitoringLog "Created directory: $dir"
        }
    }
}

# Install Prometheus
function Install-Prometheus {
    Write-MonitoringLog "Installing Prometheus..." $Blue
    
    # Create Prometheus configuration
    $prometheusConfig = @"
global:
  scrape_interval: 15s
  evaluation_interval: 15s
  external_labels:
    cluster: ism-platform
    environment: production

rule_files:
  - "rules/*.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

scrape_configs:
  # Prometheus itself
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  # ISM Backend
  - job_name: 'ism-backend'
    static_configs:
      - targets: ['localhost:5000']
    metrics_path: /api/health
    scrape_interval: 30s

  # ISM Frontend (if exposing metrics)
  - job_name: 'ism-frontend'
    static_configs:
      - targets: ['localhost:5173']
    scrape_interval: 30s

  # Node Exporter (system metrics)
  - job_name: 'node-exporter'
    static_configs:
      - targets: ['localhost:9100']
    scrape_interval: 15s
"@
    
    Set-Content -Path "$PrometheusDir/prometheus.yml" -Value $prometheusConfig
    Write-MonitoringLog "✅ Prometheus configuration created"
}

# Install Grafana
function Install-Grafana {
    Write-MonitoringLog "Installing Grafana..." $Blue
    
    # Create Grafana datasource configuration
    $datasourceConfig = @"
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: true
"@
    
    Set-Content -Path "$GrafanaDir/datasources/prometheus.yml" -Value $datasourceConfig
    Write-MonitoringLog "✅ Grafana datasource configuration created"
    
    # Create ISM Platform dashboard
    New-ISMDashboard
}

# Create ISM Platform dashboard
function New-ISMDashboard {
    Write-MonitoringLog "Creating ISM Platform dashboard..." $Blue
    
    $dashboard = @"
{
  "dashboard": {
    "id": null,
    "title": "ISM AI Platform",
    "tags": ["ism", "ai", "platform"],
    "style": "dark",
    "timezone": "browser",
    "panels": [
      {
        "id": 1,
        "title": "System Health",
        "type": "stat",
        "targets": [
          {
            "expr": "up{job=\"ism-backend\"}",
            "legendFormat": "Backend"
          },
          {
            "expr": "up{job=\"ism-frontend\"}",
            "legendFormat": "Frontend"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "color": {
              "mode": "thresholds"
            },
            "thresholds": {
              "steps": [
                {"color": "red", "value": 0},
                {"color": "green", "value": 1}
              ]
            }
          }
        },
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0}
      },
      {
        "id": 2,
        "title": "API Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_request_duration_seconds_sum[5m]) / rate(http_request_duration_seconds_count[5m])",
            "legendFormat": "Response Time"
          }
        ],
        "yAxes": [
          {
            "label": "Seconds",
            "min": 0
          }
        ],
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0}
      },
      {
        "id": 3,
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])",
            "legendFormat": "Requests/sec"
          }
        ],
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8}
      },
      {
        "id": 4,
        "title": "Error Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total{status=~\"5..\"}[5m])",
            "legendFormat": "5xx Errors"
          },
          {
            "expr": "rate(http_requests_total{status=~\"4..\"}[5m])",
            "legendFormat": "4xx Errors"
          }
        ],
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8}
      },
      {
        "id": 5,
        "title": "AI Processing Time",
        "type": "graph",
        "targets": [
          {
            "expr": "ism_ai_processing_duration_seconds",
            "legendFormat": "AI Processing"
          }
        ],
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 16}
      },
      {
        "id": 6,
        "title": "Database Connections",
        "type": "graph",
        "targets": [
          {
            "expr": "ism_database_connections_active",
            "legendFormat": "Active Connections"
          }
        ],
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 16}
      }
    ],
    "time": {
      "from": "now-1h",
      "to": "now"
    },
    "refresh": "30s"
  }
}
"@
    
    Set-Content -Path "$GrafanaDir/dashboards/ism-platform.json" -Value $dashboard
    Write-MonitoringLog "✅ ISM Platform dashboard created"
}

# Configure alerting
function Set-AlertingConfiguration {
    Write-MonitoringLog "Configuring alerting..." $Blue
    
    # Copy alerting rules
    if (Test-Path "$AlertingDir/rules/alerting-rules.yml") {
        Copy-Item "$AlertingDir/rules/alerting-rules.yml" "$PrometheusDir/rules/" -Force
        Write-MonitoringLog "✅ Alerting rules copied to Prometheus"
    }
    
    # Create alertmanager configuration
    $alertmanagerConfig = @"
global:
  resolve_timeout: 5m
  slack_api_url: 'https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK'

route:
  group_by: ['alertname', 'cluster', 'service']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'web.hook'
  routes:
    - match:
        severity: critical
      receiver: 'slack-critical'
      continue: true
    - match:
        severity: warning
      receiver: 'slack-warning'

receivers:
  - name: 'web.hook'
    webhook_configs:
      - url: 'http://127.0.0.1:5001/'

  - name: 'slack-critical'
    slack_configs:
      - channel: '#alerts-critical'
        send_resolved: true
        title: '{{ template \"slack.ism.title\" . }}'
        text: '{{ template \"slack.ism.text\" . }}'

  - name: 'slack-warning'
    slack_configs:
      - channel: '#alerts-warning'
        send_resolved: true
        title: '{{ template \"slack.ism.title\" . }}'
        text: '{{ template \"slack.ism.text\" . }}'

inhibit_rules:
  - source_match:
      severity: 'critical'
    target_match:
      severity: 'warning'
    equal: ['alertname', 'dev', 'instance']
"@
    
    Set-Content -Path "$AlertingDir/alertmanager.yml" -Value $alertmanagerConfig
    Write-MonitoringLog "✅ Alertmanager configuration created"
}

# Create Docker Compose for monitoring
function New-MonitoringCompose {
    Write-MonitoringLog "Creating Docker Compose for monitoring..." $Blue
    
    $composeContent = @"
version: '3.8'

services:
  prometheus:
    image: prom/prometheus:latest
    container_name: ism-prometheus
    restart: unless-stopped
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus:/etc/prometheus
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--storage.tsdb.retention.time=200h'
      - '--web.enable-lifecycle'
    healthcheck:
      test: ["CMD", "wget", "--no-verbose", "--tries=1", "--spider", "http://localhost:9090/-/healthy"]
      interval: 30s
      timeout: 10s
      retries: 3
    networks:
      - monitoring

  alertmanager:
    image: prom/alertmanager:latest
    container_name: ism-alertmanager
    restart: unless-stopped
    ports:
      - "9093:9093"
    volumes:
      - ./alerting/alertmanager.yml:/etc/alertmanager/alertmanager.yml
      - alertmanager_data:/alertmanager
    command:
      - '--config.file=/etc/alertmanager/alertmanager.yml'
      - '--storage.path=/alertmanager'
    healthcheck:
      test: ["CMD", "wget", "--no-verbose", "--tries=1", "--spider", "http://localhost:9093/-/healthy"]
      interval: 30s
      timeout: 10s
      retries: 3
    networks:
      - monitoring

  grafana:
    image: grafana/grafana:latest
    container_name: ism-grafana
    restart: unless-stopped
    ports:
      - "3001:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_USERS_ALLOW_SIGN_UP=false
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./grafana/datasources:/etc/grafana/provisioning/datasources
    depends_on:
      - prometheus
    healthcheck:
      test: ["CMD-SHELL", "curl -f http://localhost:3000/api/health || exit 1"]
      interval: 30s
      timeout: 10s
      retries: 3
    networks:
      - monitoring

  node-exporter:
    image: prom/node-exporter:latest
    container_name: ism-node-exporter
    restart: unless-stopped
    ports:
      - "9100:9100"
    volumes:
      - /proc:/host/proc:ro
      - /sys:/host/sys:ro
      - /:/rootfs:ro
    command:
      - '--path.procfs=/host/proc'
      - '--path.sysfs=/host/sys'
      - '--collector.filesystem.ignored-mount-points=^/(sys|proc|dev|host|etc)($$|/)'
    healthcheck:
      test: ["CMD", "wget", "--no-verbose", "--tries=1", "--spider", "http://localhost:9100/metrics"]
      interval: 30s
      timeout: 10s
      retries: 3
    networks:
      - monitoring

volumes:
  prometheus_data:
  alertmanager_data:
  grafana_data:

networks:
  monitoring:
    driver: bridge
"@
    
    Set-Content -Path "$MonitoringDir/docker-compose.monitoring.yml" -Value $composeContent
    Write-MonitoringLog "✅ Docker Compose for monitoring created"
}

# Test monitoring setup
function Test-MonitoringSetup {
    Write-MonitoringLog "Testing monitoring setup..." $Blue
    
    $services = @(
        @{Name = "Prometheus"; Url = "http://localhost:9090"; Endpoint = "/-/healthy"},
        @{Name = "Alertmanager"; Url = "http://localhost:9093"; Endpoint = "/-/healthy"},
        @{Name = "Grafana"; Url = "http://localhost:3001"; Endpoint = "/api/health"},
        @{Name = "Node Exporter"; Url = "http://localhost:9100"; Endpoint = "/metrics"}
    )
    
    foreach ($service in $services) {
        try {
            $response = Invoke-WebRequest -Uri "$($service.Url)$($service.Endpoint)" -TimeoutSec 10 -ErrorAction Stop
            if ($response.StatusCode -eq 200) {
                Write-MonitoringLog "✅ $($service.Name) is healthy" $Green
            }
            else {
                Write-MonitoringLog "⚠️ $($service.Name) returned status $($response.StatusCode)" $Yellow
            }
        }
        catch {
            Write-MonitoringLog "❌ $($service.Name) is not responding: $($_.Exception.Message)" $Red
        }
    }
}

# Install monitoring stack
function Install-MonitoringStack {
    Write-MonitoringLog "Installing monitoring stack..." $Blue
    
    # Check prerequisites
    $prerequisites = Test-MonitoringPrerequisites
    if (!$prerequisites["Docker"] -or !$prerequisites["Docker Compose"]) {
        Write-MonitoringLog "❌ Docker and Docker Compose are required" $Red
        return $false
    }
    
    # Create directory structure
    New-MonitoringStructure
    
    # Install components
    Install-Prometheus
    Install-Grafana
    Set-AlertingConfiguration
    New-MonitoringCompose
    
    # Start monitoring stack
    Write-MonitoringLog "Starting monitoring stack..." $Blue
    Set-Location $MonitoringDir
    docker-compose -f docker-compose.monitoring.yml up -d
    
    # Wait for services to start
    Start-Sleep -Seconds 30
    
    # Test setup
    Test-MonitoringSetup
    
    Write-MonitoringLog "✅ Monitoring stack installed successfully!" $Green
    Write-MonitoringLog "Access points:" $Blue
    Write-MonitoringLog "  - Prometheus: http://localhost:9090" $Blue
    Write-MonitoringLog "  - Grafana: http://localhost:3001 (admin/admin)" $Blue
    Write-MonitoringLog "  - Alertmanager: http://localhost:9093" $Blue
    Write-MonitoringLog "  - Node Exporter: http://localhost:9100" $Blue
    
    return $true
}

# Configure monitoring
function Configure-Monitoring {
    Write-MonitoringLog "Configuring monitoring..." $Blue
    
    # Import dashboard to Grafana
    try {
        $dashboard = Get-Content "$GrafanaDir/dashboards/ism-platform.json" -Raw
        $headers = @{
            "Content-Type" = "application/json"
            "Authorization" = "Basic " + [Convert]::ToBase64String([Text.Encoding]::ASCII.GetBytes("admin:admin"))
        }
        
        $response = Invoke-RestMethod -Uri "http://localhost:3001/api/dashboards/db" -Method POST -Body $dashboard -Headers $headers
        Write-MonitoringLog "✅ Dashboard imported to Grafana" $Green
    }
    catch {
        Write-MonitoringLog "⚠️ Could not import dashboard: $($_.Exception.Message)" $Yellow
    }
    
    # Configure alerting channels
    Write-MonitoringLog "Please configure alerting channels in Alertmanager:" $Blue
    Write-MonitoringLog "  - Edit $AlertingDir/alertmanager.yml" $Blue
    Write-MonitoringLog "  - Add your Slack webhook URL" $Blue
    Write-MonitoringLog "  - Restart Alertmanager: docker-compose restart alertmanager" $Blue
}

# Main function
function Start-MonitoringSetup {
    Write-MonitoringLog "Starting ISM Platform monitoring setup..." $Blue
    
    switch ($Command) {
        "install" {
            Install-MonitoringStack
        }
        "configure" {
            Configure-Monitoring
        }
        "test" {
            Test-MonitoringSetup
        }
        "dashboard" {
            New-ISMDashboard
            Write-MonitoringLog "✅ Dashboard created. Import it to Grafana manually." $Green
        }
        "help" {
            Write-Host "ISM Platform Monitoring Setup Script"
            Write-Host ""
            Write-Host "Usage: .\setup-monitoring.ps1 [command]"
            Write-Host ""
            Write-Host "Commands:"
            Write-Host "  install     Install complete monitoring stack"
            Write-Host "  configure   Configure monitoring after installation"
            Write-Host "  test        Test monitoring services"
            Write-Host "  dashboard   Create dashboard configuration"
            Write-Host "  help        Show this help message"
        }
        default {
            Install-MonitoringStack
        }
    }
}

# Run main function
Start-MonitoringSetup 