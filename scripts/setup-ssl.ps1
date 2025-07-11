# SSL Certificate Setup Script for ISM AI Platform
# Automates Let's Encrypt SSL certificate setup and renewal

param(
    [Parameter(Mandatory=$true)]
    [string]$Domain,
    
    [Parameter(Mandatory=$true)]
    [string]$Email,
    
    [Parameter(Position=0)]
    [ValidateSet("install", "renew", "status", "help")]
    [string]$Command = "install"
)

# Configuration
$CertDir = "ssl"
$CertbotDir = "$CertDir/certbot"
$NginxDir = "$CertDir/nginx"
$RenewalScript = "$CertDir/renew-certs.ps1"

# Colors for output
$Red = [System.ConsoleColor]::Red
$Green = [System.ConsoleColor]::Green
$Yellow = [System.ConsoleColor]::Yellow
$Blue = [System.ConsoleColor]::Blue

# Logging functions
function Write-SSLLog {
    param([string]$Message, [System.ConsoleColor]$Color = $Green)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logMessage = "[$timestamp] SSL: $Message"
    Write-Host $logMessage -ForegroundColor $Color
}

# Check prerequisites
function Test-SSLPrerequisites {
    Write-SSLLog "Checking prerequisites..." $Blue
    
    $prerequisites = @{
        "Docker" = $false
        "Docker Compose" = $false
        "Domain DNS" = $false
    }
    
    # Check Docker
    try {
        $dockerVersion = docker --version 2>$null
        if ($dockerVersion) {
            $prerequisites["Docker"] = $true
            Write-SSLLog "✅ Docker is available" $Green
        }
    } catch {
        Write-SSLLog "❌ Docker not found" $Red
    }
    
    # Check Docker Compose
    try {
        $composeVersion = docker-compose --version 2>$null
        if ($composeVersion) {
            $prerequisites["Docker Compose"] = $true
            Write-SSLLog "✅ Docker Compose is available" $Green
        }
    } catch {
        Write-SSLLog "❌ Docker Compose not found" $Red
    }
    
    # Check domain DNS
    try {
        $dnsResult = Resolve-DnsName $Domain -ErrorAction Stop
        if ($dnsResult) {
            $prerequisites["Domain DNS"] = $true
            Write-SSLLog "✅ Domain $Domain resolves to $($dnsResult.IPAddress)" $Green
        }
    } catch {
        Write-SSLLog "❌ Domain $Domain does not resolve" $Red
    }
    
    return $prerequisites
}

# Create SSL directory structure
function New-SSLStructure {
    Write-SSLLog "Creating SSL directory structure..." $Blue
    
    $directories = @(
        $CertDir,
        $CertbotDir,
        $NginxDir,
        "$CertbotDir/conf",
        "$CertbotDir/www",
        "$NginxDir/conf.d"
    )
    
    foreach ($dir in $directories) {
        if (!(Test-Path $dir)) {
            New-Item -ItemType Directory -Path $dir -Force | Out-Null
            Write-SSLLog "Created directory: $dir"
        }
    }
}

# Create certbot configuration
function New-CertbotConfig {
    Write-SSLLog "Creating certbot configuration..." $Blue
    
    $certbotConfig = @"
# Certbot configuration for $Domain
email = $Email
agree-tos = true
no-eff-email = true
webroot-path = /var/www/certbot
domains = $Domain, api.$Domain
"@
    
    Set-Content -Path "$CertbotDir/conf/certbot.conf" -Value $certbotConfig
    Write-SSLLog "✅ Certbot configuration created"
}

# Create Nginx configuration for SSL
function New-NginxSSLConfig {
    Write-SSLLog "Creating Nginx SSL configuration..." $Blue
    
    $nginxConfig = @"
# Nginx configuration for $Domain with SSL
server {
    listen 80;
    server_name $Domain api.$Domain;
    
    # Redirect all HTTP traffic to HTTPS
    location / {
        return 301 https://`$server_name`$request_uri;
    }
    
    # Certbot challenge location
    location /.well-known/acme-challenge/ {
        root /var/www/certbot;
    }
}

server {
    listen 443 ssl http2;
    server_name $Domain;
    
    # SSL Configuration
    ssl_certificate /etc/letsencrypt/live/$Domain/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/$Domain/privkey.pem;
    
    # SSL Security Headers
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 10m;
    
    # Security Headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;
    add_header Content-Security-Policy "default-src 'self'; script-src 'self' 'unsafe-inline' 'unsafe-eval'; style-src 'self' 'unsafe-inline'; img-src 'self' data: https:; font-src 'self' data:; connect-src 'self' https:; frame-ancestors 'self';" always;
    
    # Frontend
    location / {
        proxy_pass http://frontend:3000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade `$http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host `$host;
        proxy_set_header X-Real-IP `$remote_addr;
        proxy_set_header X-Forwarded-For `$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto `$scheme;
        proxy_cache_bypass `$http_upgrade;
    }
    
    # API
    location /api/ {
        proxy_pass http://backend:3001;
        proxy_http_version 1.1;
        proxy_set_header Host `$host;
        proxy_set_header X-Real-IP `$remote_addr;
        proxy_set_header X-Forwarded-For `$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto `$scheme;
        proxy_read_timeout 86400;
    }
}

server {
    listen 443 ssl http2;
    server_name api.$Domain;
    
    # SSL Configuration
    ssl_certificate /etc/letsencrypt/live/$Domain/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/$Domain/privkey.pem;
    
    # SSL Security Headers
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 10m;
    
    # Security Headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;
    
    # API Gateway
    location / {
        proxy_pass http://backend:3001;
        proxy_http_version 1.1;
        proxy_set_header Host `$host;
        proxy_set_header X-Real-IP `$remote_addr;
        proxy_set_header X-Forwarded-For `$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto `$scheme;
        proxy_read_timeout 86400;
    }
    
    # AI Services
    location /ai/ {
        proxy_pass http://ai-services:5000;
        proxy_http_version 1.1;
        proxy_set_header Host `$host;
        proxy_set_header X-Real-IP `$remote_addr;
        proxy_set_header X-Forwarded-For `$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto `$scheme;
        proxy_read_timeout 86400;
    }
}
"@
    
    Set-Content -Path "$NginxDir/conf.d/default.conf" -Value $nginxConfig
    Write-SSLLog "✅ Nginx SSL configuration created"
}

# Create Docker Compose for SSL setup
function New-SSLCompose {
    Write-SSLLog "Creating Docker Compose for SSL setup..." $Blue
    
    $composeContent = @"
version: '3.8'

services:
  nginx:
    image: nginx:alpine
    container_name: ism-nginx-ssl
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/conf.d:/etc/nginx/conf.d
      - ./certbot/conf:/etc/letsencrypt
      - ./certbot/www:/var/www/certbot
    depends_on:
      - certbot
    networks:
      - ssl-network

  certbot:
    image: certbot/certbot
    container_name: ism-certbot
    volumes:
      - ./certbot/conf:/etc/letsencrypt
      - ./certbot/www:/var/www/certbot
    command: certonly --webroot --webroot-path=/var/www/certbot --email $Email --agree-tos --no-eff-email -d $Domain -d api.$Domain
    networks:
      - ssl-network

volumes:
  certbot-conf:
  certbot-www:

networks:
  ssl-network:
    driver: bridge
"@
    
    Set-Content -Path "$CertDir/docker-compose.ssl.yml" -Value $composeContent
    Write-SSLLog "✅ Docker Compose for SSL created"
}

# Create renewal script
function New-RenewalScript {
    Write-SSLLog "Creating certificate renewal script..." $Blue
    
    $renewalScript = @"
# Certificate Renewal Script for $Domain
# Run this script to renew SSL certificates

Write-Host "Renewing SSL certificates for $Domain..." -ForegroundColor Blue

# Stop nginx temporarily
docker-compose -f $CertDir/docker-compose.ssl.yml stop nginx

# Renew certificates
docker-compose -f $CertDir/docker-compose.ssl.yml run --rm certbot renew

# Start nginx
docker-compose -f $CertDir/docker-compose.ssl.yml start nginx

Write-Host "Certificate renewal completed!" -ForegroundColor Green
"@
    
    Set-Content -Path $RenewalScript -Value $renewalScript
    Write-SSLLog "✅ Renewal script created"
}

# Install SSL certificates
function Install-SSLCertificates {
    Write-SSLLog "Installing SSL certificates..." $Blue
    
    # Check prerequisites
    $prerequisites = Test-SSLPrerequisites
    if (!$prerequisites["Docker"] -or !$prerequisites["Docker Compose"] -or !$prerequisites["Domain DNS"]) {
        Write-SSLLog "❌ Prerequisites not met. Please fix the issues above." $Red
        return $false
    }
    
    # Create directory structure
    New-SSLStructure
    
    # Create configurations
    New-CertbotConfig
    New-NginxSSLConfig
    New-SSLCompose
    New-RenewalScript
    
    # Start SSL services
    Write-SSLLog "Starting SSL services..." $Blue
    Set-Location $CertDir
    docker-compose -f docker-compose.ssl.yml up -d
    
    # Wait for certificates to be generated
    Write-SSLLog "Waiting for certificates to be generated..." $Blue
    Start-Sleep -Seconds 60
    
    # Check certificate status
    $certStatus = Test-CertificateStatus
    if ($certStatus) {
        Write-SSLLog "✅ SSL certificates installed successfully!" $Green
        Write-SSLLog "Your site is now available at:" $Blue
        Write-SSLLog "  - https://$Domain" $Blue
        Write-SSLLog "  - https://api.$Domain" $Blue
        Write-SSLLog "" $Blue
        Write-SSLLog "To renew certificates automatically, add to crontab:" $Blue
        Write-SSLLog "  0 12 * * * powershell -ExecutionPolicy Bypass -File `"$((Get-Location).Path)\$RenewalScript`"" $Blue
        return $true
    } else {
        Write-SSLLog "❌ SSL certificate installation failed" $Red
        return $false
    }
}

# Test certificate status
function Test-CertificateStatus {
    Write-SSLLog "Testing certificate status..." $Blue
    
    try {
        $response = Invoke-WebRequest -Uri "https://$Domain" -TimeoutSec 10 -ErrorAction Stop
        if ($response.StatusCode -eq 200) {
            Write-SSLLog "✅ Certificate is valid and working" $Green
            return $true
        }
    } catch {
        Write-SSLLog "❌ Certificate test failed: $($_.Exception.Message)" $Red
        return $false
    }
}

# Renew certificates
function Renew-SSLCertificates {
    Write-SSLLog "Renewing SSL certificates..." $Blue
    
    if (Test-Path $RenewalScript) {
        & $RenewalScript
        Write-SSLLog "✅ Certificate renewal completed" $Green
    } else {
        Write-SSLLog "❌ Renewal script not found" $Red
    }
}

# Main function
function Start-SSLSetup {
    Write-SSLLog "Starting SSL setup for domain: $Domain" $Blue
    
    switch ($Command) {
        "install" {
            Install-SSLCertificates
        }
        "renew" {
            Renew-SSLCertificates
        }
        "status" {
            Test-CertificateStatus
        }
        "help" {
            Write-Host "ISM Platform SSL Setup Script"
            Write-Host ""
            Write-Host "Usage: .\setup-ssl.ps1 -Domain <domain> -Email <email> [command]"
            Write-Host ""
            Write-Host "Commands:"
            Write-Host "  install     Install SSL certificates"
            Write-Host "  renew       Renew existing certificates"
            Write-Host "  status      Check certificate status"
            Write-Host "  help        Show this help message"
            Write-Host ""
            Write-Host "Example:"
            Write-Host "  .\setup-ssl.ps1 -Domain ism.yourdomain.com -Email admin@yourdomain.com install"
        }
        default {
            Install-SSLCertificates
        }
    }
}

# Run main function
Start-SSLSetup 