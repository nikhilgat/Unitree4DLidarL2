<#
Starts the full native LiDAR pipeline -- no Docker, ROS2, or RViz needed:

  sensor --UDP--> Windows adapter (192.168.1.2:6201)
                        |
                  udp_relay.py (Windows, forwards into WSL)
                        |
                  lidar_bridge (WSL Ubuntu, parses via Unitree SDK)
                        |
                  local TCP :9899
                        |
            visualize_lidar.py / save_scan.py  (Windows, your Python env)

Usage:
  .\start_lidar.ps1          # start relay + bridge
  .\start_lidar.ps1 -Stop    # stop both
#>

param(
    [switch]$Stop
)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

if ($Stop) {
    Write-Host "Stopping udp_relay.py..."
    Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
        Where-Object { $_.CommandLine -match "udp_relay\.py" } |
        ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }

    Write-Host "Stopping lidar_bridge..."
    wsl.exe -d Ubuntu -u root -- pkill -f lidar_bridge 2>$null

    Write-Host "Done."
    return
}

# --- lidar_bridge (WSL) ---
$bridgeRunning = wsl.exe -d Ubuntu -u root -- pgrep -f lidar_bridge
if ($bridgeRunning) {
    Write-Host "lidar_bridge already running."
} else {
    Write-Host "Starting lidar_bridge..."
    wsl.exe -d Ubuntu -u root -- bash -c "cd /root/lidar_bridge && nohup ./lidar_bridge > /root/bridge.log 2>&1 & disown; sleep 2"
    $bridgeRunning = wsl.exe -d Ubuntu -u root -- pgrep -f lidar_bridge
    if (-not $bridgeRunning) {
        Write-Error "lidar_bridge failed to start. Check: wsl -d Ubuntu -u root -- cat /root/bridge.log"
        exit 1
    }
}

# --- udp_relay.py (Windows) ---
$relayRunning = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
    Where-Object { $_.CommandLine -match "udp_relay\.py" }
if ($relayRunning) {
    Write-Host "udp_relay.py already running."
} else {
    Write-Host "Starting udp_relay.py..."
    Start-Process -FilePath python -ArgumentList "-u", "$ScriptDir\udp_relay.py" `
        -RedirectStandardOutput "$ScriptDir\relay.log" `
        -RedirectStandardError "$ScriptDir\relay_err.log" `
        -WindowStyle Hidden
    Start-Sleep -Seconds 2
    $relayLog = Get-Content "$ScriptDir\relay.log" -ErrorAction SilentlyContinue
    if (-not ($relayLog -match "Relaying UDP")) {
        Write-Error "udp_relay.py failed to start. Check $ScriptDir\relay_err.log"
        exit 1
    }
}

Write-Host ""
Write-Host "Ready. Now run:"
Write-Host "  python visualize_lidar.py"
Write-Host "  python save_scan.py --name 203"
