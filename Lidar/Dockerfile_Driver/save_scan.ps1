<#
Captures a point cloud from the running LiDAR container and saves it as a
.pcd file directly into the SMCODE data/scans folder.

Requires the lidar container to already be running (run_lidar.ps1 first).

Usage:
  .\save_scan.ps1 -Name 203              # capture for 10 seconds (default)
  .\save_scan.ps1 -Name 203 -Seconds 20  # capture for 20 seconds
#>

param(
    [Parameter(Mandatory = $true)][string]$Name,
    [int]$Seconds = 10
)

$ContainerName = "unilidar"
<# Path to the scans folder. Adjust if you installed somewhere else. #>
$DestDir       = "..\data\scans"

$runningNames = (wsl.exe -d Ubuntu -u root -- docker ps --format "{{.Names}}" 2>$null) -join "`n"
if ($runningNames -notmatch [regex]::Escape($ContainerName)) {
    Write-Error "Container '$ContainerName' is not running. Start it first: Dockerfile_Driver\run_lidar.ps1"
    exit 1
}

Write-Host "Capturing point cloud for $Seconds seconds..."
$captureCmd = @"
mkdir -p /root/scans && \
source /opt/ros/foxy/setup.bash && \
source /root/ros2_ws/install/setup.bash && \
timeout $Seconds ros2 run pcl_ros pointcloud_to_pcd --ros-args -r input:=/unilidar/cloud -p prefix:=/root/scans/${Name}_
"@
wsl.exe -d Ubuntu -u root -- docker exec $ContainerName bash -c $captureCmd
if ($LASTEXITCODE -ne 0) {
    Write-Error "Capture command failed (exit $LASTEXITCODE). Is the driver publishing /unilidar/cloud? Check: wsl -d Ubuntu -u root -- docker exec $ContainerName cat /root/launch.log"
    exit 1
}

Write-Host "Finding latest capture..."
$latestLines = wsl.exe -d Ubuntu -u root -- docker exec $ContainerName bash -c "ls -t /root/scans/${Name}_*.pcd 2>/dev/null | head -1"
$latest = ($latestLines -join "").Trim()

if (-not $latest) {
    Write-Error "No point cloud captured. Is the driver actually publishing /unilidar/cloud? Check: wsl -d Ubuntu -u root -- docker exec $ContainerName cat /root/launch.log"
    exit 1
}
Write-Host "Captured: $latest"

if (-not (Test-Path $DestDir)) {
    New-Item -ItemType Directory -Path $DestDir -Force | Out-Null
}

$destFile = "$Name.pcd"
Write-Host "Copying to SMCODE scans folder as $destFile..."
wsl.exe -d Ubuntu -u root -- docker cp "${ContainerName}:$latest" "/mnt/c/Users/nikhi/Documents/SMART-MARK/SMCODE/data/scans/$destFile"

Write-Host "Done: $DestDir\$destFile"
Write-Host "Note: this is one accumulated frame (~18 sub-scans internally). Open it in CloudCompare if you want to export as .ply or merge with other captures for denser coverage."
