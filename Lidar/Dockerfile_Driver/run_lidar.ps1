<#
Starts Docker Desktop (if needed), runs the unilidar_sdk2_foxy container inside
the Ubuntu WSL distro with WSLg GUI passthrough, and launches the ROS2 LiDAR
driver + RViz.

Usage:
  .\run_lidar.ps1          # start everything
  .\run_lidar.ps1 -Stop    # stop the container
#>

param(
    [switch]$Stop
)

$ContainerName = "unilidar"
$ImageName     = "unilidar_sdk2_foxy"

<# Path to Docker Desktop executable. Adjust if you installed it somewhere else. #>
$DockerExe     = "C:\Program Files\Docker\Docker\Docker Desktop.exe"

function Test-DockerReady {
    docker info *> $null
    return $LASTEXITCODE -eq 0
}

if ($Stop) {
    Write-Host "Stopping container '$ContainerName'..."
    wsl.exe -d Ubuntu -u root -- docker stop $ContainerName 2>$null
    Write-Host "Done."
    return
}

if (-not (Test-DockerReady)) {
    Write-Host "Starting Docker Desktop..."
    Start-Process -FilePath $DockerExe

    $ready = $false
    for ($i = 0; $i -lt 40; $i++) {
        Start-Sleep -Seconds 5
        if (Test-DockerReady) { $ready = $true; break }
    }
    if (-not $ready) {
        Write-Error "Docker engine did not come up in time. Check Docker Desktop and retry."
        exit 1
    }
}
Write-Host "Docker engine is up."

# Clear out any leftover container from a previous crashed run
wsl.exe -d Ubuntu -u root -- docker rm -f $ContainerName 2>$null | Out-Null

Write-Host "Starting container..."
$runCmd = @"
docker run -d --rm --net=host --privileged \
  --device=/dev/dxg \
  -e DISPLAY=`$DISPLAY \
  -e WAYLAND_DISPLAY=`$WAYLAND_DISPLAY \
  -e XDG_RUNTIME_DIR=`$XDG_RUNTIME_DIR \
  -e PULSE_SERVER=`$PULSE_SERVER \
  -e LD_LIBRARY_PATH=/usr/lib/wsl/lib \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /mnt/wslg:/mnt/wslg \
  -v /usr/lib/wsl:/usr/lib/wsl \
  --name $ContainerName $ImageName sleep infinity
"@
wsl.exe -d Ubuntu -u root -- bash -c $runCmd

Write-Host "Launching LiDAR driver + RViz..."
$launchCmd = "docker exec -d $ContainerName bash -c 'source /opt/ros/foxy/setup.bash && source /root/ros2_ws/install/setup.bash && ros2 launch unitree_lidar_ros2 launch.py > /root/launch.log 2>&1'"
wsl.exe -d Ubuntu -u root -- bash -c $launchCmd

Write-Host "Done. RViz should appear shortly."
Write-Host "Logs:  wsl -d Ubuntu -u root -- docker exec $ContainerName cat /root/launch.log"
Write-Host "Shell: wsl -d Ubuntu -u root -- docker exec -it $ContainerName bash"
Write-Host "Stop:  .\run_lidar.ps1 -Stop"
