Set Host network adapter to 192.168.1.2 (see: https://github.com/unitreerobotics/unilidar_sdk2)

Tested with windows as host!

build:
docker build -t unilidar_sdk2_foxy .

run:
docker run -it --rm \
  --net=host \
  --privileged \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  unilidar_sdk2_foxy
  
 launch:
 ros2 launch unitree_lidar_ros2 launch.py
 
 
 additional terminal:
 docker exec -it 918a7de4b9dd bash
 
 
 docker id:
 docker ps -a
