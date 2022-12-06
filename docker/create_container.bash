./docker/reset_xauth.bash

echo ""
echo "Running docker..."
XAUTH=/tmp/.docker.xauth
docker run -it \
      --name=mmyolo \
      --runtime=nvidia \
      --shm-size=8g \
      --ulimit memlock=-1 \
      --net=host \
      --privileged \
      --env="LANG=C.UTF-8" \
      --env="DISPLAY=$DISPLAY" \
      --env="QT_X11_NO_MITSHM=1" \
      --env="XAUTHORITY=$XAUTH" \
      --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
      --volume="$XAUTH:$XAUTH" \
      --volume="$(pwd):/opt/project" \
      --workdir="/opt/project/" \
      stliu2022/mmyolo:master
