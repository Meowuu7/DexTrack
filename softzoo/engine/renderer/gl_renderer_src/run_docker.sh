#!/bin/bash

### Run container and compile
sudo nvidia-docker run \
  -v ${PWD}:/workspace \
  -v $HOME/anaconda3:$HOME/anaconda3 \
  --gpus all \
  -e DISPLAY=$DISPLAY \
  -e QT_X11_NO_MITSHM=1 \
  -it flex:latest bash
