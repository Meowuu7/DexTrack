#!/bin/bash

### Build image
sudo chmod 666 /var/run/docker.sock
sudo docker build -t flex .