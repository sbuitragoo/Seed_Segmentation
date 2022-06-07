#!/bin/bash
cd
cd Desktop
date=$(date)
sudo python3 opencv_capture.py auto --path capturas_$date --step 30m --duration 10d