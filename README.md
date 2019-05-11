sudo sed -i "s/MAX_SPEED=.*/MAX_SPEED=1510000/" /etc/default/cpufrequtils && sudo service cpufrequtils restart

Pi
1. sudo nano /etc/dphys-swapfile CONF_SWAPSIZE=2048
2. sudo /etc/init.d/dphys-swapfile stop
3. sudo /etc/init.d/dphys-swapfile start

Debian
1. sudo fallocate -l 2G /swapfile && sudo dd if=/dev/zero of=/swapfile bs=1024 count=2097152 && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile

# coral-pi

For Raspberry Pi 3B+ and Raspbian Lite 2018-11-13 - https://www.raspberrypi.org/downloads/raspbian/

Prep
1. sudo apt-get update -y && sudo apt-get upgrade -y
2. sudo apt-get install -y feh git python3-pip python3-dev python3-numpy libsdl-dev libsdl-image1.2-dev libsdl-mixer1.2-dev libsdl-ttf2.0-dev libsmpeg-dev libportmidi-dev libavformat-dev libswscale-dev libjpeg-dev libfreetype6-dev python3-setuptools && sudo -H pip3 install wheel && sudo -H pip3 install pygame
3. cd ~ && wget https://dl.google.com/coral/edgetpu_api/edgetpu_api_latest.tar.gz -O edgetpu_api.tar.gz --trust-server-names && tar xzf edgetpu_api.tar.gz && cd edgetpu_api && bash ./install.sh
4. Unplug / reinsert TPU
3. cd ~ && mkdir models && cd models && curl -O https://dl.google.com/coral/canned_models/mobilenet_ssd_v2_face_quant_postprocess_edgetpu.tflite && curl -O https://dl.google.com/coral/canned_models/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite
4. cd ~ && git clone https://github.com/spinoza1791/detection.git
5. cd ~/detection && python3 pi-tpu.py --model=/home/libre/models/mobilenet_ssd_v2_face_quant_postprocess_edgetpu.tflite --dims=320
3. Verify python version: python3 --version (must be Python 3.5.x or higher)
4. Install Pi camera v2.1 - https://www.makeuseof.com/tag/set-up-raspberry-pi-camera-module/
5. echo "bcm2835_v4l2" | sudo tee -a /etc/modules >/dev/null
6. Set Pi memory split to 128 - https://www.raspberrypi.org/documentation/configuration/config-txt/memory.md
