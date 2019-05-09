# detection
Pi
sudo nano /etc/dphys-swapfile CONF_SWAPSIZE=2048

sudo /etc/init.d/dphys-swapfile stop

sudo /etc/init.d/dphys-swapfile start

Debian
sudo fallocate -l 2G /swapfile
sudo dd if=/dev/zero of=/swapfile bs=1024 count=2097152
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
