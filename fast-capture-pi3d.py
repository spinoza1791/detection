#!/usr/bin/python
from __future__ import absolute_import, division, print_function, unicode_literals

import demo
import pi3d
import numpy as np
import picamera
import picamera.array
import threading
import time
import io
from math import cos, sin, radians

CAMW, CAMH = 320, 320
NBYTES = CAMW * CAMH * 3
npa = np.zeros((CAMH, CAMW, 4), dtype=np.uint8)
npa[:,:,3] = 255
new_pic = False

# Create a pool of image processors
done = False
lock = threading.Lock()
pool = []

class ImageProcessor(threading.Thread):
    def __init__(self):
        super(ImageProcessor, self).__init__()
        self.stream = io.BytesIO()
        self.event = threading.Event()
        self.terminated = False
        self.start()

    def run(self):
        # This method runs in a separate thread
        global done, npa, new_pic, CAMH, CAMW, NBYTES
        while not self.terminated:
            # Wait for an image to be written to the stream
            if self.event.wait(1):
                try:
                    if self.stream.tell() >= NBYTES:
                      self.stream.seek(0)
                      # python2 doesn't have the getbuffer() method
                      #bnp = np.fromstring(self.stream.read(NBYTES),
                      #              dtype=np.uint8).reshape(CAMH, CAMW, 3)
                      bnp = np.array(self.stream.getbuffer(),
                                    dtype=np.uint8).reshape(CAMH, CAMW, 3)
                      npa[:,:,0:3] = bnp
                      new_pic = True
                except Exception as e:
                  print(e)
                finally:
                    # Reset the stream and event
                    self.stream.seek(0)
                    self.stream.truncate()
                    self.event.clear()
                    # Return ourselves to the pool
                    with lock:
                        pool.append(self)

def streams():
    while not done:
        with lock:
            if pool:
                processor = pool.pop()
            else:
                processor = None
        if processor:
            yield processor.stream
            processor.event.set()
        else:
            # When the pool is starved, wait a while for it to refill
            time.sleep(0.1)


def start_capture(): # has to be in yet another thread as blocking
  global CAMW, CAMH, pool
  with picamera.PiCamera() as camera:
    pool = [ImageProcessor() for i in range(3)]
    camera.resolution = (CAMW, CAMH)
    camera.framerate = 30
    #camera.start_preview()
    time.sleep(2)
    camera.capture_sequence(streams(), format='rgb', use_video_port=True)

t = threading.Thread(target=start_capture)
t.start()

while not new_pic:
    time.sleep(0.1)

########################################################################
DISPLAY = pi3d.Display.create(x=320, y=320, frames_per_second=30)
DISPLAY.set_background(0.0, 0.0, 0.0, 0.0)
shader = pi3d.Shader("uv_reflect")
flatsh = pi3d.Shader('uv_flat')
#========================================
# this is a bit of a one off because the texture has transparent parts
# comment out and google to see why it's included here.
from pi3d import opengles, GL_CULL_FACE
opengles.glDisable(GL_CULL_FACE)
#========================================

# Fetch key presses
mykeys = pi3d.Keyboard()
#CAMERA = pi3d.Camera.instance()
CAMERA = pi3d.Camera(is_3d=False)

while DISPLAY.loop_running():
  k = mykeys.read()
  if k==27:
      mykeys.close()
      DISPLAY.destroy()
      break

  if new_pic:
    #shinetex.update_ndarray(npa)
    new_pic = False

# Shut down the processors in an orderly fashion
while pool:
  done = True
  with lock:
    processor = pool.pop()
  processor.terminated = True
  processor.join()
