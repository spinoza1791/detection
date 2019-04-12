#!/usr/bin/python
from __future__ import absolute_import, division, print_function, unicode_literals

import pi3d
import numpy as np
import picamera
import picamera.array
import threading
import time
import io
from math import cos, sin, radians
import tkinter

mdl_dims = 320
max_fps = 30

root = tkinter.Tk()
screen_W = root.winfo_screenwidth()
screen_H = root.winfo_screenheight()
preview_W = mdl_dims
preview_H = mdl_dims
preview_mid_X = int(screen_W/2 - preview_W/2)
preview_mid_Y = int(screen_H/2 - preview_H/2)

CAMW, CAMH = mdl_dims, mdl_dims
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
    camera.framerate = max_fps
    #camera.start_preview()
    time.sleep(2)
    camera.capture_sequence(streams(), format='rgb', use_video_port=True)

t = threading.Thread(target=start_capture)
t.start()

while not new_pic:
    time.sleep(0.1)

########################################################################
DISPLAY = pi3d.Display.create(preview_mid_X, preview_mid_Y, w=preview_W, h=preview_H, layer=0, frames_per_second=max_fps)
DISPLAY.set_background(0.0, 0.0, 0.0, 0.0)
shader = pi3d.Shader("uv_reflect")
flatsh = pi3d.Shader('uv_flat')
#========================================
# this is a bit of a one off because the texture has transparent parts
# comment out and google to see why it's included here.
from pi3d import opengles, GL_CULL_FACE
opengles.glDisable(GL_CULL_FACE)
#========================================
# load bump and reflection textures
#bumptex = pi3d.Texture("textures/floor_nm.jpg")
shinetex = pi3d.Texture(npa)
# load model_loadmodel
#mymodel = pi3d.Model(file_string='models/teapot.obj', name='teapot')
#mymodel.set_shader(shader)
#mymodel.set_normal_shine(bumptex, 0.0, shinetex, 0.7)

#mysphere = pi3d.Sphere(radius=400.0, rx=180, ry=180, invert=True)
#mysphere.set_draw_details(flatsh, [shinetex], vmult=3.0, umult=3.0)

# Fetch key presses
mykeys = pi3d.Keyboard()
#mymouse = pi3d.Mouse(restrict=False)
#mymouse.start()

CAMERA = pi3d.Camera.instance()

while DISPLAY.loop_running():
  k = mykeys.read()
  if k >-1:
    if k==27:
      mykeys.close()
      DISPLAY.destroy()
      break

  if new_pic:
    shinetex.update_ndarray(npa)
    new_pic = False

  #mymodel.draw()
  #mysphere.draw()
  #mymodel.rotateIncY(0.41)
  #mymodel.rotateIncZ(0.12)
  #mymodel.rotateIncX(0.23)

# Shut down the processors in an orderly fashion
while pool:
  done = True
  with lock:
    processor = pool.pop()
  processor.terminated = True
  processor.join()
