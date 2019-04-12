#!/usr/bin/python
from __future__ import absolute_import, division, print_function, unicode_literals

""" Example showing what can be left out. ESC to quit"""
#import demo
import pi3d
import numpy as np
import picamera
import picamera.array
import threading
import time
import tkinter

npa = None # this is the array for the camera to fill
new_pic = False # this is the flag to signal when array refilled

max_fps = 60
mdl_dims = 320
start_ms = 1000
elapsed_ms = 1000

root = tkinter.Tk()
screen_W = root.winfo_screenwidth()
screen_H = root.winfo_screenheight()
preview_W = mdl_dims
preview_H = mdl_dims
preview_mid_X = int(screen_W/2 - preview_W/2)
preview_mid_Y = int(screen_H/2 - preview_H/2)

DISPLAY = pi3d.Display.create(x=mdl_dims, y=mdl_dims, frames_per_second=max_fps)
DISPLAY.set_background(0.0, 0.0, 0.0, 0.0)
txtshader = pi3d.Shader("uv_flat")
font = pi3d.Font("fonts/FreeMono.ttf", font_size=30, color=(0, 255, 0, 255)) # blue green 1.0 alpha
CAMERA = pi3d.Camera(is_3d=False)

fps = "00.0 fps"
N = 10
fps_txt = pi3d.String(camera=CAMERA, is_3d=False, font=font, string=fps, x=0, y=preview_H/2 - 10, z=1.0)
fps_txt.set_shader(txtshader)
i = 0
last_tm = time.time()
ms = str(elapsed_ms)
ms_txt = pi3d.String(camera=CAMERA, is_3d=False, font=font, string=ms, x=0, y=preview_H/2 - 30, z=1.0)
ms_txt.set_shader(txtshader)

def ms_display(elapsed_ms):
  global ms_txt
  ms = str(elapsed_ms*1000)
  ms_txt.draw()
  ms_txt.quick_change(ms)

def get_pics():
  # function to run in thread
  global npa, new_pic
  with picamera.PiCamera() as camera:
    camera.resolution = (mdl_dims, mdl_dims)
    with picamera.array.PiRGBArray(camera) as output:
      while True: # loop for ever
        output.truncate(0)
        start_ms = time.time()
        camera.capture(output, format='rgb', use_video_port=True)
        elapsed_ms = time.time() - start_ms
        ms_display(elapsed_ms)
        if npa is None: # do this once only
          npa = np.zeros(output.array.shape[:2] + (4,), dtype=np.uint8)
          npa[:,:,3] = 255 # fill alpha value
        npa[:,:,0:3] = output.array # copy in rgb bytes
        new_pic = True
        time.sleep(0.05)

##########################################################################
t = threading.Thread(target=get_pics) # set up and start capture thread
t.daemon = True
t.start()

while not new_pic: # wait for array to be filled first time
    time.sleep(0.1)

########################################################################
tex = pi3d.Texture(npa)
sprite = pi3d.Sprite(w=tex.ix, h=tex.iy, z=5.0)
sprite.set_draw_details(txtshader, [tex])
mykeys = pi3d.Keyboard()

while DISPLAY.loop_running():
  if new_pic:
    tex.update_ndarray(npa)
    new_pic = False
  sprite.draw()
  fps_txt.draw()
  i += 1
  if i > N:
    tm = time.time()
    fps = "{:5.1f}FPS".format(i / (tm - last_tm))
    fps_txt.quick_change(fps)
    i = 0
    last_tm = tm


  if mykeys.read() == 27:
    mykeys.close()
    DISPLAY.destroy()
    break
