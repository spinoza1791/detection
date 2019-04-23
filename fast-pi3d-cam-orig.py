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
import math
import tkinter
import edgetpu.detection.engine
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
  '--model', help='File path of Tflite model.', required=True)
parser.add_argument(
  '--dims', help='Model input dimension', required=True)
args = parser.parse_args()

mdl_dims = int(args.dims) #dims must be a factor of 32/16 for picamera resolution
engine = edgetpu.detection.engine.DetectionEngine(args.model)

root = tkinter.Tk()
screen_W = root.winfo_screenwidth()
screen_H = root.winfo_screenheight()
preview_W = mdl_dims
preview_H = mdl_dims
preview_mid_X = int(screen_W/2 - preview_W/2)
preview_mid_Y = int(screen_H/2 - preview_H/2)

max_obj = 15
max_fps = 60
max_cam = 33

CAMW, CAMH = mdl_dims, mdl_dims
NBYTES = mdl_dims * mdl_dims * 3
npa = np.zeros((mdl_dims, mdl_dims, 4), dtype=np.uint8)
npa[:,:,3] = 255
new_pic = False
empty_results = 0
g_input = None
results = None

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
    global done, npa, new_pic, CAMH, CAMW, NBYTES, bnp, g_input
    while not self.terminated:
      # Wait for an image to be written to the stream
      if self.event.wait(1):
        try:
          if self.stream.tell() >= NBYTES:
            self.stream.seek(0)
            g_input = np.frombuffer(self.stream.getvalue(), dtype=np.uint8)
            bnp = np.array(self.stream.getbuffer(), dtype=np.uint8).reshape(CAMW, CAMH, 3)
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
      time.sleep(0.01)


def start_capture(): # has to be in yet another thread as blocking
  global CAMW, CAMH, pool, camera
  with picamera.PiCamera(resolution=(CAMW, CAMH), framerate=max_cam) as camera:
    pool = [ImageProcessor() for i in range(4)]
    #camera.start_preview(fullscreen=False, layer=0, window=(preview_mid_X, preview_mid_Y, preview_W, preview_H))
    camera.capture_sequence(streams(), format='rgb', use_video_port=True)

t = threading.Thread(target=start_capture)
t.daemon = True
t.start()

while not new_pic:
  time.sleep(0.01)

########################################################################
DISPLAY = pi3d.Display.create(preview_mid_X, preview_mid_Y, w=preview_W, h=preview_H, layer=1, frames_per_second=max_fps)
DISPLAY.set_background(0.0, 0.0, 0.0, 0.0)
txtshader = pi3d.Shader("uv_flat")
linshader = pi3d.Shader('mat_flat')
CAMERA = pi3d.Camera(is_3d=False)

#Use pi3d as the camera preview
tex = pi3d.Texture(npa)
sprite_display = pi3d.Sprite(w=tex.ix, h=tex.iy, z=2.0)
sprite_display.set_draw_details(txtshader, [tex])

font = pi3d.Font("fonts/FreeMono.ttf", font_size=18, color=(255, 255, 255, 255)) # blue green 1.0 alpha
elapsed_ms = 0
ms_str = "000 ms"
ms_txt = pi3d.String(camera=CAMERA, is_3d=False, font=font, string=ms_str, x=0, y=preview_H/2 - 30, z=1.0)
ms_txt.set_shader(txtshader)
ms_total = 0
ms_avg = 0
fps = "00.0 fps"
N = 100
frame_rate_set = True
  
fps_txt = pi3d.String(camera=CAMERA, is_3d=False, font=font, string=fps, x=0, y=preview_H/2 - 10, z=1.0)
fps_txt.set_shader(txtshader)
i = 0
last_tm = time.time()

X_OFF = np.array([0, 0, -1, -1, 0, 0, 1, 1])
Y_OFF = np.array([-1, -1, 0, 0, 1, 1, 0, 0])
X_IX = np.array([0, 1, 1, 1, 1, 0, 0, 0])
Y_IX = np.array([0, 0, 0, 1, 1, 1, 1, 0])
verts = [[0.0, 0.0, 1.0] for i in range(8 * max_obj)] # need a vertex for each end of each side 
bbox = pi3d.Lines(vertices=verts, material=(1.0,0.8,0.05), closed=False, strip=False, line_width=3) 
bbox.set_shader(linshader)

# Fetch key presses
mykeys = pi3d.Keyboard()

while DISPLAY.loop_running():
  k = mykeys.read()
  if k >-1:
    if k==27:
      mykeys.close()
      DISPLAY.destroy()
      break

  ms_total = ms_total + (elapsed_ms*1000)
  i += 1
  if i > N:
    tm = time.time()
    fps = "{:5.1f} fps".format(i / (tm - last_tm))
    fps_txt.quick_change(fps)
    i = 0
    last_tm = tm
    ms_avg = int(ms_total / (N + 1))
    ms_str = "Detection in" + str(ms_avg) + " ms"
    ms_txt.quick_change(ms_str)
    ms_total = 0
    if frame_rate_set and ms_avg > 0:
      max_cam = int(1000 / ms_avg)
      print("Matching framerate to:" +  str(max_cam) + " fps")
      camera.framerate = max_cam
      frame_rate_set = False
    
  if new_pic:
    tex.update_ndarray(npa)
    start_ms = time.time()
    results = engine.DetectWithInputTensor(g_input, top_k=max_obj)
    elapsed_ms = time.time() - start_ms
    if results:
      num_obj = 0
      for obj in results:
        num_obj = num_obj + 1   
        buf = bbox.buf[0] # alias for brevity below
        buf.array_buffer[:,:3] = 0.0;
      for j, obj in enumerate(results):
        coords = (obj.bounding_box - 0.5) * [[1.0, -1.0]] * mdl_dims # broadcasting will fix the arrays size differences
        score = round(obj.score,2)
        ix = 8 * j
        buf.array_buffer[ix:(ix + 8), 0] = coords[X_IX, 0] + 2 * X_OFF
        buf.array_buffer[ix:(ix + 8), 1] = coords[Y_IX, 1] + 2 * Y_OFF
      buf.re_init(); # 
      new_pic = False
  sprite_display.draw()
  bbox.draw()
  fps_txt.draw()   
  ms_txt.draw()

# Shut down the processors in an orderly fashion
while pool:
  done = True
  with lock:
    processor = pool.pop()
  processor.terminated = True
  processor.join()
