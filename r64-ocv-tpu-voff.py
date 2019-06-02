import argparse
import platform
import numpy as np
import cv2
import time
from PIL import Image
from time import sleep
import multiprocessing as mp
from edgetpu.detection.engine import DetectionEngine

lastresults = None
processes = []
frameBuffer = None
results = None
fps = ""
detectfps = ""
framecount = 0
detectframecount = 0
time1 = 0
time2 = 0
box_color = (255, 128, 0)
box_thickness = 1
label_background_color = (125, 175, 75)
label_text_color = (255, 255, 255)
percentage = 0.0

# Function to read labels from text files.
def ReadLabelFile(file_path):
  with open(file_path, 'r') as f:
    lines = f.readlines()
  ret = {}
  for line in lines:
    pair = line.strip().split(maxsplit=1)
    ret[int(pair[0])] = pair[1].strip()
  return ret


def camThread(label, results, frameBuffer, camera_width, camera_height, vidfps, cam_num):

    global fps
    global detectfps
    global framecount
    global detectframecount
    global time1
    global time2
    global lastresults
    global cam
    global window_name

    cam = cv2.VideoCapture(cam_num)
    cam.set(cv2.CAP_PROP_FPS, vidfps)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)

    while True:
        t1 = time.perf_counter()

        ret, color_image = cam.read()
        if not ret:
            continue
        if frameBuffer.full():
            frameBuffer.get()
        frames = color_image
        frameBuffer.put(color_image.copy())
        res = None

        if not results.empty():
            res = results.get(False)
            detectframecount += 1
            lastresults = res

        if cv2.waitKey(1)&0xFF == ord('q'):
            break

        # FPS calculation
        framecount += 1
        if framecount >= 15:
            fps       = "(Playback) {:.1f} FPS".format(time1/15)
            detectfps = "(Detection) {:.1f} FPS".format(detectframecount/time2)
            framecount = 0
            detectframecount = 0
            time1 = 0
            time2 = 0
            print("Playback FPS: " + fps + "Detection FPS: " + detectfps)
        t2 = time.perf_counter()
        elapsedTime = t2-t1
        time1 += 1/elapsedTime
        time2 += elapsedTime

def inferencer(results, frameBuffer, model, camera_width, camera_height):
    engine = DetectionEngine(model)
    while True:
        if frameBuffer.empty():
            continue
        # Run inference.
        color_image = frameBuffer.get()
        prepimg = color_image[:, :, ::-1].copy()
        prepimg = Image.fromarray(prepimg)
        tinf = time.perf_counter()
        ans = engine.DetectWithImage(prepimg, threshold=0.3, keep_aspect_ratio=True, relative_coord=False, top_k=10)
        #print(time.perf_counter() - tinf, "sec")
        results.put(ans)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/home/rock64/models/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite", help="Path of the detection model.")
    parser.add_argument("--label", default="/home/rock64/detection/coco_labels.txt", help="Path of the labels file.")
    parser.add_argument("--cam", type=int, default=0, help="Camera number, ex. 0")
    parser.add_argument("--cam_w", type=int, default=320, help="Camera width")
    parser.add_argument("--cam_h", type=int, default=240, help="Camera height")
    parser.add_argument('--video_off', help='Video display off, for increased FPS', action='store_true', default=False)
    args = parser.parse_args()

    model = args.model
    label = ReadLabelFile(args.label)
    cam_arg = args.cam
    video_off = args.video_off
    camera_width = args.cam_w
    camera_height = args.cam_h
    vidfps = 120

    try:
        mp.set_start_method('forkserver')
        frameBuffer = mp.Queue(10)
        results = mp.Queue()

        # Start streaming
        p = mp.Process(target=camThread,
                       args=(label, results, frameBuffer, camera_width, camera_height, vidfps, cam_arg),
                       daemon=True)
        p.start()
        processes.append(p)

        # Activation of inferencer
        p = mp.Process(target=inferencer,
                       args=(results, frameBuffer, model, camera_width, camera_height),
                       daemon=True)
        p.start()
        processes.append(p)

        while True:
            sleep(1)

    finally:
        for p in range(len(processes)):
            processes[p].terminate()
