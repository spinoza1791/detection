# USAGE
# python detect_video.py --model mobilenet_ssd_v2/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite --labels mobilenet_ssd_v2/coco_labels.txt

# import the necessary packages
from edgetpu.detection.engine import DetectionEngine
from imutils.video import VideoStream
from PIL import Image
import argparse
import imutils
import time
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", default="/home/rock64/models/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite",
	help="path to TensorFlow Lite object detection model")
ap.add_argument("-l", "--labels", default="/home/rock64/detection/coco_labels.txt",
	help="path to labels file")
ap.add_argument("-c", "--confidence", type=float, default=0.3,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

global fps
global detectfps
global framecount
global detectframecount
global time1
global time2
global lastresults
global cam
global window_name

# initialize the labels dictionary
print("[INFO] parsing class labels...")
labels = {}

# loop over the class labels file
for row in open(args["labels"]):
	# unpack the row and update the labels dictionary
	(classID, label) = row.strip().split(maxsplit=1)
	labels[int(classID)] = label.strip()

# load the Google Coral object detection model
print("[INFO] loading Coral model...")
model = DetectionEngine(args["model"])

# initialize the video stream and allow the camera sensor to warmup
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
#vs = VideoStream(usePiCamera=False).start()
time.sleep(2.0)

# loop over the frames from the video stream
while True:
	t1 = time.perf_counter()
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 500 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=500)
	orig = frame.copy()

	# prepare the frame for object detection by converting (1) it
	# from BGR to RGB channel ordering and then (2) from a NumPy
	# array to PIL image format
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	frame = Image.fromarray(frame)

	# make predictions on the input frame
	tinf = time.perf_counter()
	results = model.DetectWithImage(frame, threshold=args["confidence"],
		keep_aspect_ratio=True, relative_coord=False)
	print(time.perf_counter() - tinf, "sec")

	# loop over the results
	for r in results:
		# extract the bounding box and box and predicted class label
		box = r.bounding_box.flatten().astype("int")
		(startX, startY, endX, endY) = box
		label = labels[r.label_id]

		# draw the bounding box and label on the image
		detectframecount += 1
		cv2.rectangle(orig, (startX, startY), (endX, endY),
			(0, 255, 0), 2)
		y = startY - 15 if startY - 15 > 15 else startY + 15
		text = "{}: {:.2f}%".format(label, r.score * 100)
		cv2.putText(orig, text, (startX, y),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

	# show the output frame and wait for a key press
	cv2.imshow("Frame", orig)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
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
	
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
