"""Edge TPU Face detection with bounding boxes, labels and scores via Pygame stream - AUTHOR: Andrew Craton 03/2019"""

import argparse
import io
import time
import sys
import pygame
import pygame.camera
import numpy as np
import edgetpu.detection.engine
import os

os.environ['SDL_VIDEO_CENTERED'] = '1'

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument(
	  '--model', help='File path of Tflite model.', default="/home/rock64/models/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite")
	parser.add_argument(
	  '--labels', help="Path of the labels file.", default="/home/rock64/detection/coco_labels.txt")
	parser.add_argument(
	  '--dims', help='Model input dimension', default=320)
	parser.add_argument(
	  '--max_obj', help='Maximum objects detected [>= 1], default=1', default=1)
	parser.add_argument(
	  '--thresh', help='Threshold confidence [0.1-1.0], default=0.3', default=0.3)
	parser.add_argument(
	  '--video_off', help='Video display off, for increased FPS', action='store_true', default=False)
	parser.add_argument(
	  '--cam_w', help='Set camera resolution, examples: 96, 128, 256, 352, 384, 480, 640, 1920', default=320)
	parser.add_argument(
	  '--cam_h', help='Set camera resolution, examples: 96, 128, 256, 352, 384, 480, 640, 1920', default=320)
	if len(sys.argv[0:])==0:
		parser.print_help()
		#parser.print_usage() # for just the usage line
		parser.exit()
	args = parser.parse_args()
	
	if args.labels:
		with open(args.labels, 'r') as f:
			pairs = (l.strip().split(maxsplit=1) for l in f.readlines())
			labels = dict((int(k), v) for k, v in pairs)
	else:
		lbl_input = input("Type label name for this single class model:")
		if lbl_input:
			labels = {0: lbl_input}
		else:
			labels = {0: 'object'}
			
	mdl_dims = int(args.dims)
	max_obj = int(args.max_obj)
	cam_w= int(args.cam_w)
	cam_h= int(args.cam_h)
	thresh = float(args.thresh)
	video_off = args.video_off
	engine = edgetpu.detection.engine.DetectionEngine(args.model)

	pygame.init()
	pygame.camera.init()
	if not video_off :
		screen = pygame.display.set_mode((cam_w,cam_w), pygame.DOUBLEBUF | pygame.HWSURFACE)
		pygame.display.set_caption('Object Detection')
		pygame.font.init()
		fnt_sz = 18
		fnt = pygame.font.SysFont('Arial', fnt_sz)
	camlist = pygame.camera.list_cameras()
	if camlist:
	    pycam = pygame.camera.Camera(camlist[0],(cam_w,cam_h))
	else:
		print("No camera found!")
		exit
	pycam.start() 
	time.sleep(1)
	x1=x2=y1=y2=i=j=fps_last=fps_total=0
	last_tm = time.time()
	start_ms = time.time()
	elapsed_ms = time.time()
	results = None
	fps_avg = "00.0"
	N = 10
	ms = "00"
	if not video_off :
		screen = pygame.display.get_surface() #get the surface of the current active display
		resized_x,resized_y = screen.get_width(), screen.get_height()
	img = pycam.get_image()
	img = pygame.transform.scale(img,(mdl_dims,mdl_dims))
	
	while True:
		if not video_off :
			screen = pygame.display.get_surface() #get the surface of the current active display
			resized_x,resized_y = screen.get_width(), screen.get_height()
		if pycam.query_image():
			img = pycam.get_image()
		if not video_off:
			img = pygame.transform.scale(img,(resized_x, resized_y))
			screen.blit(img, (0,0))
			detect_img = pygame.transform.scale(img,(mdl_dims,mdl_dims))
		img_arr = pygame.surfarray.pixels3d(detect_img)
		img_arr = np.swapaxes(img_arr,0,1)
		img_arr = np.ascontiguousarray(img_arr)
		frame = io.BytesIO(img_arr)
		frame_buf_val = np.frombuffer(frame.getvalue(), dtype=np.uint8)
		start_ms = time.time()
		results = engine.DetectWithInputTensor(frame_buf_val, threshold=thresh, top_k=max_obj)
		elapsed_ms = time.time() - start_ms
		i += 1
		if results:
			obj_cnt = 0
			obj_id = 0
			for obj in results:
				obj_cnt += 1
			for obj in results:
				obj_id += 1
				bbox = obj.bounding_box.flatten().tolist()
				label_id = int(round(obj.label_id,1))
				class_label = "%s" % (labels[label_id])
				if not video_off:
					fnt_class_label = fnt.render(class_label, True, (255,255,255))
					fnt_class_label_width = fnt_class_label.get_rect().width
					screen.blit(fnt_class_label,(x1, y1-fnt_sz))
				score = round(obj.score,2)
				x1 = round(bbox[0] * resized_x) 
				y1 = round(bbox[1] * resized_y) 
				x2 = round(bbox[2] * resized_x) 
				y2 = round(bbox[3] * resized_y) 
				rect_width = x2 - x1
				rect_height = y2 - y1
				class_score = "%.2f" % (score)
				if not video_off:
					fnt_class_score = fnt.render(class_score, True, (0,255,255))
					fnt_class_score_width = fnt_class_score.get_rect().width
					screen.blit(fnt_class_score,(x2-fnt_class_score_width, y1-fnt_sz))
				if i > N:
					ms = "(%d%s%d) %s%.2fms" % (obj_cnt, "/", max_obj, "objects detected in ", elapsed_ms*1000)
					print(ms)
				if not video_off:
					fnt_ms = fnt.render(ms, True, (255,255,255))
					fnt_ms_width = fnt_ms.get_rect().width
					screen.blit(fnt_ms,((resized_x / 2 ) - (fnt_ms_width / 2), 0))
					bbox_rect = pygame.draw.rect(screen, (0,255,0), (x1, y1, rect_width, rect_height), 4)
				output = "%s%d %s%s %s%s %s%d %s%d %s%d %s%d %s" % ("id:",obj_id,"class:", class_label, "conf:", class_score, "x1:",x1, "y1:",y1, "x2:",x2,"y2:", y2, fps_avg)
				print(output)
		else:
			if i > N:
				ms = "%s %.2fms %s" % ("No objects detected in", elapsed_ms*1000, fps_avg)
				print(ms)
			if not video_off:
				fnt_ms = fnt.render(ms, True, (255,0,0))
				fnt_ms_width = fnt_ms.get_rect().width
				screen.blit(fnt_ms,((resized_x / 2 ) - (fnt_ms_width / 2), 0))
				
		if i > N:
			tm = time.time()
			fps_last = i / (tm - last_tm)
			if j < 5:
				j += 1
				fps_total = fps_total + fps_last
			else:
				fps_avg = "AVG_FPS:{:5.1f} ".format(fps_total / 5)
				fps_total = 0
				j = 0
			i = 0
			last_tm = tm

		if not video_off:
			fps_thresh = fps_avg + "    thresh:" + str(thresh)
			fps_fnt = fnt.render(fps_thresh, True, (255,255,0))
			fps_width = fps_fnt.get_rect().width
			screen.blit(fps_fnt,((resized_x / 2) - (fps_width / 2), 20))

		for event in pygame.event.get():
			keys = pygame.key.get_pressed()
			if(keys[pygame.K_ESCAPE] == 1):
				pycam.stop()
				pygame.display.quit()
				sys.exit()
			#elif event.type == pygame.VIDEORESIZE and not video_off:
			#	screen = pygame.display.set_mode((event.w,event.h),pygame.RESIZABLE)
		
		if not video_off:
			pygame.display.update()
				

if __name__ == '__main__':
	main()
