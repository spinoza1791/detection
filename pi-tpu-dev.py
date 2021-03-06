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
from threading import Thread
import threading

os.environ['SDL_VIDEO_CENTERED'] = '1'

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument(
	  '--model', help='File path of Tflite model.', required=True)
	parser.add_argument(
	  '--labels', help='labels file path OR no arg will prompt for label name', required=False)
	parser.add_argument(
	  '--dims', help='Model input dimension', required=True)
	parser.add_argument(
	  '--max_obj', help='Maximum objects detected [>= 1], default=1', default=1, required=False)
	parser.add_argument(
	  '--thresh', help='Threshold confidence [0.1-1.0], default=0.3', default=0.3, required=False)
	parser.add_argument(
	  '--video_off', help='Video display on/off, for increased FPS', action='store_true', required=False)
	parser.add_argument(
	  '--cam_res_x', help='Set camera X resolution, examples: 96, 128, 256, 352, 384, 480, 640, 1920', default=320, required=False)
	parser.add_argument(
	  '--cam_res_y', help='Set camera Y resolution, examples: 96, 128, 256, 352, 384, 480, 640, 1920', default=320, required=False)
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
		lbl_input = input("Type label name for this single object model:")
		if lbl_input:
			labels = {0: lbl_input}
		else:
			labels = {0: 'object'}
			

	mdl_dims = int(args.dims)
	
	if args.max_obj:
		max_obj = int(args.max_obj)
		if max_obj < 1:
			max_obj = 1

	if args.thresh:
		thresh = float(args.thresh)
		if thresh < 0.1 or thresh > 1.0:
			thresh = 0.3	
	
	video_off = False
	if args.video_off :
		video_off = True
		
	if args.cam_res_x:
		cam_res_x= int(args.cam_res_x)
	else:		
		cam_res_x= 320
	if args.cam_res_y:
		cam_res_y= int(args.cam_res_y)
	else:		
		cam_res_y= 320
		
	#c = threading.Condition()
	#frame_buf_val = None
	
	engine = edgetpu.detection.engine.DetectionEngine(args.model)
	
	pygame.init()
	pygame.camera.init()
	camlist = pygame.camera.list_cameras()
	if camlist:
		pycam = pygame.camera.Camera(camlist[0], (cam_res_x, cam_res_y))
		pycam.start()
		time.sleep(2)
	else:
		print("No camera found!")
		exit
		
	if video_off == False:	
		screen = pygame.display.set_mode((cam_res_x, cam_res_y), pygame.RESIZABLE)
		pygame.display.set_caption('Object Detection')
		screen = pygame.display.get_surface() #get the surface of the current active display
		resized_x,resized_y = size = screen.get_width(), screen.get_height()	
		pygame.font.init()
		fnt_sz = 18
		fnt = pygame.font.SysFont('Arial', fnt_sz)
		lbl_fnt_sz = 10
		lbl_fnt = pygame.font.SysFont('Arial', lbl_fnt_sz)
		x1=x2=y1=y2=0
		
	last_tm = time.time()
	start_ms = time.time()
	elapsed_ms = time.time()
	i = 0
	results = None
	fps = "00.0 fps"
	N = 10
	ms = "00"
			     
	#py_thread = PyThread().start()
	#detection_thread = Detection(args.model).start()
	
	while True:
		start_ms = time.time()	
		#if pycam.query_image():
		img = pycam.get_image()
		#img = pycam.get_image()
		if video_off == False:			
			img = pygame.transform.scale(img,(resized_x, resized_y))	
			screen.blit(img, (0,0))
			
		detect_img = pygame.transform.scale(img,(mdl_dims,mdl_dims))
		img_arr = pygame.surfarray.pixels3d(img)			
		img_arr = np.swapaxes(img_arr,0,1)
		img_arr = np.ascontiguousarray(img_arr)
		frame = io.BytesIO(img_arr)
		frame_buf_val = np.frombuffer(frame.getvalue(), dtype=np.uint8)
		#print(frame_buf_val)
		#start_ms = time.time()
		#results = detection_thread.get_results()
		results = engine.DetectWithInputTensor(frame_buf_val, threshold=thresh, top_k=max_obj)
		#elapsed_ms = time.time() - start_ms
		#pygame.surfarray.blit_array(screen, img_arr)
		if i > 5:
			tm = time.time()
			fps = "fps:{:5.1f} ".format(i / (tm - last_tm))
			i = 0
			last_tm = tm
		i += 1
		if results:
			num_obj = 0
			for obj in results:
				bbox = obj.bounding_box.flatten().tolist()
				label_id = int(round(obj.label_id,1))
				class_label = "%s" % (labels[label_id])
				score = round(obj.score,2)
				class_score = "%.2f" % (score)
				if video_off == False:
					x1 = round(bbox[0] * resized_x) 
					y1 = round(bbox[1] * resized_y) 
					x2 = round(bbox[2] * resized_x) 
					y2 = round(bbox[3] * resized_y) 				
					rect_width = x2 - x1
					rect_height = y2 - y1				
					fnt_class_label = lbl_fnt.render(class_label, True, (255,255,255))
					fnt_class_label_width = fnt_class_label.get_rect().width				
					screen.blit(fnt_class_label,(x1, y1-lbl_fnt_sz))
					fnt_class_score = lbl_fnt.render(class_score, True, (0,255,255))
					fnt_class_score_width = fnt_class_score.get_rect().width
					screen.blit(fnt_class_score,(x2-fnt_class_score_width, y1-lbl_fnt_sz))
					bbox_rect = pygame.draw.rect(screen, (0,255,0), (x1, y1, rect_width, rect_height), 4)
					results_line = "%d, %s, %s, %d,%d,%d,%d" % (num_obj, class_label,class_score,x1,y1,x2,y2)
					print(results_line)
					elapsed_ms = time.time() - start_ms
					if i > N:
						ms = "(%d%s%d) %s%.2fms" % (num_obj, "/", max_obj, "objects detected in ", elapsed_ms*1000)
					fnt_ms = fnt.render(ms, True, (255,255,255))
					fnt_ms_width = fnt_ms.get_rect().width
					screen.blit(fnt_ms,((resized_x / 2 ) - (fnt_ms_width / 2), 0))
				else: #video_off == True
					x1 = round(bbox[0] * mdl_dims) 
					y1 = round(bbox[1] * mdl_dims) 
					x2 = round(bbox[2] * mdl_dims) 
					y2 = round(bbox[3] * mdl_dims)
					results_line = "%s, %d, %s, %s, %d,%d,%d,%d" % (fps,num_obj,class_label,class_score,x1,y1,x2,y2)
					print(results_line)
				num_obj = num_obj + 1
		else:
			if video_off == False:
				elapsed_ms = time.time() - start_ms
				if i > N:
					ms = "%s %.2fms" % ("No objects detected in", elapsed_ms*1000)
				fnt_ms = fnt.render(ms, True, (255,0,0))
				fnt_ms_width = fnt_ms.get_rect().width
				screen.blit(fnt_ms,((resized_x / 2 ) - (fnt_ms_width / 2), 0))
			else:
				print("No results")
	
		if video_off == False:	
			fps_thresh = fps + "    thresh:" + str(thresh)
			fps_fnt = fnt.render(fps_thresh, True, (255,255,0))
			fps_width = fps_fnt.get_rect().width
			screen.blit(fps_fnt,((resized_x / 2) - (fps_width / 2), 20))
			pygame.display.update()

			for event in pygame.event.get():
				keys = pygame.key.get_pressed()
				if(keys[pygame.K_ESCAPE] == 1):
					#pycam.stop()
					#pygame.display.quit()
					pygame.quit()
					sys.exit()
				elif event.type == pygame.VIDEORESIZE:
					screen = pygame.display.set_mode((event.w,event.h),pygame.RESIZABLE)
				

if __name__ == '__main__':
	main()
