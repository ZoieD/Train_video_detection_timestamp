import numpy as np
import argparse
import cv2 as cv
import subprocess
import time
import os
from yolo_utils import infer_image, show_image, infer_image_video

FLAGS = []

if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument('-m', '--model-path',
		type=str,
		default='./yolov3-coco/',
		help='The directory where the model weights and \
			  configuration files are.')

	parser.add_argument('-w', '--weights',
		type=str,
		default='./yolov3-coco/yolov3.weights',
		help='Path to the file which contains the weights \
			 	for YOLOv3.')

	parser.add_argument('-cfg', '--config',
		type=str,
		default='./yolov3-coco/yolov3.cfg',
		help='Path to the configuration file for the YOLOv3 model.')

	parser.add_argument('-i', '--image-path',
		type=str,
		help='The path to the image file')

	parser.add_argument('-v', '--video-path',
		type=str,
		help='The path to the video file')


	parser.add_argument('-vo', '--video-output-path',
		type=str,
        default='./output.avi',
		help='The path of the output video file')

	parser.add_argument('-l', '--labels',
		type=str,
		default='./yolov3-coco/coco-labels',
		help='Path to the file having the \
					labels in a new-line seperated way.')

	parser.add_argument('-c', '--confidence',
		type=float,
		default=0.5,
		help='The model will reject boundaries which has a \
				probabiity less than the confidence value. \
				default: 0.5')

	parser.add_argument('-th', '--threshold',
		type=float,
		default=0.3,
		help='The threshold to use when applying the \
				Non-Max Suppresion')

	parser.add_argument('--download-model',
		type=bool,
		default=False,
		help='Set to True, if the model weights and configurations \
				are not present on your local machine.')

	parser.add_argument('-t', '--show-time',
						type=bool,
						default=False,
						help='Show the time taken to infer each image.')

	FLAGS, unparsed = parser.parse_known_args()

	# Download the YOLOv3 models if needed
	if FLAGS.download_model:
		subprocess.call(['./yolov3-coco/get_model.sh'])

	# Get the labels
	labels = open(FLAGS.labels).read().strip().split('\n')

	# Intializing colors to represent each label uniquely
	colors = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

	# Load the weights and configutation to form the pretrained YOLOv3 model
	net = cv.dnn.readNetFromDarknet(FLAGS.config, FLAGS.weights)

	# Get the output layer names of the model
	layer_names = net.getLayerNames()
	layer_names = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        
	# If both image and video files are given then raise error
	if FLAGS.image_path is None and FLAGS.video_path is None:
	    print ('Neither path to an image or path to video provided')
	    print ('Starting Inference on Webcam')

	# Do inference with given image
	if FLAGS.image_path:
		# Read the image
		try:
			img = cv.imread(FLAGS.image_path)
			height, width = img.shape[:2]
		except:
			raise 'Image cannot be loaded!\n\
                               Please check the path provided!'

		finally:
			img, _, confidences, _, _ = infer_image(net, layer_names, height, width, img, colors, labels, FLAGS)
			# show_image(img)
			# cv.imshow('img-windows', img)
			# cv.waitKey(0)
			cv.imwrite('01.png', img)
			cv.destroyAllWindows()
			cv.waitKey(1)

	elif FLAGS.video_path:
		# Read the video
		try:
			vid = cv.VideoCapture(FLAGS.video_path)
			# frame_count = int(vid.get(cv.CAP_PROP_FRAME_COUNT))
			fps = vid.get(cv.CAP_PROP_FPS)
		except:
			raise 'Video cannot be loaded!\n\
                               Please check the path provided!'
		finally:
			count = 0
			time_period = []
			while True:
				grabbed, frame = vid.read()
				count+=1
				# print("time stamp current frame:",count/fps)

			    # Checking if the complete video is read
				if not grabbed:
					break

				is_train = infer_image_video(net, layer_names, frame, count, FLAGS)

				if is_train > 0:
					time_period.append(count)
				print("Time Period is: ", time_period, "Index is: ", count)

			ranges = sum((list(t) for t in zip(time_period, time_period[1:]) if all( [t[1] - t[0] > 2*fps] )), [])
			iranges = iter(time_period[0:1] + ranges + time_period[-1:])
			result = ', '.join([str(round(n/fps,2)) + '-' + str(round(next(iranges)/fps,2)) for n in iranges])

			print ("[INFO] Detect Train Time Period:", result)
			text_file = open("output.txt", "w")
			text_file.write("Detect Train Time Period: %s" % result)
			text_file.close()
			# writer.release()
			vid.release()

	else:
		# Infer real-time on webcam
		count = 0

		vid = cv.VideoCapture(0)
		while True:
			_, frame = vid.read()
			height, width = frame.shape[:2]

			if count == 0:
				frame, boxes, confidences, classids, idxs = infer_image(net, layer_names, \
		    						height, width, frame, colors, labels, FLAGS)
				count += 1
			else:
				frame, boxes, confidences, classids, idxs = infer_image(net, layer_names, \
		    						height, width, frame, colors, labels, FLAGS, boxes, confidences, classids, idxs, infer=False)
				count = (count + 1) % 6

			cv.imshow('webcam', frame)

			if cv.waitKey(1) & 0xFF == ord('q'):
				break
		vid.release()
		cv.destroyAllWindows()
