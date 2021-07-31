import os
import subprocess
import time
import object_detection
print("LOADING MODEL...")
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util
import cv2 
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('TKagg', force=True)


CUSTOM_MODEL_NAME = 'SignLanguageDetector'
LABEL_MAP_NAME = 'label_map.pbtxt'
paths = {
    'ANNOTATION_PATH': os.path.join('Tensorflow', 'workspace','annotations'),
    'MODEL_PATH': os.path.join('Tensorflow', 'workspace','models'),
    'CHECKPOINT_PATH': os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'checkpoint'), 
    'IMAGE_PATH': 'Images',
    'VID_PATH': 'Videos'
}
files = {
    'PIPELINE_CONFIG':os.path.join('Tensorflow', 'workspace','models', CUSTOM_MODEL_NAME, 'pipeline.config'),
    'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], LABEL_MAP_NAME)
}

# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(paths['CHECKPOINT_PATH'], 'ckpt-0')).expect_partial()

@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

category_index = label_map_util.create_category_index_from_labelmap(files['LABELMAP'])

subprocess.run('cls', shell=True)
subprocess.run('COLOR 2', shell=True)
print("AMERICAN SIGN LANGUAGE DETECTION")
print("--------------------------------")
run = True
while run:
	print("1. Image Detection")
	print("2. Video Detection")
	print("3. Real-time Detection")
	print("4. Exit")
	print("----------------------")
	opt = input("Option: ")
	if opt == '1':
		img_name = input("Enter image name: ")
		IMAGE_PATH = os.path.join(paths['IMAGE_PATH'], img_name)
		img = cv2.imread(IMAGE_PATH)
		image_np = np.array(img)
		print("PLEASE WAIT A MOMENT...")
		input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
		detections = detect_fn(input_tensor)

		num_detections = int(detections.pop('num_detections'))
		detections = {key: value[0, :num_detections].numpy()
		              for key, value in detections.items()}
		detections['num_detections'] = num_detections

		#detection_classes should be ints.
		detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

		label_id_offset = 1
		image_np_with_detections = image_np.copy()

		viz_utils.visualize_boxes_and_labels_on_image_array(
		            image_np_with_detections,
		            detections['detection_boxes'],
		            detections['detection_classes']+label_id_offset,
		            detections['detection_scores'],
		            category_index,
		            use_normalized_coordinates=True,
		            max_boxes_to_draw=1,
		            min_score_thresh=.85,
		            agnostic_mode=False)

		plt.imshow(cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB))
		plt.show()
		
	elif opt == '2':
		vid_name = input("Enter video name: ")
		VID_PATH = os.path.join(paths['VID_PATH'], vid_name)
		cap = cv2.VideoCapture(VID_PATH)
		width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
		height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

		noti = True
		print("PLEASE WAIT A MOMENT...")
		while cap.isOpened(): 
		    ret, frame = cap.read()
		    image_np = np.array(frame)
		    
		    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
		    detections = detect_fn(input_tensor)
		    
		    num_detections = int(detections.pop('num_detections'))
		    detections = {key: value[0, :num_detections].numpy()
		                  for key, value in detections.items()}
		    detections['num_detections'] = num_detections

		    # detection_classes should be ints.
		    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

		    label_id_offset = 1
		    image_np_with_detections = image_np.copy()

		    viz_utils.visualize_boxes_and_labels_on_image_array(
		                image_np_with_detections,
		                detections['detection_boxes'],
		                detections['detection_classes']+label_id_offset,
		                detections['detection_scores'],
		                category_index,
		                use_normalized_coordinates=True,
		                max_boxes_to_draw=2,
		                min_score_thresh=.8,
		                agnostic_mode=False)

		    cv2.imshow('Video Detection',  cv2.resize(image_np_with_detections, (1000, 750)))
		    
		    if cv2.waitKey(1) & 0xFF == ord('q'):
		        cap.release()
		        cv2.destroyAllWindows()
		        break
		    if noti:
		    	print("Detecting...")
		    	print("PRESS Q TO EXIT")
		    	noti = False

	elif opt == '3':
		print("PLEASE WAIT A MOMENT...")
		cap = cv2.VideoCapture(0)
		width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
		height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

		noti = True
		while cap.isOpened(): 
		    ret, frame = cap.read()
		    image_np = np.array(frame)
		    
		    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
		    detections = detect_fn(input_tensor)
		    
		    num_detections = int(detections.pop('num_detections'))
		    detections = {key: value[0, :num_detections].numpy()
		                  for key, value in detections.items()}
		    detections['num_detections'] = num_detections

		    # detection_classes should be ints.
		    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

		    label_id_offset = 1
		    image_np_with_detections = image_np.copy()

		    viz_utils.visualize_boxes_and_labels_on_image_array(
		                image_np_with_detections,
		                detections['detection_boxes'],
		                detections['detection_classes']+label_id_offset,
		                detections['detection_scores'],
		                category_index,
		                use_normalized_coordinates=True,
		                max_boxes_to_draw=1,
		                min_score_thresh=.85,
		                agnostic_mode=False)

		    cv2.imshow('Real-time Detection',  cv2.resize(image_np_with_detections, (1000, 750)))
		    
		    if cv2.waitKey(1) & 0xFF == ord('q'):
		        cap.release()
		        cv2.destroyAllWindows()
		        break
		    if noti:
		    	print("Detecting...")
		    	print("PRESS Q TO EXIT")
		    	noti = False

	else:
		print("Exiting...")
		time.sleep(0.5)
		subprocess.run('cls', shell=True)	
		break

	subprocess.run('cls', shell=True)

subprocess.run('COLOR f', shell=True)