import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2 
import os, glob
import tqdm.notebook as tq

PROJECT_DIR = "/content/drive/Shareddrives/DS 303 Course Project/"

import dlib
from PIL import Image
from skimage import io
import matplotlib.pyplot as plt
from skimage.color import rgb2gray

face_detector = dlib.get_frontal_face_detector()

def detect_faces(path, dace_detector=face_detector):
	"""
	Detects the face in the image at the specified path and returns a cropped image
	containing just the face.

	Args:
		path (str): path to image

	Returns:
		np.ndarray: cropped image with just the detected face
	"""
	image = io.imread(path)
	detected_faces = face_detector(image, 1)
	face_frames = [(x.left(), x.top(),
					x.right(), x.bottom()) for x in detected_faces]
	
	for n, face_rect in enumerate(face_frames):
		face = Image.fromarray(image).crop(face_rect)
		face=face.convert('L')
		face = np.array(face)
		face = cv2.resize(face, (96,96), interpolation=cv2.INTER_NEAREST)
	
	return face

face_mapping = {i: j for i,j in zip(range(10000),os.listdir('/content/lfw'))}
face_inverse = {j: i for i,j in zip(range(10000),os.listdir('/content/lfw'))}
num_faces = [[x[0].split('/')[-1],len(x[2])] for x in os.walk('/content/lfw')][1:]
multiple_faces = [n for n,num in enumerate(num_faces) if num[1]>=2]
single_faces = [n for n,num in enumerate(num_faces) if num[1]==1]
all_image_paths = glob.glob('/content/lfw/*/*.jpg')



def get_image_paths_from_ids(ids):
	"""
	Gets the paths of all images for all people with ids in `ids` and store it
	in a dictionary.

	Args:
		ids (array like): list or array containing ids for which image paths need to 
		be found

	Returns:
		dict: dictionary with the same keys as `ids`. Each key contains a list with 
		paths to images.
	"""
	out_dict = dict()
	for idx in ids:
		name = face_mapping[idx]
		dir = '/content/lfw/'+name+'/'
		images = os.listdir(dir)
		paths = [dir + image for image in images]
		out_dict[idx] = paths
	return out_dict

from sklearn.model_selection import train_test_split

# np.random.seed(420)
# np.random.shuffle(all_image_paths)

# train_, test_ = train_test_split(all_image_paths,test_size=0.2,random_state=69)

def get_dict_from_ids(ids):
    out_dict = dict()
    for path in ids:
        name = path.split('/')[3]
        id = face_inverse[name]
        out_dict[id] = []

    for path in ids:
        name = path.split('/')[3]
        id = face_inverse[name]
        out_dict[id].append(path)
    return out_dict

# train_dict = get_dict_from_ids(train_)
# test_dict = get_dict_from_ids(test_)

def separate_multi(dict_):
    out_multi_dict = dict_.copy()
    keys = list(out_multi_dict.keys())

    for key in keys:
        if len(out_multi_dict[key]) < 2:
            out_multi_dict.pop(key)
    return out_multi_dict

# train_multi_dict = separate_multi(train_dict)

def get_points_from_dict(path_dict, learner):
	"""
	Creates a dictionary of points predicted using the facial points detector model.

	Args:
		path_dict (dict): dictionary containing paths of images under the respective 
		id of the person
		learner (fastai `cnn_learner` model): model which has been trained to predict
		key facial features of the face supplied to it

	Returns:
		dict: dictionary of points predicted using the input model, one list corresponding 
		to each path in the input `path_dict`.
	"""
	keys = path_dict.keys()
	pts_dict = dict()
	for key in tq.tqdm(keys):
		one_dudes_images = path_dict[key]
		one_dudes_points = []
		for image in one_dudes_images:
			image = detect_faces(image)
			assert image.shape == (96,96)
			pts = learner.predict(image)[0]
			pts_req = pts[:-2]
			one_dudes_points.append(pts_req)
		pts_dict[key] = one_dudes_points
		
	return pts_dict