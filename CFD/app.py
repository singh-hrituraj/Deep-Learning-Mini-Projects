
from flask import Flask,render_template, request,json
import numpy as np
import random
import json
import h5py

from glob import glob
import matplotlib.pyplot as plt
from skimage import io, color, img_as_float
from skimage.exposure import adjust_gamma
from skimage.segmentation import mark_boundaries
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.metrics import classification_report
from keras.models import Sequential, model_from_json, Model
from keras.layers import concatenate, Input, merge
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Conv2D, UpSampling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten,Reshape
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l1_l2
from keras.optimizers import Adam
from keras.constraints import maxnorm
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import np_utils
from keras.initializers import constant
from keras import backend as K
from keras.layers.advanced_activations import LeakyReLU
from keras.legacy.layers import MaxoutDense
from keras.models import load_model
import progressbar
smooth = 1.
model = load_model('/media/hrituraj/New Volume/BRATS2015_Training/BRATS2015_Training/bm_03-0.86.hdf5')

def predict_image(model, test_img, show = False):
    imgs = io.imread(test_img).astype('float').reshape(5,240,240)
    plist = []
    
    for img in imgs[:-1]:
        if np.max(img) != 0:
            img /= np.max(img)
        p = extract_patches_2d(img, (33,33))
        plist.append(p)
        
    patches = np.array(zip(np.array(plist[0]), np.array(plist[1]), np.array(plist[2]), np.array(plist[3])))
    print("Patches Shape = ")
    print(patches.shape)
    progress = progressbar.ProgressBar(widgets=[progressbar.Bar('*', '[', ']'), progressbar.Percentage(), ' '])
    imgs = []
    y_prob = []
    for i in progress(xrange(len(patches))):
        patch = patches[i]
        patch = np.expand_dims(patch, axis = 0)
        y_prob.append(model.predict(patch))
        #y = np.array(y_prob)
        #y = y_prob.argmax(axis = -1)
        #imgs.append(y)
    y_prob = np.array(y_prob)
    
    f = np.reshape(y_prob, (208,208,5))
    f = np.argmax(f, axis = -1)
    f = f/float(np.max(f))
    io.imsave('static/images/Output.png', f)

    return None
app = Flask(__name__)
 


@app.route("/")
def output():
	return render_template('index.html')


@app.route('/uploadImage', methods=['POST'])
def uploadImage():
    file = request.files['image']

    file.save('static/images/Input.png')
    predict_image(model, 'static/images/Input.png', True)

    return render_template('index2.html')

 
if __name__ == "__main__":
    app.run("")
