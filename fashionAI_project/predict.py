import pandas as pd
from keras.layers import *
from keras.models import *

from dataset import *
import cv2
import numpy as np
from keras.applications.resnet50 import preprocess_input
from collections import Counter

from keras.models import *
from keras.optimizers import *
from keras.applications import *
from keras.regularizers import l2

from keras.preprocessing.image import *
import matplotlib.pyplot as plt
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input



def load_model(label_name):
    df = pd.read_csv('data/Annotations/label.csv', header=None)
    df.columns = ['filename', 'label_name', 'label']
    df = df.sample(frac=1).reset_index(drop=True) # shuffle
    df.label_name = df.label_name.str.replace('_labels', '')

    df = df[df.label_name == label_name]
    c = Counter(df.label_name)
    
    width = 399
    base_model = ResNet50(weights='imagenet', input_shape=(width, width, 3), include_top=False, pooling='avg')
    label_count = dict([(x, len(df[df.label_name == x].label.values[0])) for x in c.keys()])
    label_names = list(label_count.keys())
    
    input_tensor = Input((width, width, 3))
    x = input_tensor
    x = Lambda(preprocess_input)(x)
    x = base_model(x)
    x = Dropout(0.5)(x)
    x = [Dense(count, activation='softmax', name=name)(x) for name, count in label_count.items()]
    new_model = Model(input_tensor, x)
    
    return new_model

def load_model2(label_name):
    df = pd.read_csv('data/Annotations/label.csv', header=None)
    df.columns = ['filename', 'label_name', 'label']
    df.label_name = df.label_name.str.replace('_labels', '')
    df = df[df.label_name == label_name]
    df = df.sample(frac=1).reset_index(drop=True) # shuffle
    c = Counter(df.label_name)
    width = 399
    label_count = dict([(x, len(df[df.label_name == x].label.values[0])) for x in c.keys()])
    label_names = list(label_count.keys())
    base_model = InceptionV3(weights='imagenet', input_shape=(width, width, 3), include_top=False, pooling = 'avg')
    input_tensor = Input((width, width, 3))
    x = input_tensor
    x = Lambda(preprocess_input)(x)
    x = base_model(x)
    x = Dropout(0.5)(x)
    x = [Dense(count, activation='softmax', name=name)(x) for name, count in label_count.items()]
    model = Model(input_tensor, x)
    
    return model
    

def load_photo(path):
    img = cv2.imread(path)
    width = 399
    img = cv2.resize(img, (width, width))
    plt.imshow(img)
    n = 1
    x_test = np.zeros((n, width, width, 3), dtype=np.uint8)
    x_test[0] = img[:,:,::-1]
    return x_test

def predict(path, model_1, model_2, model_3, model_4):
    pred_result = []
    class_str_1 = ['Invisible', 'Short Length', 'Knee Length', 'Midi Length', 'Ankle Length', 'Floor Length']
    class_str_2 = ['Invisible', 'Turtle Neck', 'Ruffle Semi-High Collar', 'Low Turtle Neck', 'Draped Collar']
    class_str_3 = ['Invisible', 'High Waist Length', 'Regular Length', 'Long Length', 'Micro Length', 
                   'Knee Length', 'Midi Length', 'Ankle&Floor Length']
    class_str_4 = ['Invisible', 'Shirt Collar', 'Peter Pan', 'Puritan Collar', 'Rib Collar']
    x_test = load_photo(path)
    if model_1 != None:
        pred_prob = model_1.predict(x_test, batch_size=1, verbose=1)
        pred_class = np.argmax(pred_prob)
        pred_class_str = class_str_1[pred_class]
        pred_result.append(pred_class_str)
    if model_2 != None:
        pred_prob = model_2.predict(x_test, batch_size=1, verbose=1)
        pred_class = np.argmax(pred_prob)
        pred_class_str = class_str_2[pred_class]
        pred_result.append(pred_class_str)
    if model_3 != None:
        pred_prob = model_3.predict(x_test, batch_size=1, verbose=1)
        pred_class = np.argmax(pred_prob)
        pred_class_str = class_str_3[pred_class]
        pred_result.append(pred_class_str)
    if model_4 != None:
        pred_prob = model_4.predict(x_test, batch_size=1, verbose=1)
        pred_class = np.argmax(pred_prob)
        pred_class_str = class_str_4[pred_class]
        pred_result.append(pred_class_str)
    print('The skirt length of the image below is {}'.format(pred_result[0]),'\n',
          'The coat length of the image below is {}'.format(pred_result[2]),'\n',
          'The neck design of the image below is {}'.format(pred_result[1]),'\n',
          'The collar design of the image below is {}'.format(pred_result[3]),'\n'
         )
    return pred_result
        
def predict2(path, model_1, model_2, model_3, model_4):
    pred_result = []
    class_str_1 = ['Invisible', 'Notched', 'Collarless', 'Shawl Collar', 'Plus Size Shawl']
    class_str_2 = ['Invisible', 'Strapless Neck', 'Deep V Neckline', 'Straight Neck', 'V Neckline', 'Square Neckline', 'Off Shoulder', 'Round Neckline', 'Sweat Heart Neck', 'One Shoulder Neckline']
    class_str_3 = ['Invisible', 'Short Pant', 'Mid Length', '3/4 Length', 'Cropped Pant', 'Full Length']
    class_str_4 = ['Invisible', 'Sleeveless', 'Cup Sleeves', 'Short Sleeves', 'Elbow Sleeves', '3/4 Sleeves', 'Wrist Length', 'Long Sleeves', 'Extra Long Sleeves']
    x_test = load_photo(path)
    if model_1 != None:
        pred_prob = model_1.predict(x_test, batch_size=1, verbose=1)
        pred_class = np.argmax(pred_prob)
        pred_class_str = class_str_1[pred_class]
        pred_result.append(pred_class_str)
    if model_2 != None:
        pred_prob = model_2.predict(x_test, batch_size=1, verbose=1)
        pred_class = np.argmax(pred_prob)
        pred_class_str = class_str_2[pred_class]
        pred_result.append(pred_class_str)
    if model_3 != None:
        pred_prob = model_3.predict(x_test, batch_size=1, verbose=1)
        pred_class = np.argmax(pred_prob)
        pred_class_str = class_str_3[pred_class]
        pred_result.append(pred_class_str)
    if model_4 != None:
        pred_prob = model_4.predict(x_test, batch_size=1, verbose=1)
        pred_class = np.argmax(pred_prob)
        pred_class_str = class_str_4[pred_class]
        pred_result.append(pred_class_str)
    print('The lapel design of the image below is {}'.format(pred_result[0]),'\n',
          'The neckline design of the image below is {}'.format(pred_result[1]),'\n',
          'The pant length of the image below is {}'.format(pred_result[2]),'\n',
          'The sleeve length of the image below is {}'.format(pred_result[3]),'\n'
         )
    return pred_result
        