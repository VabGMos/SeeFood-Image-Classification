#%%
from PIL import Image
from PIL import ImageFilter
import matplotlib.pyplot as plt
import numpy as np 
%matplotlib inline
import glob
import pandas as pd
import time
import random
from sklearn.model_selection import train_test_split

# %%
df = pd.read_csv("food101/meta/meta/classes.txt")
class_list = list(df["apple_pie"])
class_list = ["apple_pie"] + class_list
df = pd.DataFrame(class_list)
df
# %%
base_path = "food101/images/"
image_paths = []
for food in df[0]:
    images = glob.glob(base_path + food + "/*.jpg")
    for i in images:
        image_paths.append(i)
image_paths = np.array(image_paths)
np.random.shuffle(image_paths)
# %%
#use this to get image label from path later
def get_label(p):
    s = p.split('/')
    return s[2]

#%%
#will need to encode labels
label_dict = dict(zip(list(df[0]), df.index))
label_dict
#%%
#probably also good to have other way round
number_label = dict(zip(df.index, list(df[0])))
number_label
# %%
#function to get random batch of images
def get_batch(size):
    paths = random.sample(list(image_paths), size)
    X = [np.asarray(Image.open(p)) for p in paths]
    Y = [label_dict[get_label(p)] for p in paths]
    return X, Y
# %%
#image transforming/ preprocessing, takes in images in the form of arrays for 
# training data
def train_preprocess(image_arrs):
    ims = []
    angles = [30, 45, 60, 90, 120, 135, 150, 180, 210, 225, 240, 270, 300, 315, 330, 360]
    for im in image_arrs:
        i = Image.fromarray(im)
        #first convert all images to one standard size
        i = i.resize((200, 200))
        #randomly flip images horizontally and vertically
        flipper = random.randint(1, 3)
        if flipper == 2:
            i = i.transpose(Image.FLIP_LEFT_RIGHT)
        if flipper == 3:
            i = i.transpose(Image.FLIP_TOP_BOTTOM)
        #now randomly rotate some images
        rotate_bool = random.randint(0, 1)
        if rotate_bool == 1:
            i = i.rotate(random.choice(angles))
        #randomly blur some images
        blur_bool = random.randint(0, 2)
        if blur_bool == 2:
            i = i.filter(ImageFilter.BLUR)
        i = np.asarray(i)
        ims.append(i)
    
    return ims


# %%
#function to normalize all image values, using local centering
def norm_images(image_arrs):
    ims = []
    for i in image_arrs:
        i = i.astype('float64')
        means = i.mean(axis=(0,1), dtype='float64')
        i -= means
        ims.append(i)
    return ims


# %%
#run small test
X, Y = get_batch(10000)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25, random_state = 42)
prepdx = train_preprocess(X)
normdtrain = norm_images(prepdx)
normdtest = norm_images(x_test)
# %%
