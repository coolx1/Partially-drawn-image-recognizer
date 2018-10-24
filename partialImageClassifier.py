from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dropout
from keras.models import model_from_json
import cv2
import numpy as np
import segmentation as seg
import keras
from keras.preprocessing import image as i

def preprocess(s):
    array = seg.segment(s)
    if(len(array)<1):
        keras.backend.clear_session()
        return "Not segmentable!"
    image = array[0]
    image = i.img_to_array(image)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    image = np.expand_dims(image, axis=0)
    return image

json_file = open('model1.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model1.h5")

# model_json = loaded_model.to_json()
# with open("model1.json", "w") as json_file:
#     json_file.write(model_json)
#     # serialize weights to HDF5
# loaded_model.save_weights("model1.h5")
# print("Saved model to disk")
referencelst = ["cards","combobox", "tabview", "textbox", "window_header", "enable_disable","forward","home", "information_icon",
                "zoom", "checkbox"]

reference = {"cards": 0,
             "combobox": 1,
             "tabiew": 2,
             "textbox": 3,
             "window_header": 4,
             "enable_disable": 5,
             "forward": 6,
             "home": 7,
             "informaton_icon": 8,
             "zoom": 9,
             "checkbox": 10}
#
# filename = "card.png"
# image = preprocess(filename)
# result= loaded_model.predict(image)
# a = result[0]
#
# filename = "comb.png"
# image = preprocess(filename)
# result= loaded_model.predict(image)
# b = result[0]
#
# filename = "tabv.png"
# image = preprocess(filename)
# result= loaded_model.predict(image)
# c = result[0]
#
# filename = "texb.png"
# image = preprocess(filename)
# result= loaded_model.predict(image)
# d = result[0]
#
# filename = "winh.png"
# image = preprocess(filename)
# result= loaded_model.predict(image)
# e = result[0]
#
# filename = "enad.png"
# image = preprocess(filename)
# result= loaded_model.predict(image)
# f = result[0]
#
# filename = "forw.png"
# image = preprocess(filename)
# result= loaded_model.predict(image)
# g = result[0]
#
# filename = "home.png"
# image = preprocess(filename)
# result= loaded_model.predict(image)
# h = result[0]
#
# filename = "infi.png"
# image = preprocess(filename)
# result= loaded_model.predict(image)
# v = result[0]
#
# filename = "zoom.png"
# image = preprocess(filename)
# result= loaded_model.predict(image)
# j = result[0]

# filename = "cheb.png"
# image = preprocess(filename)
# result= loaded_model.predict(image)
# k = result[0]

# baseDiagrams = np.array([a,b,c,d,e,f,g,h,v,j,k])
# np.save("baseDiagrams.npy", baseDiagrams)
# print("Saved!")
def probability(lst):
    s = sum(lst)
    return [i/s for i in lst]


filename = "sample_diagrams//sample4.png"
image = preprocess(filename)
result= loaded_model.predict(image)
l = result[0]
baseDiagrams = np.load("baseDiagrams.npy")

# distances = [np.linalg.norm(l - baseDiagrams[i]) for i in range(11)]
# inverses = [100000.0/np.square(i) for i in distances]
# probs = probability(inverses)
# referencelst = [x for _,x in sorted(zip(probs, referencelst), reverse= True)]
# print(referencelst)
# print(sorted(probs, reverse=True))

from scipy.spatial import distance
cosineSimilarity = [1- distance.cosine(l, baseDiagrams[i]) for i in range(11)]
probs = probability(cosineSimilarity)
referencelst = [x for _,x in sorted(zip(probs, referencelst), reverse= True)]
print(referencelst)
print(sorted(probs, reverse=True))

# for i in range(11):
#     print(referencelst[i], probs[i])
#     print()

# print("cards: ",np.linalg.norm(l-a))
# print("combobox: ",np.linalg.norm(l-b))
# print("tabview: ",np.linalg.norm(l-c))
# print("textbox: ",np.linalg.norm(l-d))
# print("window_header: ",np.linalg.norm(l-e))
# print("enable_disable: ",np.linalg.norm(l-f))
# print("forward: ",np.linalg.norm(l-g))
# print("home: ",np.linalg.norm(l-h))
# print("information_icon: ",np.linalg.norm(l-v))
# print("zoom: ",np.linalg.norm(l-j))
# print("checkbox: ",np.linalg.norm(l-k))