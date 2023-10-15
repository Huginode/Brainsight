import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

#setting the path to the directory containing the pics
pathn = '/dataset/no'
pathy = '/dataset/yes'

#appending the the datasets lists
trainingDataN = []
trainingDataY = []
for img in os.listdir(pathn):
    pic = cv2.IMREAD_ANYCOLOR(os.path.join(pathn, img))
    pic = cv2.cvtColor()
    pic = cv2.resize(pic, (80, 80))
    trainingDataN.append([pic])


for img in os.listdir(pathy):
    pic = cv2.IMREAD_ANYCOLOR(os.path.join(pathy, img))
    pic = cv2.cvtColor(pic, cv2.COLOR_BAYER_BG2GRAY)
    pic = cv2.resize(pic, (80, 80))
    trainingDataY.append([pic])

#converting the list to numpy array and saving it to a file using
np.save(os.path.join(pathn,'features'),np.array(trainingDataN))
np.save(os.path.join(pathy,'features'),np.array(trainingDataY))
