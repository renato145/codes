import cv2, numpy as np, imutils
import matplotlib.pyplot as plt
from pathlib import Path

def show_img(x, ax=None, figsize=(12,13)):
    gray = True if len(x.shape) == 2 else False
    if ax is None: _, ax = plt.subplots(figsize=figsize)
    if gray: ax.imshow(x, 'gray')
    else: ax.imshow(imutils.opencv2matplotlib(x))
    ax.set_axis_off()
    return ax

def binarize(x, k=5, thresh=127, thresh_type=cv2.THRESH_BINARY):
    gray = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (k,k), 0)
    thresh = cv2.threshold(blurred, thresh, 255, thresh_type)[1]
    return thresh

def get_contours(thresh):
    return cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
