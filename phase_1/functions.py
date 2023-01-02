# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
from numba import njit, prange
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from commonfunctions import *
from collections import defaultdict
from collections import Counter
from sklearn import svm
from sklearn import metrics
import seaborn as sns

from sklearn.linear_model import SGDClassifier

import joblib

# # %%
# %%capture
# %run  letters_ext
# raction.ipynb

# %%
model_0_1, model_letters, model_all, model_E_F = 0, 0, 0, 0
d = 0

# %%
winSize = (16, 16)
blockSize = (8, 8)
blockStride = (8, 8)
cellSize = (8, 8)
nbins = 9

hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)


# %%
def prepare_image(img):
    image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image = get_letters(image)[0]
    return image


def images_resize(directory):
    list_target_names = []
    list_images = []

    for path, subdirs, files in os.walk(directory):
        if (path.startswith(directory + '.')):
            continue
        files = [f for f in files if not f[0] == '.']  # Ignore '.directory' file
        print(path, len(files))
        # limit = 600
        # if len(files) > limit:
        #     files = files[:limit]

        for name in files:
            image = cv2.imread(os.path.join(path, name))
            image = prepare_image(image)
            # image=cv2.resize(image, (100, 100))
            list_target_names.append(os.path.basename(path))
            list_images.append(image)

    return list_target_names, list_images


# %%
def load_data(directory):
    global d;
    Name = []
    for file in os.listdir(directory):
        Name += [file]

    #################################
    d = defaultdict(int)
    co = 0
    for x in sorted(os.listdir(directory)):
        if not x.startswith('.') and not d[x]:
            d[x] = co
            co += 1
    #########################
    target_names, images = images_resize(directory)
    #########################
    # c = Counter(sorted(target_names))
    # target_names = [ d[key] for key in target_names ]
    target_names_shuffled, images_shuffled = shuffle(np.array(target_names), np.array(images))

    ############reshaping#############
    n_samples, nx, ny = images_shuffled.shape
    # n_samples,nx,ny= np.array(images).shape

    images_shuffled2 = np.array([hog.compute(image) for image in images_shuffled])

    images_shuffled2 = images_shuffled2.reshape(n_samples, -1)
    # images2 = images2.reshape(n_samples,-1)

    Xtrain, Xtest, ytrain, ytest = train_test_split(images_shuffled2, target_names_shuffled, random_state=0,
                                                    test_size=0.2)
    # Xtrain, Xtest, ytrain, ytest = train_test_split(images2, target_names, train_size= 0.2, random_state=5, shuffle= True)

    return Xtrain, Xtest, ytrain, ytest


# %%
def train_model(directory, filename, verbose=False, is_e_f=False):
    Xtrain, Xtest, ytrain, ytest = load_data(directory)

    ####### training #######
    if is_e_f:
        model = SGDClassifier(loss="hinge", penalty="l2")
    else:
        model = svm.SVC(gamma=0.001, C=100)
    model.fit(Xtrain, ytrain)
    ypred = model.predict(Xtest)

    ########### save model ########
    joblib.dump(model, filename)

    if (verbose):
        # sns.set(rc={'figure.figsize':(15,12)})
        mat = confusion_matrix(ytest, ypred)
        # sns.heatmap(mat.T/np.sum(mat.T), annot=True, 
        #     fmt='.2%', cmap='Blues')
        sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False, xticklabels=list(d.keys()),
                    yticklabels=list(d.keys()))
        plt.xlabel('true label')
        plt.ylabel('predicted label')
        plt.show()

    return model


# %%
def initialize_models(verbose=False):
    global model_all;
    global model_0_1;
    global model_letters;
    global model_E_F;
    model_all = train_model(directory='./all_symbols/', filename='./saved_models/model_all.sav', verbose=verbose)
    model_0_1 = train_model(directory='./0_1_symbols/', filename='./saved_models/model_0_1.sav', verbose=verbose)
    model_letters = train_model(directory='./letters_only_symbols/', filename='./saved_models/model_letters.sav',
                                verbose=verbose)
    model_E_F = train_model(directory='./E_F_symbols/', filename='./saved_models/model_E_F.sav', verbose=verbose,
                            is_e_f=True)


# %%
def load_models(is_expression=True, is_table=True):
    global model_all;
    global model_0_1;
    global model_letters;
    global model_E_F;
    if (is_table):
        model_0_1 = joblib.load('./saved_models/model_0_1.sav')
        model_letters = joblib.load('./saved_models/model_letters.sav')

    if (is_expression):
        model_all = joblib.load('./saved_models/model_all.sav')

    model_E_F = joblib.load('./saved_models/model_E_F.sav')


# %%
def classify(img, is_expression=False, is_0_1=False, is_letter=False, verbose=False, is_table=False):
    # print(img)
    letters_res = np.array(get_letters(img, verbose=verbose, single_letter=is_table))
    hog_images = np.array([hog.compute(image) for image in letters_res])

    if is_letter:
        results = model_letters.predict(hog_images)
        for i, r in enumerate(results):
            if r == 'E' or r == 'F':
                results[i] = model_E_F.predict([hog_images[i]])[0]
        return results

    if (is_0_1):
        return model_0_1.predict(hog_images)

    results = model_all.predict(hog_images)
    for i, r in enumerate(results):
        if r == 'E' or r == 'F':
            results[i] = model_E_F.predict([hog_images[i]])[0]

    return results


# %%
# letters= cv2.cvtColor(cv2.imread(r"./test_images/failed.png"), cv2.COLOR_BGR2GRAY)
# # show_images([letters])
# # letters_res = np.array(get_letters(letters, verbose= False))
# # show_images(letters_res)

# results = classify(letters, is_expression=True)
# print(results)

# # %%
# %%capture
# %run final_chars_classification.ipynb

# %%
import joblib
import cv2
import numpy as np
from commonfunctions import *


# %%

def get_table(results):
    # results = predict_image(img, verbose= False)
    results[1] = '='  # the second element is always the equal sign
    # print(results)
    # s = "".join(results)
    # splited_first , splited_second = s.split('=')
    splited_second = results[2:]
    all_valid_literals = ['A', 'B', 'D', 'C', 'F', 'E']

    def delete_invalid(variable):
        # all_valid_literals = ['A','B', 'C', 'F']
        # all_valid_chars.remove(splited_first.upper().strip())
        return variable in all_valid_literals

    final_str = ""
    all_valid_chars = all_valid_literals
    for ind, char in enumerate(splited_second):
        if (char == '+'):
            char = 'or'

        let = lambda x: (x in all_valid_chars)
        # the prev was a literal and the current is a literal or ( 
        # or the prev was ) and the current is a literal or (
        # 
        if (ind != 0 and
                (
                        (let(splited_second[ind - 1]) and (let(char) or char == '(' or char == '~')) or
                        (splited_second[ind - 1] == ')' and (let(char) or char == '(' or char == '~'))
                )
        ):
            final_str += "and "
        prev_was_literal = char in all_valid_literals
        final_str += (char + " ")

    print(final_str)

    from ttg import Truths
    filters_chars = list(set(filter(delete_invalid, splited_second)))
    try:
        table = Truths(filters_chars, [final_str])
        return table
    except:
        return None


# # img =  cv2.imread(r".\test_images\classification\test4.png")
# img =  cv2.imread(r".\test_images\classification\imp.png")
# print(get_table(img))


# %%
# %%capture
# %run final_chars_classification.ipynb


# # %%
# %run table_detector.ipynb

# # %%
# %run final_functions.ipynb

# # %%
# %run preprocessing.ipynb

# # %%
# %run get_rows_number.ipynb

# %%
initialize_models(True)
load_models()

# %%
from tabular import tabular
import math


# %%
def solve_expression(img, is_table=False, showTrace=False):
    img = cv2.resize(img, (2448, 3264))  # size of A4
    img = img.astype(np.uint8)
    if (is_table):
        # show_images([table_tany(img)])
        # table_tany(img) 

        img = table_preprocessing(img, showTrace)
        if showTrace:
            show_images([img])

        cells, col_num, row_num = box_extraction(img, showTrace)  # , table= True)

        letters = [classify(i, is_letter=True, is_table=is_table) for i in cells[0:col_num]]

        numbers = [classify(i, is_0_1=True, is_table=is_table) for i in cells[col_num:]]

        for i in cells[col_num:]:
            show_images([i])
            print("out", classify(i, is_0_1=True, is_table=is_table))

        numbers = [int(i) for i in numbers]

        if True:
            print(letters)
            print(numbers)
            print('rows numbers ' + str(row_num))
            print('cols numbers ' + str(col_num))

        solver = tabular.McCluskey()
        num_outputs = col_num - int(math.log(row_num - 1, 2))
        print(f"num_outputs: {num_outputs}")
        solver.solve(cells=numbers, num_col=col_num, num_outputs=num_outputs)

    else:

        img = expression_preprocessing(img, showTrace)
        if showTrace:
            show_images([img])

        expression_rows = get_rows_number(img, showTrace=showTrace)

        if showTrace:
            print("ROWS")
            show_images(expression_rows)
        expressions = [classify(i, is_expression=True, verbose=showTrace) for i in expression_rows]

        if showTrace: print(expressions)
        result = []
        for ex in expressions:
            try:
                ex[1] = '='
                result.append(get_table(ex))
            except:
                result.append('error')

        for r in result:
            print(r)


# %%
image = cv2.imread("./test_images/final_test_4.jpg")

solve_expression(image, is_table=False, showTrace=True)

# %%
# image = cv2.imread("./test_images/exs/t36.jpg") # TEST R
# image = cv2.imread("./test_images/exs/t35.jpg") # TEST R
# # image = cv2.imread("./test_images/exs/t40.jpg") # TEST R


# # image = cv2.imread("./test_images/exs/t34.jpg") # TEST R

# # image = cv2.imread("./test_images/exs/t33.jpg") 
# image = cv2.imread("./test_images/exs/t39.jpg") # TEST R
# # image = cv2.imread("./test_images/exs/t38.jpg") # TEST R

# result = solve_expression(image, is_table=True, showTrace=False)


# %%
import commonfunctions as cf
import cv2
import numpy as np


# %%
def get_rows_number(i, showTrace=False):
    # _,inverted_image = cv2.threshold(image,127,255,cv2.THRESH_BINARY_INV)

    image = i.copy()
    image = 255 - image
    h = image.shape[0]
    w = image.shape[1]
    h = cf.math.ceil(h * 0.01)

    # the structure element used
    se = np.ones(shape=(h, w // 5))
    # dilated = cv2.morphologyEx(inverted_image, cv2.MORPH_CLOSE, kernel = se)
    dilated = cv2.dilate(image, kernel=se)
    # cf.io.imshow(dilated)
    if (showTrace):
        cf.show_images([image, dilated], ["original", "dilated"])

    contours, _ = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = list(filter(lambda cnt: cv2.contourArea(cnt, True) < 0, contours))
    contours_list = [(x, y, w, h) for x, y, w, h in [cv2.boundingRect(c) for c in contours]]
    contours_list = sorted(contours_list, key=lambda ctr: ctr[1])
    max_height = sorted(contours_list, key=lambda ctr: ctr[3])[-1][3]
    max_width = sorted(contours_list, key=lambda ctr: ctr[2])[-1][2]

    # print("maxxxxxxxx",max_height)
    height_margin = max_height // 3
    width_margin = max_width // 5

    rows = []
    for (x, y, w, h) in (contours_list):
        if (h > height_margin and w > width_margin):
            rows.append(255 - image[y:y + h, x:x + w])

    if (showTrace):
        print(len(rows))
    return rows


# %%
# img1 = cf.io.imread('./test_images/2.png')
# img2 = cf.io.imread('./test_images/handwritten.png') 
# img3 = cf.io.imread('./test_images/6_lines.png') 
# img4 = cf.io.imread('./test_images/3_separate_lines.png') 


# result1 = count_rows(img3)
# result2,len2,ret = count_rows(img2)
# # result3,len3,ret = count_rows(img3)
# # result4,len4,ret = count_rows(img4)


# %%


# %%
import cv2
import commonfunctions as cf
import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange
# import nbimporter
# from count_rows import count_rows

# %%
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize


# Load the image
def get_letters(img, verbose=False, single_letter=False):
    # convert to grayscale
    # img = cv2.resize(img, (0,0),fx=0.5, fy=0.5)
    h, w = img.shape
    retSize = (64, 64)

    tolerance = 0.05 * w
    if verbose:
        print(f'img shape: {img.shape}, max={img.max()}, min={img.min()},median={np.median(img)} and type {img.dtype}')
        print(f'tolerance: {tolerance}')

        cf.show_images([img], ['img'])

    ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    if verbose:
        cf.show_images([thresh], ['thresh'])
    # Find the contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if verbose:
        print('contours before area filtering')
        for cnt in contours:
            print(cv2.contourArea(cnt, True), end=', ')
        print()

    average_area = 0.0
    max_area = 0.0
    max_width = 0.0
    max_height = 0.0
    average_width = 0.0
    average_height = 0.0

    # only keep the contours that are black  and i needed to discard the small parts
    contours = list(filter(lambda cnt: cv2.contourArea(cnt, True) > 0, contours))
    contours_list = [[box, c] for box, c in [(cv2.boundingRect(c), c) for c in contours]]
    if (single_letter):
        contours_list = sorted(contours_list, key=lambda ctr: cv2.contourArea(ctr[1]))
    else:
        contours_list = sorted(contours_list, key=lambda ctr: ctr[0][0])

    if len(contours_list) != 0:
        average_area = sum([cv2.contourArea(cnt[1]) for cnt in contours_list]) / len(contours_list)
        max_area = max([cv2.contourArea(cnt[1]) for cnt in contours_list])
        max_width = max([cnt[0][2] for cnt in contours_list])
        max_height = max([cnt[0][3] for cnt in contours_list])
        average_width = sum([cnt[0][2] for cnt in contours_list]) / len(contours_list)
        average_height = sum([cnt[0][3] for cnt in contours_list]) / len(contours_list)

    if (verbose):
        print('average_area', average_area)
        print('max_width', max_width)
        print('max_height', max_height)

    # only keep the contours that are black  and i needed to discard the small parts
    # x,y,w,h 

    def filter_contours(cnt):
        (x, y, w, h), q = cnt
        # filter contours that are too small
        return cv2.contourArea(q) >= max_area * 0.95 if single_letter else cv2.contourArea(
            q) > average_area * 0.1  # and  w > max_width * 0.25

    contours_list = list(filter(filter_contours, contours_list))

    if verbose:
        print('contours after area filtering')
        for cnt in contours_list:
            print(cnt[0][2] * cnt[0][3], end=', ')
        print()
    # sort contours from left to right

    # max_height =sorted(contours_list, key=lambda ctr: ctr[3])[-1][3]
    # max_width =sorted(contours_list, key=lambda ctr: ctr[2])[-1][2]

    # # print("maxxxxxxxx",max_height)
    # height_margin = max_height // 10
    # width_margin = max_width // 15

    # put masks on the image to get the each letter individually
    masks = []
    for cont in contours_list:
        mask = np.zeros(img.shape, np.float32)
        cv2.drawContours(mask, [cont[1]], 0, (1, 1, 1), -1)
        masks.append(mask)

    # sort contours from left to right
    # contours_list = sorted(contours_list, key=lambda ctr: ctr[0])3

    # merge list that are too close in x axis

    def union(a, b):
        '''
        union of two BoxRectangles 
        '''
        x = min(a[0], b[0])
        y = min(a[1], b[1])
        w = max(a[0] + a[2], b[0] + b[2]) - x
        h = max(a[1] + a[3], b[1] + b[3]) - y
        return (x, y, w, h)

    for ind, (box, _) in enumerate(contours_list):
        x, y, w, h = box
        prev_x = float('-inf') if ind == 0 else contours_list[ind - 1][0][0]
        prev_h = float('-inf') if ind == 0 else contours_list[ind - 1][0][3]
        if (x - prev_x < max_width * 0.2 and abs(h - prev_h) < max_height * 0.2):
            # merge contours 
            contours_list[ind] = (union(contours_list[ind][0], contours_list[ind - 1][0]), contours_list[ind][1])
            contours_list.pop(ind - 1)
            # merge masks of the letters
            masks[ind] = masks[ind] + masks[ind - 1]
            masks.pop(ind - 1)
            ind -= 1

    if verbose:
        print('masks')
        cf.show_images(masks)

    # For each contour, find the bounding rectangle and draw it
    ret_images = []
    for ind, (box, _) in enumerate(contours_list):
        # print(box)
        x, y, w, h = tuple(box)
        new_img = np.logical_and(~img, masks[ind])[y:y + h, x:x + w].astype(np.uint8)
        pad_h = int(h * 0.02)
        pad_w = int(w * 0.2)
        if h > w:
            pad_w = (h - w) // 2
        else:
            pad_h = (w - h) // 2
        new_img = np.pad(new_img, ((pad_h, pad_h), (pad_w, pad_w)), 'constant')
        ret_images.append(new_img)
        if verbose:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)  # crop
            # cv2.putText(img, str(ind), (x,y), cv2.FONT_ITALIC, 1, (0,0,255), 2, cv2.LINE_AA)

    if verbose:
        print('contours after merging')
        plt.show()

    for i in range(len(ret_images)):
        if (ret_images[i].shape != (0, 0)):
            ret_images[i] = cv2.resize(ret_images[i], retSize)
            ret_images[i] = cv2.morphologyEx(ret_images[i], cv2.MORPH_DILATE,
                                             cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE,
                                                                       ksize=(5, 5)))  # , iterations=1)

    if verbose:
        cf.show_images(ret_images)

    if (len(ret_images) == 0):
        return [cv2.resize(img.max() - img, retSize)]

    return ret_images


# from skimage.morphology import skeletonize

img = cv2.cvtColor(cv2.imread("./test_images/b.png"), cv2.COLOR_BGR2GRAY)

# # # apply skeletonization 
# # # perform skeletonization
letters = get_letters(img, verbose=True)

# # skeleton = skeletonize(letters[0])
cf.show_images(letters)

# letter = cv2.cvtColor(cv2.imread('./datasets/Img/img005-047.png'), cv2.COLOR_BGR2GRAY)
# # draw histogram 


# cf.show_images(get_letters(letter,verbose=True))

# img =  cv2.cvtColor(cv2.imread(r"D:\academic_material\third_year\imageProcessing\repos\LogIm\phase_1\symbols\0\0_517.jpg"), cv2.COLOR_BGR2GRAY)
# letters = get_letters(img,verbose=True, single_letter=True)
# cf.show_images(letters, )
# # skeleton = skeletonize(letters[0])
# # cf.show_images([skeleton],['skeleton'])

# %%
# import os
# from os import listdir

# dataset = []
# #load dataset
# folder_dir = r"./datasets/symbols/eval/plus val/"
# for image in os.listdir(folder_dir):
#     dataset.append(cv2.cvtColor(cv2.imread(folder_dir+'/'+image), cv2.COLOR_BGR2GRAY))


# # get the letters from the dataset
# letters = []
# for img in dataset:
#     letters.append(get_letters(img,single_letter=True))

# count = 0
# # write the letters to the disk
# for i in range(len(letters)):
#     if(len(letters[i]) >1):
#         count +=1
#     for j in range(len(letters[i])):
#         cv2.imwrite('./datasets/letters/'+str(i)+'-'+str(j)+'.png',(1- letters[i][j])*255)


# %%


import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import numpy as np
from sklearn import *
from skimage.morphology import binary_erosion, binary_dilation, binary_closing, skeletonize, thin

from sklearn.metrics import accuracy_score

# %%
# #importing required libraries
# from skimage.io import imread, imshow
# from skimage.transform import resize
# from skimage.feature import hog
# from skimage import exposure
# import matplotlib.pyplot as plt
# %matplotlib inline


# #reading the image
# # img = imread(r"D:\academic_material\third_year\imageProcessing\repos\LogIm\phase_1\symbols\!\!_7928.jpg")
# # imshow(img)
# # print(img.shape)

# def hog_func(img):
#     #resizing image 
#     resized_img = resize(img, (128,64)) 
#     # imshow(resized_img) 
#     # print(resized_img.shape)

#     #creating hog features 
#     fd, hog_image = hog(resized_img, orientations=9, pixels_per_cell=(8, 8), 
#                         cells_per_block=(2, 2), visualize=True)

#     return fd


# %%
import sys
from commonfunctions import *
import matplotlib.pyplot as plt
import numpy as np
import skimage.io as io
from skimage.color import rgb2gray
from scipy.ndimage import interpolation as inter
from skimage.morphology import binary_erosion, binary_dilation

import cv2
from utlis import *
import functools


# %%
def skew_correction(image, showTrace):
    # correct skew
    def RotationAngle(binImg):
        def find_score(arr, angle):
            data = inter.rotate(arr, angle, reshape=False, order=0)
            hist = np.sum(data, axis=1)
            score = np.sum((hist[1:] - hist[:-1]) ** 2)
            return hist, score

        delta = 1
        limit = 50
        angles = np.arange(-limit, limit + delta, delta)
        scores = []
        for angle in angles:
            hist, score = find_score(binImg, angle)
            scores.append(score)
        best_score = max(scores)
        best_angle = angles[scores.index(best_score)]
        return best_angle

    def RotateImage(thresh2, angle):
        (h, w) = thresh2.shape
        (cX, cY) = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
        rotated_image = cv2.warpAffine(thresh2, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
        return rotated_image

    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11,
                                    5)  # increasing box size will make elements delated in high ratio
    rotated = RotateImage(thresh2, RotationAngle(thresh2))
    if (showTrace):
        show_images([rotated])

    return rotated


# %%
def trepozoidal_correction(image, showTrace):
    def biggestContour(contours):
        biggest = np.array([])
        max_area = 0
        for i in contours:
            area = cv2.contourArea(i)
            if area > 10000:
                peri = cv2.arcLength(i, True)
                approx = cv2.approxPolyDP(i, 0.02 * peri, True)
                if area > max_area and len(approx) == 4:
                    biggest = approx
                    max_area = area
        return biggest, max_area

    # Image modification
    img_original = image.copy()
    img = image.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 50, 15, 15)
    edged = cv2.Canny(gray, 50, 50)
    kernel = np.ones((7, 7))
    dilatedImg = cv2.dilate(edged, kernel, iterations=2)
    edged = cv2.erode(dilatedImg, kernel, iterations=1)
    # cv2.imwrite("mine.jpg",edged)

    # Contour detection
    contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    ContourFrame = img.copy()
    ContourFrame = cv2.drawContours(ContourFrame, contours, -1, (0, 255, 0), 10)

    biggest, max_area = biggestContour(contours)

    new_img = cv2.drawContours(img, [biggest], -1, (0, 255, 0), 3)
    if (showTrace): show_images([ContourFrame, new_img])

    # Pixel values in the original image
    points = biggest.reshape(4, 2)
    input_points = np.zeros((4, 2), dtype="float32")

    points_sum = points.sum(axis=1)
    input_points[0] = points[np.argmin(points_sum)]
    input_points[3] = points[np.argmax(points_sum)]

    points_diff = np.diff(points, axis=1)
    input_points[1] = points[np.argmin(points_diff)]
    input_points[2] = points[np.argmax(points_diff)]

    (top_left, top_right, bottom_right, bottom_left) = input_points
    bottom_width = np.sqrt(((bottom_right[0] - bottom_left[0]) ** 2) + ((bottom_right[1] - bottom_left[1]) ** 2))
    top_width = np.sqrt(((top_right[0] - top_left[0]) ** 2) + ((top_right[1] - top_left[1]) ** 2))
    right_height = np.sqrt(((top_right[0] - bottom_right[0]) ** 2) + ((top_right[1] - bottom_right[1]) ** 2))
    left_height = np.sqrt(((top_left[0] - bottom_left[0]) ** 2) + ((top_left[1] - bottom_left[1]) ** 2))

    # Output image size
    max_width = max(int(bottom_width), int(top_width))
    # max_height = max(int(right_height), int(left_height))
    max_height = int(max_width * 1.414)  # for A4

    # Desired points values in the output image
    converted_points = np.float32([[0, 0], [max_width, 0], [0, max_height], [max_width, max_height]])

    # Perspective transformation
    matrix = cv2.getPerspectiveTransform(input_points, converted_points)
    img_output = cv2.warpPerspective(img_original, matrix, (max_width, max_height))

    # Image shape modification for hstack
    gray = np.stack((gray,) * 3, axis=-1)
    edged = np.stack((edged,) * 3, axis=-1)

    if (showTrace):
        show_images([img_original, gray, edged])
        show_images([ContourFrame])
        show_images([img, img_output])
    return img_output


# %%
def CheckAlgo(image, showTrace):
    img = cv2.resize(image, (480 * 2, 640 * 2))
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgThre = cv2.adaptiveThreshold(imgGray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY_INV, 1001, 5)

    imgThre2 = cv2.bitwise_not(imgThre)
    imgThre3 = cv2.medianBlur(imgThre2, 3)
    w, h = imgThre3.shape

    if (showTrace):
        show_images([img, imgThre, imgThre2, imgThre3])

    top = np.sum(imgThre3[0:20, 0:h]) / (20 * h)
    bottom = np.sum(imgThre3[w - 20:w, 0:h]) / (20 * h)
    right = np.sum(imgThre3[0:w, 0:20]) / (20 * w)
    left = np.sum(imgThre3[0:w, h - 20:h]) / (20 * w)
    check = (top + bottom + right + left) / 4
    return check


# %%
def remove_black(img, showTrace):
    image = img.copy()
    # image = 255-image  

    h = image.shape[0]
    w = image.shape[1]

    image = image[int(0.05 * h):int(0.95 * h), int(0.05 * w):int(0.95 * w)]
    image = cv2.resize(image, (w, h))

    if (showTrace):
        show_images([image])

    return image


# %%
# initializing
def table_preprocessing(image, showTrace=False):
    check = CheckAlgo(image, showTrace)

    blockSize = image.shape[1] // 10

    if (blockSize % 2 != 1):
        blockSize = blockSize + 1

    if (check < 100):
        img = trepozoidal_correction(image, showTrace)
        # img = trap(image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, blockSize, 12)  # 51 12
    else:
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, blockSize, 5)  # 101 5
        # img = skew_correction(image, showTrace)

        img = cv2.medianBlur(img, 5)

    if showTrace: show_images([img])

    return img


# initializing
def expression_preprocessing(image, showTrace=False):
    check = CheckAlgo(image, showTrace)
    img = image.copy()
    if (check < 100):
        img = trepozoidal_correction(image, showTrace)
    if showTrace: show_images([img], ['after check'])

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    bluredImg = cv2.GaussianBlur(img, (5, 5), 1.2)
    bluredImg = cv2.medianBlur(img, 5)

    blockSize = image.shape[1] // 10

    if (blockSize % 2 != 1):
        blockSize = blockSize + 1

    thresholdedImg = cv2.adaptiveThreshold(bluredImg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY, blockSize=blockSize, C=25)
    if showTrace: show_images([thresholdedImg])

    kernel = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(5, 5))
    thresholdedImg = cv2.morphologyEx(thresholdedImg, cv2.MORPH_OPEN, kernel, iterations=3)
    thresholdedImg = cv2.morphologyEx(thresholdedImg, cv2.MORPH_ERODE,
                                      cv2.getStructuringElement(shape=cv2.MORPH_CROSS, ksize=(3, 3)), iterations=1)
    thresholdedImg = remove_black(thresholdedImg, showTrace=showTrace)

    if showTrace: show_images([thresholdedImg])
    return thresholdedImg


# %%

# img = cv2.imread("grade sheet/2.jpg") 

# print(sorted)
# print(counter)
# plt.figure(),plt.imshow(img),plt.title('Hough Lines'),plt.axis('off')
# plt.show()


# %%
from commonfunctions import *
import math

from utlis import *
import functools


# # %%
# %%capture
# %run  preprocessing.ipynb

# %%
def count_rows(arr, showTrace):
    def sort_horizantal_lines(br_a, br_b):
        return br_a[1] - br_b[1]

    sorted_lines = sorted(arr, key=functools.cmp_to_key(sort_horizantal_lines))

    def filter_sorted_lines(array):
        length = len(array) - 1
        i = 0
        while i < length:
            if abs(array[i][1] - array[i + 1][1]) <= 20:
                array.pop(i)
                length = length - 1
                i = i - 1
            i = i + 1

    result = filter_sorted_lines(sorted_lines)

    if (showTrace):
        print("count rows", len(sorted_lines))

    return len(sorted_lines)


# %%
def box_extraction(image, showTrace):
    img = image.copy()
    (thresh, img_bin) = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)  # Thresholding the image
    img_bin = 255 - img_bin
    kernel_length = np.array(img).shape[1] // 40

    verticle_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))
    hori_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    img_temp1 = cv2.erode(img_bin, verticle_kernel, iterations=3)
    verticle_lines_img = cv2.dilate(img_temp1, verticle_kernel, iterations=5)
    img_temp2 = cv2.erode(img_bin, hori_kernel, iterations=3)
    horizontal_lines_img = cv2.dilate(img_temp2, hori_kernel, iterations=5)

    contours, hierarchy = cv2.findContours(horizontal_lines_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # num_rows=3 # comment this line when detect number of rows

    # Weighting parameters, this will decide the quantity of an image to be added to make a new image.
    alpha = 0.5
    beta = 1.0 - alpha
    # This function helps to add two image with specific weight parameter to get a third image as summation of two image.
    img_final_bin = cv2.addWeighted(verticle_lines_img, alpha, horizontal_lines_img, beta, 0.0)

    img_final_bin = cv2.erode(~img_final_bin, kernel, iterations=2)
    (thresh, img_final_bin) = cv2.threshold(img_final_bin, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # to get the countours clear of the image
    kernel_two = np.ones((7, 7))
    img_final_bin = cv2.morphologyEx(img_final_bin, cv2.MORPH_CLOSE, kernel_two, iterations=5)
    # dilatedImg = cv2.dilate(img_final_bin, kernel_two, iterations=7)
    # img_final_bin = cv2.erode(dilatedImg, kernel_two, iterations=2)

    cv2.imwrite("img_final_bin.jpg", img_final_bin)

    # Find contours for image, which will detect all the boxes
    contours, hierarchy = cv2.findContours(img_final_bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = list(filter(lambda cnt: cv2.contourArea(cnt, True) > 0, contours))
    print("LEN Contours", len(contours))

    boundingBox_arr = []

    for c in contours:
        # Returns the location and width,height for every contour
        x, y, w, h = cv2.boundingRect(c)
        # if w < img_final_bin.shape[1]//30 or h < img_final_bin.shape[0]//30 or w > img_final_bin.shape[1]//3 or h > img_final_bin.shape[0]//3:
        if w > img_final_bin.shape[1] // 3 or h > img_final_bin.shape[0] // 3:
            continue
        boundingBox_arr.append(cv2.boundingRect(c))

    def contour_sort(br_a, br_b):

        if abs(br_a[1] - br_b[1]) <= 15:
            return br_a[0] - br_b[0]

        return br_a[1] - br_b[1]

    boundingBox_arr = sorted(boundingBox_arr, key=functools.cmp_to_key(contour_sort))

    num_rows = count_rows(boundingBox_arr, showTrace)

    num_cols = int(len(boundingBox_arr) / num_rows)

    # if(showTrace):
    print(len(boundingBox_arr))
    print(num_cols, num_rows)

    arr = []
    for c in boundingBox_arr:
        arr.append(img[c[1]:c[1] + c[3], c[0]:c[0] + c[2]])
        if (showTrace):
            new_img = img[c[1]:c[1] + c[3], c[0]:c[0] + c[2]]
            show_images([new_img])
            print(new_img.shape[1], img_final_bin.shape[1])
            print(new_img.shape[0], img_final_bin.shape[0])

    return arr, num_cols, num_rows




path = ("enter path")
image = cv2.imread('./test_images/{path}')
solve_expression(image, True, True)
