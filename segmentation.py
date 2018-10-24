'''
    THIS FILE IS USED FOR PREPROCESSING
'''
import cv2
import os
import shutil
import numpy as np

SEG_IMG_NAME = 'segmented.png'
DIR_NAME = 'segmented_chars'

# taken an array and appended boolean values corresponding to each row i.e.
# if a row contains atleast one white pixel then true is appended else false is appended
def rows_text_present(img, rows, columns):
    row_intensities = []
    for row in rows:
        zero_pixels = False
        for column in columns:
            if img[row][column] == 255:
                zero_pixels = True
                break
        row_intensities.append(zero_pixels)
    return row_intensities

def make_dir(dir_name):
    if(os.path.exists(DIR_NAME)):
        shutil.rmtree(DIR_NAME)
    os.mkdir(DIR_NAME)

# left right top and bottom borders are drawn for the image
def draw_border(img, **kwargs):
    o = kwargs["left_col"]
    q = kwargs["right_col"]
    vert_up = kwargs["top_row"]
    vert_down = kwargs["bottom_row"]
    color = kwargs.get("color", (0, 255, 0))
    thickness = kwargs.get("thickness", 1)
    cv2.line(img, (o, vert_up), (o, vert_down), color, thickness)
    cv2.line(img, (q, vert_up), (q, vert_down), color, thickness)
    cv2.line(img, (o, vert_up), (q, vert_up), color, thickness)
    cv2.line(img, (o, vert_down), (q, vert_down), color, thickness)

# returns image between the borders drawn on the image
def make_square(im, ds):
    desired_size = ds
    old_size = im.shape[:2]  # old_size is in (height, width) format
    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])
    # new_size should be in (width, height) format
    im = cv2.resize(im, (new_size[1], new_size[0]))
    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                value=color)
    return new_im


import time
def segment(input_img, **kwargs):
    segmented_imgs = []
    save_letters = kwargs.get("save_letters", False)
    orig_seg_out = kwargs.get("orig_seg_out", False)
    seg_img_size = kwargs.get("seg_img_size", 50)
    seg_img_num = 1
    start_time = time.time()
    # if input image is a file location i.e. a string, then we read the image using opencv
    if(isinstance(input_img,str)):
        grayscale = cv2.imread(input_img, cv2.IMREAD_GRAYSCALE)
    # else input image is an array which can be converted to gray scale using cvtcolor
    else:
        grayscale = cv2.cvtColor(input_img, cv2.COLOR_RGB2GRAY)

    # Finding threshold value using otsu, performed inverse binary operation i.e. pixels with
    # intensities less than threshold value are made 255 and that grater are made 0
    _, thresh = cv2.threshold(grayscale,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    end_time = time.time()
    print("Time taken by otsu:",)
    # Find shape of thresholded image
    rows, columns = thresh.shape
    rows = range(rows)
    columns = range(columns)
    row_intensities = rows_text_present(thresh, rows, columns)
    img_for_extraction = thresh
    thresh = cv2.cvtColor(thresh,cv2.COLOR_GRAY2RGB)
    i = 0
    j = -1
    if save_letters:
        make_dir(DIR_NAME)
    # image is scanned leaving the first and last row of the image
    for i in rows[1:-1]:
        # Row with atleast one white pixel is stored in j until it finds row with no white pixel
        if row_intensities[i]:
            if j == -1:
                j = i

        # if we find a row with no white pixels after finding a row with white pixels
        # then the row number is stored in k and the columns are scanned to find the borders
        else:
            if j != -1:
                k = i
                o = -1
                vert_up = k
                vert_down = j
                # scan every column between j and k, first row with white pixel in that column
                # is stored in vert_up, likewise we scan every column and take the min of all as vert_up border
                # same way we take max of last rows with white pixels and store it in vert_down
                for m in columns[1:-1]:
                    text_present = False
                    for n in range(j, k+1):
                        # if we have a column with no white pixels between j and k rows then text_present is false
                        if all(x == 255 for x in thresh[n][m]):
                            text_present = True
                            vert_up = min(vert_up, n)
                            for d in reversed(range(j, k+1)):
                                if all(x == 255 for x in thresh[d][m]):
                                    vert_down = max(vert_down, d)
                                    break
                            break
                    # initial column number with atleast one white pixel is stored in o
                    if text_present:
                        if o == -1:
                            o = m
                    # text_present is flase implies we got a column with no white pixels, so we store it in q
                    # so now we draw border between rows as vet_up, vert_down and columns as o, q
                    else:
                        if o != -1:
                            q = m
                            # image between the borders is extracted
                            seg_img = img_for_extraction[vert_up-1:vert_down+1, o-1:q+1]
                            seg_img = make_square(seg_img, seg_img_size)
                            # _, seg_img = cv2.threshold(seg_img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                            segmented_imgs.append(seg_img)
                            if save_letters:
                                seg_img_path = "{0}/char_{1}.png".format(DIR_NAME, seg_img_num)
                                cv2.imwrite(seg_img_path, seg_img)
                            seg_img_num += 1
                            if orig_seg_out:
                                draw_border(thresh, left_col=o, right_col=q, top_row=vert_up,
                                            bottom_row=vert_down)
                            o = -1
                            vert_up = k
                            vert_down = j
                j = -1
    if orig_seg_out:
        cv2.imwrite(SEG_IMG_NAME, thresh)
    # returns array of segmented images
    return np.array(segmented_imgs)