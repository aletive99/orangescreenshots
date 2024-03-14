import numpy as np
import cv2 as cv
import requests
from bs4 import BeautifulSoup
import os
import urllib
import time


def get_widgets():
    """
    This function downloads the widgets from the Orange Data Mining website and stores them in a folder called Widgets,
    and it is to be called only once before using the other functions for the first time. Every time a new widget is
    added to the website, this function should be called again to update the widgets.
    """
    main = 'https://orangedatamining.com/'
    sub = 'widget-catalog/'
    url = main + sub
    wd = os.getcwd()
    destination_dir = wd + '/Widgets'
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.169 Safari/537.36'}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        img_tags = soup.find_all('img')
        os.makedirs(destination_dir, exist_ok=True)
        for img in img_tags:
            img_url = img.get('src')
            img_url_parsed = urllib.parse.quote(img_url)
            filename = img_url_parsed.split('/')[-1]
            if sub in img_url and not os.path.exists(os.path.join(destination_dir, filename)):
                img_url = urllib.parse.urljoin(main, img_url)
                img_url = urllib.parse.quote(img_url, safe=':/')
                img_name = os.path.join(destination_dir, os.path.basename(img_url))
                req = urllib.request.Request(img_url, headers=headers)
                with urllib.request.urlopen(req) as response:
                    with open(img_name, 'wb') as outfile:
                        outfile.write(response.read())
                        print(f"Downloaded {img_name}")
                time.sleep(0.5)
            else:
                print(f"File {filename} already exists")
    else:
        print(f"Failed to fetch {url}")


def size_identification(img_names_to_check):
    """
    This function identifies the size of the widgets in the image
    :param img_names_to_check: str
    :return: size_widget: int
    """
    image = cv.imread(img_names_to_check, cv.IMREAD_GRAYSCALE)
    blurred = cv.GaussianBlur(image, (5, 5), 0)
    circles = cv.HoughCircles(blurred, cv.HOUGH_GRADIENT, dp=1, minDist=20,
                              param1=90, param2=62.5, minRadius=17, maxRadius=60)
    if circles is None:
        return None
    size_widget = np.round(np.median(circles[0, :, 2])*2)+1
    return size_widget


def get_sizes(img_names_to_check):
    """
    This function calculates the final size of the widgets to be used in the image processing and the number of pixels to
    keep in the widgets
    :param img_names_to_check: str
    :return: final_size: int, pixels_to_keep: int
    """
    widget_size = 100
    target_size = size_identification(img_names_to_check)
    if target_size is None:
        return None, None
    pixels_to_keep = 70
    ratio = target_size/widget_size
    final_size = np.floor(ratio*pixels_to_keep).astype(dtype='int64')
    return final_size, pixels_to_keep


def get_filenames(direct):
    """
    This function gets the filenames of the widgets from the folder created by the get_widgets function
    :param direct: str
    :return: img_names_tgt: np.array
    """
    img_names_tgt = []
    for root, dirs, filenames in os.walk(direct):
        for filename in filenames:
            # Check if the file has the desired extension
            if filename.endswith('png'):
                # If it does, append its full path to the list
                img_names_tgt.append(os.path.join(root, filename))
    img_names_tgt = np.array(img_names_tgt).astype(dtype='str_')
    return img_names_tgt


def widget_loading(img_names_tgt, img_names_to_check):
    """
    This function loads the widgets to be used in the image processing. The output is a 3D array with the number of
    widgets in the first dimension and the final size of the widgets in the second and third dimensions
    :param img_names_tgt: np.array
    :param img_names_to_check: str
    :return: check_img: np.array
    """
    final_size, pixels_to_keep = get_sizes(img_names_to_check)
    if final_size is None:
        return None
    image_size = len(cv.imread(img_names_tgt[0]))
    check_img = np.zeros((len(img_names_tgt), final_size, final_size), dtype='uint8')
    for i in range(0, len(img_names_tgt)):
        check_img[i, :, :] = cv.resize(cv.imread(img_names_tgt[i], cv.IMREAD_GRAYSCALE)
                                       [round((image_size-pixels_to_keep)/2):round((image_size+pixels_to_keep)/2)-1,
                                        round((image_size-pixels_to_keep)/2):round((image_size+pixels_to_keep)/2)-1],
                                        (final_size, final_size), interpolation=4)
    return check_img


def is_there_widget_creation(img_names_to_check, value_thresh=0.81):
    """
    This function creates a matrix with the information of the widgets present in the image. The first column of the
    matrix indicates the widget presence, the second column indicates the location of the widget in the image, the third
    and fourth columns indicate the size of the image, and the fifth column indicates the value of the match.
    :param img_names_to_check: str
    :param value_thresh: float
    :return: is_there_widget: np.array
    """
    widget_size = size_identification(img_names_to_check)
    if widget_size is None:
        return None, None
    img_names_tgt = get_filenames('Widgets')
    check_img = widget_loading(img_names_tgt, img_names_to_check)
    is_there_widget = np.zeros((len(check_img), 5), dtype='int64')
    image_to_check = cv.imread(img_names_to_check, cv.IMREAD_GRAYSCALE)
    for j in range(len(img_names_tgt)):
        res = cv.matchTemplate(image_to_check, check_img[j, :, :], cv.TM_CCOEFF_NORMED)
        tmp = np.sum(res > value_thresh)
        form = res.shape
        if tmp:
            tmp2 = np.argmax(res)
            loc_x_to_check, loc_y_to_check = np.unravel_index(tmp2, form)
            is_there_widget[j, 0] = 1
            is_there_widget[j, 1] = np.array(tmp2)
            is_there_widget[j, 2] = np.array(form)[0, None]
            is_there_widget[j, 3] = np.array(form)[1, None]
            is_there_widget[j, 4] = np.round(np.max(res)*1000)
            if tmp > 1:
                is_far = 0
                for k in range(1, tmp):
                    raveled_loc = res.argsort(axis=None)[-(k+1), None]
                    loc_x, loc_y = np.unravel_index(raveled_loc, form)
                    if all(np.abs(loc_x - loc_x_to_check) > widget_size) or all(np.abs(loc_y - loc_y_to_check) > widget_size):
                        is_far = is_far + 1
                        loc_x_to_check = np.append(loc_x_to_check, loc_x)
                        loc_y_to_check = np.append(loc_y_to_check, loc_y)
                        if is_there_widget[j - is_far, 0] == 0:
                            is_there_widget[j - is_far, 0] = -1 * is_far
                            is_there_widget[j - is_far, 1] = np.array(raveled_loc)
                            is_there_widget[j - is_far, 2] = np.array(form)[0, None]
                            is_there_widget[j - is_far, 3] = np.array(form)[1, None]
                            is_there_widget[j - is_far, 4] = np.round(res[loc_x, loc_y]*1000)
                        else:
                            found = 0
                            while found == 0:
                                if is_there_widget[j - is_far, 0] == 0:
                                    is_there_widget[j - is_far, 0] = -1 * is_far
                                    is_there_widget[j - is_far, 1] = np.array(raveled_loc)
                                    is_there_widget[j - is_far, 2] = np.array(form)[0, None]
                                    is_there_widget[j - is_far, 3] = np.array(form)[1, None]
                                    is_there_widget[j - is_far, 4] = np.round(res[loc_x, loc_y]*1000)
                                    found = 1
                                else:
                                    is_far = is_far + 1
    j = 0
    ind_to_check = np.where(is_there_widget[:, 0] != 0)[0]
    while len(ind_to_check) != 0:
        form = tuple([is_there_widget[ind_to_check[j], 2], is_there_widget[ind_to_check[j], 3]])
        before = 0
        loc_x, loc_y = np.unravel_index(is_there_widget[ind_to_check[j], 1].astype('int64'),
                                        tuple([is_there_widget[ind_to_check[j], 2], is_there_widget
                                        [ind_to_check[j], 3]]))
        loc_x_to_check, loc_y_to_check = np.unravel_index(is_there_widget[ind_to_check, 1], form)
        if np.any(np.logical_and(np.abs(loc_x - loc_x_to_check[np.arange(len(ind_to_check)) != j]) < widget_size,
                                 np.abs(loc_y - loc_y_to_check[np.arange(len(ind_to_check)) != j]) < widget_size)):
            which = np.array(np.where(np.logical_and(np.abs(loc_x - loc_x_to_check) < widget_size,
                                                     np.abs(loc_y - loc_y_to_check) < widget_size) == 1)[0][:, None])
            which_greatest = np.argmax(is_there_widget[ind_to_check[[which]], 4])
            is_there_widget[ind_to_check[which[np.arange(len(which)) != which_greatest]], :] = 0
            ind_to_check = np.where(is_there_widget[:, 0])[0]
            if which_greatest != 0:
                before = 1

        if before == 0:
            j = j+1
        if j >= len(ind_to_check):
            break
        return is_there_widget


def widgets_from_image(img_names_to_check):
    """
    This function returns the list of widgets present in the image
    :param img_names_to_check: str
    :return: widget_list: list
    """
    is_there_widget = is_there_widget_creation(img_names_to_check)
    if is_there_widget is None:
        return None
    img_names_tgt = get_filenames('Widgets/')
    ind_present = np.where(is_there_widget[:, 0] != 0)[0].astype(dtype='int64')
    adjusted_element_index = np.zeros_like(ind_present)
    for j in range(len(ind_present)):
        adjusted_element_index[j] = ind_present[j] - min(is_there_widget[ind_present[j], 0], 0)
    widget_list = list(img_names_tgt[adjusted_element_index])
    return widget_list


def widget_pairs_from_image(img_names_to_check):
    """
    This function returns the list of widget pairs present in the image
    :param img_names_to_check: str
    :return: link_list: list
    """
    is_there_widget = is_there_widget_creation(img_names_to_check)
    if is_there_widget is None:
        return None
    img_names_tgt = get_filenames('Widgets/')
    final_size, _ = get_sizes(img_names_to_check)
    widget_size = size_identification(img_names_to_check)
    x_tol = round(widget_size/2)
    y_tol = round(widget_size/1.8)
    # first dimension of the matrix will indicate what's the receiving widget and second dimension will be the widget
    # from which the link is coming
    links = np.zeros(([len(img_names_tgt), len(img_names_tgt)]))
    # image processing to extract connected components
    tmp_img = cv.imread(img_names_to_check, cv.IMREAD_GRAYSCALE)
    indexes = np.where(is_there_widget[:, 0] != 0)[0]
    form = (is_there_widget[indexes[0], 2], is_there_widget[indexes[0], 3])
    coord_x, coord_y = np.unravel_index(is_there_widget[indexes, 1], form) + np.floor(final_size/2).astype(dtype='int64')
    binary_image = cv.threshold(tmp_img, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)[1]
    binary_image = np.where(binary_image > 0, 0, 255).astype(np.uint8)
    # identifying connected components location and potential links
    num_lab, labels_im = cv.connectedComponents(binary_image)
    labels_in = np.zeros((len(indexes), num_lab))
    labels_out = np.zeros((len(indexes), num_lab))
    iterate = len(coord_x)
    for j in range(iterate):
        if coord_x[j] >= x_tol and coord_y[j] >= y_tol:
            tmp_in = np.unique(labels_im[(coord_x[j]-x_tol):(coord_x[j]+x_tol), (coord_y[j]-y_tol):(coord_y[j])])
        else:
            tmp_in = np.unique(labels_im[max((coord_x[j]-x_tol), 0):(coord_x[j]+x_tol), max((coord_y[j]-y_tol), 0):(coord_y[j])])
        tmp_out = np.unique(labels_im[(coord_x[j]-x_tol):(coord_x[j]+x_tol), (coord_y[j]):(coord_y[j]+y_tol)])
        labels_in[j, 0:len(tmp_in)-1] = tmp_in[tmp_in != 0]
        labels_out[j, 0:len(tmp_out)-1] = tmp_out[tmp_out != 0]
    for j in range(iterate):
        in_to_check = labels_in[j, labels_in[j, :] != 0]
        for k in range(len(in_to_check)):
            except_element = np.arange(iterate) != j
            check_presence = labels_out[except_element, :] == in_to_check[k]
            which_present = np.any(check_presence, axis=1)
            if np.any(which_present):
                which_indexes = indexes[except_element][which_present] - np.where(is_there_widget[indexes[except_element]
                [which_present], 0] > 0, 0, is_there_widget[indexes[except_element][which_present], 0])
                adjusted_element_index = indexes[j] - min(is_there_widget[indexes[j], 0], 0)
                links[adjusted_element_index, which_indexes] = (links[adjusted_element_index, which_indexes] +
                                                                   np.sum(check_presence, axis=1)[which_present])
    a, b = np.where(links != 0)
    link_list = list([])
    for i in range(len(a)):
        link_list.append((img_names_tgt[b[i]], img_names_tgt[a[i]]))
    return link_list


