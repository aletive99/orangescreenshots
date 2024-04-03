import numpy as np
import cv2 as cv
import requests
from bs4 import BeautifulSoup
import os
import urllib
import time
from tqdm import tqdm
import imageio
from datetime import date, timedelta
import yaml


"""
This operations guarantee that the widgets are downloaded and updated everytime the library is imported.
"""
main = 'https://orangedatamining.com/'
sub = 'widget-catalog/'
url = main + sub
wd = os.getcwd()
destination_dir = wd + '/widgets'
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.169 Safari/537.36'}
response = requests.get(url, headers=headers)
if response.status_code == 200:
    soup = BeautifulSoup(response.content, 'html.parser')
    img_tags = soup.find_all('img')
    os.makedirs(destination_dir, exist_ok=True)
    n_iter = len(img_tags)
    i = 0
    remember = -1
    progress_bar = tqdm(total=n_iter, desc="Progress")
    print('Checking and downloading widgets')
    while i < n_iter:
        img = img_tags[i]
        img_url = img.get('src')
        img_url_parsed = urllib.parse.quote(img_url)
        filename = img_url_parsed.split('/')[-1]
        if sub in img_url and not os.path.exists(os.path.join(destination_dir, filename)):
            img_url = urllib.parse.urljoin(main, img_url)
            img_url = urllib.parse.quote(img_url, safe=':/')
            img_name = os.path.join(destination_dir, os.path.basename(img_url))
            req = urllib.request.Request(img_url, headers=headers)
            try:
                with urllib.request.urlopen(req) as response:
                    with open(img_name, 'wb') as outfile:
                        outfile.write(response.read())
                time.sleep(0.1)
            except urllib.error.URLError:
                remember = i
                i = 0
        i += 1
        if i > remember:
            progress_bar.update(1)
    progress_bar.close()
else:
    print(f"Failed to fetch {url}")
print('Widgets downloaded and updated')
img_name = None
req = None
response = None
outfile = None
del main, sub, url, wd, destination_dir, headers, response, soup, img_tags, img_url, img_url_parsed, filename, \
    img_name, req, remember, i, n_iter, progress_bar, outfile, img


def size_identification(img_names_to_check):
    """
    This function identifies the size of the widgets in the image
    :param img_names_to_check: str
    :return: size_widget: int
    """
    image_to_check = screenshot_loading(img_names_to_check)
    if image_to_check is None:
        return None
    blurred = cv.GaussianBlur(image_to_check, (5, 5), 0)
    circles = cv.HoughCircles(blurred, cv.HOUGH_GRADIENT, dp=1, minDist=20,
                              param1=90, param2=62.5, minRadius=15, maxRadius=60)
    if circles is None or len(circles) > 15:
        return None
    size_widget = (np.round(np.median(circles[0, :, 2])*2)+1).astype(dtype='int64')
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


def get_filenames(direct, ext='image'):
    """
    This function gets the filenames of the widgets from the folder created by the get_widgets function. By default, the
    extension is set to 'image' meaning that only images of the type '.png' and '.jpg' will be extracted, but it can be
    changed according to the needs of the user by specifying an actual extension. The function returns a list with the
    full path of the widgets with the desired extension
    :param direct: str
    :param ext: str
    :return: img_names_tgt: np.array
    """
    img_names_tgt = []
    for root, dirs, filenames in os.walk(direct):
        for filename in filenames:
            if ext == 'image':
                if filename.endswith('.png') or filename.endswith('.jpg'):
                    img_names_tgt.append(os.path.join(root, filename))
            elif filename.endswith(ext):
                img_names_tgt.append(os.path.join(root, filename))
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


def screenshot_loading(img_names_to_check):
    """
    This function loads the screenshot to be used to identify the widgets and their links. It loads a color image and
    converts it to grayscale
    :param img_names_to_check:
    :return: image_to_check: np.array
    """
    reader = imageio.get_reader(img_names_to_check)
    if len(reader) > 1:
        print('The image', img_names_to_check, 'has more than one frame and it will not be considered')
        return None
    for pict in reader:
        frame = cv.cvtColor(pict, cv.COLOR_RGB2BGR)
        image_to_check = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    return image_to_check


def is_there_widget_creation(img_names_to_check, value_thresh=0.80):
    """
    This function creates a matrix with the information of the widgets present in the image. The first column of the
    matrix indicates the widget presence, the second column indicates the location of the widget in the image, the third
    and fourth columns indicate the size of the image, and the fifth column indicates the value of the match. The default
    value of the threshold is 0.81, but it can be changed according to the needs of the user. A lower value will increase
    the number of widgets detected, but it will also increase the number of false positives. A higher value will
    decrease the number of widgets detected, but it will also increase the number of false negatives
    :param img_names_to_check: str
    :param value_thresh: float
    :return: is_there_widget: np.array
    """
    widget_size = size_identification(img_names_to_check)
    if widget_size is None:
        return None
    img_names_tgt = get_filenames('widgets')
    check_img = widget_loading(img_names_tgt, img_names_to_check)
    is_there_widget = np.zeros((len(check_img), 5), dtype='int64')
    image_to_check = screenshot_loading(img_names_to_check)
    for j in range(len(img_names_tgt)):
        res = cv.matchTemplate(image_to_check, check_img[j, :, :], cv.TM_CCOEFF_NORMED)
        if 'Datasets' in img_names_tgt[j]:
            tmp = np.sum(res > 0.7)
        elif 'Correlogram' in img_names_tgt[j]:
            tmp = np.sum(res > 0.85)
        elif 'Periodogram' in img_names_tgt[j]:
            tmp = np.sum(res > 0.90)
        else:
            tmp = np.sum(res > value_thresh)
        form = res.shape
        if tmp:
            loc_y_to_check, loc_x_to_check = (-widget_size, -widget_size)
            is_far = 0
            ordered_res = res.argsort(axis=None)
            for k in range(tmp):
                raveled_loc = ordered_res[-(k+1), None]
                loc_y, loc_x = np.unravel_index(raveled_loc, form)
                if all(np.abs(loc_x - loc_x_to_check) > widget_size) or all(np.abs(loc_y - loc_y_to_check) > widget_size):
                    is_far = is_far + 1
                    loc_x_to_check = np.append(loc_x_to_check, loc_x)
                    loc_y_to_check = np.append(loc_y_to_check, loc_y)
                    if is_there_widget[j - is_far, 0] == 0:
                        is_there_widget[j - is_far, 0] = -1 * is_far
                        is_there_widget[j - is_far, 1] = np.array(raveled_loc)
                        is_there_widget[j - is_far, 2] = np.array(form)[0, None]
                        is_there_widget[j - is_far, 3] = np.array(form)[1, None]
                        is_there_widget[j - is_far, 4] = np.round(res[loc_y, loc_x]*1000)
                    else:
                        found = 0
                        while found == 0:
                            if is_there_widget[j - is_far, 0] == 0:
                                is_there_widget[j - is_far, 0] = -1 * is_far
                                is_there_widget[j - is_far, 1] = np.array(raveled_loc)
                                is_there_widget[j - is_far, 2] = np.array(form)[0, None]
                                is_there_widget[j - is_far, 3] = np.array(form)[1, None]
                                is_there_widget[j - is_far, 4] = np.round(res[loc_y, loc_x]*1000)
                                found = 1
                            else:
                                is_far = is_far + 1
    j = 0
    ind_to_check = np.where(is_there_widget[:, 0] != 0)[0]
    while j < len(ind_to_check):
        form = tuple([is_there_widget[ind_to_check[j], 2], is_there_widget[ind_to_check[j], 3]])
        before = 0
        loc_y, loc_x = np.unravel_index(is_there_widget[ind_to_check[j], 1].astype('int64'),
                                        tuple([is_there_widget[ind_to_check[j], 2], is_there_widget
                                        [ind_to_check[j], 3]]))
        loc_y_to_check, loc_x_to_check = np.unravel_index(is_there_widget[ind_to_check, 1], form)
        if np.any(np.logical_and(np.abs(loc_x - loc_x_to_check[np.arange(len(ind_to_check)) != j]) < widget_size,
                                 np.abs(loc_y - loc_y_to_check[np.arange(len(ind_to_check)) != j]) < widget_size)):
            which = np.array(np.where(np.logical_and(np.abs(loc_x - loc_x_to_check) < widget_size,
                                                     np.abs(loc_y - loc_y_to_check) < widget_size) == 1)[0])
            which_greatest = np.argmax(is_there_widget[ind_to_check[which], 4])
            is_there_widget[ind_to_check[which[np.arange(len(which)) != which_greatest]], :] = 0
            ind_to_check = np.where(is_there_widget[:, 0])[0]
            if which_greatest != 0:
                before = 1
        if before == 0:
            j = j+1
    if np.sum(is_there_widget[:, 0] != 0) == 0:
        return None
    return is_there_widget


def draw_locations(img_names_to_check, return_img=False):
    """
    This function shows the locations of the widgets in the image by highlighting them with a rectangle. The widgets are
    also labeled with their name. If the return_img parameter is set to True, the function will return the image with
    the widgets highlighted
    :param img_names_to_check: str
    :param return_img: bool
    """
    widget_size = size_identification(img_names_to_check)
    if widget_size is None:
        print('There is no widget in the image')
        return
    thickness = np.round(widget_size/2).astype(dtype='int64')
    final_size, _ = get_sizes(img_names_to_check)
    is_there_widget = is_there_widget_creation(img_names_to_check)
    if is_there_widget is None:
        print('There is no widget in the image')
        return
    indexes = np.where(is_there_widget[:, 0] != 0)[0]
    form = (is_there_widget[indexes[0], 2], is_there_widget[indexes[0], 3])
    coord_y, coord_x = np.unravel_index(is_there_widget[indexes, 1], form) + np.floor(final_size/2).astype(dtype='int64')
    image = screenshot_loading(img_names_to_check)
    image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    adjusted_element_index = np.zeros_like(indexes)
    for j in range(len(indexes)):
        cv.rectangle(image, (coord_x[j]-thickness, coord_y[j]-thickness), (coord_x[j]+thickness, coord_y[j]+thickness),
                     (0, 0, 255), 2)
        adjusted_element_index[j] = indexes[j] - min(is_there_widget[indexes[j], 0], 0)
        label = urllib.parse.unquote(get_filenames('widgets/')[adjusted_element_index[j]])
        if len(label.split('-')) == 2:
            label = label.split('-')[1]
        else:
            label = label.split('-')[1] + '-' + label.split('-')[2]
        label = label.split('.')[0]
        cv.putText(image, label, (coord_x[j]-thickness, coord_y[j]-thickness-10), cv.FONT_HERSHEY_SIMPLEX,
                   0.25, (0, 0, 255), 1)
    if not return_img:
        cv.imshow('widget locations highlighted in red', image)
        cv.waitKey(0)
        cv.destroyAllWindows()
        cv.waitKey(1)
    else:
        return image


def draw_links(img_names_to_check):
    """
    This function shows the links between the widgets in the image by drawing a line between the widgets. If the antennas
    are also shown in the image it means that the links are detected only through the connected components of the links,
    otherwise the links are detected through the circle intersection algorithm used on the link direction.
    :param img_names_to_check: str
    """
    widget_size = size_identification(img_names_to_check)
    if widget_size is None:
        print('There is no widget in the image')
        return
    _, link_img = link_detection(img_names_to_check)
    image = screenshot_loading(img_names_to_check)
    image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    image[link_img == 255] = (0, 0, 255)
    cv.imshow('links highlighted in red', image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    cv.waitKey(1)


def draw_links_and_positions(img_names_to_check):
    """
    This function shows the widget positions as well as the links between the widgets in the image.
    :param img_names_to_check:
    :return:
    """
    image = draw_locations(img_names_to_check, return_img=True)
    _, link_img = link_detection(img_names_to_check)
    if link_img is not None:
        image[link_img == 255] = (0, 0, 255)
        cv.imshow('links and widget locations highlighted in red', image)
        cv.waitKey(0)
        cv.destroyAllWindows()
        cv.waitKey(1)


def widgets_from_image(img_names_to_check):
    """
    This function returns the list of widgets present in the image
    :param img_names_to_check: str
    :return: widget_list: list of tuples
    """
    is_there_widget = is_there_widget_creation(img_names_to_check)
    if is_there_widget is None:
        return None
    img_names_tgt = get_filenames('widgets/')
    ind_present = np.where(is_there_widget[:, 0] != 0)[0].astype(dtype='int64')
    adjusted_element_index = np.zeros_like(ind_present)
    widget_list = list([])
    for j in range(len(ind_present)):
        adjusted_element_index[j] = ind_present[j] - min(is_there_widget[ind_present[j], 0], 0)
        adjusted_element_index[j] = np.where(adjusted_element_index[j] > len(is_there_widget) - 1, adjusted_element_index[j] - len(is_there_widget), adjusted_element_index[j])
        tmp = urllib.parse.unquote(img_names_tgt[adjusted_element_index[j]])
        _, module_name = os.path.split(tmp)
        if len(module_name.split('-')) == 2:
            module, name = module_name.split('-')
        else:
            module = module_name.split('-')[0]
            name = module_name.split('-')[1] + '-' + module_name.split('-')[2]
        name = name.split('.')[0]
        widget_list.append((module, name))
    return widget_list


def find_circle_intersection(label_binary_image, center, radius_size, prev_direction):
    """
    This function finds the intersection of a hand build circle with the white pixels of the binary image. The function
    returns the points of intersection, the direction of the points relative to the center, and the index of the point
    that best fits the previous direction
    :param label_binary_image: np.array
    :param center: tuple
    :param radius_size: int
    :param prev_direction: float
    :return: found_points: np.array, found_direction: np.array, best_fit_index: int
    """
    circle_image = np.zeros_like(label_binary_image)
    cv.circle(circle_image, center, radius_size, 255, 1)
    output = cv.bitwise_and(label_binary_image, circle_image)
    num_links, labeled_link_image = cv.connectedComponents(output)
    found_points = np.zeros((num_links - 1, 2)).astype(dtype='int64')
    for i in range(1, num_links):
        loc_y, loc_x = np.where(labeled_link_image == i)
        loc_x = np.mean(loc_x).astype(dtype='int64')
        loc_y = np.mean(loc_y).astype(dtype='int64')
        found_points[i-1, :] = [loc_x, loc_y]
    found_direction = np.arctan2(found_points[:, 1] - center[1], found_points[:, 0] - center[0]) * 180/np.pi
    found_direction = np.where(found_direction >= 0, found_direction, 360 + found_direction)
    best_fit_index = np.argmin(np.abs(found_direction - prev_direction))
    return found_points, found_direction, best_fit_index


def link_detection(img_names_to_check):
    is_there_widget = is_there_widget_creation(img_names_to_check)
    if is_there_widget is None:
        return None, None
    img_names_tgt = get_filenames('widgets/')
    final_size, _ = get_sizes(img_names_to_check)
    widget_size = size_identification(img_names_to_check)
    tol = round(widget_size/2)
    # first dimension of the matrix will indicate what's the receiving widget and second dimension will be the widget
    # from which the link is coming
    links = np.zeros(([len(img_names_tgt), len(img_names_tgt)]), dtype='int64')
    # image processing to extract connected components
    tmp_img = screenshot_loading(img_names_to_check)
    indexes = np.where(is_there_widget[:, 0] != 0)[0]
    form = (is_there_widget[indexes[0], 2], is_there_widget[indexes[0], 3])
    coord_y, coord_x = np.unravel_index(is_there_widget[indexes, 1], form) + np.floor(final_size/2).astype(dtype='int64')
    binary_image = np.where(tmp_img > 180, 0, 255).astype(np.uint8)
    # identifying connected components location and potential links
    num_lab, labels_im = cv.connectedComponents(binary_image)
    labels_in = np.zeros((len(indexes), num_lab))
    labels_out = np.zeros((len(indexes), num_lab))
    iterate = len(coord_x)
    link_img = np.zeros_like(tmp_img)
    for j in range(iterate):
        tmp_in = np.unique(labels_im[(coord_y[j]-tol):(coord_y[j]+tol), (coord_x[j]-tol):(coord_x[j])])
        tmp_out = np.unique(labels_im[(coord_y[j]-tol):(coord_y[j]+tol), (coord_x[j]):(coord_x[j]+tol)])
        labels_in[j, 0:len(tmp_in)-1] = tmp_in[tmp_in != 0]
        labels_out[j, 0:len(tmp_out)-1] = tmp_out[tmp_out != 0]
    for j in range(iterate):
        in_to_check = labels_in[j, labels_in[j, :] != 0]
        for k in range(len(in_to_check)):
            except_element = np.arange(iterate) != j
            check_presence = labels_out[except_element, :] == in_to_check[k]
            which_present = np.any(check_presence, axis=1)
            if np.sum(which_present) == 1:
                label_binary_image = np.where(labels_im == in_to_check[k], 255, 0).astype(dtype='uint8')
                link_img[label_binary_image == 255] = 255
                which_indexes = indexes[except_element][which_present] - np.where(is_there_widget[indexes[except_element]
                [which_present], 0] > 0, 0, is_there_widget[indexes[except_element][which_present], 0])
                which_indexes = np.where(which_indexes > len(is_there_widget) - 1, which_indexes - len(is_there_widget),
                                         which_indexes)
                adjusted_element_index = indexes[j] - min(is_there_widget[indexes[j], 0], 0)
                adjusted_element_index = np.where(adjusted_element_index >= len(is_there_widget),
                                                  adjusted_element_index - len(is_there_widget),
                                                  adjusted_element_index)
                links[adjusted_element_index, which_indexes] = (links[adjusted_element_index, which_indexes] +
                                                                np.sum(check_presence, axis=1)[which_present])
            elif np.sum(which_present) > 1:
                label_binary_image = np.where(labels_im == in_to_check[k], 255, 0).astype(dtype='uint8')
                if widget_size < 36:
                    struct_elem = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))
                    label_binary_image = cv.dilate(label_binary_image, struct_elem, iterations=1)
                start_points, _, _ = find_circle_intersection(label_binary_image, (coord_x[j]-round(widget_size/3.8),
                                                                                   coord_y[j]), round(widget_size/2), 0)
                actual_start = start_points[start_points[:, 0] < coord_x[j]-tol, :]
                in_element_index = indexes[j] - min(is_there_widget[indexes[j], 0], 0)
                in_element_index = np.where(in_element_index > len(is_there_widget) - 1,
                                            in_element_index - len(is_there_widget),
                                            in_element_index)
                for i in range(len(actual_start)):
                    prev_direction = 180
                    center = tuple(actual_start[i])
                    n_iter = 0
                    while True:
                        found_points, found_direction, best_fit_index = find_circle_intersection(label_binary_image, center,
                                                                                                 round(widget_size/7), prev_direction)
                        if abs(abs(found_direction[best_fit_index] - prev_direction) - 180) < 20:
                            center = (center[0]+1, center[1])
                        else:
                            cv.line(link_img, center, tuple(found_points[best_fit_index, :]), 255, 2)
                            center = tuple(found_points[best_fit_index, :])
                            prev_direction = found_direction[best_fit_index]
                        """image_to_show = cv.cvtColor(tmp_img, cv.COLOR_GRAY2BGR)
                        cv.drawMarker(image_to_show, center, (0, 0, 255), cv.MARKER_CROSS, 10, 1)
                        cv.imshow('image', image_to_show)
                        cv.waitKey(200)"""
                        n_iter += 1
                        if n_iter > 100:
                            break
                        if np.any(np.logical_and(abs(center[0] - coord_x[except_element]) - tol*1.5 < 0,
                                                 abs(center[1] - coord_y[except_element]) - tol*0.30 < 0)):
                            which_present = np.logical_and(abs(center[0] - coord_x[except_element]) - tol*1.5 < 0,
                                                           abs(center[1] - coord_y[except_element]) - tol*0.30 < 0)
                            out_element_index = indexes[except_element][which_present] - np.where(is_there_widget[
                                                                                                      indexes[except_element][which_present], 0] > 0, 0, is_there_widget[
                                                                                                      indexes[except_element][which_present], 0])
                            out_element_index = np.where(out_element_index > len(is_there_widget) - 1,
                                                         out_element_index - len(is_there_widget),
                                                         out_element_index)
                            links[in_element_index, out_element_index] = (links[in_element_index, out_element_index] + 1)
                            break

    """cv.destroyAllWindows()
    cv.waitKey(1)"""
    return links, link_img


def widget_pairs_from_image(img_names_to_check):
    """
    This function returns the list of widget pairs present in the image
    :param img_names_to_check: str
    :return: link_list: list of tuples of tuples
    """
    links, _ = link_detection(img_names_to_check)
    if links is None:
        return None
    img_names_tgt = get_filenames('widgets/')
    if np.sum(links) == 0:
        return None
    a, b = np.where(links != 0)
    link_list = list([])
    for i in range(len(a)):
        tmp = urllib.parse.unquote(img_names_tgt[b[i]])
        _, module_name = os.path.split(tmp)
        if len(module_name.split('-')) == 2:
            module1, name1 = module_name.split('-')
        else:
            module1 = module_name.split('-')[0]
            name1 = module_name.split('-')[1] + '-' + module_name.split('-')[2]
        name1 = name1.split('.')[0]
        tmp = urllib.parse.unquote(img_names_tgt[a[i]])
        _, module_name = os.path.split(tmp)
        if len(module_name.split('-')) == 2:
            module2, name2 = module_name.split('-')
        else:
            module2 = module_name.split('-')[0]
            name2 = module_name.split('-')[1] + '-' + module_name.split('-')[2]
        name2 = name2.split('.')[0]
        for j in range(links[a[i], b[i]]):
            link_list.append(((module1, name1), (module2, name2)))
    return link_list


def update_image_list():
    """
    This function finds all the subdirectories inside the orange-lecture-notes-web/public/chapters directory and checks
    if the  information about the images inside the subdirectories are updated. The function creates a yaml file with
    the information about the images present in the notebooks. If an image file has not been updated for 30 days it will
    be updated, otherwise it will stay the same.
    """
    markdown_names = get_filenames('orange-lecture-notes-web/public/chapters', 'index.md')
    today = date.today()
    yaml_direct = 'image-analysis-results'
    os.makedirs(yaml_direct, exist_ok=True)
    progress_bar = tqdm(total=len(markdown_names), desc="Progress")
    try:
        with open(yaml_direct+'/image-list.yaml', 'r') as file:
            images = yaml.safe_load(file)
    except FileNotFoundError:
        images = dict({})
    for i in range(len(markdown_names)):
        direct = markdown_names[i].split('/')[3:-1]
        direct = '/'.join(direct)
        with open(markdown_names[i], 'r') as file:
            data = file.read()
        data = data.split('\n')
        for j in range(len(data)):
            if 'title:' in data[j]:
                data = data[j].split('title: ')[1]
                if '"' in data:
                    title = data.split('"')[1]
                    break
                elif "'" in data:
                    title = data.split("'")[1]
                    break
                else:
                    title = data
                    break
        img_names = get_filenames('orange-lecture-notes-web/public/chapters/' + direct)
        for j in range(len(img_names)):
            img_names[j] = img_names[j].split('/')[-1]
        if len(img_names) > 0:
            try:
                prev_date = images[direct]['date']
                prev_date = date.fromisoformat(prev_date)
            except KeyError:
                prev_date = date(2000, 1, 1)
            if direct not in images or direct in images and today - prev_date > timedelta(days=30):
                images[direct] = {'document-title': str(title),
                                  'date': str(today),
                                  'images': img_names}
        progress_bar.update(1)
    with open(yaml_direct+'/image-list.yaml', 'w') as file:
        yaml.dump(images, file, sort_keys=False)
    progress_bar.close()


def update_widget_list():
    """
    This function finds all the images inside the orange-lecture-notes-web/public/chapters directory and checks if the
    information about the widgets present in the images are updated. The function creates a yaml file with the
    information about the widgets present in the images. If an image file has not been updated for 30 days it will be
    updated, otherwise it will stay the same.
    """
    today = date.today()
    yaml_direct = 'image-analysis-results'
    os.makedirs(yaml_direct, exist_ok=True)
    try:
        with open(yaml_direct+'/image-widgets.yaml', 'r') as file:
            widgets = yaml.safe_load(file)
    except FileNotFoundError:
        widgets = dict({})
    img_names_to_check = get_filenames('orange-lecture-notes-web/public/chapters/')
    progress_bar = tqdm(total=len(img_names_to_check), desc="Progress")
    for i in range(len(img_names_to_check)):
        size = size_identification(img_names_to_check[i])
        img_name = img_names_to_check[i].split('/')[-1]
        path = img_names_to_check[i].split('/')[3:-1]
        path = '/'.join(path)
        key = path + '---' + img_name
        if size is not None:
            try:
                prev_date = widgets[key]['date']
                prev_date = date.fromisoformat(prev_date)
            except KeyError:
                prev_date = date(2000, 1, 1)
            if key not in widgets or key in widgets and today - prev_date > timedelta(days=30):
                widget_list = widgets_from_image(img_names_to_check[i])
                if widget_list is not None:
                    list_of_widgets = list([])
                    for widget_names in widget_list:
                        list_of_widgets.append(widget_names[1])
                    output, count = np.unique(list_of_widgets, return_counts=True)
                    list_of_widgets = list([])
                    for j in range(len(output)):
                        list_of_widgets.append(output[j] + '/' + str(count[j]))
                    widgets[key] = {'path': path,
                                    'filename': img_name,
                                    'date': str(today),
                                    'widgets': list_of_widgets}
                else:
                    widgets[key] = {'path': path,
                                    'filename': img_name,
                                    'date': str(today),
                                    'widgets': None}
        else:
            try:
                prev_date = widgets[key]['date']
                prev_date = date.fromisoformat(prev_date)
            except KeyError:
                prev_date = date(2000, 1, 1)
            if key not in widgets or key in widgets and today - prev_date > timedelta(days=30):
                widgets[key] = {'path': path,
                                'filename': img_name,
                                'date': str(today),
                                'widgets': None}
        progress_bar.update(1)
    with open(yaml_direct+'/image-widgets.yaml', 'w') as file:
        yaml.dump(widgets, file, sort_keys=False)
    progress_bar.close()


def update_image_links():
    """
    This function finds all the images inside the orange-lecture-notes-web/public/chapters directory and checks if the
    information about the links between the widgets present in the images are updated. The function creates a yaml file
    with the information about the links between the widgets present in the images. If an image file has not been
    updated for 30 days it will be updated, otherwise it will stay the same.
    """
    today = date.today()
    yaml_direct = 'image-analysis-results'
    os.makedirs(yaml_direct, exist_ok=True)
    try:
        with open(yaml_direct+'/image-links.yaml', 'r') as file:
            links = yaml.safe_load(file)
    except FileNotFoundError:
        links = dict({})
    img_names_to_check = get_filenames('orange-lecture-notes-web/public/chapters/')
    progress_bar = tqdm(total=len(img_names_to_check), desc="Progress")
    for i in range(len(img_names_to_check)):
        size = size_identification(img_names_to_check[i])
        img_name = img_names_to_check[i].split('/')[-1]
        path = img_names_to_check[i].split('/')[3:-1]
        path = '/'.join(path)
        key = path + '---' + img_name
        if size is not None:
            try:
                prev_date = links[key]['date']
                prev_date = date.fromisoformat(prev_date)
            except KeyError:
                prev_date = date(2000, 1, 1)
            if key not in links or key in links and today - prev_date > timedelta(days=30):
                link_list = widget_pairs_from_image(img_names_to_check[i])
                if link_list is not None:
                    list_of_links = list([])
                    for link_pairs in link_list:
                        list_of_links.append(link_pairs[0][1] + ' -> ' + link_pairs[1][1])
                    output, count = np.unique(list_of_links, return_counts=True)
                    list_of_links = list([])
                    for j in range(len(output)):
                        list_of_links.append(output[j] + '/' + str(count[j]))
                    links[key] = {'path': path,
                                  'filename': img_name,
                                  'date': str(today),
                                  'links': list_of_links}
                else:
                    links[key] = {'path': path,
                                  'filename': img_name,
                                  'date': str(today),
                                  'links': None}
        else:
            try:
                prev_date = links[key]['date']
                prev_date = date.fromisoformat(prev_date)
            except KeyError:
                prev_date = date(2000, 1, 1)
            if key not in links or key in links and today - prev_date > timedelta(days=30):
                links[key] = {'path': path,
                              'filename': img_name,
                              'date': str(today),
                              'links': None}
        progress_bar.update(1)
    with open(yaml_direct+'/image-links.yaml', 'w') as file:
        yaml.dump(links, file, sort_keys=False)
    progress_bar.close()


#%%
