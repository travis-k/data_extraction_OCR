import pytesseract
import numpy as np
from noteshrink import sample_pixels, get_palette, apply_palette, load, make_image
import cv2
import matplotlib.pyplot as plt
from pycpd import rigid_registration
from skimage.measure import label, regionprops
from skimage.segmentation import clear_border
from pytesseract import Output
import re
from string import punctuation


class Part:
    def __init__(self):
        group = None
        part_number = None
        description = None
        figure_index = (None, None)
        units_per_assembly = {}


def extract_textregion(data, points, s, R, t):
    # Given a location area on the template, we transform the area to the original scan coordinate system and
    # extract the text. This is done for the part descriptions, because extracting this from adj_page is less reliable
    glob_points = template_global(points, s, R, t) # The locations of the bounding box on the original scan

    margin = 0.01 # 1% margin on the item height in the list
    n = len(data['text'])
    text = []
    for i in range(0, n):  # For all results from OCR
        if any(data['text'][i].strip()):  # If this is real text
            # If this text fits in our box
            if data['left'][i] >= glob_points[0, 1] and ((data['left'][i] + data['width'][i]) <= glob_points[1, 1])\
                    and data['top'][i] >= glob_points[0, 0]*(1 - margin)\
                    and ((data['top'][i] + data['height'][i]) <= glob_points[1, 0]*(1 + margin)):
                text.append(data['text'][i])

    return ' '.join(text)


def template_global(points, s, R, t):
    # This transforms us from template coordinates to original scan coordinates
    return np.dot((points - t)/s, np.linalg.inv(R))


def global_template(points, s, R, t):
    # This transforms us from scan coordinates to template coordinates
    return s * np.dot(points, R) + t


def scrub(str_in):
    # This just scrubs out unwanted characters
    str_out = str_in.strip()
    str_out = str_out.strip(' ')
    str_out = str_out.lstrip(punctuation)
    str_out = re.sub(r'[\|!@${}\[\]¢+]', '', str_out)
    str_out = re.sub(r'“‘|”’|[“”]|‘‘', '"', str_out)
    str_out = re.sub('—', ' - ', str_out)
    str_out = re.sub('  ', ' ', str_out)
    str_out = str_out.strip()
    return str_out


def extract_subregion(img, region):
    # This will pull the region from the image (used to crop to specific areas)
    # region format is [[top, left], [bottom, right]]
    return img[region[0, 0]:region[1, 0], region[0, 1]:region[1, 1]]


def align_images(t_img, cp_img, p_img):
    # Aligning page with template (template size and location is stationary, origin is top left corner of image)

    # Binarizing the template and the cleaned_page, pre-processing for the alignment
    threshold_level = 250
    t_img = (t_img > threshold_level) * np.uint8(255)
    cp_img = (cp_img > threshold_level) * np.uint8(255)

    # Getting the point clouds that we will match via ICP
    X = np.column_stack(np.where(t_img < threshold_level))
    Y = np.column_stack(np.where(cp_img < threshold_level))

    # This can be done with 3000 points each
    maxpoints = 3000
    X = X[::int(X.shape[0] / maxpoints)]
    Y = Y[::int(Y.shape[0] / maxpoints)]

    # Point matching via ICP from pycpd, which scales, rotates and translates
    reg = rigid_registration(**{'X': X, 'Y': Y, 'max_iterations': 1000, 'tolerance': 0.0001})
    reg.register()

    # Applying the transformation from the point cloud registration with an affine transformation
    adj_page = p_img.copy()  # Copying the page, adj_page will be the output of this (adjusted page)
    M = np.zeros((2, 3))  # Creating the affine transformation matrix
    M[:, 0:2] = reg.s * reg.R
    M[0, 2] = reg.t[1]
    M[1, 2] = reg.t[0]
    adj_page = cv2.warpAffine(adj_page, M, (adj_page.shape[1], adj_page.shape[0]), flags=cv2.INTER_AREA)

    # Cropping or padding image to make it the same size as the template
    old_sz = adj_page.shape
    new_sz = t_img.shape

    # First dim (removing or padding bottom of image)
    if old_sz[0] > new_sz[0]:
        adj_page = adj_page[0:new_sz[0], :]
    elif old_sz[0] < new_sz[0]:
        adj_page = cv2.copyMakeBorder(adj_page, 0, 0, 0, new_sz[0] - old_sz[0], cv2.BORDER_CONSTANT, value=255)

    # Second dim (removing or padding right edge of image)
    if old_sz[1] > new_sz[1]:
        adj_page = adj_page[:, 0:new_sz[1]]
    elif old_sz[1] < new_sz[1]:
        adj_page = cv2.copyMakeBorder(adj_page, 0, new_sz[1] - old_sz[1], 0, 0, cv2.BORDER_CONSTANT, value=255)

    return adj_page, reg.s, reg.R, reg.t


######################################################################################################################
# Get and clean the scanned page image
# this section leaves us with page (the untouched image) and cleaned_page (stripped for template matching)
filename = 'images/page_23.jpg'

sample_fraction = 5
value_threshold = .2
sat_threshold = .2
num_colors = 2
saturate = True
white_bg = True

# This is based on noteshrink (https://github.com/mzucker/noteshrink)
cleaned_page, dpi = load(filename)
samples = sample_pixels(cleaned_page, sample_fraction)
palette = get_palette(samples, num_colors, value_threshold, sat_threshold)
labels = apply_palette(cleaned_page, palette, value_threshold, sat_threshold)
page_pil = make_image(labels, palette, dpi, saturate, white_bg)
page_pil = page_pil.convert('RGB')
cleaned_page = cv2.cvtColor(np.array(page_pil), cv2.COLOR_BGR2GRAY)

page = cv2.imread(filename)  # This is the untouched image

######################################################################################################################
# Loading and scaling the template. The template is scaled so we don't have to downscale the image
template = cv2.imread('images/template_tight.png', 0)

area_ratio = (page.shape[0] * page.shape[1]) / (template.shape[0] * template.shape[1])
len_ratio = page.shape[0] / template.shape[0]

template = cv2.resize(template, (int(template.shape[1] * len_ratio), int(template.shape[0] * len_ratio)),
                      interpolation=cv2.INTER_AREA)

# These are the known predefined regions, based on the template
template_regions = {}
# format is [[top, left], [bottom, right]]
template_regions['group'] = np.int32(np.array([[4, 60], [24, 444]]) * len_ratio)
template_regions['part_number'] = np.int32(np.array([[59, 59], [893, 125]]) * len_ratio)
template_regions['figure_index'] = np.int32(np.array([[59, 4], [893, 57]]) * len_ratio)
template_regions['parts'] = np.int32(np.array([[60, 120], [893, 450]]) * len_ratio)

######################################################################################################################
# Aligning the scanned page to the template
# This section will leave us with a transformation (scale, rotation and translation)
# between our template and the scanned page

# Stripping the page to expose the features that we will match to the template
label_img = label(cleaned_page)
regions = regionprops(label_img)

threshold_area = 1000
for props in regions:
    if props.area < threshold_area:
        for coord in props.coords:
            cleaned_page[coord[0], coord[1]] = 255

cleaned_page = clear_border(cleaned_page, bgval=255)  # Removing any black border that may exist due to the scan

# Page transformation to template, using ICP from pycpd
[adj_page, s, R, t] = align_images(template, cleaned_page, page)

# This is a shortcut to bypass ICP during testing
# adj_page = cv2.imread('images/adj_page.png')
# s = 1.1784858241565166
# R = np.array([[0.99999582,  0.00289168], [-0.00289168,  0.99999582]])
# t = np.array([-348.01448184, -375.05771191])

######################################################################################################################
# OCR
# Reading in page-wide data via tesseract-ocr
cfg = '--oem 1 --psm 4 --user-words RR_IPC.user-words'
page_data = pytesseract.image_to_data(page, config='--oem 1 --psm 4 --user-words RR_IPC.user-words',
                                      output_type=Output.DICT)

# Identifying the group name for this page
str_group = extract_textregion(page_data, template_regions['group'], s, R, t)

# Identifying part numbers on this page
img_part_number = extract_subregion(adj_page, template_regions['part_number'])
part_data = pytesseract.image_to_data(img_part_number, config='--oem 1 --psm 4 --user-words RR_IPC.user-words',
                                 output_type=Output.DICT)

# Identifying individual part numbers and their vertical locations in the list
part_numbers = []
lines = [template_regions['part_number'][0, 0]]
for i in range(0, len(part_data['text'])):
    p = re.compile(r'\S*\d+')
    text = part_data['text'][i].strip()
    if any(text) and re.match(p, text):
        part_numbers.append(text)
            # <--- clean here
        if len(part_numbers) > 1:
            lines.append(part_data['top'][i] + template_regions['part_number'][0, 0])

# Looping through the individual part numbers, extracting other information based on its vertical location on the page
parts_list = []
for i in range(0, len(part_numbers)-1):
    parts_list.append(Part())

    # Group
    parts_list[-1].group = str_group

    # Part number
    parts_list[-1].part_number = part_numbers[i]

    # Description
    # region format is [[top, left], [bottom, right]]
    tmp_region = np.array([[lines[i], template_regions['parts'][0, 1]], [lines[i+1], template_regions['parts'][1, 1]]])
    str_description = scrub(extract_textregion(page_data, tmp_region, s, R, t))
    parts_list[-1].description = str_description

    # Figure and Index
    tmp_region = np.array([[lines[i], template_regions['figure_index'][0, 1]],
                           [lines[i + 1], template_regions['figure_index'][1, 1]]])
    img_figure_index = extract_subregion(adj_page, tmp_region)
    figure_index = pytesseract.image_to_string(img_figure_index,
                                               config='--oem 1 --psm 7 --user-words RR_IPC.user-words').strip()
    p = re.compile(r'\s*\d+\s+\d+\s*')
    if any(figure_index) and re.match(p, figure_index):
        parts_list[-1].figure_index = (int(figure_index.split()[0]), int(figure_index.split()[1]))

# Printing out the parts list
for part in parts_list:
    if hasattr(part, 'figure_index'):
        s = 'Part Number: %s\n\tGroup: %s\n\tFigure and Index: (%d, %d)\n\tDescription: %s\n\n' \
            % (part.part_number, part.group, part.figure_index[0], part.figure_index[1], part.description)
    else:
        s = 'Part Number: %s\n\tGroup: %s\n\tDescription: %s\n\n' \
            % (part.part_number, part.group, part.description)
    print(s)

print('End')