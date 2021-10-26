import numpy as np
import tensorflow as tf
import cv2
import pytesseract as tess
import re
from matplotlib import pyplot as plt

# Set pytesseract Path
tess.pytesseract.tesseract_cmd = r'C:\Users\ary7ban\AppData\Local\Tesseract-OCR\tesseract.exe'


# For showing OpenCV image in Notebook
def show(image, title='Image'):
    image = np.array(image)
    plt.imshow(image, cmap='Greys')
    plt.axis('off')
    if title:
        plt.title(title + ' %sx%s.' % image.shape[0:2])
    return plt.show()


def show_multiple(images, title_list=None):
    f = plt.figure(figsize=(10, 3))
    title_list = list(title_list)
    for i, image in enumerate(images):
        f.add_subplot(1, len(images), i + 1)
        title = '(%sx%s)' % image.shape[0:2]
        if title_list:
            title = title_list[i] + title
        plt.axis('off')
        plt.title(title)
        p = plt.imshow(image, cmap='Greys')
    return plt.show()


def read_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def normalize(image):
    input_size = 416
    image_data = cv2.resize(image, (input_size, input_size))
    image_data = image_data / 255.
    images_data = np.asarray([image_data]).astype(np.float32)
    return tf.constant(images_data)


def get_model_infer(model_path):
    saved_model_loaded = tf.saved_model.load(model_path, tags="serve")
    infer = saved_model_loaded.signatures['serving_default']
    return infer


def process_boxes(pred_bbox):
    for key, value in pred_bbox.items():
        boxes = value[:, :, 0:4]
        pred_conf = value[:, :, 4:]

    # run non max suppression on detections
    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
        scores=tf.reshape(
            pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
        max_output_size_per_class=5,
        max_total_size=5,
        iou_threshold=0.45,
        score_threshold=0.50
    )
    num_boxes = valid_detections.numpy()[0]
    boxes = boxes.numpy()[0][0:num_boxes]
    return boxes


# helper function to convert bounding boxes from normalized to original
# ymin, xmin, ymax, xmax ---> xmin, ymin, xmax, ymax
def format_boxes(bboxes, image_height, image_width):
    for box in bboxes:
        ymin = int(box[0] * image_height)
        xmin = int(box[1] * image_width)
        ymax = int(box[2] * image_height)
        xmax = int(box[3] * image_width)
        box[0], box[1], box[2], box[3] = xmin, ymin, xmax, ymax
    return bboxes


def get_plate_images(infer, image_path, show_to_notebook=False):
    image = read_image(image_path)
    image_height, image_width, _ = image.shape

    images_data = normalize(image)
    boxes = infer(images_data)
    boxes = process_boxes(boxes)
    boxes = format_boxes(boxes, image_height, image_width)

    plate_images = []
    for coor in boxes:
        # separate coordinates from box
        xmin, ymin, xmax, ymax = coor
        # get the subimage that makes up the bounded region and take an additional 5 pixels on each side
        box = image[int(ymin) - 5:int(ymax) + 5, int(xmin) - 5:int(xmax) + 5]
        plate_images.append(box)

    if show_to_notebook:
        show(image, 'Original Image')
        show(images_data[0], 'Normalized Image')

        disp_img = image.copy()
        for coor in boxes:
            c1, c2 = (coor[0], coor[1]), (coor[2], coor[3])
            cv2.rectangle(disp_img, c1, c2, (0, 255, 0), int((image_height + image_width) / 600))
        show(disp_img, '%s plate(s) detected' % len(boxes))

    return plate_images


def valid_rect(img_shape, rect_shape):
    im_h, im_w = img_shape
    rect_h, rect_w = rect_shape

    reason = 0
    # if height of box is not tall enough relative to total height then skip
    if not im_h / rect_h < 6:
        reason += 1

    # if height to width ratio is less than 1.5 skip
    if not 1.5 < rect_h / rect_w < 4.5:
        reason += 2

    # if width is not wide enough relative to total width then skip
    if not rect_w > im_w / 40:
        reason += 4

    # if area is less than 100 pixels skip
    if not rect_h * rect_w > 100:
        reason += 8

    print(img_shape, rect_shape, reason)
    return reason == 0


def get_char_images(plate_image, show_to_notebook=False):
    # resize image to three times as large as original for better readability
    plate_img = cv2.resize(plate_image, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    # grayscale region within bounding box
    gray = cv2.cvtColor(plate_img, cv2.COLOR_RGB2GRAY)
    # perform gaussian blur to smoothen image
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # threshold the image using Otsus method to preprocess for tesseract
    ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    # # create rectangular kernel for dilation
    rect_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    # apply dilation to make regions more clear
    erosion = cv2.erode(thresh, rect_kern, iterations=1)
    # find contours of regions of interest within license plate
    try:
        contours, hierarchy = cv2.findContours(erosion, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    except:
        ret_img, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Get Bounding Rectangles for all Contours
    rects = [cv2.boundingRect(contour) for contour in contours]
    rects = sorted(rects, key=lambda r: r[0])
    # Filter for valid Bounding Rectangles
    filtered_rects = [(x, y, w, h) for (x, y, w, h) in rects if valid_rect(gray.shape, (h, w))]
    median_h = np.median([h for (x, y, w, h) in filtered_rects])
    filtered_rects = [(x, y, w, h) for (x, y, w, h) in rects if abs((h - median_h) / median_h) < 0.1]
    # grab character regions of image
    char_images = [gray[y - 5:y + h + 5, x - 5:x + w + 5] for (x, y, w, h) in filtered_rects]

    if show_to_notebook:
        show(plate_img, 'Plate')
        show(gray, 'gray')
        show(blur, 'blur')
        show(thresh, 'thresh')
        show(erosion, 'erosion')
        # show(dilation, 'dilation')

        disp_image = plate_img.copy()
        cv2.drawContours(image=disp_image, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2,
                         lineType=cv2.LINE_AA)
        show(disp_image, 'All contours')
        print('No.of Contours =', len(contours))

        disp_image = plate_img.copy()
        for (x, y, w, h) in rects:
            cv2.rectangle(disp_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        show(disp_image, 'All Rectangles')
        print('No.of Rectangles =', len(rects))

        disp_image = plate_img.copy()
        for (x, y, w, h) in filtered_rects:
            cv2.rectangle(disp_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        show(disp_image, 'Filtered Rectangles')
        print('No.of Filtered Rectangles =', len(filtered_rects))

    return char_images


def read_char_image(image, show_to_notebook=False):
    # Sharpen the Image
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharp = cv2.filter2D(image, -1, kernel)
    # threshold the image using Otsus method to preprocess for tesseract
    ret, thresh = cv2.threshold(sharp, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)
    # perform another blur on character region
    blur = cv2.GaussianBlur(thresh, (3, 3), 0)
    # create rectangular kernel for dilation
    rect_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    # apply dilation to make regions more clear
    dilation = cv2.dilate(thresh, rect_kern, iterations=1)

    final_image = dilation
    # try:
    text = tess.image_to_string(final_image, config='-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
                                                    ' --psm 8 --oem 3')
    clean_text = re.sub(r'[\W_]+', '', text)

    if show_to_notebook:
        images = {
            'CharImage': image,
            'Sharp': sharp,
            'Thresh': thresh,
            'Blur': blur,
            'Dilated': dilation,
        }

        show_multiple(images.values(), title_list=images.keys())
        if clean_text:
            print('Character Detected:', clean_text)
            # print(tess.image_to_data(final_image, output_type='data.frame'))
        else:
            print('No Character Detected in Below image')
            show(final_image, title='')

    return clean_text


def read_plate(infer, image_path, show_to_notebook=False):
    plate_images = get_plate_images(infer, image_path, show_to_notebook=show_to_notebook)
    plate_texts = []
    error_images = []
    for plate_img in plate_images:
        char_images = get_char_images(plate_img, show_to_notebook=show_to_notebook)

        # concat_image = concat_resize(char_images)
        # show(concat_image)
        # plate_text = read_char_image(concat_image, show_to_notebook=show_to_notebook)
        # plate_texts.append(plate_text)

        plate_text = ''
        for img in char_images:
            char = read_char_image(img, show_to_notebook=show_to_notebook)
            if char:
                plate_text += char
            else:
                error_images.append(img)
        plate_texts.append(plate_text)

    return {'plate_texts': plate_texts, 'error_images': error_images}


# horizontally concatenating images of different heights
def concat_resize(img_list, interpolation=cv2.INTER_CUBIC):
    # take minimum heights
    h_min = min(img.shape[0] for img in img_list)

    # image resizing
    im_list_resize = [cv2.resize(img, (int(img.shape[1] * h_min / img.shape[0]), h_min), interpolation=interpolation)
                      for img in img_list]

    # return final image
    return cv2.hconcat(im_list_resize)
