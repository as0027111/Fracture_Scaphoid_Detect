import matplotlib.pyplot as plt
import numpy as np
import cv2

PICTURE_BOX_HEIGHT = 281
SCAPHID_BOX_HEIGHT = 161
def image_convert(image):
    image = image.clone().cpu().numpy()
    image = image.transpose((1,2,0))
    image = (image * 255).astype(np.uint8)
    return image

def stage1_plot_img(data, idx):
    out = data.__getitem__(idx)
    image = image_convert(out[0])
    image = np.ascontiguousarray(image)
    bb = out[1]['boxes'].numpy()
    for i in bb:
        cv2.rectangle(image, (int(i[0]),int(i[1])), (int(i[2]),int(i[3])), (0,255,0), thickness=2)
    scaling = PICTURE_BOX_HEIGHT / image.shape[0]
    image = cv2.resize(image, None, fx=scaling, fy=scaling, interpolation=cv2.INTER_AREA)

    return image

def stage1_predict_plot_img(pred_list, data, idx):
    out = data.__getitem__(idx)
    image = image_convert(out[0])
    image = np.ascontiguousarray(image)
    bb = out[1]['boxes'].numpy()
    for i in bb:
        cv2.rectangle(image, (int(i[0]),int(i[1])), (int(i[2]),int(i[3])), (0,255,0), thickness=2)
    bb = pred_list[idx]['boxes'].detach().numpy()
    for i in bb[:1]:
        cv2.rectangle(image, (int(i[0]),int(i[1])), (int(i[2]),int(i[3])), (0,0,255), thickness=2)
    scaling = PICTURE_BOX_HEIGHT / image.shape[0]
    image = cv2.resize(image, None, fx=scaling, fy=scaling, interpolation=cv2.INTER_AREA)

    return image


def classifier_plot_img(data, idx):
    out = data.__getitem__(idx)
    image = classifier_image_convert(out[0])
    image = np.ascontiguousarray(image)
    scaling = SCAPHID_BOX_HEIGHT / image.shape[0]
    image = cv2.resize(image, None, fx=scaling, fy=scaling, interpolation=cv2.INTER_AREA)
    return image

def classifier_image_convert(image):
    # image = image.clone().cpu().numpy()
    image = image.transpose((1,2,0))
    image = (image * 255).astype(np.uint8)
    return image



def fracture_predict_plot_img(data, idx, pred_list):
    out = data.__getitem__(idx)
    image = image_convert(out[0])
    image = np.ascontiguousarray(image)
    bb = out[1]['boxes'].numpy()
    theta = out[1]['angle'].numpy()
    print("Angle: ", theta)
    for i in bb:
        cv2.rectangle(image, (int(i[0]),int(i[1])), (int(i[2]),int(i[3])), (0,255,0), thickness=1)

    for i in bb:
        image = draw_rectangle(image, (int(i[0]),int(i[1])), (int(i[2]),int(i[3])), theta, (0, 200, 0))

    bb = pred_list[idx]['boxes'].detach().numpy() # predicted
    for i in bb[:1]:
        image = draw_rectangle(image, (int(i[0]),int(i[1])), (int(i[2]),int(i[3])), theta, (200, 0, 0))
        cv2.rectangle(image, (int(i[0]),int(i[1])), (int(i[2]),int(i[3])), (255,0,0), thickness=1)

    scaling = SCAPHID_BOX_HEIGHT / image.shape[0]
    image = cv2.resize(image, None, fx=scaling, fy=scaling, interpolation=cv2.INTER_AREA)
    return image


def draw_rectangle(image, x1y1, x2y2, theta, color):
    centre = (int((x1y1[0]+x2y2[0])/2), int((x1y1[1]+x2y2[1])/2))
    width = int(x2y2[0] - x1y1[0])
    height= int(x2y2[1] - x1y1[1])
    
    theta = np.radians(theta)
    c, s = np.cos(theta), np.sin(theta)
    R = np.matrix('{} {}; {} {}'.format(c, -s, s, c))
    # print(R)
    p1 = [ + width / 2,  + height / 2]
    p2 = [- width / 2,  + height / 2]
    p3 = [ - width / 2, - height / 2]
    p4 = [ + width / 2,  - height / 2]
    p1_new = np.dot(p1, R)+ centre
    p2_new = np.dot(p2, R)+ centre
    p3_new = np.dot(p3, R)+ centre
    p4_new = np.dot(p4, R)+ centre
    img = cv2.line(image, (int(p1_new[0, 0]), int(p1_new[0, 1])), (int(p2_new[0, 0]), int(p2_new[0, 1])), color, 1)
    img = cv2.line(img, (int(p2_new[0, 0]), int(p2_new[0, 1])), (int(p3_new[0, 0]), int(p3_new[0, 1])), color, 1)
    img = cv2.line(img, (int(p3_new[0, 0]), int(p3_new[0, 1])), (int(p4_new[0, 0]), int(p4_new[0, 1])), color, 1)
    img = cv2.line(img, (int(p4_new[0, 0]), int(p4_new[0, 1])), (int(p1_new[0, 0]), int(p1_new[0, 1])), color, 1)

    return img
