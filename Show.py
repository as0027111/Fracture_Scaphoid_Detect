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
    real_bb = out[1]['boxes'].numpy()
    for i in real_bb:
        cv2.rectangle(image, (int(i[0]),int(i[1])), (int(i[2]),int(i[3])), (0,255,0), thickness=4)
    pred_bb = pred_list[idx]['boxes'].detach().numpy()
    for i in pred_bb[:1]:
        cv2.rectangle(image, (int(i[0]),int(i[1])), (int(i[2]),int(i[3])), (0,0,255), thickness=4)

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
    # print("Angle: ", theta)
    # for i in bb:
    #     cv2.rectangle(image, (int(i[0]),int(i[1])), (int(i[2]),int(i[3])), (0,255,0), thickness=1)

    for i in bb:
        image = draw_rectangle(image, (int(i[0]),int(i[1])), (int(i[2]),int(i[3])), theta, (0, 255, 0))

    bb = pred_list[idx]['boxes'].detach().numpy() # predicted
    for i in bb[:1]:
        # image = draw_rectangle(image, (int(i[0]),int(i[1])), (int(i[2]),int(i[3])), theta, (200, 0, 0))
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

def score_computing(actual, predict):
    TP, TN, FP, FN = 0, 0, 0, 0
    for i in range(len(actual)):
        if actual[i] == 1 and predict[i] == 1:
            TP += 1
        elif actual[i] == 0 and predict[i] == 0:
            TN += 1
        elif actual[i] == 0 and predict[i] == 1:
            FP += 1
        elif actual[i] == 1 and predict[i] == 0:
            FN += 1
    recall = TP / (TP + FN)
    precision = TP / (TP + FP)
    f1_score = 2 * precision * recall / (precision + recall)
    return recall, precision, f1_score

def compute_iou(rec1, rec2):
    """
    computing IoU
    :param rec1: (y0, x0, y1, x1), which reflects (top, left, bottom, right)
    :param rec2: (y0, x0, y1, x1)
    :return: scala value of IoU
    """
    # computing area of each rectangles
    S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
    S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])
    # computing the sum_area
    sum_area = S_rec1 + S_rec2
    # find the each edge of intersect rectangle
    left_line = max(rec1[1], rec2[1])
    right_line = min(rec1[3], rec2[3])
    top_line = max(rec1[0], rec2[0])
    bottom_line = min(rec1[2], rec2[2])
    # judge if there is an intersect
    if left_line >= right_line or top_line >= bottom_line:
        return 0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        return intersect / (sum_area - intersect)

def evaluationIOU(predict, groundTruth): # pos = [[x0, y0], [x1, y1], [x2, y2], [x3, y3]]
    if len(predict) == 0:
        return 0

    predictPos = boundingBoxAreaPos(predict)
    groundTruthPos = boundingBoxAreaPos(groundTruth)

    IOU = len(predictPos & groundTruthPos) / len(predictPos | groundTruthPos)

    return IOU


def boundingBoxAreaPos(position):
    position = np.array(position)
    pos1, pos2, pos3, pos4 = position

    def line1(pos):
        a, b = np.array(pos1) - np.array(pos2)
        return a * pos[0] + b * pos[1]

    def line2(pos):
        a, b = np.array(pos2) - np.array(pos3)
        return a * pos[0] + b * pos[1]

    line1_range = sorted([line1(pos1), line1(pos2)])
    line2_range = sorted([line2(pos2), line2(pos3)])

    found = set()
    minX = np.min(position[:, 0])
    minY = np.min(position[:, 1])
    maxX = np.max(position[:, 0])
    maxY = np.max(position[:, 1])
    for i in range(minX, maxX + 1):
        for j in range(minY, maxY + 1):
            pos = (i, j)
            if line1_range[0] <= line1(pos) <= line1_range[1] and line2_range[0] <= line2(pos) <= line2_range[1]:
                found.add(pos)

    return found
