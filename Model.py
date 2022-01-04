import torchvision, torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm
import numpy as np
import Show

def get_stage1_model(num_classes):
    # load a model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # num_classes which is user-defined
    num_classes = 2  # 1 class (person) + background
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def load_stage1_model(device, load_name):
    # print(device)
    pred_model = get_stage1_model(num_classes=2)
    if device == torch.device('cpu'):
        print("CPU GOGO")
        pred_model.load_state_dict(torch.load(load_name, map_location='cpu'))
    else:
        pred_model.load_state_dict(torch.load(load_name))
    # model.load_state_dict(torch.load('model_state.pth', map_location='cpu'))

    pred_model.to(device) # For inference
    pred_model.eval()
    return pred_model

def predict_stage1(valid_data_loader, load_name='fasterrcnn_resnet50_fpn_cloud_1216.pth'):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    pred_model = load_stage1_model(device, load_name)
    real_list = []
    pred_list = []
    with torch.no_grad():
        for images, targets, image_ids in tqdm(valid_data_loader):
            
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            prediction = pred_model(images, targets)
            
            for i, j in zip(prediction, targets):
                pred_list.append(i)
                real_list.append(j)
                

    iou_list = []
    for i in range(len(pred_list)):
        real_bb = real_list[i]['boxes'].numpy().astype(np.int)[0]
        pred_bb = pred_list[i]['boxes'].detach().numpy().astype(np.int)[0]
        score_iou = Show.compute_iou(real_bb, pred_bb)
        iou_list.append(score_iou)
    
        
    return pred_list, iou_list


def predict_classifier(valid_data_loader, load_name='classifier_stage2_1231.pth'):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    classifier_model = torch.nn.Sequential(
                torchvision.models.resnet50(pretrained=True), 
                torch.nn.ReLU(),
                torch.nn.Linear(1000, 2)
            ).to(device)
    if device == torch.device('cpu'):
        print("CPU GOGO")
        classifier_model.load_state_dict(torch.load(load_name, map_location=torch.device('cpu')))
    else:
        classifier_model.load_state_dict(torch.load(load_name))

    classifier_model.eval()
    classifier_pred_list = []
    classifier_GT = []
    correct_predict_num = 0
    for step, (images, targets, image_ids) in enumerate(tqdm(valid_data_loader)):
        images = list(image for image in images)
        images = torch.tensor(images).to(device)
        targets = torch.tensor(targets).to(device)
        output = classifier_model(images)
        # print(output)
        
        _, predict = torch.max(torch.nn.functional.softmax(output, dim = 1), 1)
        correct_predict_num += (predict.data.cpu() == targets).sum()

        for i, j in zip(predict,targets):
            classifier_pred_list.append(i.item())
            classifier_GT.append(j.item())
    # print(correct_predict_num.item(), len(classifier_pred_list))

    return classifier_pred_list, classifier_GT, correct_predict_num.item()/len(classifier_pred_list)


def get_frac_detect_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    num_classes = 2  # 1 class (person) + background
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def predict_frac_detect(valid_data_loader, load_name='stage2bbox_resnet50_fpn.pth'):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    pred_model = get_frac_detect_model(num_classes=2)
    if device == torch.device('cpu'):
        print("CPU GOGO")
        pred_model.load_state_dict(torch.load(load_name, map_location=torch.device('cpu')))
    else:
        pred_model.load_state_dict(torch.load(load_name))
    # model.load_state_dict(torch.load('model_state.pth', map_location='cpu'))

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    pred_model.to(device) # For inference
    pred_model.eval()
    pred_list =[]

    for images, targets, image_ids in tqdm(valid_data_loader):
        
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        prediction = pred_model(images, targets)
        
        for i in prediction:
            pred_list.append(i)

    return pred_list
