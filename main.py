from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QFileDialog
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtGui import QImage, QPixmap
import cv2
from torch.nn.modules.container import ModuleList

import Load, Show, Model
import sys


class Ui(QtWidgets.QMainWindow):
    def __init__(self):
        super(Ui, self).__init__() # Call the inherited classes __init__ method
        uic.loadUi('HW2_main.ui', self) # Load the .ui file
        self.fpath_btn.clicked.connect(self.open_folder)
        self.stage1_btn.clicked.connect(self.detect_scaphid)
        self.stage2_btn.clicked.connect(self.classifier_and_bbox)
        self.img_ScrollBar.setRange(0, 8-1)
        self.img_ScrollBar.valueChanged.connect(lambda: self.Scrollbar_action())
        self.idx = 0
        self.label_ScrollBar.setText("Index of Images: "+ str(self.idx+1))
        self.ResultDisplay(self.idx)
        self.show() # Show the GUI

    def open_folder(self):
        self.folder_path = QFileDialog.getExistingDirectory(self, "Open folder", "./")                 # start path
        print("Open floder name: ", self.folder_path)
        if self.folder_path: # Avoid didn't open any floder
            self.frac_imgs_path, self.norm_imgs_path, self.frac_cord_path, self.slice_path = Load.data_load(self.folder_path)
            stage1_df = Load.stage_one_df_create(self.slice_path, self.folder_path)
            print("Stage one's dataframe:\n", stage1_df.head())
            self.val_data, self.valid_data_loader = Load.stage_one_data_loader(stage1_df, self.folder_path)
            img = Show.stage1_plot_img(self.val_data, 0) # initialize display windows
            self.display_img(img, self.label_original_img)
            self.img_ScrollBar.setRange(0, len(self.val_data)-1) # set the range of scrollbar
            

    def detect_scaphid(self): # stage 1
        print("[Ming] Start the Stage 1: Scraphoid Detection\n")
        self.pred_list, self.iou_list = Model.predict_stage1(self.valid_data_loader, "fasterrcnn_resnet50_fpn_cloud_1216.pth")
        img = Show.stage1_predict_plot_img(self.pred_list, self.val_data, 0)
        self.display_img(img, self.label_detected_img)
        self.ResultDisplay(self.idx)

    def classifier_and_bbox(self): # stage 2
        ########### CLASSIFICATION ###########
        print("[Ming] Start the Stage 2: Scraphoid Classification\n")
        classifier_df = Load.classifier_df_create(self.slice_path, self.folder_path)
        self.frac_index_lsit = classifier_df[classifier_df['frac']==1].index
        # print("Stage classifier's dataframe:\n", classifier_df.head())
        self.classifier_val_data, self.classifier_valid_data_loader = Load.stage_classifier_data_loader(classifier_df, self.folder_path)
        img = Show.classifier_plot_img(self.classifier_val_data, 0)
        self.display_img(img, self.label_classifier_img)
        self.classifier_pred_list, self.classifier_GT, classifier_acc = Model.predict_classifier(self.classifier_valid_data_loader, 'classifier_stage2_1231.pth')
        # print(self.classifier_GT)
        # print(self.classifier_pred_list)
        self.recall, self.precision, self.f1_score = Show.score_computing(self.classifier_GT, self.classifier_pred_list)
        print("Classification Score (Mean): ", self.recall, self.precision, self.f1_score)
        
        ########### FRACTURE BBOX DETECTION ###########  # stage 3
        print("[Ming] Start the Stage 3: Fracture Detection\n")

        fracture_df = Load.fracture_df_create(self.slice_path, self.folder_path)
        print("Stage fracture detection's dataframe:\n", fracture_df.head())
        self.frac_val_data, self.frac_valid_data_loader = Load.stage_fracture_detect_data_loader(fracture_df, self.folder_path)
        self.frac_pred_list, self.frac_iou_list = Model.predict_frac_detect(self.frac_valid_data_loader, "stage2bbox_resnet50_fpn.pth")
        frac_image = Show.fracture_predict_plot_img(self.frac_val_data, 0, self.frac_pred_list)
        self.display_img(frac_image, self.label_frac_img)
        self.ResultDisplay(self.idx)
        # print(self.frac_pred_list)
    

    def display_img(self, img, label):
        height, width, channel = img.shape
        bytesPerline = 3 * width
        qimg = QImage(img, width, height, bytesPerline, QImage.Format_RGB888).rgbSwapped()
        label.setPixmap(QPixmap.fromImage(qimg))

    def Scrollbar_action(self): # for display image
        # getting current value
        value = self.img_ScrollBar.value()
        self.idx = value
        if hasattr(self, 'val_data'): # ????????????
            ori_img = Show.stage1_plot_img(self.val_data, value)
            self.display_img(ori_img, self.label_original_img)
        if hasattr(self, 'pred_list'): # ??????????????????(???bbox)
            pre_img = Show.stage1_predict_plot_img(self.pred_list, self.val_data, value)
            self.display_img(pre_img, self.label_detected_img)
        if hasattr(self, 'classifier_val_data'): # ??????????????????
            cls_img = Show.classifier_plot_img(self.classifier_val_data, value)
            self.display_img(cls_img, self.label_classifier_img)
        if hasattr(self, 'frac_val_data'): # ??????????????????(???bbox) 
            if value in self.frac_index_lsit:
                pos = 0
                for idx, i in enumerate(self.frac_index_lsit): 
                    if value == i: 
                        pos = idx
                        break
                frac_image = Show.fracture_predict_plot_img(self.frac_val_data, pos, self.frac_pred_list)
                self.display_img(frac_image, self.label_frac_img)
            else:
                cls_img = Show.classifier_plot_img(self.classifier_val_data, value) 
                self.display_img(cls_img, self.label_frac_img)
        self.ResultDisplay(value)
        self.label_ScrollBar.setText("Index of Images: " + str(self.idx+1))

    def ResultDisplay(self, value): # for display result, value->index of display image
        # [stage 1] Scaphoid detection
        stage1_iou_text = ""
        stage1_average_iou_text = ""
        if hasattr(self, 'pred_list'): # ??????????????????(???bbox)
            stage1_iou_text +=  "{:.2f}".format(self.iou_list[value]*100)
            stage1_average_iou_text += str(round(sum(self.iou_list)*100/float(len(self.iou_list)), 3))

        # [stage 2] Cliped Scaphoid classification
        stage2_type_text = ""
        stage2_pred_text = ""
        stage2_recall_text = ""
        stage2_precision_text = ""
        stage2_f1score = ""
        if hasattr(self, 'classifier_val_data'): # ??????????????????
            if self.classifier_GT[value] == 1:
                stage2_type_text += "Fracture"
            elif self.classifier_GT[value] == 0:
                stage2_type_text += "Normal"
            if self.classifier_pred_list[value] == 1:
                stage2_pred_text += "Fracture"
            elif self.classifier_pred_list[value] == 0:
                stage2_pred_text += "Normal"
            # show_acc_text += "\n\nClassification Score (Average): \n"
            stage2_recall_text += "{:.1f}".format(self.recall*100)
            stage2_precision_text += "{:.1f}".format(self.precision*100)
            stage2_f1score += "{:.1f}".format(self.f1_score*100)

        # [stage 3] Fracrure detection
        stage3_iou_text = ""
        stage3_average_iou_text = ""
        if hasattr(self, 'frac_val_data'): # ??????????????????(???bbox) 
            if value in self.frac_index_lsit:
                pos = 0
                for idx, i in enumerate(self.frac_index_lsit): 
                    if value == i: 
                        pos = idx
                        break
                stage3_iou_text += "{:.2f}".format(self.frac_iou_list[pos]*100)
            else:
                stage3_iou_text += "-"
            stage3_average_iou_text += str(round(sum(self.frac_iou_list)*100/float(len(self.frac_iou_list)), 3))
        self.label_Show_acc.setText("Type: " + stage2_type_text + "\n"+
                                    "Predicted as: " + stage2_pred_text + "\n\n\n"+
                                    "Current Image:\n" +
                                    "  Scaphoid's IOU Score: " + stage1_iou_text + " %\n" +
                                    "  Fracture's IOU Score: " + stage3_iou_text + " %\n\n\n" +
                                    "Floder (Average):\n"+
                                    "  Scaphoid's IOU Score: " + stage1_average_iou_text + " %\n" +
                                    "  Fracture's IOU Score: " + stage3_average_iou_text + " %\n" +
                                    "  Recall: " + stage2_recall_text + " %\n" +
                                    "  Precision: " + stage2_precision_text  + " %\n" +
                                    "  F1-score: " + stage2_f1score + " %\n"
                                    )




if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = Ui()
    app.exec_()
