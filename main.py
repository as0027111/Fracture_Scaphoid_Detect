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
        self.label_ScrollBar.setText("0")

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

    def detect_scaphid(self):
        self.pred_list, self.iou_list = Model.predict_stage1(self.valid_data_loader, "fasterrcnn_resnet50_fpn_cloud_1216.pth")
        img = Show.stage1_predict_plot_img(self.pred_list, self.val_data, 0)
        self.display_img(img, self.label_detected_img)
        # print("1")

    def classifier_and_bbox(self): # stage 2
        ########### CLASSIFICATION ###########
        classifier_df = Load.classifier_df_create(self.slice_path, self.folder_path)
        self.frac_index_lsit = classifier_df[classifier_df['frac']==1].index
        print("Stage classifier's dataframe:\n", classifier_df.head())
        self.classifier_val_data, self.classifier_valid_data_loader = Load.stage_classifier_data_loader(classifier_df, self.folder_path)
        img = Show.classifier_plot_img(self.classifier_val_data, 0)
        self.display_img(img, self.label_classifier_img)
        self.classifier_pred_list, self.classifier_GT, classifier_acc = Model.predict_classifier(self.classifier_valid_data_loader, 'classifier_stage2_1231.pth')
        # print(self.classifier_GT)
        # print(self.classifier_pred_list)
        self.recall, self.precision, self.f1_score = Show.score_computing(self.classifier_GT, self.classifier_pred_list)
        print("Classification Score (Mean): ", self.recall, self.precision, self.f1_score)
        
        show_acc_text = ""
        if self.classifier_GT[0] == 1:
            show_acc_text += "Type: Fracture\n"
        elif self.classifier_GT[0] == 0:
            show_acc_text += "Type: Normal\n"
        if self.classifier_pred_list[0] == 1:
            show_acc_text += "Predicted as: Fracture\n"
        elif self.classifier_pred_list[0] == 0:
            show_acc_text += "Predicted as: Normal\n"
        self.label_Show_acc.setText(show_acc_text)

        ########### FRACTURE BBOX DETECTION ###########
        fracture_df = Load.fracture_df_create(self.slice_path, self.folder_path)
        print("Stage fracture detection's dataframe:\n", fracture_df.head())
        self.frac_val_data, self.frac_valid_data_loader = Load.stage_fracture_detect_data_loader(fracture_df, self.folder_path)
        self.frac_pred_list = Model.predict_frac_detect(self.frac_valid_data_loader, "stage2bbox_resnet50_fpn.pth")
        frac_image = Show.fracture_predict_plot_img(self.frac_val_data, 0, self.frac_pred_list)
        self.display_img(frac_image, self.label_frac_img)

        # print(self.frac_pred_list)
    

    def display_img(self, img, label):
        height, width, channel = img.shape
        bytesPerline = 3 * width
        qimg = QImage(img, width, height, bytesPerline, QImage.Format_RGB888).rgbSwapped()
        label.setPixmap(QPixmap.fromImage(qimg))

    def Scrollbar_action(self):
        # getting current value
        value = self.img_ScrollBar.value()
        show_acc_text = ""
        if hasattr(self, 'val_data'): # 原始圖片
            ori_img = Show.stage1_plot_img(self.val_data, value)
            self.display_img(ori_img, self.label_original_img)
        if hasattr(self, 'pred_list'): # 手骨預測圖片(含bbox)
            pre_img = Show.stage1_predict_plot_img(self.pred_list, self.val_data, value)
            self.display_img(pre_img, self.label_detected_img)
            show_acc_text += "Stage1's IOU Score: " + "{:.2f}".format(self.iou_list[value]*100) + " %\n"
            show_acc_text += "Average IOU Score: " + str(round(sum(self.iou_list)*100/float(len(self.iou_list)), 3)) + " %\n\n\n"
        if hasattr(self, 'classifier_val_data'): # 手骨裁切畫面
            cls_img = Show.classifier_plot_img(self.classifier_val_data, value)
            self.display_img(cls_img, self.label_classifier_img)

            if self.classifier_GT[value] == 1:
                show_acc_text += "Type: Fracture\n"
            elif self.classifier_GT[value] == 0:
                show_acc_text += "Type: Normal\n"
            if self.classifier_pred_list[value] == 1:
                show_acc_text += "Predicted as: Fracture\n"
            elif self.classifier_pred_list[value] == 0:
                show_acc_text += "Predicted as: Normal\n"
            show_acc_text += "\n\nClassification Score (Average): \n"
            show_acc_text += "  Recall: " + "{:.1f}".format(self.recall*100) + " %\n"
            show_acc_text += "  Precision: " + "{:.1f}".format(self.precision*100) + " %\n"
            show_acc_text += "  F1-score: " + "{:.1f}".format(self.f1_score*100) + " %\n"
        if hasattr(self, 'frac_val_data'): # 骨折預測圖片(含bbox) 
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
        self.label_Show_acc.setText(show_acc_text)
        self.label_ScrollBar.setText(str(value))




if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = Ui()
    app.exec_()
