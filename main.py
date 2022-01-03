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
        self.pred_list = Model.predict_stage1(self.valid_data_loader, "fasterrcnn_resnet50_fpn_cloud_1216.pth")
        img = Show.stage1_predict_plot_img(self.pred_list, self.val_data, 0)
        self.display_img(img, self.label_detected_img)
        # print("1")

    def classifier_and_bbox(self): # stage 2
        classifier_df = Load.classifier_df_create(self.slice_path, self.folder_path)
        self.frac_index_lsit = classifier_df[classifier_df['frac']==1].index
        print("Stage classifier's dataframe:\n", classifier_df.head())
        self.classifier_val_data, self.classifier_valid_data_loader = Load.stage_classifier_data_loader(classifier_df, self.folder_path)
        img = Show.classifier_plot_img(self.classifier_val_data, 0)
        self.display_img(img, self.label_classifier_img)
        self.classifier_pred_list, self.classifier_GT, classifier_acc = Model.predict_classifier(self.classifier_valid_data_loader, 'classifier_stage2_1231.pth')
        
        fracture_df = Load.fracture_df_create(self.slice_path, self.folder_path)
        print("Stage fracture detection's dataframe:\n", fracture_df.head())
        self.frac_val_data, self.frac_valid_data_loader = Load.stage_fracture_detect_data_loader(fracture_df, self.folder_path)
        self.frac_pred_list = Model.predict_frac_detect(self.frac_valid_data_loader, "stage2bbox_resnet50_fpn.pth")
        # print(self.frac_pred_list)
        frac_image = Show.fracture_predict_plot_img(self.frac_val_data, 0, self.frac_pred_list)
        self.display_img(frac_image, self.label_frac_img)
        # show_acc_text = ""
        # if self.classifier_GT[0] == 1:
        #     show_acc_text += "Type: Fracture\n"
        # elif self.classifier_GT[0] == 0:
        #     show_acc_text += "Type: Normal\n"
        # if self.classifier_pred_list[0] == 1:
        #     show_acc_text += "Predicted as: Fracture\n"
        # elif self.classifier_pred_list[0] == 0:
        #     show_acc_text += "Predicted as: Normal\n"
        # self.label_Show_acc.setText(show_acc_text)
        # print(self.classifier_GT)
        # print(self.classifier_pred_list)
        # print(acc)
    

    def display_img(self, img, label):
        height, width, channel = img.shape
        bytesPerline = 3 * width
        qimg = QImage(img, width, height, bytesPerline, QImage.Format_RGB888).rgbSwapped()
        label.setPixmap(QPixmap.fromImage(qimg))

    def Scrollbar_action(self):
        # getting current value
        value = self.img_ScrollBar.value()
        if hasattr(self, 'val_data'):  
            ori_img = Show.stage1_plot_img(self.val_data, value) # initialize display windows
            self.display_img(ori_img, self.label_original_img)
        if hasattr(self, 'pred_list'):  
            pre_img = Show.stage1_predict_plot_img(self.pred_list, self.val_data, value)
            self.display_img(pre_img, self.label_detected_img)
        if hasattr(self, 'classifier_val_data'):  
            cls_img = Show.classifier_plot_img(self.classifier_val_data, value)
            self.display_img(cls_img, self.label_classifier_img)
            show_acc_text = ""
            if self.classifier_GT[value] == 1:
                show_acc_text += "Type: Fracture\n"
            elif self.classifier_GT[value] == 0:
                show_acc_text += "Type: Normal\n"
            if self.classifier_pred_list[value] == 1:
                show_acc_text += "Predicted as: Fracture\n"
            elif self.classifier_pred_list[value] == 0:
                show_acc_text += "Predicted as: Normal\n"
            self.label_Show_acc.setText(show_acc_text)
        if hasattr(self, 'frac_val_data'):  
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


        self.label_ScrollBar.setText(str(value))




if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = Ui()
    app.exec_()
