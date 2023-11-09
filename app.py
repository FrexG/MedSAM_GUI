import cv2
import threading
import numpy as np

import matplotlib

matplotlib.use("QtAgg")

from matplotlib.patches import Rectangle
from matplotlib.widgets import RectangleSelector
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon, QFont
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QToolBar,
    QWidget,
    QLabel,
    QFileDialog,
    QVBoxLayout,
    QPushButton,
)


import torch
import torchvision.transforms.functional as TF
from infer import infer, infer_auto, birads_infer, load_sam, load_unet, load_resnet


class Toolbar(QToolBar):
    def __init__(self, controller):
        super(Toolbar, self).__init__()

        self.controller = controller

        self.open_file = QPushButton("Open")
        self.open_file.setIcon(QIcon("./icons/folder.png"))
        self.open_file.clicked.connect(self.on_open_clicked)

        self.auto_segment_btn = QPushButton("Auto-Detect")
        self.auto_segment_btn.setIcon(QIcon("./icons/robot.png"))
        self.auto_segment_btn.clicked.connect(self.on_auto_btn_clicked)

        # draw box btn
        self.draw_selection_btn = QPushButton("Select")
        self.draw_selection_btn.setIcon(QIcon("./icons/add-selection.png"))
        self.draw_selection_btn.setEnabled(False)
        self.draw_selection_btn.clicked.connect(self.on_selection_box_clicked)

        self.cancer_probality = QLabel("Risk of Malignancy:")
        self.prob_lbl = QLabel("None")
        font = QFont()
        font.setPointSize(16)
        self.prob_lbl.setFont(font)

        self.addWidget(self.open_file)
        self.show()

    def show_controls(self):
        self.addWidget(self.auto_segment_btn)
        self.addWidget(self.draw_selection_btn)
        self.addWidget(self.cancer_probality)
        self.addWidget(self.prob_lbl)

    def on_open_clicked(self):
        self.controller.open_file()

    def on_auto_btn_clicked(self):
        # start inference on a new thread
        self.controller.image_widget.perform_auto_sam()

    def on_selection_box_clicked(self):
        self.controller.image_widget.perform_semiauto_sam()


class ImageWidget(QWidget):
    def __init__(self, controller):
        super(ImageWidget, self).__init__()
        self.controller = controller

        layout = QVBoxLayout()
        self.fig = Figure(figsize=(20, 20), dpi=100)
        self.fig.tight_layout()
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.mat_toolbar = NavigationToolbar2QT(self.canvas, self)

        self.x1 = 0
        self.x2 = 0
        self.y1 = 0
        self.y2 = 0

        layout.addWidget(self.mat_toolbar)
        layout.addWidget(self.canvas)

        self.setLayout(layout)
        self.show()

    def show_image(self, image_path):
        # create an axes
        self.image_np = cv2.imread(image_path)
        img_gray_scale = cv2.imread(image_path, 0)

        self.axes = self.fig.add_subplot(1, 1, 1)
        self.axes.imshow(self.image_np)
        self.canvas.draw()
        # get birads predictioin

        pred_class = birads_infer(TF.to_tensor(img_gray_scale), self.controller.resnet)
        self.controller.toolbar.prob_lbl.setText(pred_class)

    def perform_auto_sam(self):
        thread = threading.Thread(target=self.run_inference, args=("auto",))
        thread.start()

    def perform_semiauto_sam(self):
        """Attach a rectangle selector to an axes"""

        self.rect_selector = RectangleSelector(
            self.axes,
            self.select_callback,
            drawtype="line",
            useblit=True,
            button=[1],
            minspanx=5,
            minspany=5,
            spancoords="pixels",
            interactive=True,
        )

    def select_callback(self, eclick, erelease):
        """Callback for line selection.
        *eclick* and *erelease* are the press and release events.
        https://matplotlib.org/stable/gallery/widgets/rectangle_selector.html
        """
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata

        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

        thread = threading.Thread(target=self.run_inference, args=("semi",))
        thread.start()

        del self.rect_selector

        print(f"({x1:3.2f}, {y1:3.10f}) --> ({x2:3.2f}, {y2:3.10f})")

    def run_inference(self, infer_type="auto"):
        cv2_image = self.image_np.copy()

        if infer_type == "auto":
            tensor_img = TF.to_tensor(
                cv2.cvtColor(cv2_image, cv2.COLOR_BGR2GRAY)
            )  # min-max normalize and scale

            tensor_img = (tensor_img - tensor_img.min()) / (
                tensor_img.max() - tensor_img.min()
            )

            mask_pred = infer_auto(tensor_img, self.controller.elunet)
            mask_pred = (mask_pred > 0.7).float() * 255.0

            self.display_result(
                mask_pred, cv2_image, (cv2_image.shape[1], cv2_image.shape[0])
            )

        else:
            tensor_img = TF.to_tensor(self.image_np)

            tensor_img = (tensor_img - tensor_img.min()) / (
                tensor_img.max() - tensor_img.min()
            )

            mask_pred = infer(
                tensor_img,
                torch.tensor([self.x1, self.y1, self.x2, self.y2]),
                self.controller.sam,
            )
            self.display_result(
                mask_pred, cv2_image, (cv2_image.shape[1], cv2_image.shape[0])
            )
        # get class

    def display_result(self, mask_pred, cv2_image, display_shape):
        # reshape to channels last
        mask_pred = mask_pred[0].permute(1, 2, 0)
        # threshold mask
        mask_pred = np.uint8(mask_pred.numpy())
        # resize to original
        mask_pred = cv2.resize(mask_pred, display_shape)

        mask_pred = np.expand_dims(mask_pred, axis=-1)

        # find the contours of the mask
        contours, _ = cv2.findContours(
            mask_pred, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        ## color the mask
        # color_mask = np.zeros(
        #    (mask_pred.shape[0], mask_pred.shape[1], 3), dtype=np.uint8
        # )
        # color_mask[:, :] = (255, 55, 5)
        ## bit wise and
        # color_mask = cv2.bitwise_and(color_mask, color_mask, mask=mask_pred)

        # output_image = cv2.addWeighted(cv2_image, 0.8, color_mask, 0.2, 0.0)

        output_image = cv2.drawContours(cv2_image, contours, -1, (255, 5, 0), 2)

        self.axes.imshow(output_image)
        self.canvas.draw()


class MedsamApp(QMainWindow):
    def __init__(self):
        super(MedsamApp, self).__init__()
        self.setWindowTitle("PIERWSI MedSAM")
        self.resize(1024, 720)

        self.elunet = load_unet()
        self.sam = load_sam()
        self.resnet = load_resnet()
        # define a layout
        layout = QVBoxLayout()
        self.toolbar = Toolbar(self)
        self.image_widget = ImageWidget(self)

        self.addToolBar(self.toolbar)
        layout.addWidget(self.image_widget)

        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

    def open_file(self):
        self.image_widget.fig.clear()
        # Open a file dialog
        file_filters = ".png files (*.png);;.jpg files (*.jpg);;.jpeg files (*.jpeg)"

        self.filename = QFileDialog.getOpenFileName(self)[0]

        if not self.filename:
            return

        if self.filename:
            self.toolbar.show_controls()
            self.toolbar.draw_selection_btn.setEnabled(True)
            self.image_widget.show_image(self.filename)


app = QApplication([])
window = MedsamApp()
window.show()

app.exec()
