import os
import sys
import time
import json
import csv
import math
from urllib.parse import quote
import webbrowser
import requests
import numpy as np
import cv2

# PyQt5 imports
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QFileDialog, QGraphicsView, QGraphicsScene, QMessageBox, QInputDialog, QTreeWidget, 
    QTreeWidgetItem, QCheckBox, QDialog, QFormLayout, QSpinBox, QDialogButtonBox, QGridLayout,
    QGraphicsEllipseItem, QGraphicsLineItem, QGraphicsRectItem, QGraphicsPathItem, 
    QColorDialog, QFontDialog, QStyle, QSlider, QTabWidget, QScrollArea, QSizePolicy, QSpacerItem, QGraphicsTextItem
)
from PyQt5.QtGui import (
    QPixmap, QImage, QPainter, QPen, QColor, QTransform, QIcon, QPainterPath, QFont, QMovie
)
from PyQt5.QtCore import Qt, QRectF, QLineF, QPointF, QThread, pyqtSignal, QCoreApplication, QPoint

# PIL (Pillow) imports
from PIL import Image, ImageDraw, ImageFont

# Astropy and Astroquery imports
from astropy.io import fits
import tifffile as tiff
from astroquery.simbad import Simbad
from astroquery.mast import Mast
from astroquery.vizier import Vizier
from astropy.coordinates import SkyCoord, Angle
import astropy.units as u
from astropy.wcs import WCS
from astropy.utils.data import conf

# Math imports
from math import sqrt



class AstroEditingSuite(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        # Create tab widget
        self.tabs = QTabWidget()

        # Add individual tabs for each tool
        self.tabs.addTab(StatisticalStretchTab(), "Statistical Stretch")
        self.tabs.addTab(NBtoRGBstarsTab(), "NB to RGB Stars")  # Placeholder        
        self.tabs.addTab(StarStretchTab(), "Star Stretch")  # Placeholder
        self.tabs.addTab(HaloBGonTab(), "Halo-B-Gon")  # Placeholder
        self.tabs.addTab(ContinuumSubtractTab(), "Continuum Subtraction")
        self.tabs.addTab(MainWindow(),"What's In My Image")
        

        # Add the tab widget to the main layout
        layout.addWidget(self.tabs)

        # Set the layout for the main window
        self.setLayout(layout)
        self.setWindowTitle('Seti Astro\'s Suite V1.3')


class StatisticalStretchTab(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.image = None
        self.filename = None
        self.zoom_factor = 1.0
        self.original_header = None

    def initUI(self):
        main_layout = QHBoxLayout()

        # Left column for controls
        left_widget = QWidget(self)
        left_layout = QVBoxLayout(left_widget)
        left_widget.setFixedWidth(400)  # You can adjust this width as needed

        instruction_box = QLabel(self)
        instruction_box.setText("""
            Instructions:
            1. Select an image to stretch.
            2. Adjust the target median and optional settings.
            3. Preview the result.
            4. Save the stretched image in your desired format.
        """)
        instruction_box.setWordWrap(True)
        left_layout.addWidget(instruction_box)

        # File selection button
        self.fileButton = QPushButton('Select Image', self)
        self.fileButton.clicked.connect(self.openFileDialog)
        left_layout.addWidget(self.fileButton)

        self.fileLabel = QLabel('No file selected', self)
        left_layout.addWidget(self.fileLabel)

        # Target median slider
        self.medianLabel = QLabel('Target Median: 0.25', self)
        self.medianSlider = QSlider(Qt.Horizontal)
        self.medianSlider.setMinimum(1)
        self.medianSlider.setMaximum(100)
        self.medianSlider.setValue(25)
        self.medianSlider.valueChanged.connect(self.updateMedianLabel)
        left_layout.addWidget(self.medianLabel)
        left_layout.addWidget(self.medianSlider)

        # Linked/Unlinked stretch checkbox
        self.linkedCheckBox = QCheckBox('Linked Stretch', self)
        self.linkedCheckBox.setChecked(True)
        left_layout.addWidget(self.linkedCheckBox)

        # Normalization checkbox
        self.normalizeCheckBox = QCheckBox('Normalize Image', self)
        left_layout.addWidget(self.normalizeCheckBox)

        # Curves adjustment checkbox
        self.curvesCheckBox = QCheckBox('Apply Curves Adjustment', self)
        self.curvesCheckBox.stateChanged.connect(self.toggleCurvesSlider)
        left_layout.addWidget(self.curvesCheckBox)

        # Curves Boost slider (initially hidden)
        self.curvesBoostLabel = QLabel('Curves Boost: 0.00', self)
        self.curvesBoostSlider = QSlider(Qt.Horizontal)
        self.curvesBoostSlider.setMinimum(0)
        self.curvesBoostSlider.setMaximum(50)
        self.curvesBoostSlider.setValue(0)
        self.curvesBoostSlider.valueChanged.connect(self.updateCurvesBoostLabel)
        self.curvesBoostLabel.hide()
        self.curvesBoostSlider.hide()

        left_layout.addWidget(self.curvesBoostLabel)
        left_layout.addWidget(self.curvesBoostSlider)

        # Progress indicator (spinner) label
        self.spinnerLabel = QLabel(self)
        self.spinnerLabel.setAlignment(Qt.AlignCenter)
        # Use the resource path function to access the GIF
        self.spinnerMovie = QMovie(resource_path("spinner.gif"))  # Updated path
        self.spinnerLabel.setMovie(self.spinnerMovie)
        self.spinnerLabel.hide()  # Hide spinner by default
        left_layout.addWidget(self.spinnerLabel)      

        # Preview button
        self.previewButton = QPushButton('Preview Stretch', self)
        self.previewButton.clicked.connect(self.previewStretch)
        left_layout.addWidget(self.previewButton)

        # Zoom buttons
        zoom_layout = QHBoxLayout()
        self.zoomInButton = QPushButton('Zoom In', self)
        self.zoomInButton.clicked.connect(self.zoom_in)
        zoom_layout.addWidget(self.zoomInButton)

        self.zoomOutButton = QPushButton('Zoom Out', self)
        self.zoomOutButton.clicked.connect(self.zoom_out)
        zoom_layout.addWidget(self.zoomOutButton)

        left_layout.addLayout(zoom_layout)

        # Save button
        self.saveButton = QPushButton('Save Stretched Image', self)
        self.saveButton.clicked.connect(self.saveImage)
        left_layout.addWidget(self.saveButton)

                        # Footer
        footer_label = QLabel("""
            Written by Franklin Marek<br>
            <a href='http://www.setiastro.com'>www.setiastro.com</a>
        """)
        footer_label.setAlignment(Qt.AlignCenter)
        footer_label.setOpenExternalLinks(True)
        footer_label.setStyleSheet("font-size: 10px;")
        left_layout.addWidget(footer_label)

        left_layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

        # Add the left widget to the main layout
        main_layout.addWidget(left_widget)

        # Right side for the preview inside a QScrollArea
        self.scrollArea = QScrollArea(self)
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scrollArea.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        # QLabel for the image preview
        self.imageLabel = QLabel(self)
        self.imageLabel.setAlignment(Qt.AlignCenter)
        self.scrollArea.setWidget(self.imageLabel)
        self.scrollArea.setMinimumSize(400, 400)

        main_layout.addWidget(self.scrollArea)

        self.setLayout(main_layout)
        self.zoom_factor = 0.25
        self.scrollArea.viewport().setMouseTracking(True)
        self.scrollArea.viewport().installEventFilter(self)
        self.dragging = False
        self.last_pos = QPoint()

    def eventFilter(self, source, event):
        if event.type() == event.MouseButtonPress and event.button() == Qt.LeftButton:
            self.dragging = True
            self.last_pos = event.pos()
        elif event.type() == event.MouseButtonRelease and event.button() == Qt.LeftButton:
            self.dragging = False
        elif event.type() == event.MouseMove and self.dragging:
            delta = event.pos() - self.last_pos
            self.scrollArea.horizontalScrollBar().setValue(self.scrollArea.horizontalScrollBar().value() - delta.x())
            self.scrollArea.verticalScrollBar().setValue(self.scrollArea.verticalScrollBar().value() - delta.y())
            self.last_pos = event.pos()

        return super().eventFilter(source, event)


    def openFileDialog(self):
        self.filename, _ = QFileDialog.getOpenFileName(self, 'Open Image File', '', 'Images (*.png *.tiff *.tif *.fits *.fit);;All Files (*)')
        if self.filename:
            self.fileLabel.setText(self.filename)
            # Unpack all four values returned by load_image
            self.image, self.original_header, self.bit_depth, self.is_mono = load_image(self.filename)


    def updateMedianLabel(self, value):
        self.medianLabel.setText(f'Target Median: {value / 100:.2f}')

    def updateCurvesBoostLabel(self, value):
        self.curvesBoostLabel.setText(f'Curves Boost: {value / 100:.2f}')

    def toggleCurvesSlider(self, state):
        if state == Qt.Checked:
            self.curvesBoostLabel.show()
            self.curvesBoostSlider.show()
        else:
            self.curvesBoostLabel.hide()
            self.curvesBoostSlider.hide()

    def previewStretch(self):
        if self.image is not None:
            # Show spinner before starting processing
            self.showSpinner()

            # Start background processing
            self.processing_thread = StatisticalStretchProcessingThread(self.image,
                                                                        self.medianSlider.value(),
                                                                        self.linkedCheckBox.isChecked(),
                                                                        self.normalizeCheckBox.isChecked(),
                                                                        self.curvesCheckBox.isChecked(),
                                                                        self.curvesBoostSlider.value() / 100.0)
            self.processing_thread.preview_generated.connect(self.update_preview)
            self.processing_thread.start()

    def update_preview(self, stretched_image):
        # Save the stretched image for later use in zoom functions
        self.stretched_image = stretched_image

        # Update the preview once the processing thread emits the result
        img = (stretched_image * 255).astype(np.uint8)
        h, w = img.shape[:2]

        if img.ndim == 3:
            bytes_per_line = 3 * w
            q_image = QImage(img.tobytes(), w, h, bytes_per_line, QImage.Format_RGB888)
        else:
            bytes_per_line = w
            q_image = QImage(img.tobytes(), w, h, bytes_per_line, QImage.Format_Grayscale8)

        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(pixmap.size() * self.zoom_factor, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.imageLabel.setPixmap(scaled_pixmap)
        self.imageLabel.resize(scaled_pixmap.size())

        # Hide the spinner after processing is done
        self.hideSpinner()


    def showSpinner(self):
        self.spinnerLabel.show()
        self.spinnerMovie.start()

    def hideSpinner(self):
        self.spinnerLabel.hide()
        self.spinnerMovie.stop()    

    def zoom_in(self):
        if hasattr(self, 'stretched_image'):
            self.zoom_factor *= 1.2
            self.update_preview(self.stretched_image)  # Pass the latest stretched image

    def zoom_out(self):
        if hasattr(self, 'stretched_image'):
            self.zoom_factor /= 1.2
            self.update_preview(self.stretched_image)  # Pass the latest stretched image



    def saveImage(self):
        if hasattr(self, 'stretched_image') and self.stretched_image is not None:
            # Pre-populate the save dialog with the original image name
            base_name = os.path.basename(self.filename)
            default_save_name = os.path.splitext(base_name)[0] + '_stretched.tif'
            original_dir = os.path.dirname(self.filename)

            # Open the save file dialog
            save_filename, _ = QFileDialog.getSaveFileName(
                self, 
                'Save Image As', 
                os.path.join(original_dir, default_save_name), 
                'Images (*.tiff *.tif *.png *.fit *.fits);;All Files (*)'
            )

            if save_filename:
                original_format = save_filename.split('.')[-1].lower()

                # For TIFF and FITS files, prompt the user to select the bit depth
                if original_format in ['tiff', 'tif', 'fits', 'fit']:
                    bit_depth_options = ["16-bit", "32-bit unsigned", "32-bit floating point"]
                    bit_depth, ok = QInputDialog.getItem(self, "Select Bit Depth", "Choose bit depth for saving:", bit_depth_options, 0, False)
                    
                    if ok and bit_depth:
                        # Call save_image with the necessary parameters
                        save_image(self.stretched_image, save_filename, original_format, bit_depth, self.original_header, self.is_mono)
                        self.fileLabel.setText(f'Image saved as: {save_filename}')
                    else:
                        self.fileLabel.setText('Save canceled.')
                else:
                    # For non-TIFF/FITS formats, save directly without bit depth selection
                    save_image(self.stretched_image, save_filename, original_format)
                    self.fileLabel.setText(f'Image saved as: {save_filename}')
            else:
                self.fileLabel.setText('Save canceled.')
        else:
            self.fileLabel.setText('No stretched image to save. Please generate a preview first.')



# Thread for Stat Stretch background processing
class StatisticalStretchProcessingThread(QThread):
    preview_generated = pyqtSignal(np.ndarray)  # Signal to send the generated preview image back to the main thread

    def __init__(self, image, target_median, linked, normalize, apply_curves, curves_boost):
        super().__init__()
        self.image = image
        self.target_median = target_median / 100.0  # Ensure proper scaling
        self.linked = linked
        self.normalize = normalize
        self.apply_curves = apply_curves
        self.curves_boost = curves_boost

    def run(self):
        # Perform the image stretching in the background
        if self.image.ndim == 2:  # Mono image
            stretched_image = stretch_mono_image(self.image, self.target_median, self.normalize, self.apply_curves, self.curves_boost)
        else:  # Color image
            stretched_image = stretch_color_image(self.image, self.target_median, self.linked, self.normalize, self.apply_curves, self.curves_boost)

        # Emit the result once done
        self.preview_generated.emit(stretched_image)

# Thread for star stretch background processing
class ProcessingThread(QThread):
    preview_generated = pyqtSignal(np.ndarray)

    def __init__(self, image, stretch_factor, sat_amount, scnr_enabled):
        super().__init__()
        self.image = image
        self.stretch_factor = stretch_factor
        self.sat_amount = sat_amount
        self.scnr_enabled = scnr_enabled

    def run(self):
        stretched_image = self.applyPixelMath(self.image, self.stretch_factor)
        stretched_image = self.applyColorSaturation(stretched_image, self.sat_amount)
        if self.scnr_enabled:
            stretched_image = self.applySCNR(stretched_image)
        self.preview_generated.emit(stretched_image)

    def applyPixelMath(self, image_array, amount):
        expression = (3 ** amount * image_array) / ((3 ** amount - 1) * image_array + 1)
        return np.clip(expression, 0, 1)

    def applyColorSaturation(self, image_array, satAmount):
        saturationLevel = [
            [0.0, satAmount * 0.4],
            [0.5, satAmount * 0.7],
            [1.0, satAmount * 0.4]
        ]
        return self.adjust_saturation(image_array, saturationLevel)

    def adjust_saturation(self, image_array, saturation_level):
        hsv_image = np.array(Image.fromarray((image_array * 255).astype(np.uint8)).convert('HSV')) / 255.0
        hsv_image[..., 1] *= saturation_level[1][1]
        hsv_image[..., 1] = np.clip(hsv_image[..., 1], 0, 1)
        rgb_image = Image.fromarray((hsv_image * 255).astype(np.uint8), 'HSV').convert('RGB')
        return np.array(rgb_image) / 255.0

    def applySCNR(self, image_array):
        red_channel = image_array[..., 0]
        green_channel = image_array[..., 1]
        blue_channel = image_array[..., 2]

        # Apply green neutralization where green is higher than red and blue
        mask = green_channel > np.maximum(red_channel, blue_channel)
        green_channel[mask] = np.maximum(red_channel[mask], blue_channel[mask])

        # Recombine the channels
        image_array[..., 1] = green_channel
        return np.clip(image_array, 0, 1)


class StarStretchTab(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.image = None  # Store the selected image
        self.stretch_factor = 5.0
        self.sat_amount = 1.0
        self.is_mono = True
        self.remove_green = False
        self.filename = None  # Store the selected file path
        self.preview_image = None  # Store the preview result
        self.zoom_factor = 0.25  # Initialize zoom factor for preview scaling
        self.dragging = False
        self.last_pos = None
        self.processing_thread = None  # Thread for background processing
        self.original_header = None

    def initUI(self):
        main_layout = QHBoxLayout()

        # Left column for controls
        left_widget = QWidget(self)
        left_layout = QVBoxLayout(left_widget)
        left_widget.setFixedWidth(400)  # Fix the left column width

        instruction_box = QLabel(self)
        instruction_box.setText("""
            Instructions:
            1. Select a stars-only image.
            2. Adjust the stretch and optional settings.
            3. Preview the result.
        """)
        instruction_box.setWordWrap(True)
        left_layout.addWidget(instruction_box)

        # File selection button
        self.fileButton = QPushButton("Select Stars Only Image", self)
        self.fileButton.clicked.connect(self.selectImage)
        left_layout.addWidget(self.fileButton)

        self.fileLabel = QLabel('No file selected', self)
        left_layout.addWidget(self.fileLabel)

        # Stretch Amount slider with more precision
        self.stretchLabel = QLabel("Stretch Amount: 5.00", self)
        self.stretchSlider = QSlider(Qt.Horizontal)
        self.stretchSlider.setMinimum(0)
        self.stretchSlider.setMaximum(800)  # Allow two decimal places of precision
        self.stretchSlider.setValue(500)  # 500 corresponds to 5.00
        self.stretchSlider.valueChanged.connect(self.updateStretchLabel)
        left_layout.addWidget(self.stretchLabel)
        left_layout.addWidget(self.stretchSlider)

        # Color Boost Amount slider
        self.satLabel = QLabel("Color Boost: 1.00", self)
        self.satSlider = QSlider(Qt.Horizontal)
        self.satSlider.setMinimum(0)
        self.satSlider.setMaximum(200)
        self.satSlider.setValue(100)  # 100 corresponds to 1.0 boost
        self.satSlider.valueChanged.connect(self.updateSatLabel)
        left_layout.addWidget(self.satLabel)
        left_layout.addWidget(self.satSlider)

        # SCNR checkbox
        self.scnrCheckBox = QCheckBox("Remove Green via SCNR (Optional)", self)
        left_layout.addWidget(self.scnrCheckBox)

        # Refresh preview button
        self.refreshButton = QPushButton("Refresh Preview", self)
        self.refreshButton.clicked.connect(self.generatePreview)
        left_layout.addWidget(self.refreshButton)

        # Progress indicator (spinner) label
        self.spinnerLabel = QLabel(self)
        self.spinnerLabel.setAlignment(Qt.AlignCenter)
        # Use the resource path function to access the GIF
        self.spinnerMovie = QMovie(resource_path("spinner.gif"))  # Updated path
        self.spinnerLabel.setMovie(self.spinnerMovie)
        self.spinnerLabel.hide()  # Hide spinner by default
        left_layout.addWidget(self.spinnerLabel)

        # Zoom buttons for preview
        zoom_layout = QHBoxLayout()
        self.zoomInButton = QPushButton('Zoom In', self)
        self.zoomInButton.clicked.connect(self.zoom_in)
        zoom_layout.addWidget(self.zoomInButton)

        self.zoomOutButton = QPushButton('Zoom Out', self)
        self.zoomOutButton.clicked.connect(self.zoom_out)
        zoom_layout.addWidget(self.zoomOutButton)
        left_layout.addLayout(zoom_layout)

        # Save As button (replaces Execute button)
        self.saveAsButton = QPushButton("Save As", self)
        self.saveAsButton.clicked.connect(self.saveImage)
        left_layout.addWidget(self.saveAsButton)

                        # Footer
        footer_label = QLabel("""
            Written by Franklin Marek<br>
            <a href='http://www.setiastro.com'>www.setiastro.com</a>
        """)
        footer_label.setAlignment(Qt.AlignCenter)
        footer_label.setOpenExternalLinks(True)
        footer_label.setStyleSheet("font-size: 10px;")
        left_layout.addWidget(footer_label)

        left_layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

        # Add the left widget to the main layout
        main_layout.addWidget(left_widget)

        # Right side for the preview inside a QScrollArea
        self.scrollArea = QScrollArea(self)
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scrollArea.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scrollArea.viewport().installEventFilter(self)

        # QLabel for the image preview
        self.imageLabel = QLabel(self)
        self.imageLabel.setAlignment(Qt.AlignCenter)
        self.scrollArea.setWidget(self.imageLabel)
        self.scrollArea.setMinimumSize(400, 400)

        main_layout.addWidget(self.scrollArea)

        self.setLayout(main_layout)

    def saveImage(self):
        # Use the processed/stretched image for saving
        if hasattr(self, 'stretched_image') and self.stretched_image is not None:
            # Pre-populate the save dialog with the original image name
            base_name = os.path.basename(self.filename)
            default_save_name = os.path.splitext(base_name)[0] + '_stretched.tif'
            original_dir = os.path.dirname(self.filename)

            # Open the save file dialog
            save_filename, _ = QFileDialog.getSaveFileName(
                self, 
                'Save Image As', 
                os.path.join(original_dir, default_save_name), 
                'Images (*.tiff *.tif *.png *.fit *.fits);;All Files (*)'
            )

            if save_filename:
                original_format = save_filename.split('.')[-1].lower()

                # For TIFF and FITS files, prompt the user to select the bit depth
                if original_format in ['tiff', 'tif', 'fits', 'fit']:
                    bit_depth_options = ["16-bit", "32-bit unsigned", "32-bit floating point"]
                    bit_depth, ok = QInputDialog.getItem(self, "Select Bit Depth", "Choose bit depth for saving:", bit_depth_options, 0, False)
                    
                    if ok and bit_depth:
                        # Call save_image with the necessary parameters
                        save_image(self.stretched_image, save_filename, original_format, bit_depth, self.original_header, self.is_mono)
                        self.fileLabel.setText(f'Image saved as: {save_filename}')
                    else:
                        self.fileLabel.setText('Save canceled.')
                else:
                    # For non-TIFF/FITS formats, save directly without bit depth selection
                    save_image(self.stretched_image, save_filename, original_format)
                    self.fileLabel.setText(f'Image saved as: {save_filename}')
            else:
                self.fileLabel.setText('Save canceled.')
        else:
            self.fileLabel.setText('No stretched image to save. Please generate a preview first.')


    def selectImage(self):
        selected_file, _ = QFileDialog.getOpenFileName(self, "Select Stars Only Image", "", "Images (*.png *.tif *.fits *.fit)")
        if selected_file:
            try:
                self.image, self.original_header, _, self.is_mono = load_image(selected_file)  # Load image with header
                self.filename = selected_file  # Store the selected file path
                self.fileLabel.setText(os.path.basename(selected_file))
                self.generatePreview()

            except Exception as e:
                self.fileLabel.setText(f"Error: {str(e)}")
                print(f"Failed to load image: {e}")

    def updateStretchLabel(self, value):
        self.stretch_factor = value / 100.0  # Precision of two decimals
        self.stretchLabel.setText(f"Stretch Amount: {self.stretch_factor:.2f}")

    def updateSatLabel(self, value):
        self.sat_amount = value / 100.0
        self.satLabel.setText(f"Color Boost: {self.sat_amount:.2f}")

    def generatePreview(self):
        if self.image is not None and self.image.size > 0:
            # Show spinner before starting processing
            self.showSpinner()

            # Start background processing
            self.processing_thread = ProcessingThread(self.image, self.stretch_factor, self.sat_amount, self.scnrCheckBox.isChecked())
            self.processing_thread.preview_generated.connect(self.updatePreview)
            self.processing_thread.start()

    def updatePreview(self, stretched_image):
        # Store the stretched image for saving
        self.stretched_image = stretched_image

        # Update the preview once the processing thread emits the result
        preview_image = (stretched_image * 255).astype(np.uint8)
        h, w = preview_image.shape[:2]
        if preview_image.ndim == 3:
            q_image = QImage(preview_image.data, w, h, 3 * w, QImage.Format_RGB888)
        else:
            q_image = QImage(preview_image.data, w, h, w, QImage.Format_Grayscale8)

        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(pixmap.size() * self.zoom_factor, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.imageLabel.setPixmap(scaled_pixmap)
        self.imageLabel.resize(scaled_pixmap.size())

        # Hide the spinner after processing is done
        self.hideSpinner()

    def eventFilter(self, source, event):
        if event.type() == event.MouseButtonPress and event.button() == Qt.LeftButton:
            self.dragging = True
            self.last_pos = event.pos()
        elif event.type() == event.MouseButtonRelease and event.button() == Qt.LeftButton:
            self.dragging = False
        elif event.type() == event.MouseMove and self.dragging:
            delta = event.pos() - self.last_pos
            self.scrollArea.horizontalScrollBar().setValue(self.scrollArea.horizontalScrollBar().value() - delta.x())
            self.scrollArea.verticalScrollBar().setValue(self.scrollArea.verticalScrollBar().value() - delta.y())
            self.last_pos = event.pos()

        return super().eventFilter(source, event)
    

    def showSpinner(self):
        self.spinnerLabel.show()
        self.spinnerMovie.start()

    def hideSpinner(self):
        self.spinnerLabel.hide()
        self.spinnerMovie.stop()

    def zoom_in(self):
        self.zoom_factor *= 1.2
        self.generatePreview()

    def zoom_out(self):
        self.zoom_factor /= 1.2
        self.generatePreview()

    def applyStretch(self):
        if self.image is not None and self.image.size > 0:
            print(f"Applying stretch: {self.stretch_factor}, Color Boost: {self.sat_amount:.2f}, SCNR: {self.scnrCheckBox.isChecked()}")
            self.generatePreview()

class NBtoRGBstarsTab(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.ha_image = None
        self.oiii_image = None
        self.sii_image = None
        self.osc_image = None
        self.combined_image = None
        self.is_mono = False
        self.filename = None  # Store the selected file path
        self.zoom_factor = 0.25
        self.dragging = False
        self.last_pos = QPoint()
        self.processing_thread = None
        self.original_header = None

    def initUI(self):
        main_layout = QHBoxLayout()

        # Left column for controls
        left_widget = QWidget(self)
        left_layout = QVBoxLayout(left_widget)
        left_widget.setFixedWidth(400)

        instruction_box = QLabel(self)
        instruction_box.setText("""
            Instructions:
            1. Select Ha, OIII, and SII (optional) narrowband images, or an OSC stars-only image.
            2. Adjust the Ha to OIII Ratio if needed.
            3. Preview the combined result.
            4. Save the final composite image.
        """)
        instruction_box.setWordWrap(True)
        left_layout.addWidget(instruction_box)

        # Ha, OIII, SII image selections
        self.haButton = QPushButton('Select Ha Image', self)
        self.haButton.clicked.connect(lambda: self.selectImage('Ha'))
        left_layout.addWidget(self.haButton)
        self.haLabel = QLabel('No Ha image selected', self)
        left_layout.addWidget(self.haLabel)

        self.oiiiButton = QPushButton('Select OIII Image', self)
        self.oiiiButton.clicked.connect(lambda: self.selectImage('OIII'))
        left_layout.addWidget(self.oiiiButton)
        self.oiiiLabel = QLabel('No OIII image selected', self)
        left_layout.addWidget(self.oiiiLabel)

        self.siiButton = QPushButton('Select SII Image (Optional)', self)
        self.siiButton.clicked.connect(lambda: self.selectImage('SII'))
        left_layout.addWidget(self.siiButton)
        self.siiLabel = QLabel('No SII image selected', self)
        left_layout.addWidget(self.siiLabel)

        self.oscButton = QPushButton('Select OSC Stars Image (Optional)', self)
        self.oscButton.clicked.connect(lambda: self.selectImage('OSC'))
        left_layout.addWidget(self.oscButton)
        self.oscLabel = QLabel('No OSC stars image selected', self)
        left_layout.addWidget(self.oscLabel)

        # Ha to OIII Ratio slider
        self.haToOiiRatioLabel, self.haToOiiRatioSlider = self.createRatioSlider("Ha to OIII Ratio", 30)
        left_layout.addWidget(self.haToOiiRatioLabel)
        left_layout.addWidget(self.haToOiiRatioSlider)

        # Star Stretch checkbox and sliders
        self.starStretchCheckBox = QCheckBox("Enable Star Stretch", self)
        self.starStretchCheckBox.setChecked(True)
        self.starStretchCheckBox.toggled.connect(self.toggleStarStretchControls)
        left_layout.addWidget(self.starStretchCheckBox)

        self.stretchSliderLabel, self.stretchSlider = self.createStretchSlider("Stretch Factor", 5.0)
        left_layout.addWidget(self.stretchSliderLabel)
        left_layout.addWidget(self.stretchSlider)

        # Progress indicator (spinner) label
        self.spinnerLabel = QLabel(self)
        self.spinnerLabel.setAlignment(Qt.AlignCenter)
        # Use the resource path function to access the GIF
        self.spinnerMovie = QMovie(resource_path("spinner.gif"))  # Updated path
        self.spinnerLabel.setMovie(self.spinnerMovie)
        self.spinnerLabel.hide()  # Hide spinner by default
        left_layout.addWidget(self.spinnerLabel)

        # Preview and Save buttons
        self.previewButton = QPushButton('Preview Combined Image', self)
        self.previewButton.clicked.connect(self.previewCombine)
        left_layout.addWidget(self.previewButton)

        # File label for displaying save status
        self.fileLabel = QLabel('', self)
        left_layout.addWidget(self.fileLabel)

        self.saveButton = QPushButton('Save Combined Image', self)
        self.saveButton.clicked.connect(self.saveImage)
        left_layout.addWidget(self.saveButton)

                        # Footer
        footer_label = QLabel("""
            Written by Franklin Marek<br>
            <a href='http://www.setiastro.com'>www.setiastro.com</a>
        """)
        footer_label.setAlignment(Qt.AlignCenter)
        footer_label.setOpenExternalLinks(True)
        footer_label.setStyleSheet("font-size: 10px;")
        left_layout.addWidget(footer_label)

        left_layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))
        main_layout.addWidget(left_widget)

        # Right side for the preview inside a QScrollArea
        self.scrollArea = QScrollArea(self)
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scrollArea.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.imageLabel = QLabel(self)
        self.imageLabel.setAlignment(Qt.AlignCenter)
        self.scrollArea.setWidget(self.imageLabel)
        self.scrollArea.setMinimumSize(400, 400)
        main_layout.addWidget(self.scrollArea)

        self.setLayout(main_layout)
        self.scrollArea.viewport().setMouseTracking(True)
        self.scrollArea.viewport().installEventFilter(self)

    def createRatioSlider(self, label_text, default_value):
        label = QLabel(f"{label_text}: {default_value / 100:.2f}", self)
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(0)
        slider.setMaximum(100)
        slider.setValue(default_value)
        slider.valueChanged.connect(lambda value: label.setText(f"{label_text}: {value / 100:.2f}"))
        return label, slider

    def createStretchSlider(self, label_text, default_value):
        label = QLabel(f"{label_text}: {default_value:.2f}", self)
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(0)
        slider.setMaximum(800)
        slider.setValue(int(default_value * 100))  # Scale to handle float values
        slider.valueChanged.connect(lambda value: label.setText(f"{label_text}: {value / 100:.2f}"))
        return label, slider

    def toggleStarStretchControls(self):
        enabled = self.starStretchCheckBox.isChecked()
        self.stretchSliderLabel.setVisible(enabled)
        self.stretchSlider.setVisible(enabled)

    def selectImage(self, image_type):
        selected_file, _ = QFileDialog.getOpenFileName(self, f"Select {image_type} Image", "", "Images (*.png *.tif *.fits *.fit)")
        if selected_file:
            try:
                if image_type == 'Ha':
                    self.ha_image, self.original_header, _, _ = load_image(selected_file)  # Store header
                    self.filename = selected_file
                    self.haLabel.setText(f"{os.path.basename(selected_file)} selected")
                elif image_type == 'OIII':
                    self.oiii_image, self.original_header, _, _ = load_image(selected_file)
                    self.oiiiLabel.setText(f"{os.path.basename(selected_file)} selected")
                elif image_type == 'SII':
                    self.sii_image, self.original_header, _, _ = load_image(selected_file)
                    self.siiLabel.setText(f"{os.path.basename(selected_file)} selected")
                elif image_type == 'OSC':
                    self.osc_image, self.original_header, _, _ = load_image(selected_file)
                    self.oscLabel.setText(f"{os.path.basename(selected_file)} selected")
            except Exception as e:
                print(f"Failed to load {image_type} image: {e}")


    def previewCombine(self):
        ha_to_oii_ratio = self.haToOiiRatioSlider.value() / 100.0
        enable_star_stretch = self.starStretchCheckBox.isChecked()
        stretch_factor = self.stretchSlider.value() / 100.0

        # Show spinner before starting processing
        self.showSpinner()

        # Start background processing
        self.processing_thread = NBtoRGBProcessingThread(
            self.ha_image, self.oiii_image, self.sii_image, self.osc_image,
            ha_to_oii_ratio=ha_to_oii_ratio, enable_star_stretch=enable_star_stretch, stretch_factor=stretch_factor
        )
        self.processing_thread.preview_generated.connect(self.updatePreview)
        self.processing_thread.start()

    def updatePreview(self, combined_image):
        # Set the combined image for saving
        self.combined_image = combined_image

        # Convert the image to display format
        preview_image = (combined_image * 255).astype(np.uint8)
        h, w = preview_image.shape[:2]
        q_image = QImage(preview_image.data, w, h, 3 * w, QImage.Format_RGB888)

        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(pixmap.size() * self.zoom_factor, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.imageLabel.setPixmap(scaled_pixmap)
        self.imageLabel.resize(scaled_pixmap.size())

        # Hide the spinner after processing is done
        self.hideSpinner()

    def showSpinner(self):
        self.spinnerLabel.show()
        self.spinnerMovie.start()

    def hideSpinner(self):
        self.spinnerLabel.hide()
        self.spinnerMovie.stop()

    def saveImage(self):
        if self.combined_image is not None:
            # Pre-populate the save dialog with the original image name
            base_name = os.path.basename(self.filename) if self.filename else "output"
            default_save_name = 'NBtoRGBstars.tif'
            original_dir = os.path.dirname(self.filename) if self.filename else ""

            # Open the save file dialog
            save_filename, _ = QFileDialog.getSaveFileName(
                self, 
                'Save Image As', 
                os.path.join(original_dir, default_save_name), 
                'Images (*.tiff *.tif *.png *.fit *.fits);;All Files (*)'
            )

            if save_filename:
                original_format = save_filename.split('.')[-1].lower()

                # For TIFF and FITS files, prompt the user to select the bit depth
                if original_format in ['tiff', 'tif', 'fits', 'fit']:
                    bit_depth_options = ["16-bit", "32-bit unsigned", "32-bit floating point"]
                    bit_depth, ok = QInputDialog.getItem(self, "Select Bit Depth", "Choose bit depth for saving:", bit_depth_options, 0, False)
                    
                    if ok and bit_depth:
                        # Call save_image with the necessary parameters
                        save_image(self.combined_image, save_filename, original_format, bit_depth, self.original_header, self.is_mono)
                        self.fileLabel.setText(f'Image saved as: {save_filename}')
                    else:
                        self.fileLabel.setText('Save canceled.')
                else:
                    # For non-TIFF/FITS formats, save directly without bit depth selection
                    save_image(self.combined_image, save_filename, original_format)
                    self.fileLabel.setText(f'Image saved as: {save_filename}')
            else:
                self.fileLabel.setText('Save canceled.')
        else:
            self.fileLabel.setText("No combined image to save.")




    # Add event filter for mouse dragging in preview area
    def eventFilter(self, source, event):
        if event.type() == event.MouseButtonPress and event.button() == Qt.LeftButton:
            self.dragging = True
            self.last_pos = event.pos()
        elif event.type() == event.MouseButtonRelease and event.button() == Qt.LeftButton:
            self.dragging = False
        elif event.type() == event.MouseMove and self.dragging:
            delta = event.pos() - self.last_pos
            self.scrollArea.horizontalScrollBar().setValue(self.scrollArea.horizontalScrollBar().value() - delta.x())
            self.scrollArea.verticalScrollBar().setValue(self.scrollArea.verticalScrollBar().value() - delta.y())
            self.last_pos = event.pos()

        return super().eventFilter(source, event)

class NBtoRGBProcessingThread(QThread):
    preview_generated = pyqtSignal(np.ndarray)

    def __init__(self, ha_image, oiii_image, sii_image=None, osc_image=None, ha_to_oii_ratio=0.3, enable_star_stretch=True, stretch_factor=5.0):
        super().__init__()
        self.ha_image = ha_image
        self.oiii_image = oiii_image
        self.sii_image = sii_image
        self.osc_image = osc_image
        self.ha_to_oii_ratio = ha_to_oii_ratio
        self.enable_star_stretch = enable_star_stretch
        self.stretch_factor = stretch_factor

    def run(self):
        # Check if an OSC image is provided and handle as RGB channels
        if self.osc_image is not None:
            r_channel = self.osc_image[..., 0]
            g_channel = self.osc_image[..., 1]
            b_channel = self.osc_image[..., 2]
            
            # Handle cases where narrowband images are missing
            # If Ha is None, use the red channel of the OSC image for g_combined
            if self.ha_image is None:
                self.ha_image = r_channel
            # If OIII is None, use the green channel of the OSC image for b_combined
            if self.oiii_image is None:
                self.oiii_image = g_channel
            # If SII is None, use the red channel of the OSC image for r_combined
            if self.sii_image is None:
                self.sii_image = r_channel

            # Combined RGB channels with defaults as fallbacks
            r_combined = 0.5 * r_channel + 0.5 * self.sii_image
            g_combined = self.ha_to_oii_ratio * self.ha_image + (1 - self.ha_to_oii_ratio) * g_channel
            b_combined = b_channel
        else:
            # If no OSC image, use Ha, OIII, and SII images directly
            r_combined = 0.5 * self.ha_image + 0.5 * (self.sii_image if self.sii_image is not None else self.ha_image)
            g_combined = self.ha_to_oii_ratio * self.ha_image + (1 - self.ha_to_oii_ratio) * self.oiii_image
            b_combined = self.oiii_image

        # Stack the channels to create an RGB image
        combined_image = np.stack((r_combined, g_combined, b_combined), axis=-1)

        # Apply star stretch if enabled
        if self.enable_star_stretch:
            combined_image = self.apply_star_stretch(combined_image)

        # Apply SCNR (remove green cast)
        combined_image = self.apply_scnr(combined_image)
        self.preview_generated.emit(combined_image)


    def apply_star_stretch(self, image):
        stretched = ((3 ** self.stretch_factor) * image) / ((3 ** self.stretch_factor - 1) * image + 1)
        return np.clip(stretched, 0, 1)

    def apply_scnr(self, image):
        green_channel = image[..., 1]
        max_rg = np.maximum(image[..., 0], image[..., 2])
        green_channel[green_channel > max_rg] = max_rg[green_channel > max_rg]
        image[..., 1] = green_channel
        return image

class HaloBGonTab(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.image = None  # Selected image
        self.filename = None  # Store the selected file path
        self.preview_image = None  # Store the preview result
        self.processed_image = None
        self.zoom_factor = 0.25  # Initialize zoom factor for preview scaling
        self.dragging = False
        self.is_mono = True
        self.last_pos = None
        self.processing_thread = None  # For background processing
        self.original_header = None
        

    def initUI(self):
        main_layout = QHBoxLayout()

        # Left column for controls
        left_widget = QWidget(self)
        left_layout = QVBoxLayout(left_widget)
        left_widget.setFixedWidth(400)  # Fixed width for left column

        # Instructions label
        instruction_box = QLabel(self)
        instruction_box.setText("""
            Instructions:
            1. Select a stars-only image.
            2. Adjust the reduction amount as needed.
            3. Click Execute to apply the halo reduction.
        """)
        instruction_box.setWordWrap(True)
        left_layout.addWidget(instruction_box)

        # File selection button
        self.fileButton = QPushButton("Load Image", self)
        self.fileButton.clicked.connect(self.selectImage)
        left_layout.addWidget(self.fileButton)

        self.fileLabel = QLabel('No file selected', self)
        left_layout.addWidget(self.fileLabel)

        # Reduction amount slider
        self.reductionLabel = QLabel("Reduction Amount:", self)
        self.reductionSlider = QSlider(Qt.Horizontal, self)
        self.reductionSlider.setMinimum(0)
        self.reductionSlider.setMaximum(3)
        self.reductionSlider.setValue(1)
        self.reductionSlider.setToolTip("Adjust the amount of halo reduction (Extra Low, Low, Medium, High)")
        left_layout.addWidget(self.reductionLabel)
        left_layout.addWidget(self.reductionSlider)

        # Linear data checkbox
        self.linearDataCheckbox = QCheckBox("Linear Data", self)
        self.linearDataCheckbox.setToolTip("Check if the data is linear")
        left_layout.addWidget(self.linearDataCheckbox)

        # Progress indicator (spinner) label
        self.spinnerLabel = QLabel(self)
        self.spinnerLabel.setAlignment(Qt.AlignCenter)
        # Use the resource path function to access the GIF
        self.spinnerMovie = QMovie(resource_path("spinner.gif"))  # Updated path
        self.spinnerLabel.setMovie(self.spinnerMovie)
        self.spinnerLabel.hide()  # Hide spinner by default
        left_layout.addWidget(self.spinnerLabel)

        # Execute and Save buttons
        self.executeButton = QPushButton("Refresh Preview", self)
        self.executeButton.clicked.connect(self.generatePreview)
        left_layout.addWidget(self.executeButton)

        self.saveButton = QPushButton("Save Image", self)
        self.saveButton.clicked.connect(self.saveImage)
        left_layout.addWidget(self.saveButton)

        # Zoom in and out buttons
        zoom_layout = QHBoxLayout()
        self.zoomInButton = QPushButton("Zoom In", self)
        self.zoomInButton.clicked.connect(self.zoomIn)
        zoom_layout.addWidget(self.zoomInButton)

        self.zoomOutButton = QPushButton("Zoom Out", self)
        self.zoomOutButton.clicked.connect(self.zoomOut)
        zoom_layout.addWidget(self.zoomOutButton)
        left_layout.addLayout(zoom_layout)

                # Footer
        footer_label = QLabel("""
            Written by Franklin Marek<br>
            <a href='http://www.setiastro.com'>www.setiastro.com</a>
        """)
        footer_label.setAlignment(Qt.AlignCenter)
        footer_label.setOpenExternalLinks(True)
        footer_label.setStyleSheet("font-size: 10px;")
        left_layout.addWidget(footer_label)

        left_layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))
        main_layout.addWidget(left_widget)

        

        # Right side for the preview inside a QScrollArea
        self.scrollArea = QScrollArea(self)
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scrollArea.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scrollArea.viewport().installEventFilter(self)

        # QLabel for the image preview
        self.imageLabel = QLabel(self)
        self.imageLabel.setAlignment(Qt.AlignCenter)
        self.scrollArea.setWidget(self.imageLabel)
        self.scrollArea.setMinimumSize(400, 400)

        main_layout.addWidget(self.scrollArea)
        self.setLayout(main_layout)

    def zoomIn(self):
        self.zoom_factor *= 1.2  # Increase zoom by 20%
        self.updatePreview(self.image)

    def zoomOut(self):
        self.zoom_factor /= 1.2  # Decrease zoom by 20%
        self.updatePreview(self.image)
    

    def selectImage(self):
        selected_file, _ = QFileDialog.getOpenFileName(self, "Select Stars Only Image", "", "Images (*.png *.tif *.fits *.fit)")
        if selected_file:
            try:
                # Match the StarStretchTab loading method here
                self.image, self.original_header, _, self.is_mono = load_image(selected_file)  # Load image with header
                self.filename = selected_file 
                self.fileLabel.setText(os.path.basename(selected_file))
                self.generatePreview()  # Generate preview after loading
            except Exception as e:
                self.fileLabel.setText(f"Error: {str(e)}")
                print(f"Failed to load image: {e}")


    def applyHaloReduction(self):
        if self.image is None:
            print("No image selected.")
            return

        reduction_amount = self.reductionSlider.value()
        is_linear = self.linearDataCheckbox.isChecked()

        # Show spinner and start background processing
        self.showSpinner()
        self.processing_thread = QThread()
        self.processing_worker = self.HaloProcessingWorker(self.image, reduction_amount, is_linear)
        self.processing_worker.moveToThread(self.processing_thread)
        self.processing_worker.processing_complete.connect(self.updateImage)
        self.processing_thread.started.connect(self.processing_worker.process)
        self.processing_thread.start()

    def updatePreview(self, processed_image):
        # Store the processed image for saving
        self.processed_image = processed_image

        # Update the preview once the processing thread emits the result
        preview_image = (processed_image * 255).astype(np.uint8)
        h, w = preview_image.shape[:2]
        if preview_image.ndim == 3:
            q_image = QImage(preview_image.data, w, h, 3 * w, QImage.Format_RGB888)
        else:
            q_image = QImage(preview_image.data, w, h, w, QImage.Format_Grayscale8)

        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(pixmap.size() * self.zoom_factor, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.imageLabel.setPixmap(scaled_pixmap)
        self.imageLabel.resize(scaled_pixmap.size())

        # Hide the spinner after processing is done
        self.hideSpinner()

    def saveImage(self):
        if self.processed_image is not None:
            # Pre-populate the save dialog with the original image name
            base_name = os.path.basename(self.filename)
            default_save_name = os.path.splitext(base_name)[0] + '_reduced.tif'
            original_dir = os.path.dirname(self.filename)

            # Open the save file dialog
            save_filename, _ = QFileDialog.getSaveFileName(
                self, 
                'Save Image As', 
                os.path.join(original_dir, default_save_name), 
                'Images (*.tiff *.tif *.png *.fit *.fits);;All Files (*)'
            )

            if save_filename:
                original_format = save_filename.split('.')[-1].lower()

                # For TIFF and FITS files, prompt the user to select the bit depth
                bit_depth_options = ["16-bit", "32-bit unsigned", "32-bit floating point"]
                bit_depth, ok = QInputDialog.getItem(self, "Select Bit Depth", "Choose bit depth for saving:", bit_depth_options, 0, False)
                
                if ok and bit_depth:
                    # If linear data is checked, revert to linear before saving
                    if self.linearDataCheckbox.isChecked():
                        saved_image = np.clip(self.processed_image ** 5, 0, 1)  # Revert to linear state
                    else:
                        saved_image = self.processed_image  # Save as is (non-linear)

                    # Call save_image with the necessary parameters
                    save_image(saved_image, save_filename, original_format, bit_depth, self.original_header, self.is_mono)
                    self.fileLabel.setText(f'Image saved as: {save_filename}')
                else:
                    self.fileLabel.setText('Save canceled.')
            else:
                self.fileLabel.setText('Save canceled.')



    def showSpinner(self):
        self.spinnerLabel.show()
        self.spinnerMovie.start()

    def hideSpinner(self):
        self.spinnerLabel.hide()
        self.spinnerMovie.stop()

    # Updated generatePreview method in HaloBGonTab to use HaloProcessingThread
    def generatePreview(self):
        if self.image is not None and self.image.size > 0:
            # Show spinner before starting processing
            self.showSpinner()

            # Start background processing with HaloProcessingThread
            self.processing_thread = HaloProcessingThread(self.image, self.reductionSlider.value(), self.linearDataCheckbox.isChecked())
            self.processing_thread.preview_generated.connect(self.updatePreview)
            self.processing_thread.start()

    def updatePreview(self, processed_image):
        # Update the preview with the processed (non-linear if is_linear is checked) image
        self.processed_image = processed_image  # Save for use in saving
        preview_image = (processed_image * 255).astype(np.uint8)
        h, w = preview_image.shape[:2]
        if preview_image.ndim == 3:
            q_image = QImage(preview_image.tobytes(), w, h, 3 * w, QImage.Format_RGB888)
        else:
            q_image = QImage(preview_image.tobytes(), w, h, w, QImage.Format_Grayscale8)

        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(pixmap.size() * self.zoom_factor, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.imageLabel.setPixmap(scaled_pixmap)
        self.imageLabel.resize(scaled_pixmap.size())

        # Hide the spinner after processing is done
        self.hideSpinner()

    def eventFilter(self, source, event):
        if event.type() == event.MouseButtonPress and event.button() == Qt.LeftButton:
            self.dragging = True
            self.last_pos = event.pos()
        elif event.type() == event.MouseButtonRelease and event.button() == Qt.LeftButton:
            self.dragging = False
        elif event.type() == event.MouseMove and self.dragging:
            delta = event.pos() - self.last_pos
            self.scrollArea.horizontalScrollBar().setValue(self.scrollArea.horizontalScrollBar().value() - delta.x())
            self.scrollArea.verticalScrollBar().setValue(self.scrollArea.verticalScrollBar().value() - delta.y())
            self.last_pos = event.pos()

        return super().eventFilter(source, event)


    def create_lightness_mask(image):
        # Convert to grayscale to get the lightness mask
        lightness_mask = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply Unsharp Mask
        blurred = cv2.GaussianBlur(lightness_mask, (0, 0), sigmaX=2)
        lightness_mask = cv2.addWeighted(lightness_mask, 1.66, blurred, -0.66, 0)
        
        # Normalize to the [0, 1] range
        return lightness_mask / 255.0

    def createDuplicateImage(self, original):
        return np.copy(original)

    def invert_mask(mask):
        return 1.0 - mask  # Assuming mask is normalized between 0 and 1


    def apply_mask_to_image(image, mask):
        # Ensure mask is 3-channel to match the image dimensions
        mask_rgb = np.stack([mask] * 3, axis=-1)
        return cv2.multiply(image, mask_rgb)


    def apply_curves_to_image(image, reduction_amount):
        # Define the curve based on reduction amount
        if reduction_amount == 0:
            curve = [int((i / 255.0) ** 0.575 * 255) for i in range(256)]
        else:
            curve = [int((i / 255.0) ** 0.4 * 255) for i in range(256)]
        
        lut = np.array(curve, dtype=np.uint8)
        return cv2.LUT((image * 255).astype(np.uint8), lut).astype(np.float32) / 255.0


    def load_image(self, filename):
        original_header = None
        file_extension = filename.split('.')[-1].lower()

        # Handle different file types and normalize them to [0, 1] range
        if file_extension in ['tif', 'tiff']:
            image = tiff.imread(filename).astype(np.float32) / 65535.0  # For 16-bit TIFF images
        elif file_extension == 'png':
            image = np.array(Image.open(filename).convert('RGB')).astype(np.float32) / 255.0  # Normalize to [0, 1]
        elif file_extension in ['fits', 'fit']:
            with fits.open(filename) as hdul:
                image = hdul[0].data.astype(np.float32)
                original_header = hdul[0].header
                # Normalize if data is 16-bit or higher
                if image.max() > 1:
                    image /= np.max(image)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")

        return image, original_header

    def save_image(self, image, filename, file_format, bit_depth="16-bit", original_header=None):
        img = Image.fromarray((image * 255).astype(np.uint8))
        img.save(filename)

class HaloProcessingThread(QThread):
    preview_generated = pyqtSignal(np.ndarray)

    def __init__(self, image, reduction_amount, is_linear):
        super().__init__()
        self.image = image
        self.reduction_amount = reduction_amount
        self.is_linear = is_linear


    def run(self):
        processed_image = self.applyHaloReduction(self.image, self.reduction_amount, self.is_linear)
        self.preview_generated.emit(processed_image)

    def applyHaloReduction(self, image, reduction_amount, is_linear):
        image = np.clip(image, 0, 1)
        if is_linear:
            image = image ** (1 / 5)  # Convert linear to non-linear (approx gamma correction)

        # Apply halo reduction logic
        lightness_mask = self.createLightnessMask(image)
        inverted_mask = 1.0 - lightness_mask
        duplicated_mask = cv2.GaussianBlur(lightness_mask, (0, 0), sigmaX=2)
        enhanced_mask = inverted_mask - duplicated_mask * reduction_amount * 0.33
        masked_image = cv2.multiply(image, np.stack([enhanced_mask] * 3, axis=-1))
        final_image = self.applyCurvesToImage(masked_image, reduction_amount)

        #if is_linear:
        #    final_image = final_image ** 5  # Convert back to linear

        return np.clip(final_image, 0, 1)

    def createLightnessMask(self, image):
        # Convert image to grayscale to create a lightness mask
        lightness_mask = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        blurred = cv2.GaussianBlur(lightness_mask, (0, 0), sigmaX=2)
        return cv2.addWeighted(lightness_mask, 1.66, blurred, -0.66, 0)

    def createDuplicateMask(self, mask):
        # Duplicate the mask and apply additional processing (simulating MMT)
        duplicated_mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=2)
        return duplicated_mask

    def applyMaskToImage(self, image, mask):
        # Blend the original image with the mask based on the reduction level
        mask_rgb = np.stack([mask] * 3, axis=-1)  # Convert to 3-channel
        return cv2.multiply(image, mask_rgb)

    def applyCurvesToImage(self, image, reduction_amount):
        # Apply a curves transformation based on reduction_amount
        if reduction_amount == 0:
            # Extra Low setting, mild curve
            curve = [int((i / 255.0) ** 1.2 * 255) for i in range(256)]
        elif reduction_amount == 1:
            # Low setting, slightly stronger darkening
            curve = [int((i / 255.0) ** 1.5 * 255) for i in range(256)]
        elif reduction_amount == 2:
            # Medium setting, moderate darkening
            curve = [int((i / 255.0) ** 1.8 * 255) for i in range(256)]
        else:
            # High setting, strong darkening effect
            curve = [int((i / 255.0) ** 2.2 * 255) for i in range(256)]

        # Apply the curve transformation as a lookup table
        lut = np.array(curve, dtype=np.uint8)
        transformed_image = cv2.LUT((image * 255).astype(np.uint8), lut).astype(np.float32) / 255.0
        return transformed_image



class ContinuumSubtractTab(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.nb_image = None  # Changed from ha_image to nb_image
        self.filename = None  # Store the selected file path
        self.is_mono = True
        self.continuum_image = None  # Changed from red_continuum_image to continuum_image
        self.processing_thread = None  # For background processing
        self.combined_image = None  # Store the result of the continuum subtraction
        self.zoom_factor = 0.25  # Initial zoom factor
        self.original_header = None

    def initUI(self):
        main_layout = QHBoxLayout()

        # Left side controls
        left_widget = QWidget(self)
        left_layout = QVBoxLayout(left_widget)
        left_widget.setFixedWidth(400)  # Fixed width for left column

        # Instruction box
        instruction_box = QLabel(self)
        instruction_box.setText("""
            Instructions:
            1. Load your NB and Continuum images.
            2. Select for optional linear only output.
            3. Click Execute to perform continuum subtraction.
        """)
        instruction_box.setWordWrap(True)
        left_layout.addWidget(instruction_box)

        # File Selection Buttons
        self.nb_button = QPushButton("Load NB Image")
        self.nb_button.clicked.connect(lambda: self.selectImage("nb"))
        self.nb_label = QLabel("No NB image selected")  # Updated label
        left_layout.addWidget(self.nb_button)
        left_layout.addWidget(self.nb_label)

        self.continuum_button = QPushButton("Load Continuum Image")
        self.continuum_button.clicked.connect(lambda: self.selectImage("continuum"))
        self.continuum_label = QLabel("No Continuum image selected")  # Updated label
        left_layout.addWidget(self.continuum_button)
        left_layout.addWidget(self.continuum_label)

        self.linear_output_checkbox = QCheckBox("Output Linear Image Only")
        left_layout.addWidget(self.linear_output_checkbox)

        # Progress indicator (spinner) label
        self.spinnerLabel = QLabel(self)
        self.spinnerLabel.setAlignment(Qt.AlignCenter)
        # Use the resource path function to access the GIF
        self.spinnerMovie = QMovie(resource_path("spinner.gif"))  # Updated path
        self.spinnerLabel.setMovie(self.spinnerMovie)
        self.spinnerLabel.hide()  # Hide spinner by default
        left_layout.addWidget(self.spinnerLabel)

        # Status label to show what is happening in the background
        self.statusLabel = QLabel(self)
        self.statusLabel.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(self.statusLabel)


        # Execute Button
        self.execute_button = QPushButton("Execute")
        self.execute_button.clicked.connect(self.startContinuumSubtraction)
        left_layout.addWidget(self.execute_button)

        # Zoom In and Zoom Out Buttons
        zoom_layout = QHBoxLayout()
        self.zoomInButton = QPushButton("Zoom In")
        self.zoomInButton.clicked.connect(self.zoom_in)
        zoom_layout.addWidget(self.zoomInButton)

        self.zoomOutButton = QPushButton("Zoom Out")
        self.zoomOutButton.clicked.connect(self.zoom_out)
        zoom_layout.addWidget(self.zoomOutButton)
        left_layout.addLayout(zoom_layout)

        # Save Button
        self.save_button = QPushButton("Save Continuum Subtracted Image")
        self.save_button.clicked.connect(self.save_continuum_subtracted)
        left_layout.addWidget(self.save_button)

        # Footer
        footer_label = QLabel("""
            Written by Franklin Marek<br>
            <a href='http://www.setiastro.com'>www.setiastro.com</a>
        """)
        footer_label.setAlignment(Qt.AlignCenter)
        footer_label.setOpenExternalLinks(True)
        footer_label.setStyleSheet("font-size: 10px;")
        left_layout.addWidget(footer_label)

        # Spacer to push elements to the top
        left_layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

        # Add left widget layout to the main layout
        main_layout.addWidget(left_widget)

        # Right side for the preview inside a QScrollArea
        self.scrollArea = QScrollArea(self)
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scrollArea.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scrollArea.viewport().installEventFilter(self)

        # QLabel for the image preview
        self.imageLabel = QLabel(self)
        self.imageLabel.setAlignment(Qt.AlignCenter)
        self.scrollArea.setWidget(self.imageLabel)
        self.scrollArea.setMinimumSize(400, 400)

        main_layout.addWidget(self.scrollArea)
        self.setLayout(main_layout)

    def selectImage(self, image_type):
        selected_file, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.tif *.fits *.fit)")
        if selected_file:
            try:
                image, original_header, _, _ = load_image(selected_file)  # Load image with header
                self.filename = selected_file
                if image_type == "nb":
                    self.nb_image = image
                    self.nb_label.setText(os.path.basename(selected_file))  # Updated label
                elif image_type == "continuum":
                    self.continuum_image = image
                    self.continuum_label.setText(os.path.basename(selected_file))  # Updated label
            except Exception as e:
                print(f"Failed to load {image_type} image: {e}")
                if image_type == "nb":
                    self.nb_label.setText("Error loading NB image")
                elif image_type == "continuum":
                    self.continuum_label.setText("Error loading Continuum image")

    def startContinuumSubtraction(self):
        if self.nb_image is not None and self.continuum_image is not None:
            # Show spinner and start background processing
            self.showSpinner()
            self.processing_thread = ContinuumProcessingThread(self.nb_image, self.continuum_image,
                                                            self.linear_output_checkbox.isChecked())
            self.processing_thread.processing_complete.connect(self.display_image)
            self.processing_thread.finished.connect(self.hideSpinner)
            self.processing_thread.status_update.connect(self.update_status_label)
            self.processing_thread.start()
        else:
            print("Please select both NB and Continuum images.")

    def update_status_label(self, message):
        self.statusLabel.setText(message)

    def zoom_in(self):
        self.zoom_factor *= 1.2
        self.update_preview()

    def zoom_out(self):
        self.zoom_factor /= 1.2
        self.update_preview()

    def update_preview(self):
        if self.combined_image is not None:
            self.display_image(self.combined_image)        

    def load_image(self, filename):
        # Placeholder for actual image loading logic
        image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        return image, None, None, None
    
    def save_continuum_subtracted(self):
        if self.combined_image is not None:
            # Pre-populate the save dialog with the original image name
            base_name = os.path.basename(self.filename)
            default_save_name = os.path.splitext(base_name)[0] + '_continuumsubtracted.tif'
            original_dir = os.path.dirname(self.filename)

            # Open the save file dialog
            save_filename, _ = QFileDialog.getSaveFileName(
                self, 
                'Save Image As', 
                os.path.join(original_dir, default_save_name), 
                'Images (*.tiff *.tif *.png *.fit *.fits);;All Files (*)'
            )

            if save_filename:
                original_format = save_filename.split('.')[-1].lower()

                # For TIFF and FITS files, prompt the user to select the bit depth
                if original_format in ['tiff', 'tif', 'fits', 'fit']:
                    bit_depth_options = ["16-bit", "32-bit unsigned", "32-bit floating point"]
                    bit_depth, ok = QInputDialog.getItem(self, "Select Bit Depth", "Choose bit depth for saving:", bit_depth_options, 0, False)
                    
                    if ok and bit_depth:
                        # Call save_image with the necessary parameters
                        save_image(self.combined_image, save_filename, original_format, bit_depth, self.original_header, self.is_mono)
                        self.statusLabel.setText(f'Image saved as: {save_filename}')
                    else:
                        self.statusLabel.setText('Save canceled.')
                else:
                    # For non-TIFF/FITS formats, save directly without bit depth selection
                    save_image(self.image, save_filename, original_format)
                    self.statusLabel.setText(f'Image saved as: {save_filename}')
            else:
                self.statusLabel.setText('Save canceled.')



    def display_image(self, processed_image):
        self.combined_image = processed_image

        # Convert the processed image to a displayable format
        preview_image = (processed_image * 255).astype(np.uint8)
        
        # Check if the image is mono or RGB
        if preview_image.ndim == 2:  # Mono image
            # Create a 3-channel RGB image by duplicating the single channel
            preview_image = np.stack([preview_image] * 3, axis=-1)  # Stack to create RGB

        h, w = preview_image.shape[:2]

        # Change the format to RGB888 for displaying an RGB image
        q_image = QImage(preview_image.data, w, h, 3 * w, QImage.Format_RGB888)

        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(pixmap.size() * self.zoom_factor, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.imageLabel.setPixmap(scaled_pixmap)
        self.imageLabel.resize(scaled_pixmap.size())


    def showSpinner(self):
        self.spinnerLabel.show()
        self.spinnerMovie.start()

    def hideSpinner(self):
        self.spinnerLabel.hide()
        self.spinnerMovie.stop()

    def eventFilter(self, source, event):
        if source is self.scrollArea.viewport():
            if event.type() == event.MouseButtonPress and event.button() == Qt.LeftButton:
                self.dragging = True
                self.last_pos = event.pos()
            elif event.type() == event.MouseButtonRelease and event.button() == Qt.LeftButton:
                self.dragging = False
            elif event.type() == event.MouseMove and self.dragging:
                delta = event.pos() - self.last_pos
                self.scrollArea.horizontalScrollBar().setValue(self.scrollArea.horizontalScrollBar().value() - delta.x())
                self.scrollArea.verticalScrollBar().setValue(self.scrollArea.verticalScrollBar().value() - delta.y())
                self.last_pos = event.pos()

        return super().eventFilter(source, event)


class ContinuumProcessingThread(QThread):
    processing_complete = pyqtSignal(np.ndarray)
    status_update = pyqtSignal(str)

    def __init__(self, nb_image, continuum_image, output_linear):
        super().__init__()
        self.nb_image = nb_image
        self.continuum_image = continuum_image
        self.output_linear = output_linear
        self.background_reference = None  # Store the background reference



    def run(self):
        # Ensure both images are mono
        if self.nb_image.ndim == 3 and self.nb_image.shape[2] == 3:
            self.nb_image = self.nb_image[..., 0]  # Take one channel for the NB image

        if self.continuum_image.ndim == 3 and self.continuum_image.shape[2] == 3:
            self.continuum_image = self.continuum_image[..., 0]  # Take one channel for the continuum image

        # Create RGB image
        r_combined = self.nb_image  # Use the normalized NB image as the Red channel
        g_combined = self.continuum_image # Use the normalized continuum image as the Green channel
        b_combined = self.continuum_image  # Use the normalized continuum image as the Blue channel


        # Stack the channels into a single RGB image
        combined_image = np.stack((r_combined, g_combined, b_combined), axis=-1)

        self.status_update.emit("Performing background neutralization...")
        QCoreApplication.processEvents()
            # Perform background neutralization
        self.background_neutralization(combined_image)

        # Normalize the red channel to the green channel
        combined_image[..., 0] = self.normalize_channel(combined_image[..., 0], combined_image[..., 1])

        # Perform continuum subtraction
        linear_image = combined_image[..., 0] - 0.9*(combined_image[..., 1]-np.median(combined_image[..., 1]))

            # Check if the Output Linear checkbox is checked
        if self.output_linear:
            # Emit the linear image for preview
            self.processing_complete.emit(np.clip(linear_image, 0, 1))
            return  # Exit the method if we only want to output the linear image

        self.status_update.emit("Subtraction complete.")
        QCoreApplication.processEvents()

        # Perform statistical stretch
        target_median = 0.25
        stretched_image = stretch_color_image(linear_image, target_median, True, False)

        # Final image adjustment
        final_image = stretched_image - 0.7*np.median(stretched_image)

        # Clip the final image to stay within [0, 1]
        final_image = np.clip(final_image, 0, 1)

        # Applies Curves Boost
        final_image = apply_curves_adjustment(final_image, np.median(final_image), 0.5)

        self.status_update.emit("Linear to Non-Linear Stretch complete.")
        QCoreApplication.processEvents()
        # Emit the final image for preview
        self.processing_complete.emit(final_image)

    def background_neutralization(self, rgb_image):
        height, width, _ = rgb_image.shape
        num_boxes = 200
        box_size = 25
        iterations = 25

        boxes = [(np.random.randint(0, height - box_size), np.random.randint(0, width - box_size)) for _ in range(num_boxes)]
        best_means = np.full(num_boxes, np.inf)

        for _ in range(iterations):
            for i, (y, x) in enumerate(boxes):
                if y + box_size <= height and x + box_size <= width:
                    patch = rgb_image[y:y + box_size, x:x + box_size]
                    patch_median = np.median(patch) if patch.size > 0 else np.inf

                    if patch_median < best_means[i]:
                        best_means[i] = patch_median

                    surrounding_values = []
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            surrounding_y = y + dy * box_size
                            surrounding_x = x + dx * box_size
                            
                            if (0 <= surrounding_y < height - box_size) and (0 <= surrounding_x < width - box_size):
                                surrounding_patch = rgb_image[surrounding_y:surrounding_y + box_size, surrounding_x:surrounding_x + box_size]
                                if surrounding_patch.size > 0:
                                    surrounding_values.append(np.median(surrounding_patch))

                    if surrounding_values:
                        dimmest_index = np.argmin(surrounding_values)
                        new_y = y + (dimmest_index // 3 - 1) * box_size
                        new_x = x + (dimmest_index % 3 - 1) * box_size
                        boxes[i] = (new_y, new_x)

        # After iterations, find the darkest box median
        darkest_value = np.inf
        background_box = None

        for box in boxes:
            y, x = box
            if y + box_size <= height and x + box_size <= width:
                patch = rgb_image[y:y + box_size, x:y + box_size]
                patch_median = np.median(patch) if patch.size > 0 else np.inf

                if patch_median < darkest_value:
                    darkest_value = patch_median
                    background_box = patch

        if background_box is not None:
            self.background_reference = np.median(background_box.reshape(-1, 3), axis=0)
            
            # Adjust the channels based on the median reference
            channel_medians = np.median(rgb_image, axis=(0, 1))

            # Adjust channels based on the red channel
            for channel in range(3):
                if self.background_reference[channel] < channel_medians[channel]:
                    pedestal = channel_medians[channel] - self.background_reference[channel]
                    rgb_image[..., channel] += pedestal

            # Specifically adjust G and B to match R
            r_median = self.background_reference[0]
            for channel in [1, 2]:  # Green and Blue channels
                if self.background_reference[channel] < r_median:
                    rgb_image[..., channel] += (r_median - self.background_reference[channel])

        self.status_update.emit("Background neutralization complete.")
        QCoreApplication.processEvents()
        return rgb_image
    
    def normalize_channel(self, image_channel, reference_channel):
        mad_image = np.mean(np.abs(image_channel - np.mean(image_channel)))
        mad_ref = np.mean(np.abs(reference_channel - np.mean(reference_channel)))

        median_image = np.median(image_channel)
        median_ref = np.median(reference_channel)

        # Apply the normalization formula
        normalized_channel = (
            image_channel * mad_ref / mad_image
            - (mad_ref / mad_image) * median_image
            + median_ref
        )

        self.status_update.emit("Color calibration complete.")
        QCoreApplication.processEvents()
        return np.clip(normalized_channel, 0, 1)  



    def continuum_subtraction(self, rgb_image):
        red_channel = rgb_image[..., 0]
        green_channel = rgb_image[..., 1]
        
        # Determine Q based on the selection (modify condition based on actual UI element)
        Q = 0.9 if self.output_linear else 1.0

        # Perform the continuum subtraction
        median_green = np.median(green_channel)
        result_image = red_channel - Q * (green_channel - median_green)
        
        return np.clip(result_image, 0, 1)  # Ensure values stay within [0, 1]





# Function to load and convert images to 32-bit floating point
def load_image(filename):
    bit_depth = None  # Initialize bit depth to None
    is_mono = True  # Assume monochrome by default    
    original_header = None  # Initialize an empty header for FITS files
    if filename.lower().endswith('.png'):
        img = Image.open(filename).convert('RGB')  # Ensures it's RGB
        img_array = np.array(img, dtype=np.float32) / 255.0  # Normalize to [0, 1]
    elif filename.lower().endswith('.tiff') or filename.lower().endswith('.tif'):
        img_array = tiff.imread(filename)
        # Handle different bit depths
        if img_array.dtype == np.uint8:
            img_array = img_array.astype(np.float32) / 255.0  # Normalize 8-bit to [0, 1]
        elif img_array.dtype == np.uint16:
            img_array = img_array.astype(np.float32) / 65535.0  # Normalize 16-bit to [0, 1]
        elif img_array.dtype == np.uint32:
            img_array = img_array.astype(np.float32) / 4294967295.0  # Normalize 32-bit unsigned integer to [0, 1]
        elif img_array.dtype == np.float32:
            img_array = img_array  # Already in 32-bit floating point, no need to convert
        else:
            raise ValueError("Unsupported TIFF format!")
    elif filename.lower().endswith('.fits') or filename.lower().endswith('.fit'):
        with fits.open(filename) as hdul:
            img_array = hdul[0].data
            original_header = hdul[0].header  # Capture the FITS header
            bit_depth = None

            # Determine bit depth and apply necessary transformations
            if img_array.dtype == np.uint16:
                bit_depth = "16-bit"
                img_array = img_array.astype(np.float32) / 65535.0  # Normalize 16-bit to [0, 1]
            elif img_array.dtype == np.uint32:
                bit_depth = "32-bit unsigned"
                bzero = original_header.get('BZERO', 0)
                bscale = original_header.get('BSCALE', 1)
                img_array = img_array.astype(np.float32) * bscale + bzero

                # Normalize to [0, 1] based on range
                image_min = img_array.min()
                image_max = img_array.max()
                img_array = (img_array - image_min) / (image_max - image_min)
            elif img_array.dtype == np.float32:
                bit_depth = "32-bit floating point"
                # No normalization needed for 32-bit float

            # Handle 3D FITS data (e.g., RGB or multi-layered data)
            if img_array.ndim == 3 and img_array.shape[0] == 3:
                img_array = np.transpose(img_array, (1, 2, 0))  # Reorder to (height, width, channels)
                is_mono = False
            elif img_array.ndim == 2:
                img_array = np.stack([img_array] * 3, axis=-1)  # Convert grayscale to 3-channel for consistency
                is_mono = True
            else:
                raise ValueError("Unsupported FITS format!")
    else:
        raise ValueError("Unsupported file format!")

    return img_array, original_header, bit_depth, is_mono  # Return the image array, header, bit depth, and is_mono flag






def save_image(img_array, filename, original_format, bit_depth=None, original_header=None, is_mono=False):
    img_array = ensure_native_byte_order(img_array)  # Apply native byte order correction if needed

    if original_format == 'png':
        img = Image.fromarray((img_array * 255).astype(np.uint8))  # Convert to 8-bit and save as PNG
        img.save(filename)
    elif original_format in ['tiff', 'tif']:
        if bit_depth == "16-bit":
            tiff.imwrite(filename, (img_array * 65535).astype(np.uint16))  # Save as 16-bit TIFF
        elif bit_depth == "32-bit unsigned":
            tiff.imwrite(filename, (img_array * 4294967295).astype(np.uint32))  # Save as 32-bit unsigned TIFF
        elif bit_depth == "32-bit floating point":
            tiff.imwrite(filename, img_array.astype(np.float32))  # Save as 32-bit floating point TIFF
    elif original_format in ['fits', 'fit']:
        # For grayscale (mono) FITS images
        if is_mono:
            if bit_depth == "16-bit":
                img_array_fits = (img_array[:, :, 0] * 65535).astype(np.uint16)
            elif bit_depth == "32-bit unsigned":
                img_array_fits = (img_array[:, :, 0] * 4294967295).astype(np.uint32)
            elif bit_depth == "32-bit floating point":
                img_array_fits = img_array[:, :, 0].astype(np.float32)
            hdu = fits.PrimaryHDU(img_array_fits, header=original_header)
        else:
            # Transpose RGB image to (channels, height, width) for FITS format
            img_array_fits = np.transpose(img_array, (2, 0, 1))
            if bit_depth == "16-bit":
                img_array_fits = (img_array_fits * 65535).astype(np.uint16)
            elif bit_depth == "32-bit unsigned":
                img_array_fits = (img_array_fits * 4294967295).astype(np.uint32)
            elif bit_depth == "32-bit floating point":
                img_array_fits = img_array_fits.astype(np.float32)

            # Update the original header with correct dimensions for multi-channel images
            original_header['NAXIS'] = 3
            original_header['NAXIS1'] = img_array_fits.shape[2]  # Width
            original_header['NAXIS2'] = img_array_fits.shape[1]  # Height
            original_header['NAXIS3'] = img_array_fits.shape[0]  # Channels

            hdu = fits.PrimaryHDU(img_array_fits, header=original_header)

        hdu.writeto(filename, overwrite=True)
    else:
        raise ValueError("Unsupported file format!")




def stretch_mono_image(image, target_median, normalize=False, apply_curves=False, curves_boost=0.0):
    black_point = max(np.min(image), np.median(image) - 2.7 * np.std(image))
    rescaled_image = (image - black_point) / (1 - black_point)
    median_image = np.median(rescaled_image)
    stretched_image = ((median_image - 1) * target_median * rescaled_image) / (median_image * (target_median + rescaled_image - 1) - target_median * rescaled_image)
    
    if apply_curves:
        stretched_image = apply_curves_adjustment(stretched_image, target_median, curves_boost)
    
    if normalize:
        stretched_image = stretched_image / np.max(stretched_image)
    
    return np.clip(stretched_image, 0, 1)


def stretch_color_image(image, target_median, linked=True, normalize=False, apply_curves=False, curves_boost=0.0):
    if linked:
        combined_median = np.median(image)
        combined_std = np.std(image)
        black_point = max(np.min(image), combined_median - 2.7 * combined_std)
        rescaled_image = (image - black_point) / (1 - black_point)
        median_image = np.median(rescaled_image)
        stretched_image = ((median_image - 1) * target_median * rescaled_image) / (median_image * (target_median + rescaled_image - 1) - target_median * rescaled_image)
    else:
        stretched_image = np.zeros_like(image)
        for channel in range(image.shape[2]):
            black_point = max(np.min(image[..., channel]), np.median(image[..., channel]) - 2.7 * np.std(image[..., channel]))
            rescaled_channel = (image[..., channel] - black_point) / (1 - black_point)
            median_channel = np.median(rescaled_channel)
            stretched_image[..., channel] = ((median_channel - 1) * target_median * rescaled_channel) / (median_channel * (target_median + rescaled_channel - 1) - target_median * rescaled_channel)
    
    if apply_curves:
        stretched_image = apply_curves_adjustment(stretched_image, target_median, curves_boost)
    
    if normalize:
        stretched_image = stretched_image / np.max(stretched_image)
    
    return np.clip(stretched_image, 0, 1)


def apply_curves_adjustment(image, target_median, curves_boost):
    curve = [
        [0.0, 0.0],
        [0.5 * target_median, 0.5 * target_median],
        [target_median, target_median],
        [(1 / 4 * (1 - target_median) + target_median), 
         np.power((1 / 4 * (1 - target_median) + target_median), (1 - curves_boost))],
        [(3 / 4 * (1 - target_median) + target_median), 
         np.power(np.power((3 / 4 * (1 - target_median) + target_median), (1 - curves_boost)), (1 - curves_boost))],
        [1.0, 1.0]
    ]
    adjusted_image = np.interp(image, [p[0] for p in curve], [p[1] for p in curve])
    return adjusted_image

def resource_path(relative_path):
    """ Get the absolute path to the resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temporary folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

def ensure_native_byte_order(array):
    """
    Ensures that the array is in the native byte order.
    If the array is in a non-native byte order, it will convert it.
    """
    if array.dtype.byteorder == '=':  # Already in native byte order
        return array
    elif array.dtype.byteorder in ('<', '>'):  # Non-native byte order
        return array.byteswap().newbyteorder()
    return array


# Determine if running inside a PyInstaller bundle
if hasattr(sys, '_MEIPASS'):
    # Set path for PyInstaller bundle
    data_path = os.path.join(sys._MEIPASS, "astroquery", "simbad", "data")
else:
    # Set path for regular Python environment
    data_path = "C:/Users/Gaming/Desktop/Python Code/venv/Lib/site-packages/astroquery/simbad/data"

# Ensure the final path doesn't contain 'data/data' duplication
if 'data/data' in data_path:
    data_path = data_path.replace('data/data', 'data')

conf.dataurl = f'file://{data_path}/'

# Access wrench_icon.png, adjusting for PyInstaller executable
if hasattr(sys, '_MEIPASS'):
    icon_path = os.path.join(sys._MEIPASS, 'wrench_icon.png')
    eye_icon_path = os.path.join(sys._MEIPASS, 'eye.png')
    disk_icon_path = os.path.join(sys._MEIPASS, 'disk.png')
    nuke_path = os.path.join(sys._MEIPASS, 'nuke.png')  
    hubble_path = os.path.join(sys._MEIPASS, 'hubble.png') 
    collage_path = os.path.join(sys._MEIPASS, 'collage.png') 
    annotated_path = os.path.join(sys._MEIPASS, 'annotated.png') 
    colorwheel_path = os.path.join(sys._MEIPASS, 'colorwheel.png')
    font_path = os.path.join(sys._MEIPASS, 'font.png')
    csv_icon_path = os.path.join(sys._MEIPASS, 'cvs.png')
else:
    icon_path = 'wrench_icon.png'  # Path for running as a script
    eye_icon_path = 'eye.png'  # Path for running as a script
    disk_icon_path = 'disk.png'   
    nuke_path = 'nuke.png' 
    hubble_path = 'hubble.png'
    collage_path = 'collage.png'
    annotated_path = 'annotated.png'
    colorwheel_path = 'colorwheel.png'
    font_path = 'font.png'
    csv_icon_path = 'cvs.png'

# Constants for comoving radial distance calculation
H0 = 69.6  # Hubble constant in km/s/Mpc
WM = 0.286  # Omega(matter)
WV = 0.714  # Omega(vacuum)
c = 299792.458  # speed of light in km/s
Tyr = 977.8  # coefficient to convert 1/H into Gyr
Mpc_to_Gly = 3.262e-3  # Conversion from Mpc to Gly

otype_long_name_lookup = {
    "ev": "transient event",
    "Rad": "Radio-source",
    "mR": "metric Radio-source",
    "cm": "centimetric Radio-source",
    "mm": "millimetric Radio-source",
    "smm": "sub-millimetric source",
    "HI": "HI (21cm) source",
    "rB": "radio Burst",
    "Mas": "Maser",
    "IR": "Infra-Red source",
    "FIR": "Far-Infrared source",
    "MIR": "Mid-Infrared source",
    "NIR": "Near-Infrared source",
    "blu": "Blue object",
    "UV": "UV-emission source",
    "X": "X-ray source",
    "UX?": "Ultra-luminous X-ray candidate",
    "ULX": "Ultra-luminous X-ray source",
    "gam": "gamma-ray source",
    "gB": "gamma-ray Burst",
    "err": "Not an object (error, artefact, ...)",
    "grv": "Gravitational Source",
    "Lev": "(Micro)Lensing Event",
    "LS?": "Possible gravitational lens System",
    "Le?": "Possible gravitational lens",
    "LI?": "Possible gravitationally lensed image",
    "gLe": "Gravitational Lens",
    "gLS": "Gravitational Lens System (lens+images)",
    "GWE": "Gravitational Wave Event",
    "..?": "Candidate objects",
    "G?": "Possible Galaxy",
    "SC?": "Possible Supercluster of Galaxies",
    "C?G": "Possible Cluster of Galaxies",
    "Gr?": "Possible Group of Galaxies",
    "**?": "Physical Binary Candidate",
    "EB?": "Eclipsing Binary Candidate",
    "Sy?": "Symbiotic Star Candidate",
    "CV?": "Cataclysmic Binary Candidate",
    "No?": "Nova Candidate",
    "XB?": "X-ray binary Candidate",
    "LX?": "Low-Mass X-ray binary Candidate",
    "HX?": "High-Mass X-ray binary Candidate",
    "Pec?": "Possible Peculiar Star",
    "Y*?": "Young Stellar Object Candidate",
    "TT?": "T Tau star Candidate",
    "C*?": "Possible Carbon Star",
    "S*?": "Possible S Star",
    "OH?": "Possible Star with envelope of OH/IR type",
    "WR?": "Possible Wolf-Rayet Star",
    "Be?": "Possible Be Star",
    "Ae?": "Possible Herbig Ae/Be Star",
    "HB?": "Possible Horizontal Branch Star",
    "RR?": "Possible Star of RR Lyr type",
    "Ce?": "Possible Cepheid",
    "WV?": "Possible Variable Star of W Vir type",
    "RB?": "Possible Red Giant Branch star",
    "sg?": "Possible Supergiant star",
    "s?r": "Possible Red supergiant star",
    "s?y": "Possible Yellow supergiant star",
    "s?b": "Possible Blue supergiant star",
    "AB?": "Asymptotic Giant Branch Star candidate",
    "LP?": "Long Period Variable candidate",
    "Mi?": "Mira candidate",
    "pA?": "Post-AGB Star Candidate",
    "BS?": "Candidate blue Straggler Star",
    "HS?": "Hot subdwarf candidate",
    "WD?": "White Dwarf Candidate",
    "N*?": "Neutron Star Candidate",
    "BH?": "Black Hole Candidate",
    "SN?": "SuperNova Candidate",
    "LM?": "Low-mass star candidate",
    "BD?": "Brown Dwarf Candidate",
    "mul": "Composite object",
    "reg": "Region defined in the sky",
    "vid": "Underdense region of the Universe",
    "SCG": "Supercluster of Galaxies",
    "ClG": "Cluster of Galaxies",
    "GrG": "Group of Galaxies",
    "CGG": "Compact Group of Galaxies",
    "PaG": "Pair of Galaxies",
    "IG": "Interacting Galaxies",
    "C?*": "Possible (open) star cluster",
    "Gl?": "Possible Globular Cluster",
    "Cl*": "Cluster of Stars",
    "GlC": "Globular Cluster",
    "OpC": "Open (galactic) Cluster",
    "As*": "Association of Stars",
    "St*": "Stellar Stream",
    "MGr": "Moving Group",
    "**": "Double or multiple star",
    "EB*": "Eclipsing binary",
    "Al*": "Eclipsing binary of Algol type",
    "bL*": "Eclipsing binary of beta Lyr type",
    "WU*": "Eclipsing binary of W UMa type",
    "SB*": "Spectroscopic binary",
    "El*": "Ellipsoidal variable Star",
    "Sy*": "Symbiotic Star",
    "CV*": "Cataclysmic Variable Star",
    "DQ*": "CV DQ Her type (intermediate polar)",
    "AM*": "CV of AM Her type (polar)",
    "NL*": "Nova-like Star",
    "No*": "Nova",
    "DN*": "Dwarf Nova",
    "XB*": "X-ray Binary",
    "LXB": "Low Mass X-ray Binary",
    "HXB": "High Mass X-ray Binary",
    "ISM": "Interstellar matter",
    "PoC": "Part of Cloud",
    "PN?": "Possible Planetary Nebula",
    "CGb": "Cometary Globule",
    "bub": "Bubble",
    "EmO": "Emission Object",
    "Cld": "Cloud",
    "GNe": "Galactic Nebula",
    "DNe": "Dark Cloud (nebula)",
    "RNe": "Reflection Nebula",
    "MoC": "Molecular Cloud",
    "glb": "Globule (low-mass dark cloud)",
    "cor": "Dense core",
    "SFR": "Star forming region",
    "HVC": "High-velocity Cloud",
    "HII": "HII (ionized) region",
    "PN": "Planetary Nebula",
    "sh": "HI shell",
    "SR?": "SuperNova Remnant Candidate",
    "SNR": "SuperNova Remnant",
    "of?": "Outflow candidate",
    "out": "Outflow",
    "HH": "Herbig-Haro Object",
    "*": "Star",
    "V*?": "Star suspected of Variability",
    "Pe*": "Peculiar Star",
    "HB*": "Horizontal Branch Star",
    "Y*O": "Young Stellar Object",
    "Ae*": "Herbig Ae/Be star",
    "Em*": "Emission-line Star",
    "Be*": "Be Star",
    "BS*": "Blue Straggler Star",
    "RG*": "Red Giant Branch star",
    "AB*": "Asymptotic Giant Branch Star (He-burning)",
    "C*": "Carbon Star",
    "S*": "S Star",
    "sg*": "Evolved supergiant star",
    "s*r": "Red supergiant star",
    "s*y": "Yellow supergiant star",
    "s*b": "Blue supergiant star",
    "HS*": "Hot subdwarf",
    "pA*": "Post-AGB Star (proto-PN)",
    "WD*": "White Dwarf",
    "LM*": "Low-mass star (M<1solMass)",
    "BD*": "Brown Dwarf (M<0.08solMass)",
    "N*": "Confirmed Neutron Star",
    "OH*": "OH/IR star",
    "TT*": "T Tau-type Star",
    "WR*": "Wolf-Rayet Star",
    "PM*": "High proper-motion Star",
    "HV*": "High-velocity Star",
    "V*": "Variable Star",
    "Ir*": "Variable Star of irregular type",
    "Or*": "Variable Star of Orion Type",
    "Er*": "Eruptive variable Star",
    "RC*": "Variable Star of R CrB type",
    "RC?": "Variable Star of R CrB type candidate",
    "Ro*": "Rotationally variable Star",
    "a2*": "Variable Star of alpha2 CVn type",
    "Psr": "Pulsar",
    "BY*": "Variable of BY Dra type",
    "RS*": "Variable of RS CVn type",
    "Pu*": "Pulsating variable Star",
    "RR*": "Variable Star of RR Lyr type",
    "Ce*": "Cepheid variable Star",
    "dS*": "Variable Star of delta Sct type",
    "RV*": "Variable Star of RV Tau type",
    "WV*": "Variable Star of W Vir type",
    "bC*": "Variable Star of beta Cep type",
    "cC*": "Classical Cepheid (delta Cep type)",
    "gD*": "Variable Star of gamma Dor type",
    "SX*": "Variable Star of SX Phe type (subdwarf)",
    "LP*": "Long-period variable star",
    "Mi*": "Variable Star of Mira Cet type",
    "SN*": "SuperNova",
    "su*": "Sub-stellar object",
    "Pl?": "Extra-solar Planet Candidate",
    "Pl": "Extra-solar Confirmed Planet",
    "G": "Galaxy",
    "PoG": "Part of a Galaxy",
    "GiC": "Galaxy in Cluster of Galaxies",
    "BiC": "Brightest galaxy in a Cluster (BCG)",
    "GiG": "Galaxy in Group of Galaxies",
    "GiP": "Galaxy in Pair of Galaxies",
    "rG": "Radio Galaxy",
    "H2G": "HII Galaxy",
    "LSB": "Low Surface Brightness Galaxy",
    "AG?": "Possible Active Galaxy Nucleus",
    "Q?": "Possible Quasar",
    "Bz?": "Possible Blazar",
    "BL?": "Possible BL Lac",
    "EmG": "Emission-line galaxy",
    "SBG": "Starburst Galaxy",
    "bCG": "Blue compact Galaxy",
    "LeI": "Gravitationally Lensed Image",
    "LeG": "Gravitationally Lensed Image of a Galaxy",
    "LeQ": "Gravitationally Lensed Image of a Quasar",
    "AGN": "Active Galaxy Nucleus",
    "LIN": "LINER-type Active Galaxy Nucleus",
    "SyG": "Seyfert Galaxy",
    "Sy1": "Seyfert 1 Galaxy",
    "Sy2": "Seyfert 2 Galaxy",
    "Bla": "Blazar",
    "BLL": "BL Lac - type object",
    "OVV": "Optically Violently Variable object",
    "QSO": "Quasar"
}


# Configure Simbad to include the necessary fields, including redshift
Simbad.add_votable_fields('otype', 'otypes', 'diameter', 'z_value')
Simbad.ROW_LIMIT = 0  # Remove row limit for full results
Simbad.TIMEOUT = 60  # Increase timeout for long queries

# Astrometry.net API constants
ASTROMETRY_API_URL = "http://nova.astrometry.net/api/"
ASTROMETRY_API_KEY_FILE = "astrometry_api_key.txt"


def load_api_key():
    """Load the API key from a file, if it exists."""
    if os.path.exists(ASTROMETRY_API_KEY_FILE):
        with open(ASTROMETRY_API_KEY_FILE, 'r') as file:
            return file.read().strip()
    return None  # Return None if the file doesn't exist

def save_api_key(api_key):
    """Save the API key to a file."""
    with open(ASTROMETRY_API_KEY_FILE, 'w') as file:
        file.write(api_key)

def load_image(filename):
    try:
        file_extension = filename.lower().split('.')[-1]
        bit_depth = None
        is_mono = True
        original_header = None

        if file_extension in ['tif', 'tiff']:
            image = tiff.imread(filename)
            print(f"Loaded TIFF image with dtype: {image.dtype}")
            
            if image.dtype == np.uint16:
                image = image.astype(np.float32) / 65535.0
                bit_depth = "16-bit"
            elif image.dtype == np.uint32:
                image = image.astype(np.float32) / 4294967295.0
                bit_depth = "32-bit unsigned"
            else:
                image = image.astype(np.float32)
                bit_depth = "32-bit floating point"

            print(f"Final bit depth set to: {bit_depth}")

            # Check if the image has an alpha channel and remove it if necessary
            if image.shape[-1] == 4:
                print("Detected alpha channel in TIFF. Removing it.")
                image = image[:, :, :3]  # Keep only the first 3 channels (RGB)

            if len(image.shape) == 2:
                image = np.stack([image] * 3, axis=-1)
                is_mono = True

        elif file_extension in ['fits', 'fit']:
            with fits.open(filename) as hdul:
                image_data = hdul[0].data
                original_header = hdul[0].header  # Capture the FITS header

                # Determine the bit depth based on the data type in the FITS file
                if image_data.dtype == np.uint16:
                    bit_depth = "16-bit"
                    print("Identified 16-bit FITS image.")
                elif image_data.dtype == np.uint8:
                    bit_depth = "8-bit"
                    print("Identified 8-bit FITS image")
                elif image_data.dtype == np.float32:
                    bit_depth = "32-bit floating point"
                    print("Identified 32-bit floating point FITS image.")
                elif image_data.dtype == np.uint32:
                    bit_depth = "32-bit unsigned"
                    print("Identified 32-bit unsigned FITS image.")

                # Handle 3D FITS data (e.g., RGB or multi-layered data)
                if image_data.ndim == 3 and image_data.shape[0] == 3:
                    image = np.transpose(image_data, (1, 2, 0))  # Reorder to (height, width, channels)

                    if bit_depth == "16-bit":
                        image = image.astype(np.float32) / 65535.0  # Normalize to [0, 1] for 16-bit
                    elif bit_depth == "8-bit":
                        image = image.astype(np.float32) / 255.0    
                    elif bit_depth == "32-bit unsigned":
                        # Apply BSCALE and BZERO if present
                        bzero = original_header.get('BZERO', 0)
                        bscale = original_header.get('BSCALE', 1)
                        image = image.astype(np.float32) * bscale + bzero

                        # Normalize based on the actual data range
                        image_min, image_max = image.min(), image.max()
                        image = (image - image_min) / (image_max - image_min)
                        print(f"Image range after applying BZERO and BSCALE (3D case): min={image_min}, max={image_max}")

                    is_mono = False  # RGB data

                # Handle 2D FITS data (grayscale)
                elif image_data.ndim == 2:
                    if bit_depth == "16-bit":
                        image = image_data.astype(np.float32) / 65535.0  # Normalize to [0, 1] for 16-bit
                    elif bit_depth == "8-bit":
                        image = image_data.astype(np.float32) / 255.1    
                    elif bit_depth == "32-bit unsigned":
                        # Apply BSCALE and BZERO if present
                        bzero = original_header.get('BZERO', 0)
                        bscale = original_header.get('BSCALE', 1)
                        image = image_data.astype(np.float32) * bscale + bzero

                        # Normalize based on the actual data range
                        image_min, image_max = image.min(), image.max()
                        image = (image - image_min) / (image_max - image_min)
                        print(f"Image range after applying BZERO and BSCALE (2D case): min={image_min}, max={image_max}")

                    elif bit_depth == "32-bit floating point":
                        image = image_data  # No normalization needed for 32-bit float

                    is_mono = True
                    image = np.stack([image] * 3, axis=-1)  # Convert to 3-channel for consistency
                else:
                    raise ValueError("Unsupported FITS format!")

        else:
            # For PNG, JPEG, or other standard image formats
            image = np.array(Image.open(filename).convert('RGB')).astype(np.float32) / 255.0
            is_mono = False

        return image, original_header, bit_depth, is_mono

    except Exception as e:
        print(f"Error reading image {filename}: {e}")
        return None, None, None, None


class CustomGraphicsView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.setMouseTracking(True)  # Enable mouse tracking
        self.setDragMode(QGraphicsView.NoDrag)  # Disable default drag mode to avoid hand cursor
        self.setCursor(Qt.ArrowCursor)  # Set default cursor to arrow
        self.drawing_item = None
        self.start_pos = None     
        self.annotation_items = []  # Store annotation items  
        self.drawing_measurement = False
        self.measurement_start = QPointF()     

        self.selected_object = None  # Initialize selected_object to None
        self.show_names = False 

        # Variables for drawing the circle
        self.circle_center = None
        self.circle_radius = 0
        self.drawing_circle = False  # Flag to check if we're currently drawing a circle
        self.dragging = False  # Flag to manage manual dragging


    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            if event.modifiers() == Qt.ControlModifier:
                # Start annotation mode with the current tool
                self.start_pos = self.mapToScene(event.pos())

                # Check which tool is currently selected
                if self.parent.current_tool == "Ellipse":
                    self.drawing_item = QGraphicsEllipseItem()
                    self.drawing_item.setPen(QPen(self.parent.selected_color, 2))
                    self.parent.main_scene.addItem(self.drawing_item)

                elif self.parent.current_tool == "Rectangle":
                    self.drawing_item = QGraphicsRectItem()
                    self.drawing_item.setPen(QPen(self.parent.selected_color, 2))
                    self.parent.main_scene.addItem(self.drawing_item)

                elif self.parent.current_tool == "Arrow":
                    self.drawing_item = QGraphicsLineItem()
                    self.drawing_item.setPen(QPen(self.parent.selected_color, 2))
                    self.parent.main_scene.addItem(self.drawing_item)

                elif self.parent.current_tool == "Freehand":
                    self.drawing_item = QGraphicsPathItem()
                    path = QPainterPath(self.start_pos)
                    self.drawing_item.setPath(path)
                    self.drawing_item.setPen(QPen(self.parent.selected_color, 2))
                    self.parent.main_scene.addItem(self.drawing_item)

                elif self.parent.current_tool == "Text":
                    text, ok = QInputDialog.getText(self, "Add Text", "Enter text:")
                    if ok and text:
                        text_item = QGraphicsTextItem(text)
                        text_item.setPos(self.start_pos)
                        text_item.setDefaultTextColor(self.parent.selected_color)  # Use selected color
                        text_item.setFont(self.parent.selected_font)  # Use selected font
                        self.parent.main_scene.addItem(text_item)
                        
                        # Store as ('text', text, position, color)
                        self.annotation_items.append(('text', text, self.start_pos, self.parent.selected_color))


                elif self.parent.current_tool == "Compass":
                    self.place_celestial_compass(self.start_pos)

            elif event.modifiers() == Qt.ShiftModifier:
                # Start drawing a circle for Shift+Click
                self.drawing_circle = True
                self.circle_center = self.mapToScene(event.pos())
                self.circle_radius = 0
                self.parent.status_label.setText("Drawing circle: Shift + Drag")
                self.update_circle()

            elif event.modifiers() == Qt.AltModifier:
                # Start celestial measurement for Alt+Click
                self.measurement_start = self.mapToScene(event.pos())
                self.drawing_measurement = True
                self.drawing_item = None  # Clear any active annotation item
    

            else:
                # Detect if an object circle was clicked without Shift or Ctrl
                scene_pos = self.mapToScene(event.pos())
                clicked_object = self.get_object_at_position(scene_pos)
                
                if clicked_object:
                    # Select the clicked object and redraw
                    self.parent.selected_object = clicked_object
                    self.select_object(clicked_object)
                    self.draw_query_results()
                    self.update_mini_preview()
                    
                    # Highlight the corresponding row in the TreeWidget
                    for i in range(self.parent.results_tree.topLevelItemCount()):
                        item = self.parent.results_tree.topLevelItem(i)
                        if item.text(2) == clicked_object["name"]:  # Assuming third element is 'Name'
                            self.parent.results_tree.setCurrentItem(item)
                            break
                else:
                    # Start manual dragging if no modifier is held
                    self.dragging = True
                    self.setCursor(Qt.ClosedHandCursor)  # Use closed hand cursor to indicate dragging
                    self.drag_start_pos = event.pos()  # Store starting position

        super().mousePressEvent(event)


    def mouseDoubleClickEvent(self, event):
        """Handle double-click event on an object in the main image to open SIMBAD or NED URL based on source."""
        scene_pos = self.mapToScene(event.pos())
        clicked_object = self.get_object_at_position(scene_pos)

        if clicked_object:
            object_name = clicked_object.get("name")  # Access 'name' key from the dictionary
            ra = float(clicked_object.get("ra"))  # Ensure RA is a float for precision
            dec = float(clicked_object.get("dec"))  # Ensure Dec is a float for precision
            source = clicked_object.get("source", "Simbad")  # Default to "Simbad" if source not specified

            if source == "Simbad" and object_name:
                # Open Simbad URL with encoded object name
                encoded_name = quote(object_name)
                url = f"https://simbad.cds.unistra.fr/simbad/sim-basic?Ident={encoded_name}&submit=SIMBAD+search"
                webbrowser.open(url)
            elif source == "Vizier":
                # Format the NED search URL with proper RA, Dec, and radius
                radius = 5 / 60  # Radius in arcminutes (5 arcseconds)
                dec_sign = "%2B" if dec >= 0 else "-"  # Determine sign for declination
                ned_url = (
                    f"http://ned.ipac.caltech.edu/conesearch?search_type=Near%20Position%20Search"
                    f"&ra={ra:.6f}d&dec={dec_sign}{abs(dec):.6f}d&radius={radius:.3f}"
                    "&in_csys=Equatorial&in_equinox=J2000.0"
                )
                webbrowser.open(ned_url)
            elif source == "Mast":
                # Open MAST URL using RA and Dec with a small radius for object lookup
                mast_url = f"https://mast.stsci.edu/portal/Mashup/Clients/Mast/Portal.html?searchQuery={ra}%2C{dec}%2Cradius%3D0.0006"
                webbrowser.open(mast_url)                
        else:
            super().mouseDoubleClickEvent(event)

    def mouseMoveEvent(self, event):
        scene_pos = self.mapToScene(event.pos())

        if self.drawing_circle:
            # Update the circle radius as the mouse moves
            self.circle_radius = np.sqrt(
                (scene_pos.x() - self.circle_center.x()) ** 2 +
                (scene_pos.y() - self.circle_center.y()) ** 2
            )
            self.update_circle()

        elif self.drawing_measurement:
            # Update the measurement line dynamically as the mouse moves
            if self.drawing_item:
                self.parent.main_scene.removeItem(self.drawing_item)  # Remove previous line if exists
            self.drawing_item = QGraphicsLineItem(QLineF(self.measurement_start, scene_pos))
            self.drawing_item.setPen(QPen(Qt.green, 2, Qt.DashLine))  # Use green dashed line for measurement
            self.parent.main_scene.addItem(self.drawing_item)

        elif self.drawing_item:
            # Update the current drawing item based on the selected tool and mouse position
            if isinstance(self.drawing_item, QGraphicsEllipseItem) and self.parent.current_tool == "Ellipse":
                # For Ellipse tool, update the ellipse dimensions
                rect = QRectF(self.start_pos, scene_pos).normalized()
                self.drawing_item.setRect(rect)

            elif isinstance(self.drawing_item, QGraphicsRectItem) and self.parent.current_tool == "Rectangle":
                # For Rectangle tool, update the rectangle dimensions
                rect = QRectF(self.start_pos, scene_pos).normalized()
                self.drawing_item.setRect(rect)

            elif isinstance(self.drawing_item, QGraphicsLineItem) and self.parent.current_tool == "Arrow":
                # For Arrow tool, set the line from start_pos to current mouse position
                line = QLineF(self.start_pos, scene_pos)
                self.drawing_item.setLine(line)

            elif isinstance(self.drawing_item, QGraphicsPathItem) and self.parent.current_tool == "Freehand":
                # For Freehand tool, add a line to the path to follow the mouse movement
                path = self.drawing_item.path()
                path.lineTo(scene_pos)
                self.drawing_item.setPath(path)

        elif self.dragging:
            # Handle manual dragging by scrolling the view
            delta = event.pos() - self.drag_start_pos
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            self.drag_start_pos = event.pos()
        else:
            # Update RA/Dec display as the cursor moves
            self.parent.update_ra_dec_from_mouse(event)
            
        super().mouseMoveEvent(event)
                

    def mouseReleaseEvent(self, event):
        if self.drawing_circle and event.button() == Qt.LeftButton:
            # Stop drawing the circle
            self.drawing_circle = False
            self.parent.circle_center = self.circle_center
            self.parent.circle_radius = self.circle_radius

            # Calculate RA/Dec for the circle center
            ra, dec = self.parent.calculate_ra_dec_from_pixel(self.circle_center.x(), self.circle_center.y())
            if ra is not None and dec is not None:
                self.parent.ra_label.setText(f"RA: {self.parent.convert_ra_to_hms(ra)}")
                self.parent.dec_label.setText(f"Dec: {self.parent.convert_dec_to_dms(dec)}")

                if self.parent.pixscale:
                    radius_arcmin = self.circle_radius * self.parent.pixscale / 60.0
                    self.parent.status_label.setText(
                        f"Circle set at center RA={ra:.6f}, Dec={dec:.6f}, radius={radius_arcmin:.2f} arcmin"
                    )
                else:
                    self.parent.status_label.setText("Pixscale not available for radius calculation.")
            else:
                self.parent.status_label.setText("Unable to determine RA/Dec due to missing WCS.")

            # Update circle data and redraw
            self.parent.update_circle_data()
            self.update_circle()

        elif self.drawing_measurement and event.button() == Qt.LeftButton:
            # Complete the measurement when the mouse is released
            self.drawing_measurement = False
            measurement_end = self.mapToScene(event.pos())

            # Calculate celestial distance between start and end points
            ra1, dec1 = self.parent.calculate_ra_dec_from_pixel(self.measurement_start.x(), self.measurement_start.y())
            ra2, dec2 = self.parent.calculate_ra_dec_from_pixel(measurement_end.x(), measurement_end.y())
            
            if ra1 is not None and dec1 is not None and ra2 is not None and dec2 is not None:
                # Compute the angular distance
                angular_distance = self.parent.calculate_angular_distance(ra1, dec1, ra2, dec2)
                distance_text = self.parent.format_distance_as_dms(angular_distance)

                # Create and add the line item for display
                measurement_line_item = QGraphicsLineItem(QLineF(self.measurement_start, measurement_end))
                measurement_line_item.setPen(QPen(Qt.green, 2, Qt.DashLine))
                self.parent.main_scene.addItem(measurement_line_item)

                # Create a midpoint position for the distance text
                midpoint = QPointF(
                    (self.measurement_start.x() + measurement_end.x()) / 2,
                    (self.measurement_start.y() + measurement_end.y()) / 2
                )

                # Create and add the text item at the midpoint
                text_item = QGraphicsTextItem(distance_text)
                text_item.setPos(midpoint)
                text_item.setDefaultTextColor(Qt.green)
                text_item.setFont(self.parent.selected_font)  # Use the selected font
                self.parent.main_scene.addItem(text_item)

                # Store the line and text in annotation items for future reference
                measurement_line = QLineF(self.measurement_start, measurement_end)
                self.annotation_items.append(('line', measurement_line))  # Store QLineF, not QGraphicsLineItem
                self.annotation_items.append(('text', distance_text, midpoint, Qt.green))

            # Clear the temporary measurement line item without removing the final line
            self.drawing_item = None



        elif self.drawing_item and event.button() == Qt.LeftButton:
            # Finalize the shape drawing and add its properties to annotation_items
            if isinstance(self.drawing_item, QGraphicsEllipseItem):
                rect = self.drawing_item.rect()
                color = self.drawing_item.pen().color()
                self.annotation_items.append(('ellipse', rect, color))
            elif isinstance(self.drawing_item, QGraphicsRectItem):
                rect = self.drawing_item.rect()
                color = self.drawing_item.pen().color()
                self.annotation_items.append(('rect', rect, color))
            elif isinstance(self.drawing_item, QGraphicsLineItem):
                line = self.drawing_item.line()
                color = self.drawing_item.pen().color()
                self.annotation_items.append(('line', line, color))
            elif isinstance(self.drawing_item, QGraphicsTextItem):
                pos = self.drawing_item.pos()
                text = self.drawing_item.toPlainText()
                color = self.drawing_item.defaultTextColor()
                self.annotation_items.append(('text', pos, text, color))
            elif isinstance(self.drawing_item, QGraphicsPathItem):  # Handle Freehand
                path = self.drawing_item.path()
                color = self.drawing_item.pen().color()
                self.annotation_items.append(('freehand', path, color))        

            # Clear the temporary drawing item
            self.drawing_item = None

        # Stop manual dragging and reset cursor to arrow
        self.dragging = False
        self.setCursor(Qt.ArrowCursor)
        
        # Update the mini preview to reflect any changes
        self.update_mini_preview()

        super().mouseReleaseEvent(event)


    def draw_measurement_line_and_label(self, distance_ddmmss):
        """Draw the measurement line and label with the celestial distance."""
        # Draw line
        line_item = QGraphicsLineItem(
            QLineF(self.measurement_start, self.measurement_end)
        )
        line_item.setPen(QPen(QColor(0, 255, 255), 2))  # Cyan color for measurement
        self.parent.main_scene.addItem(line_item)

        # Place distance text at the midpoint of the line
        midpoint = QPointF(
            (self.measurement_start.x() + self.measurement_end.x()) / 2,
            (self.measurement_start.y() + self.measurement_end.y()) / 2
        )
        text_item = QGraphicsTextItem(distance_ddmmss)
        text_item.setDefaultTextColor(QColor(0, 255, 255))  # Same color as line
        text_item.setPos(midpoint)
        self.parent.main_scene.addItem(text_item)
        
        # Append both line and text to annotation_items
        self.annotation_items.append(('line', line_item))
        self.annotation_items.append(('text', midpoint, distance_ddmmss, QColor(0, 255, 255)))


    
    def wheelEvent(self, event):
        """Handle zoom in and out with the mouse wheel."""
        if event.angleDelta().y() > 0:
            self.parent.zoom_in()
        else:
            self.parent.zoom_out()        

    def update_circle(self):
        """Draws the search circle on the main scene if circle_center and circle_radius are set."""
        if self.parent.main_image and self.circle_center is not None and self.circle_radius > 0:
            # Clear the main scene and add the main image back
            self.parent.main_scene.clear()
            self.parent.main_scene.addPixmap(self.parent.main_image)

            # Redraw all shapes and annotations from stored properties
            for item in self.annotation_items:
                if item[0] == 'ellipse':
                    rect = item[1]
                    color = item[2]
                    ellipse = QGraphicsEllipseItem(rect)
                    ellipse.setPen(QPen(color, 2))
                    self.parent.main_scene.addItem(ellipse)
                elif item[0] == 'rect':
                    rect = item[1]
                    color = item[2]
                    rect_item = QGraphicsRectItem(rect)
                    rect_item.setPen(QPen(color, 2))
                    self.parent.main_scene.addItem(rect_item)
                elif item[0] == 'line':
                    line = item[1]
                    color = item[2]
                    line_item = QGraphicsLineItem(line)
                    line_item.setPen(QPen(color, 2))
                    self.parent.main_scene.addItem(line_item)
                elif item[0] == 'text':
                    text = item[1]            # The text string
                    pos = item[2]             # A QPointF for the position
                    color = item[3]           # The color for the text

                    text_item = QGraphicsTextItem(text)
                    text_item.setPos(pos)
                    text_item.setDefaultTextColor(color)
                    text_item.setFont(self.parent.selected_font)
                    self.parent.main_scene.addItem(text_item)

                elif item[0] == 'freehand':  # Redraw Freehand
                    path = item[1]
                    color = item[2]
                    freehand_item = QGraphicsPathItem(path)
                    freehand_item.setPen(QPen(color, 2))
                    self.parent.main_scene.addItem(freehand_item)        

                elif item[0] == 'compass':
                    compass = item[1]
                    # North Line
                    north_line_coords = compass['north_line']
                    north_line_item = QGraphicsLineItem(
                        north_line_coords[0], north_line_coords[1], north_line_coords[2], north_line_coords[3]
                    )
                    north_line_item.setPen(QPen(Qt.red, 2))
                    self.parent.main_scene.addItem(north_line_item)
                    
                    # East Line
                    east_line_coords = compass['east_line']
                    east_line_item = QGraphicsLineItem(
                        east_line_coords[0], east_line_coords[1], east_line_coords[2], east_line_coords[3]
                    )
                    east_line_item.setPen(QPen(Qt.blue, 2))
                    self.parent.main_scene.addItem(east_line_item)
                    
                    # North Label
                    text_north = QGraphicsTextItem(compass['north_label'][2])
                    text_north.setPos(compass['north_label'][0], compass['north_label'][1])
                    text_north.setDefaultTextColor(Qt.red)
                    self.parent.main_scene.addItem(text_north)
                    
                    # East Label
                    text_east = QGraphicsTextItem(compass['east_label'][2])
                    text_east.setPos(compass['east_label'][0], compass['east_label'][1])
                    text_east.setDefaultTextColor(Qt.blue)
                    self.parent.main_scene.addItem(text_east)

                elif item[0] == 'measurement':  # Redraw celestial measurement line
                    line = item[1]
                    color = item[2]
                    text_position = item[3]
                    distance_text = item[4]
                    
                    # Draw the measurement line
                    measurement_line_item = QGraphicsLineItem(line)
                    measurement_line_item.setPen(QPen(color, 2, Qt.DashLine))  # Dashed line for measurement
                    self.parent.main_scene.addItem(measurement_line_item)
                    
                    # Draw the distance text label
                    text_item = QGraphicsTextItem(distance_text)
                    text_item.setPos(text_position)
                    text_item.setDefaultTextColor(color)
                    text_item.setFont(self.parent.selected_font)
                    self.parent.main_scene.addItem(text_item)                                
                        
            
            # Draw the search circle
            pen_circle = QPen(QColor(255, 0, 0), 2)
            self.parent.main_scene.addEllipse(
                int(self.circle_center.x() - self.circle_radius),
                int(self.circle_center.y() - self.circle_radius),
                int(self.circle_radius * 2),
                int(self.circle_radius * 2),
                pen_circle
            )
            self.update_mini_preview()
        else:
            # If circle is disabled (e.g., during save), clear without drawing
            self.parent.main_scene.clear()
            self.parent.main_scene.addPixmap(self.parent.main_image)


    def scrollContentsBy(self, dx, dy):
        """Called whenever the main preview scrolls, ensuring the green box updates in the mini preview."""
        super().scrollContentsBy(dx, dy)
        self.parent.update_green_box()

    def update_mini_preview(self):
        """Update the mini preview with the current view rectangle and any additional mirrored elements."""
        if self.parent.main_image:
            # Scale the main image to fit in the mini preview
            mini_pixmap = self.parent.main_image.scaled(
                self.parent.mini_preview.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            mini_painter = QPainter(mini_pixmap)

            try:
                # Define scale factors based on main image dimensions
                if self.parent.main_image.width() > 0 and self.parent.main_image.height() > 0:
                    scale_factor_x = mini_pixmap.width() / self.parent.main_image.width()
                    scale_factor_y = mini_pixmap.height() / self.parent.main_image.height()

                    # Draw the search circle if it's defined
                    if self.circle_center is not None and self.circle_radius > 0:
                        pen_circle = QPen(QColor(255, 0, 0), 2)
                        mini_painter.setPen(pen_circle)
                        mini_painter.drawEllipse(
                            int(self.circle_center.x() * scale_factor_x - self.circle_radius * scale_factor_x),
                            int(self.circle_center.y() * scale_factor_y - self.circle_radius * scale_factor_y),
                            int(self.circle_radius * 2 * scale_factor_x),
                            int(self.circle_radius * 2 * scale_factor_y)
                        )

                    # Draw the green box representing the current view
                    mini_painter.setPen(QPen(QColor(0, 255, 0), 2))
                    view_rect = self.parent.main_preview.mapToScene(
                        self.parent.main_preview.viewport().rect()
                    ).boundingRect()
                    mini_painter.drawRect(
                        int(view_rect.x() * scale_factor_x),
                        int(view_rect.y() * scale_factor_y),
                        int(view_rect.width() * scale_factor_x),
                        int(view_rect.height() * scale_factor_y)
                    )


                    # Draw dots for each result with a color based on selection status
                    for obj in self.parent.results:
                        ra, dec = obj['ra'], obj['dec']
                        x, y = self.parent.calculate_pixel_from_ra_dec(ra, dec)
                        if x is not None and y is not None:
                            # Change color to green if this is the selected object
                            dot_color = QColor(0, 255, 0) if obj == getattr(self.parent, 'selected_object', None) else QColor(255, 0, 0)
                            mini_painter.setPen(QPen(dot_color, 4))
                            mini_painter.drawPoint(
                                int(x * scale_factor_x),
                                int(y * scale_factor_y)
                            )

                    # Redraw annotation items on the mini preview
                    for item in self.annotation_items:
                        pen = QPen(self.parent.selected_color, 1)  # Use a thinner pen for mini preview
                        mini_painter.setPen(pen)

                        # Interpret item type and draw accordingly
                        if item[0] == 'ellipse':
                            rect = item[1]
                            mini_painter.drawEllipse(
                                int(rect.x() * scale_factor_x), int(rect.y() * scale_factor_y),
                                int(rect.width() * scale_factor_x), int(rect.height() * scale_factor_y)
                            )
                        elif item[0] == 'rect':
                            rect = item[1]
                            mini_painter.drawRect(
                                int(rect.x() * scale_factor_x), int(rect.y() * scale_factor_y),
                                int(rect.width() * scale_factor_x), int(rect.height() * scale_factor_y)
                            )
                        elif item[0] == 'line':
                            line = item[1]
                            mini_painter.drawLine(
                                int(line.x1() * scale_factor_x), int(line.y1() * scale_factor_y),
                                int(line.x2() * scale_factor_x), int(line.y2() * scale_factor_y)
                            )
                        elif item[0] == 'text':
                            text = item[1]            # The text string
                            pos = item[2]             # A QPointF for the position
                            color = item[3]           # The color for the text

                            # Create a smaller font for the mini preview
                            mini_font = QFont(self.parent.selected_font)
                            mini_font.setPointSize(int(self.parent.selected_font.pointSize() * 0.2))  # Scale down font size

                            mini_painter.setFont(mini_font)
                            mini_painter.setPen(color)  # Set the color for the text
                            mini_painter.drawText(
                                int(pos.x() * scale_factor_x), int(pos.y() * scale_factor_y),
                                text
                            )

                        elif item[0] == 'freehand':
                            # Scale the freehand path and draw it
                            path = item[1]
                            scaled_path = QPainterPath()
                            
                            # Scale each point in the path to fit the mini preview
                            for i in range(path.elementCount()):
                                point = path.elementAt(i)
                                if i == 0:
                                    scaled_path.moveTo(point.x * scale_factor_x, point.y * scale_factor_y)
                                else:
                                    scaled_path.lineTo(point.x * scale_factor_x, point.y * scale_factor_y)

                            mini_painter.drawPath(scaled_path)

                        elif item[0] == 'compass':
                            compass = item[1]
                            # Draw the North line
                            mini_painter.setPen(QPen(Qt.red, 1))
                            north_line = compass["north_line"]
                            mini_painter.drawLine(
                                int(north_line[0] * scale_factor_x), int(north_line[1] * scale_factor_y),
                                int(north_line[2] * scale_factor_x), int(north_line[3] * scale_factor_y)
                            )

                            # Draw the East line
                            mini_painter.setPen(QPen(Qt.blue, 1))
                            east_line = compass["east_line"]
                            mini_painter.drawLine(
                                int(east_line[0] * scale_factor_x), int(east_line[1] * scale_factor_y),
                                int(east_line[2] * scale_factor_x), int(east_line[3] * scale_factor_y)
                            )

                            # Draw North and East labels
                            mini_painter.setPen(QPen(Qt.red, 1))
                            north_label = compass["north_label"]
                            mini_painter.drawText(
                                int(north_label[0] * scale_factor_x), int(north_label[1] * scale_factor_y), north_label[2]
                            )

                            mini_painter.setPen(QPen(Qt.blue, 1))
                            east_label = compass["east_label"]
                            mini_painter.drawText(
                                int(east_label[0] * scale_factor_x), int(east_label[1] * scale_factor_y), east_label[2]
                            )                            

            finally:
                mini_painter.end()  # Ensure QPainter is properly ended

            self.parent.mini_preview.setPixmap(mini_pixmap)

    def place_celestial_compass(self, center):
        """Draw a celestial compass at a given point aligned with celestial North and East."""
        compass_radius = 50  # Length of the compass lines

        # Get the orientation in radians (assuming `self.parent.orientation` is in degrees)
        orientation_radians = math.radians(self.parent.orientation)

        # Calculate North vector (upwards, adjusted for orientation)
        north_dx = math.sin(orientation_radians) * compass_radius
        north_dy = -math.cos(orientation_radians) * compass_radius

        # Calculate East vector (rightwards, adjusted for orientation)
        east_dx = math.cos(orientation_radians) * -compass_radius
        east_dy = math.sin(orientation_radians) * -compass_radius

        # Draw North line
        north_line = QGraphicsLineItem(
            center.x(), center.y(),
            center.x() + north_dx, center.y() + north_dy
        )
        north_line.setPen(QPen(Qt.red, 2))
        self.parent.main_scene.addItem(north_line)

        # Draw East line
        east_line = QGraphicsLineItem(
            center.x(), center.y(),
            center.x() + east_dx, center.y() + east_dy
        )
        east_line.setPen(QPen(Qt.blue, 2))
        self.parent.main_scene.addItem(east_line)

        # Add labels for North and East
        text_north = QGraphicsTextItem("N")
        text_north.setDefaultTextColor(Qt.red)
        text_north.setPos(center.x() + north_dx - 10, center.y() + north_dy - 10)
        self.parent.main_scene.addItem(text_north)

        text_east = QGraphicsTextItem("E")
        text_east.setDefaultTextColor(Qt.blue)
        text_east.setPos(center.x() + east_dx - 15, center.y() + east_dy - 10)
        self.parent.main_scene.addItem(text_east)

        # Append all compass components as a tuple to annotation_items for later redrawing
        self.annotation_items.append((
            "compass", {
                "center": center,
                "north_line": (center.x(), center.y(), center.x() + north_dx, center.y() + north_dy),
                "east_line": (center.x(), center.y(), center.x() + east_dx, center.y() + east_dy),
                "north_label": (center.x() + north_dx - 10, center.y() + north_dy - 10, "N"),
                "east_label": (center.x() + east_dx - 15, center.y() + east_dy - 10, "E"),
                "orientation": self.parent.orientation
            }
        ))

    def draw_query_results(self):
        """Draw query results with or without names based on the show_names setting."""
        if self.parent.main_image:
            # Clear the main scene and re-add the main image
            self.parent.main_scene.clear()
            self.parent.main_scene.addPixmap(self.parent.main_image)

            # Redraw all shapes and annotations from stored properties
            for item in self.annotation_items:
                if item[0] == 'ellipse':
                    rect = item[1]
                    color = item[2]
                    ellipse = QGraphicsEllipseItem(rect)
                    ellipse.setPen(QPen(color, 2))
                    self.parent.main_scene.addItem(ellipse)
                elif item[0] == 'rect':
                    rect = item[1]
                    color = item[2]
                    rect_item = QGraphicsRectItem(rect)
                    rect_item.setPen(QPen(color, 2))
                    self.parent.main_scene.addItem(rect_item)
                elif item[0] == 'line':
                    line = item[1]
                    color = item[2]
                    line_item = QGraphicsLineItem(line)
                    line_item.setPen(QPen(color, 2))
                    self.parent.main_scene.addItem(line_item)
                elif item[0] == 'text':
                    text = item[1]            # The text string
                    pos = item[2]             # A QPointF for the position
                    color = item[3]           # The color for the text

                    text_item = QGraphicsTextItem(text)
                    text_item.setPos(pos)
                    text_item.setDefaultTextColor(color)
                    text_item.setFont(self.parent.selected_font)
                    self.parent.main_scene.addItem(text_item)

                elif item[0] == 'freehand':  # Redraw Freehand
                    path = item[1]
                    color = item[2]
                    freehand_item = QGraphicsPathItem(path)
                    freehand_item.setPen(QPen(color, 2))
                    self.parent.main_scene.addItem(freehand_item)                      
                elif item[0] == 'measurement':  # Redraw celestial measurement line
                    line = item[1]
                    color = item[2]
                    text_position = item[3]
                    distance_text = item[4]
                    
                    # Draw the measurement line
                    measurement_line_item = QGraphicsLineItem(line)
                    measurement_line_item.setPen(QPen(color, 2, Qt.DashLine))  # Dashed line for measurement
                    self.parent.main_scene.addItem(measurement_line_item)
                    
                    # Draw the distance text label
                    text_item = QGraphicsTextItem(distance_text)
                    text_item.setPos(text_position)
                    text_item.setDefaultTextColor(color)
                    text_item.setFont(self.parent.selected_font)
                    self.parent.main_scene.addItem(text_item)        
                elif item[0] == 'compass':
                    compass = item[1]
                    # North Line
                    north_line_coords = compass['north_line']
                    north_line_item = QGraphicsLineItem(
                        north_line_coords[0], north_line_coords[1], north_line_coords[2], north_line_coords[3]
                    )
                    north_line_item.setPen(QPen(Qt.red, 2))
                    self.parent.main_scene.addItem(north_line_item)
                    
                    # East Line
                    east_line_coords = compass['east_line']
                    east_line_item = QGraphicsLineItem(
                        east_line_coords[0], east_line_coords[1], east_line_coords[2], east_line_coords[3]
                    )
                    east_line_item.setPen(QPen(Qt.blue, 2))
                    self.parent.main_scene.addItem(east_line_item)
                    
                    # North Label
                    text_north = QGraphicsTextItem(compass['north_label'][2])
                    text_north.setPos(compass['north_label'][0], compass['north_label'][1])
                    text_north.setDefaultTextColor(Qt.red)
                    self.parent.main_scene.addItem(text_north)
                    
                    # East Label
                    text_east = QGraphicsTextItem(compass['east_label'][2])
                    text_east.setPos(compass['east_label'][0], compass['east_label'][1])
                    text_east.setDefaultTextColor(Qt.blue)
                    self.parent.main_scene.addItem(text_east)                               
            # Ensure the search circle is drawn if circle data is available
            if self.circle_center is not None and self.circle_radius > 0:
                self.update_circle()

            # Draw object annotations
            for obj in self.parent.results:
                ra, dec, name = obj["ra"], obj["dec"], obj["name"]
                x, y = self.parent.calculate_pixel_from_ra_dec(ra, dec)
                if x is not None and y is not None:
                    # Use green color if this is the selected object, otherwise use red
                    pen_color = QColor(0, 255, 0) if obj == self.selected_object else QColor(255, 0, 0)
                    pen = QPen(pen_color, 2)
                    self.parent.main_scene.addEllipse(int(x - 5), int(y - 5), 10, 10, pen)

                    # Conditionally draw names if the checkbox is checked
                    if self.parent.show_names:
                        text_item = QGraphicsTextItem(name)
                        text_item.setPos(x + 10, y + 10)
                        text_item.setDefaultTextColor(QColor(255, 255, 0))  # Set text color to white
                        self.parent.main_scene.addItem(text_item)
    

    def clear_query_results(self):
        """Clear query markers from the main image without removing annotations."""
        # Clear the main scene and add the main image back
        self.parent.main_scene.clear()
        if self.parent.main_image:
            self.parent.main_scene.addPixmap(self.parent.main_image)
        
        # Redraw the stored annotation items
        for item in self.annotation_items:
            if item[0] == 'ellipse':
                rect = item[1]
                color = item[2]
                ellipse = QGraphicsEllipseItem(rect)
                ellipse.setPen(QPen(color, 2))
                self.parent.main_scene.addItem(ellipse)
            elif item[0] == 'rect':
                rect = item[1]
                color = item[2]
                rect_item = QGraphicsRectItem(rect)
                rect_item.setPen(QPen(color, 2))
                self.parent.main_scene.addItem(rect_item)
            elif item[0] == 'line':
                line = item[1]
                color = item[2]
                line_item = QGraphicsLineItem(line)
                line_item.setPen(QPen(color, 2))
                self.parent.main_scene.addItem(line_item)
            elif item[0] == 'text':
                text = item[1]            # The text string
                pos = item[2]             # A QPointF for the position
                color = item[3]           # The color for the text

                text_item = QGraphicsTextItem(text)
                text_item.setPos(pos)
                text_item.setDefaultTextColor(color)
                text_item.setFont(self.parent.selected_font)
                self.parent.main_scene.addItem(text_item)

            elif item[0] == 'freehand':  # Redraw Freehand
                path = item[1]
                color = item[2]
                freehand_item = QGraphicsPathItem(path)
                freehand_item.setPen(QPen(color, 2))
                self.parent.main_scene.addItem(freehand_item)  
            elif item[0] == 'measurement':  # Redraw celestial measurement line
                line = item[1]
                color = item[2]
                text_position = item[3]
                distance_text = item[4]
                
                # Draw the measurement line
                measurement_line_item = QGraphicsLineItem(line)
                measurement_line_item.setPen(QPen(color, 2, Qt.DashLine))  # Dashed line for measurement
                self.parent.main_scene.addItem(measurement_line_item)
                
                # Draw the distance text label
                text_item = QGraphicsTextItem(distance_text)
                text_item.setPos(text_position)
                text_item.setDefaultTextColor(color)
                text_item.setFont(self.parent.selected_font)
                self.parent.main_scene.addItem(text_item)       
            elif item[0] == 'compass':
                compass = item[1]
                # North line
                north_line_item = QGraphicsLineItem(
                    compass['north_line'][0], compass['north_line'][1],
                    compass['north_line'][2], compass['north_line'][3]
                )
                north_line_item.setPen(QPen(Qt.red, 2))
                self.parent.main_scene.addItem(north_line_item)
                # East line
                east_line_item = QGraphicsLineItem(
                    compass['east_line'][0], compass['east_line'][1],
                    compass['east_line'][2], compass['east_line'][3]
                )
                east_line_item.setPen(QPen(Qt.blue, 2))
                self.parent.main_scene.addItem(east_line_item)
                # North label
                text_north = QGraphicsTextItem(compass['north_label'][2])
                text_north.setPos(compass['north_label'][0], compass['north_label'][1])
                text_north.setDefaultTextColor(Qt.red)
                self.parent.main_scene.addItem(text_north)
                # East label
                text_east = QGraphicsTextItem(compass['east_label'][2])
                text_east.setPos(compass['east_label'][0], compass['east_label'][1])
                text_east.setDefaultTextColor(Qt.blue)
                self.parent.main_scene.addItem(text_east)
        
        # Update the circle data, if any
        self.parent.update_circle_data()
                        

    def set_query_results(self, results):
        """Set the query results and redraw."""
        self.parent.results = results  # Store results as dictionaries
        self.draw_query_results()

    def get_object_at_position(self, pos):
        """Find the object at the given position in the main preview."""
        for obj in self.parent.results:
            ra, dec = obj["ra"], obj["dec"]
            x, y = self.parent.calculate_pixel_from_ra_dec(ra, dec)
            if x is not None and y is not None:
                if abs(pos.x() - x) <= 5 and abs(pos.y() - y) <= 5:
                    return obj
        return None


    def select_object(self, selected_obj):
        """Select or deselect the specified object and update visuals."""
        self.selected_object = selected_obj if self.selected_object != selected_obj else None
        self.draw_query_results()  # Redraw to reflect selection

        # Update the TreeWidget selection in MainWindow
        for i in range(self.parent.results_tree.topLevelItemCount()):
            item = self.parent.results_tree.topLevelItem(i)
            if item.text(2) == selected_obj["name"]:  # Assuming 'name' is the unique identifier
                self.parent.results_tree.setCurrentItem(item if self.selected_object else None)
                break

    def undo_annotation(self):
        """Remove the last annotation item from the scene and annotation_items list."""
        if self.annotation_items:
            # Remove the last item from annotation_items
            self.annotation_items.pop()

            # Clear the scene and redraw all annotations except the last one
            self.parent.main_scene.clear()
            if self.parent.main_image:
                self.parent.main_scene.addPixmap(self.parent.main_image)

            # Redraw remaining annotations
            self.redraw_annotations()

            # Optionally, update the mini preview to reflect changes
            self.update_mini_preview()

    def clear_annotations(self):
        """Clear all annotation items from the scene and annotation_items list."""
        # Clear all items in annotation_items and update the scene
        self.annotation_items.clear()
        self.parent.main_scene.clear()
        
        # Redraw only the main image
        if self.parent.main_image:
            self.parent.main_scene.addPixmap(self.parent.main_image)

        # Optionally, update the mini preview to reflect changes
        self.update_mini_preview()

    def redraw_annotations(self):
        """Helper function to redraw all annotations from annotation_items."""
        for item in self.annotation_items:
            if item[0] == 'ellipse':
                rect = item[1]
                color = item[2]
                ellipse = QGraphicsEllipseItem(rect)
                ellipse.setPen(QPen(color, 2))
                self.parent.main_scene.addItem(ellipse)
            elif item[0] == 'rect':
                rect = item[1]
                color = item[2]
                rect_item = QGraphicsRectItem(rect)
                rect_item.setPen(QPen(color, 2))
                self.parent.main_scene.addItem(rect_item)
            elif item[0] == 'line':
                line = item[1]
                color = item[2]
                line_item = QGraphicsLineItem(line)
                line_item.setPen(QPen(color, 2))
                self.parent.main_scene.addItem(line_item)
            elif item[0] == 'text':
                text = item[1]            # The text string
                pos = item[2]             # A QPointF for the position
                color = item[3]           # The color for the text

                text_item = QGraphicsTextItem(text)
                text_item.setPos(pos)
                text_item.setDefaultTextColor(color)
                text_item.setFont(self.parent.selected_font)
                self.parent.main_scene.addItem(text_item)

            elif item[0] == 'freehand':  # Redraw Freehand
                path = item[1]
                color = item[2]
                freehand_item = QGraphicsPathItem(path)
                freehand_item.setPen(QPen(color, 2))
                self.parent.main_scene.addItem(freehand_item) 
            elif item[0] == 'measurement':  # Redraw celestial measurement line
                line = item[1]
                color = item[2]
                text_position = item[3]
                distance_text = item[4]
                
                # Draw the measurement line
                measurement_line_item = QGraphicsLineItem(line)
                measurement_line_item.setPen(QPen(color, 2, Qt.DashLine))  # Dashed line for measurement
                self.parent.main_scene.addItem(measurement_line_item)
                
                # Draw the distance text label
                text_item = QGraphicsTextItem(distance_text)
                text_item.setPos(text_position)
                text_item.setDefaultTextColor(color)
                text_item.setFont(self.parent.selected_font)
                self.parent.main_scene.addItem(text_item)                                        
            elif item[0] == 'compass':
                compass = item[1]
                # Redraw north line
                north_line_item = QGraphicsLineItem(
                    compass['north_line'][0], compass['north_line'][1],
                    compass['north_line'][2], compass['north_line'][3]
                )
                north_line_item.setPen(QPen(Qt.red, 2))
                self.parent.main_scene.addItem(north_line_item)
                
                # Redraw east line
                east_line_item = QGraphicsLineItem(
                    compass['east_line'][0], compass['east_line'][1],
                    compass['east_line'][2], compass['east_line'][3]
                )
                east_line_item.setPen(QPen(Qt.blue, 2))
                self.parent.main_scene.addItem(east_line_item)
                
                # Redraw labels
                text_north = QGraphicsTextItem(compass['north_label'][2])
                text_north.setPos(compass['north_label'][0], compass['north_label'][1])
                text_north.setDefaultTextColor(Qt.red)
                self.parent.main_scene.addItem(text_north)
                
                text_east = QGraphicsTextItem(compass['east_label'][2])
                text_east.setPos(compass['east_label'][0], compass['east_label'][1])
                text_east.setDefaultTextColor(Qt.blue)
                self.parent.main_scene.addItem(text_east)        


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("What's In My Image")
        self.setGeometry(100, 100, 1200, 800)
        # Track the theme status
        self.is_dark_mode = True

        self.circle_center = None
        self.circle_radius = 0    
        self.show_names = False  # Boolean to toggle showing names on the main image
        self.max_results = 100  # Default maximum number of query results     
        self.current_tool = None  # Track the active annotation tool
            

        main_layout = QHBoxLayout()

        # Left Column Layout
        left_panel = QVBoxLayout()

        # Create the instruction QLabel for search region
        self.title_label = QLabel("What's In My Image")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet("font-size: 20px; color: lightgrey;")    
        left_panel.addWidget(self.title_label)    

      
        
        # Load button
        self.load_button = QPushButton("Load Image")
        self.load_button.setIcon(QApplication.style().standardIcon(QStyle.SP_FileDialogStart))
        self.load_button.clicked.connect(self.open_image)
        left_panel.addWidget(self.load_button)

        # Create the instruction QLabel for search region
        search_region_instruction_label = QLabel("Shift+Click to define a search region")
        search_region_instruction_label.setAlignment(Qt.AlignCenter)
        search_region_instruction_label.setStyleSheet("font-size: 15px; color: gray;")

        # Add this QLabel to your layout at the appropriate position above RA/Dec
        left_panel.addWidget(search_region_instruction_label)  



        # Query Simbad button
        self.query_button = QPushButton("Query Simbad")
        self.query_button.setIcon(QApplication.style().standardIcon(QStyle.SP_DialogApplyButton))
        left_panel.addWidget(self.query_button)
        self.query_button.clicked.connect(lambda: self.query_simbad(self.get_defined_radius()))


        # Create a horizontal layout for the show names checkbox and clear results button
        show_clear_layout = QHBoxLayout()

        # Create the Show Object Names checkbox
        self.show_names_checkbox = QCheckBox("Show Object Names")
        self.show_names_checkbox.stateChanged.connect(self.toggle_object_names)  # Connect to a function to toggle names
        show_clear_layout.addWidget(self.show_names_checkbox)

        # Create the Clear Results button
        self.clear_results_button = QPushButton("Clear Results")
        self.clear_results_button.setIcon(QApplication.style().standardIcon(QStyle.SP_DialogCloseButton))
        self.clear_results_button.clicked.connect(self.clear_search_results)  # Connect to a function to clear results
        show_clear_layout.addWidget(self.clear_results_button)

        # Add this horizontal layout to the left panel layout (or wherever you want it to appear)
        left_panel.addLayout(show_clear_layout)   

        # Create a horizontal layout for the two buttons
        button_layout = QHBoxLayout()

        # Show Visible Objects Only button
        self.toggle_visible_objects_button = QPushButton("Show Visible Objects Only")
        self.toggle_visible_objects_button.setCheckable(True)  # Toggle button state
        self.toggle_visible_objects_button.setIcon(QIcon(eye_icon_path))
        self.toggle_visible_objects_button.clicked.connect(self.filter_visible_objects)
        self.toggle_visible_objects_button.setToolTip("Toggle the visibility of objects based on brightness.")
        button_layout.addWidget(self.toggle_visible_objects_button)

        # Save CSV button
        self.save_csv_button = QPushButton("Save CSV")
        self.save_csv_button.setIcon(QIcon(csv_icon_path))
        self.save_csv_button.clicked.connect(self.save_results_as_csv)
        button_layout.addWidget(self.save_csv_button)

        # Add the button layout to the left panel or main layout
        left_panel.addLayout(button_layout)  

        # Advanced Search Button
        self.advanced_search_button = QPushButton("Advanced Search")
        self.advanced_search_button.setIcon(QApplication.style().standardIcon(QStyle.SP_FileDialogDetailedView))
        self.advanced_search_button.setCheckable(True)
        self.advanced_search_button.clicked.connect(self.toggle_advanced_search)
        left_panel.addWidget(self.advanced_search_button)

        # Advanced Search Panel (initially hidden)
        self.advanced_search_panel = QVBoxLayout()
        self.advanced_search_panel_widget = QWidget()
        self.advanced_search_panel_widget.setLayout(self.advanced_search_panel)
        self.advanced_search_panel_widget.setFixedWidth(300)
        self.advanced_search_panel_widget.setVisible(False)  # Hide initially        

        # Status label
        self.status_label = QLabel("Status: Ready")
        left_panel.addWidget(self.status_label)

        # Create a horizontal layout
        button_layout = QHBoxLayout()

        # Copy RA/Dec to Clipboard button
        self.copy_button = QPushButton("Copy RA/Dec to Clipboard", self)
        self.copy_button.setIcon(QApplication.style().standardIcon(QStyle.SP_CommandLink))
        self.copy_button.clicked.connect(self.copy_ra_dec_to_clipboard)
        button_layout.addWidget(self.copy_button)

        # Settings button (wrench icon)
        self.settings_button = QPushButton()
        self.settings_button.setIcon(QIcon(icon_path))  # Adjust icon path as needed
        self.settings_button.clicked.connect(self.open_settings_dialog)
        button_layout.addWidget(self.settings_button)

        # Add the horizontal layout to the main layout or the desired parent layout
        left_panel.addLayout(button_layout)
        
         # Save Plate Solved Fits Button
        self.save_plate_solved_button = QPushButton("Save Plate Solved Fits")
        self.save_plate_solved_button.setIcon(QIcon(disk_icon_path))
        self.save_plate_solved_button.clicked.connect(self.save_plate_solved_fits)
        left_panel.addWidget(self.save_plate_solved_button)       

        # RA/Dec Labels
        ra_dec_layout = QHBoxLayout()
        self.ra_label = QLabel("RA: N/A")
        self.dec_label = QLabel("Dec: N/A")
        self.orientation_label = QLabel("Orientation: N/A°")
        ra_dec_layout.addWidget(self.ra_label)
        ra_dec_layout.addWidget(self.dec_label)
        ra_dec_layout.addWidget(self.orientation_label)
        left_panel.addLayout(ra_dec_layout)

        # Mini Preview
        self.mini_preview = QLabel("Mini Preview")
        self.mini_preview.setFixedSize(300, 300)
        self.mini_preview.mousePressEvent = self.on_mini_preview_press
        self.mini_preview.mouseMoveEvent = self.on_mini_preview_drag
        self.mini_preview.mouseReleaseEvent = self.on_mini_preview_release
        left_panel.addWidget(self.mini_preview)

        # Toggle Theme button
        self.theme_toggle_button = QPushButton("Switch to Light Theme")
        self.theme_toggle_button.clicked.connect(self.toggle_theme)
        left_panel.addWidget(self.theme_toggle_button)        

        footer_label = QLabel("""
            Written by Franklin Marek<br>
            <a href='http://www.setiastro.com'>www.setiastro.com</a>
        """)
        footer_label.setAlignment(Qt.AlignCenter)
        footer_label.setOpenExternalLinks(True)
        footer_label.setStyleSheet("font-size: 10px;")
        left_panel.addWidget(footer_label)

        # Right Column Layout
        right_panel = QVBoxLayout()

        # Zoom buttons above the main preview
        zoom_controls_layout = QHBoxLayout()
        self.zoom_in_button = QPushButton("Zoom In")
        self.zoom_in_button.clicked.connect(self.zoom_in)
        self.zoom_out_button = QPushButton("Zoom Out")
        self.zoom_out_button.clicked.connect(self.zoom_out)
        zoom_controls_layout.addWidget(self.zoom_in_button)
        zoom_controls_layout.addWidget(self.zoom_out_button)
        right_panel.addLayout(zoom_controls_layout)        

        # Main Preview
        self.main_preview = CustomGraphicsView(self)
        self.main_scene = QGraphicsScene(self.main_preview)
        self.main_preview.setScene(self.main_scene)
        self.main_preview.setRenderHint(QPainter.Antialiasing)
        self.main_preview.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        right_panel.addWidget(self.main_preview)

        # Save Annotated Image and Save Collage of Objects Buttons in a Horizontal Layout between main image and treebox
        save_buttons_layout = QHBoxLayout()

        # Button to toggle annotation tools section
        self.show_annotations_button = QPushButton("Show Annotation Tools")
        self.show_annotations_button.setIcon(QApplication.style().standardIcon(QStyle.SP_DialogResetButton))
        self.show_annotations_button.clicked.connect(self.toggle_annotation_tools)
        save_buttons_layout.addWidget(self.show_annotations_button)
        
        self.save_annotated_button = QPushButton("Save Annotated Image")
        self.save_annotated_button.setIcon(QIcon(annotated_path))
        self.save_annotated_button.clicked.connect(self.save_annotated_image)
        save_buttons_layout.addWidget(self.save_annotated_button)
        
        self.save_collage_button = QPushButton("Save Collage of Objects")
        self.save_collage_button.setIcon(QIcon(collage_path))
        self.save_collage_button.clicked.connect(self.save_collage_of_objects)
        save_buttons_layout.addWidget(self.save_collage_button)

        right_panel.addLayout(save_buttons_layout)        

        # Connect scroll events to update the green box in the mini preview
        self.main_preview.verticalScrollBar().valueChanged.connect(self.main_preview.update_mini_preview)
        self.main_preview.horizontalScrollBar().valueChanged.connect(self.main_preview.update_mini_preview)

        # Tree Widget for results
        self.results_tree = QTreeWidget()
        self.results_tree.setHeaderLabels(["RA", "Dec", "Name", "Diameter", "Type", "Long Type", "Redshift", "Comoving Radial Distance (GLy)"])
        self.results_tree.setFixedHeight(150)
        self.results_tree.itemClicked.connect(self.on_tree_item_clicked)
        self.results_tree.itemDoubleClicked.connect(self.on_tree_item_double_clicked) 
        self.results_tree.setSortingEnabled(True)  # <-- Add this line
        right_panel.addWidget(self.results_tree)

        self.annotation_buttons = []

        # Annotation Tools Section (initially hidden)
        self.annotation_tools_section = QWidget()
        annotation_tools_layout = QGridLayout(self.annotation_tools_section)

        annotation_instruction_label = QLabel("Ctrl+Click to add items, Alt+Click to measure distance")
        annotation_instruction_label.setAlignment(Qt.AlignCenter)
        annotation_instruction_label.setStyleSheet("font-size: 10px; color: gray;")        

        self.draw_ellipse_button = QPushButton("Draw Ellipse")
        self.draw_ellipse_button.tool_name = "Ellipse"
        self.draw_ellipse_button.clicked.connect(lambda: self.set_tool("Ellipse"))
        self.annotation_buttons.append(self.draw_ellipse_button)

        self.freehand_button = QPushButton("Freehand (Lasso)")
        self.freehand_button.tool_name = "Freehand"
        self.freehand_button.clicked.connect(lambda: self.set_tool("Freehand"))
        self.annotation_buttons.append(self.freehand_button)

        self.draw_rectangle_button = QPushButton("Draw Rectangle")
        self.draw_rectangle_button.tool_name = "Rectangle"
        self.draw_rectangle_button.clicked.connect(lambda: self.set_tool("Rectangle"))
        self.annotation_buttons.append(self.draw_rectangle_button)

        self.draw_arrow_button = QPushButton("Draw Arrow")
        self.draw_arrow_button.tool_name = "Arrow"
        self.draw_arrow_button.clicked.connect(lambda: self.set_tool("Arrow"))
        self.annotation_buttons.append(self.draw_arrow_button)

        self.place_compass_button = QPushButton("Place Celestial Compass")
        self.place_compass_button.tool_name = "Compass"
        self.place_compass_button.clicked.connect(lambda: self.set_tool("Compass"))
        self.annotation_buttons.append(self.place_compass_button)

        self.add_text_button = QPushButton("Add Text")
        self.add_text_button.tool_name = "Text"
        self.add_text_button.clicked.connect(lambda: self.set_tool("Text"))
        self.annotation_buttons.append(self.add_text_button)

        # Add Color and Font buttons
        self.color_button = QPushButton("Select Color")
        self.color_button.setIcon(QIcon(colorwheel_path))
        self.color_button.clicked.connect(self.select_color)

        self.font_button = QPushButton("Select Font")
        self.font_button.setIcon(QIcon(font_path))
        self.font_button.clicked.connect(self.select_font)

        # Undo button
        self.undo_button = QPushButton("Undo")
        self.undo_button.setIcon(QApplication.style().standardIcon(QStyle.SP_ArrowLeft))  # Left arrow icon for undo
        self.undo_button.clicked.connect(self.main_preview.undo_annotation)  # Connect to undo_annotation in CustomGraphicsView

        # Clear Annotations button
        self.clear_annotations_button = QPushButton("Clear Annotations")
        self.clear_annotations_button.setIcon(QApplication.style().standardIcon(QStyle.SP_TrashIcon))  # Trash icon
        self.clear_annotations_button.clicked.connect(self.main_preview.clear_annotations)  # Connect to clear_annotations in CustomGraphicsView

        # Add the instruction label to the top of the grid layout (row 0, spanning multiple columns)
        annotation_tools_layout.addWidget(annotation_instruction_label, 0, 0, 1, 4)  # Span 5 columns to center it

        # Shift all other widgets down by one row
        annotation_tools_layout.addWidget(self.draw_ellipse_button, 1, 0)
        annotation_tools_layout.addWidget(self.freehand_button, 1, 1)
        annotation_tools_layout.addWidget(self.draw_rectangle_button, 2, 0)
        annotation_tools_layout.addWidget(self.draw_arrow_button, 2, 1)
        annotation_tools_layout.addWidget(self.place_compass_button, 3, 0)
        annotation_tools_layout.addWidget(self.add_text_button, 3, 1)
        annotation_tools_layout.addWidget(self.color_button, 4, 0)
        annotation_tools_layout.addWidget(self.font_button, 4, 1)
        annotation_tools_layout.addWidget(self.undo_button, 1, 4)
        annotation_tools_layout.addWidget(self.clear_annotations_button, 2, 4)

        self.annotation_tools_section.setVisible(False)  # Initially hidden
        right_panel.addWidget(self.annotation_tools_section)

        # Advanced Search Panel
        self.advanced_param_label = QLabel("Advanced Search Parameters")
        self.advanced_search_panel.addWidget(self.advanced_param_label)

        # TreeWidget for object types
        self.object_tree = QTreeWidget()
        self.object_tree.setHeaderLabels(["Object Type", "Description"])
        self.object_tree.setColumnWidth(0, 150)
        self.object_tree.setSortingEnabled(True)

        # Populate the TreeWidget with object types from otype_long_name_lookup
        for obj_type, description in otype_long_name_lookup.items():
            item = QTreeWidgetItem([obj_type, description])
            item.setCheckState(0, Qt.Checked)  # Start with all items unchecked
            self.object_tree.addTopLevelItem(item)

        self.advanced_search_panel.addWidget(self.object_tree)

        # Buttons for toggling selections
        toggle_buttons_layout = QHBoxLayout()

        # Toggle All Button
        self.toggle_all_button = QPushButton("Toggle All")
        self.toggle_all_button.clicked.connect(self.toggle_all_items)
        toggle_buttons_layout.addWidget(self.toggle_all_button)

        # Toggle Stars Button
        self.toggle_stars_button = QPushButton("Toggle Stars")
        self.toggle_stars_button.clicked.connect(self.toggle_star_items)
        toggle_buttons_layout.addWidget(self.toggle_stars_button)

        # Toggle Galaxies Button
        self.toggle_galaxies_button = QPushButton("Toggle Galaxies")
        self.toggle_galaxies_button.clicked.connect(self.toggle_galaxy_items)
        toggle_buttons_layout.addWidget(self.toggle_galaxies_button)

        # Add toggle buttons to the advanced search layout
        self.advanced_search_panel.addLayout(toggle_buttons_layout)    

        # Add Simbad Search buttons below the toggle buttons
        search_button_layout = QHBoxLayout()

        self.simbad_defined_region_button = QPushButton("Search Defined Region")
        self.simbad_defined_region_button.clicked.connect(self.search_defined_region)
        search_button_layout.addWidget(self.simbad_defined_region_button)

        self.simbad_entire_image_button = QPushButton("Search Entire Image")
        self.simbad_entire_image_button.clicked.connect(self.search_entire_image)
        search_button_layout.addWidget(self.simbad_entire_image_button)

        self.advanced_search_panel.addLayout(search_button_layout)

        # Adding the "Deep Vizier Search" button below the other search buttons
        self.deep_vizier_button = QPushButton("Caution - Deep Vizier Search")
        self.deep_vizier_button.setIcon(QIcon(nuke_path))  # Assuming `nuke_path` is the correct path for the icon
        self.deep_vizier_button.setToolTip("Perform a deep search with Vizier. Caution: May return large datasets.")

        # Connect the button to a placeholder method for the deep Vizier search
        self.deep_vizier_button.clicked.connect(self.perform_deep_vizier_search)

        # Add the Deep Vizier button to the advanced search layout
        self.advanced_search_panel.addWidget(self.deep_vizier_button)

        self.mast_search_button = QPushButton("Search M.A.S.T Database")
        self.mast_search_button.setIcon(QIcon(hubble_path))
        self.mast_search_button.clicked.connect(self.perform_mast_search)
        self.mast_search_button.setToolTip("Search Hubble, JWST, Spitzer, TESS and More.")
        self.advanced_search_panel.addWidget(self.mast_search_button)                        

        # Combine left and right panels
        main_layout.addLayout(left_panel)
        main_layout.addLayout(right_panel)
        main_layout.addWidget(self.advanced_search_panel_widget)
        
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        self.image_path = None
        self.zoom_level = 1.0
        self.main_image = None
        self.green_box = None
        self.dragging = False
        self.center_ra = None
        self.center_dec = None
        self.pixscale = None
        self.orientation = None
        self.parity = None  
        self.circle_center = None
        self.circle_radius = 0  
        self.results = []
        self.wcs = None  # Initialize WCS to None
        # Initialize selected color and font with default values
        self.selected_color = QColor(Qt.red)  # Default annotation color
        self.selected_font = QFont("Arial", 12)  # Default font for text annotations        

        # Initialize the dark theme
        self.apply_light_theme()

    def set_tool(self, tool_name):
        """Sets the current tool and updates button states."""
        self.current_tool = tool_name

        # Reset button styles and highlight the selected button
        for button in self.annotation_buttons:
            if button.tool_name == tool_name:
                button.setStyleSheet("background-color: lightblue;")  # Highlight selected button
            else:
                button.setStyleSheet("")  # Reset other buttons


    def select_color(self):
        """Opens a color dialog to choose annotation color."""
        color = QColorDialog.getColor(self.selected_color, self, "Select Annotation Color")
        if color.isValid():
            self.selected_color = color

    def select_font(self):
        """Opens a font dialog to choose text annotation font."""
        font, ok = QFontDialog.getFont(self.selected_font, self, "Select Annotation Font")
        if ok:
            self.selected_font = font                

    def toggle_annotation_tools(self):
        """Toggle the visibility of the annotation tools section."""
        is_visible = self.annotation_tools_section.isVisible()
        self.annotation_tools_section.setVisible(not is_visible)
        self.show_annotations_button.setText("Hide Annotation Tools" if not is_visible else "Show Annotation Tools")

    def save_plate_solved_fits(self):
        """Save the plate-solved FITS file with WCS header data and the desired bit depth."""
        # Prompt user to select bit depth
        bit_depth, ok = QInputDialog.getItem(
            self, 
            "Select Bit Depth", 
            "Choose the bit depth for the FITS file:",
            ["8-bit", "16-bit", "32-bit"], 
            0, False
        )

        if not ok:
            return  # User cancelled the selection

        # Open file dialog to select where to save the FITS file
        output_image_path, _ = QFileDialog.getSaveFileName(
            self, "Save Plate Solved FITS", "", "FITS Files (*.fits *.fit)"
        )

        if not output_image_path:
            return  # User cancelled save file dialog

        # Verify WCS header data is available
        if not hasattr(self, 'wcs') or self.wcs is None:
            QMessageBox.warning(self, "WCS Data Missing", "WCS header data is not available.")
            return

        # Retrieve image data and WCS header
        image_data = self.image_data  # Raw image data
        wcs_header = self.wcs.to_header(relax=True)  # WCS header, including non-standard keywords
        combined_header = self.original_header.copy() if self.original_header else fits.Header()
        combined_header.update(wcs_header)  # Combine original header with WCS data

        # Convert image data based on selected bit depth
        if self.is_mono:
            # Grayscale (2D) image
            if bit_depth == "8-bit":
                scaled_image = (image_data[:, :, 0] / np.max(image_data) * 255).astype(np.uint8)
                combined_header['BITPIX'] = 8
            elif bit_depth == "16-bit":
                scaled_image = (image_data[:, :, 0] * 65535).astype(np.uint16)
                combined_header['BITPIX'] = 16
            elif bit_depth == "32-bit":
                scaled_image = image_data[:, :, 0].astype(np.float32)
                combined_header['BITPIX'] = -32
        else:
            # RGB (3D) image: Transpose to FITS format (channels, height, width)
            transformed_image = np.transpose(image_data, (2, 0, 1))
            if bit_depth == "8-bit":
                scaled_image = (transformed_image / np.max(transformed_image) * 255).astype(np.uint8)
                combined_header['BITPIX'] = 8
            elif bit_depth == "16-bit":
                scaled_image = (transformed_image * 65535).astype(np.uint16)
                combined_header['BITPIX'] = 16
            elif bit_depth == "32-bit":
                scaled_image = transformed_image.astype(np.float32)
                combined_header['BITPIX'] = -32

            # Update header to reflect 3D structure
            combined_header['NAXIS'] = 3
            combined_header['NAXIS1'] = transformed_image.shape[2]
            combined_header['NAXIS2'] = transformed_image.shape[1]
            combined_header['NAXIS3'] = transformed_image.shape[0]

        # Save the image with combined header (including WCS and original data)
        hdu = fits.PrimaryHDU(scaled_image, header=combined_header)
        try:
            hdu.writeto(output_image_path, overwrite=True)
            QMessageBox.information(self, "File Saved", f"FITS file saved as {output_image_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save FITS file: {str(e)}")



    def save_annotated_image(self):
        """Save the annotated image as a full or cropped view, excluding the search circle."""
        # Create a custom message box
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Save Annotated Image")
        msg_box.setText("Do you want to save the Full Image or Cropped Only?")
        
        # Add custom buttons
        full_image_button = msg_box.addButton("Save Full", QMessageBox.AcceptRole)
        cropped_image_button = msg_box.addButton("Save Cropped", QMessageBox.DestructiveRole)
        msg_box.addButton(QMessageBox.Cancel)

        # Show the message box and get the user's response
        msg_box.exec_()

        # Determine the save type based on the selected button
        if msg_box.clickedButton() == full_image_button:
            save_full_image = True
        elif msg_box.clickedButton() == cropped_image_button:
            save_full_image = False
        else:
            return  # User cancelled

        # Open a file dialog to select the file name and format
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Annotated Image",
            "",
            "JPEG (*.jpg *.jpeg);;PNG (*.png);;TIFF (*.tiff *.tif)"
        )
        
        if not file_path:
            return  # User cancelled the save dialog

        # Temporarily disable the search circle in the custom graphics view
        original_circle_center = self.main_preview.circle_center
        original_circle_radius = self.main_preview.circle_radius
        self.main_preview.circle_center = None  # Hide the circle temporarily
        self.main_preview.circle_radius = 0

        # Redraw annotations without the search circle
        self.main_preview.draw_query_results()

        # Create a QPixmap to render the annotations
        if save_full_image:
            # Save the entire main image with annotations
            pixmap = QPixmap(self.main_image.size())
            pixmap.fill(Qt.transparent)
            painter = QPainter(pixmap)
            self.main_scene.render(painter)  # Render the entire scene without the search circle
        else:
            # Save only the currently visible area (cropped view)
            rect = self.main_preview.viewport().rect()
            scene_rect = self.main_preview.mapToScene(rect).boundingRect()
            pixmap = QPixmap(int(scene_rect.width()), int(scene_rect.height()))
            pixmap.fill(Qt.transparent)
            painter = QPainter(pixmap)
            self.main_scene.render(painter, QRectF(0, 0, pixmap.width(), pixmap.height()), scene_rect)

        painter.end()  # End QPainter to finalize drawing

        # Restore the search circle in the custom graphics view
        self.main_preview.circle_center = original_circle_center
        self.main_preview.circle_radius = original_circle_radius
        self.main_preview.draw_query_results()  # Redraw the scene with the circle

        # Save the QPixmap as an image file in the selected format
        try:
            if pixmap.save(file_path, file_path.split('.')[-1].upper()):
                QMessageBox.information(self, "Save Successful", f"Annotated image saved as {file_path}")
            else:
                raise Exception("Failed to save image due to format or file path issues.")
        except Exception as e:
            QMessageBox.critical(self, "Save Failed", f"An error occurred while saving the image: {str(e)}")


    def save_collage_of_objects(self):
        """Save a collage of 128x128 pixel patches centered around each object, with dynamically spaced text below."""
        # Options for display
        options = ["Name", "RA", "Dec", "Short Type", "Long Type", "Redshift", "Comoving Distance"]

        # Create a custom dialog to select information to display
        dialog = QDialog(self)
        dialog.setWindowTitle("Select Information to Display")
        layout = QVBoxLayout(dialog)
        
        # Add checkboxes for each option
        checkboxes = {}
        for option in options:
            checkbox = QCheckBox(option)
            checkbox.setChecked(True)  # Default to checked
            layout.addWidget(checkbox)
            checkboxes[option] = checkbox

        # Add OK and Cancel buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        layout.addWidget(button_box)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)

        # Show the dialog and get the user's response
        if dialog.exec_() == QDialog.Rejected:
            return  # User cancelled

        # Determine which fields to display based on user selection
        selected_fields = [key for key, checkbox in checkboxes.items() if checkbox.isChecked()]

        # Calculate required vertical space for text based on number of selected fields
        text_row_height = 15
        text_block_height = len(selected_fields) * text_row_height
        patch_size = 128
        space_between_patches = max(64, text_block_height + 20)  # Ensure enough space for text between patches

        # Set parameters for collage layout
        number_of_objects = len(self.results)

        if number_of_objects == 0:
            QMessageBox.warning(self, "No Objects", "No objects available to create a collage.")
            return

        # Determine grid size for the collage
        grid_size = math.ceil(math.sqrt(number_of_objects))
        collage_width = patch_size * grid_size + space_between_patches * (grid_size - 1) + 128
        collage_height = patch_size * grid_size + space_between_patches * (grid_size - 1) + 128

        # Create an empty black RGB image for the collage
        collage_image = Image.new("RGB", (collage_width, collage_height), (0, 0, 0))

        # Temporarily disable annotations
        original_show_names = self.show_names
        original_circle_center = self.main_preview.circle_center
        original_circle_radius = self.main_preview.circle_radius
        self.show_names = False
        self.main_preview.circle_center = None
        self.main_preview.circle_radius = 0

        try:
            for i, obj in enumerate(self.results):
                # Calculate position in the grid
                row = i // grid_size
                col = i % grid_size
                offset_x = 64 + col * (patch_size + space_between_patches)
                offset_y = 64 + row * (patch_size + space_between_patches)

                # Calculate pixel coordinates around the object
                ra, dec = obj["ra"], obj["dec"]
                x, y = self.calculate_pixel_from_ra_dec(ra, dec)

                # Render the main image without annotations onto a QPixmap
                patch = QPixmap(self.main_image.size())
                patch.fill(Qt.black)
                painter = QPainter(patch)
                self.main_scene.clear()  # Clear any previous drawings on the scene
                self.main_scene.addPixmap(self.main_image)  # Add only the main image without annotations
                self.main_scene.render(painter)  # Render the scene onto the patch

                # End the painter early to prevent QPaintDevice errors
                painter.end()

                # Crop the relevant area for the object
                rect = QRectF(x - patch_size // 2, y - patch_size // 2, patch_size, patch_size)
                cropped_patch = patch.copy(rect.toRect())
                cropped_image = cropped_patch.toImage().scaled(patch_size, patch_size).convertToFormat(QImage.Format_RGB888)

                # Convert QImage to PIL format for adding to the collage
                bytes_img = cropped_image.bits().asstring(cropped_image.width() * cropped_image.height() * 3)
                pil_patch = Image.frombytes("RGB", (patch_size, patch_size), bytes_img)

                # Paste the patch in the correct location on the collage
                collage_image.paste(pil_patch, (offset_x, offset_y))

                # Draw the selected information below the patch
                draw = ImageDraw.Draw(collage_image)
                font = ImageFont.truetype("arial.ttf", 12)  # Adjust font path as needed
                text_y = offset_y + patch_size + 5

                for field in selected_fields:
                    # Retrieve data and only display if not "N/A"
                    if field == "Name" and obj.get("name") != "N/A":
                        text = obj["name"]
                    elif field == "RA" and obj.get("ra") is not None:
                        text = f"RA: {obj['ra']:.6f}"
                    elif field == "Dec" and obj.get("dec") is not None:
                        text = f"Dec: {obj['dec']:.6f}"
                    elif field == "Short Type" and obj.get("short_type") != "N/A":
                        text = f"Type: {obj['short_type']}"
                    elif field == "Long Type" and obj.get("long_type") != "N/A":
                        text = f"{obj['long_type']}"
                    elif field == "Redshift" and obj.get("redshift") != "N/A":
                        text = f"Redshift: {float(obj['redshift']):.5f}"  # Limit redshift to 5 decimal places
                    elif field == "Comoving Distance" and obj.get("comoving_distance") != "N/A":
                        text = f"Distance: {obj['comoving_distance']} GLy"
                    else:
                        continue  # Skip if field is not available or set to "N/A"

                    # Draw the text and increment the Y position
                    draw.text((offset_x + 10, text_y), text, (255, 255, 255), font=font)
                    text_y += text_row_height  # Space between lines

        finally:
            # Restore the original annotation and search circle settings
            self.show_names = original_show_names
            self.main_preview.circle_center = original_circle_center
            self.main_preview.circle_radius = original_circle_radius

        # Save the collage
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Collage of Objects", "", "JPEG (*.jpg *.jpeg);;PNG (*.png);;TIFF (*.tiff *.tif)"
        )

        if file_path:
            collage_image.save(file_path)
            QMessageBox.information(self, "Save Successful", f"Collage saved as {file_path}")


        # Restore the search circle in the custom graphics view
        self.main_preview.circle_center = original_circle_center
        self.main_preview.circle_radius = original_circle_radius
        self.main_preview.draw_query_results()  # Redraw the scene with the circle


    def get_selected_object_types(self):
        """Return a list of selected object types from the tree widget."""
        selected_types = []
        for i in range(self.object_tree.topLevelItemCount()):
            item = self.object_tree.topLevelItem(i)
            if item.checkState(0) == Qt.Checked:
                selected_types.append(item.text(0))  # Add the object type
        return selected_types
    
    def search_defined_region(self):
        """Perform a Simbad search for the defined region and filter by selected object types."""
        selected_types = self.get_selected_object_types()
        if not selected_types:
            QMessageBox.warning(self, "No Object Types Selected", "Please select at least one object type.")
            return

        # Calculate the radius in degrees for the defined region (circle radius)
        radius_deg = self.get_defined_radius()

        # Perform the Simbad search in the defined region with the calculated radius
        self.query_simbad(radius_deg)


    def search_entire_image(self):
        """Search the entire image using Simbad with selected object types."""
        selected_types = self.get_selected_object_types()  # Get selected types from the advanced search panel
        if not selected_types:
            QMessageBox.warning(self, "No Object Types Selected", "Please select at least one object type.")
            return

        # Calculate radius as the distance from the image center to a corner
        width, height = self.main_image.width(), self.main_image.height()
        center_x, center_y = width / 2, height / 2
        corner_x, corner_y = width, height  # Bottom-right corner
        # Calculate distance in pixels from center to corner
        radius_px = np.sqrt((corner_x - center_x) ** 2 + (corner_y - center_y) ** 2)
        # Convert radius from pixels to degrees
        radius_deg = float((radius_px * self.pixscale) / 3600.0)

        # Automatically set circle_center and circle_radius for the entire image
        self.circle_center = QPointF(center_x, center_y)  # Assuming QPointF is used
        self.circle_radius = radius_px  # Set this to allow the check in `query_simbad`

        # Perform the query with the calculated radius
        self.query_simbad(radius_deg, max_results=25000)

    def toggle_advanced_search(self):
        """Toggle visibility of the advanced search panel."""
        self.advanced_search_panel.setVisible(not self.advanced_search_panel.isVisible())

    def toggle_all_items(self):
        """Toggle selection for all items in the object tree."""
        for i in range(self.object_tree.topLevelItemCount()):
            item = self.object_tree.topLevelItem(i)
            new_state = Qt.Checked if item.checkState(0) == Qt.Unchecked else Qt.Unchecked
            item.setCheckState(0, new_state)

    def toggle_star_items(self):
        """Toggle selection for items related to stars."""
        star_keywords = ["star", "Eclipsing binary of W UMa type", "Spectroscopic binary",
                         "Variable of RS CVn type", "Mira candidate", "Long Period Variable candidate",
                         "Hot subdwarf", "Eclipsing Binary Candidate", "Eclipsing binary", 
                         "Cataclysmic Binary Candidate", "Possible Cepheid", "White Dwarf", 
                         "White Dwarf Candidate"]
        for i in range(self.object_tree.topLevelItemCount()):
            item = self.object_tree.topLevelItem(i)
            description = item.text(1).lower()
            object_type = item.text(0)
            if any(keyword.lower() in description for keyword in star_keywords) or "*" in object_type:
                new_state = Qt.Checked if item.checkState(0) == Qt.Unchecked else Qt.Unchecked
                item.setCheckState(0, new_state)

    def toggle_galaxy_items(self):
        """Toggle selection for items related to galaxies."""
        for i in range(self.object_tree.topLevelItemCount()):
            item = self.object_tree.topLevelItem(i)
            description = item.text(1).lower()
            if "galaxy" in description or "galaxies" in description:
                new_state = Qt.Checked if item.checkState(0) == Qt.Unchecked else Qt.Unchecked
                item.setCheckState(0, new_state)


    def toggle_advanced_search(self):
        """Toggle the visibility of the advanced search panel."""
        is_visible = self.advanced_search_panel_widget.isVisible()
        self.advanced_search_panel_widget.setVisible(not is_visible)

    def save_results_as_csv(self):
        """Save the results from the TreeWidget as a CSV file."""
        path, _ = QFileDialog.getSaveFileName(self, "Save CSV", "", "CSV Files (*.csv)")
        if path:
            with open(path, mode='w', newline='') as file:
                writer = csv.writer(file)
                # Write header
                writer.writerow(["RA", "Dec", "Name", "Diameter", "Type", "Long Type", "Redshift", "Comoving Radial Distance (GLy)"])

                # Write data from TreeWidget
                for i in range(self.results_tree.topLevelItemCount()):
                    item = self.results_tree.topLevelItem(i)
                    row_data = [item.text(column) for column in range(self.results_tree.columnCount())]
                    writer.writerow(row_data)

            QMessageBox.information(self, "CSV Saved", f"Results successfully saved to {path}")        

    def filter_visible_objects(self):
        """Filter objects based on visibility threshold."""
        if not self.main_image:  # Ensure there's an image loaded
            QMessageBox.warning(self, "No Image", "Please load an image first.")
            return

        n = 0.2  # Threshold multiplier, adjust as needed
        median, std_dev = self.calculate_image_statistics(self.main_image)

        # Remove objects below threshold from results
        filtered_results = []
        for obj in self.results:
            if self.is_marker_visible(obj, median, std_dev, n):
                filtered_results.append(obj)

        # Update the results and redraw the markers
        self.results = filtered_results
        self.update_results_tree()
        self.main_preview.draw_query_results()

    def calculate_image_statistics(self, image):
        """Calculate median and standard deviation for a grayscale image efficiently using OpenCV."""
        
        # Convert QPixmap to QImage if necessary
        qimage = image.toImage()

        # Convert QImage to a format compatible with OpenCV
        width = qimage.width()
        height = qimage.height()
        ptr = qimage.bits()
        ptr.setsize(height * width * 4)  # 4 channels (RGBA)
        img_array = np.array(ptr).reshape(height, width, 4)  # Convert to RGBA array

        # Convert to grayscale for analysis
        gray_image = cv2.cvtColor(img_array, cv2.COLOR_RGBA2GRAY)

        # Calculate median and standard deviation
        median = np.median(gray_image)
        _, std_dev = cv2.meanStdDev(gray_image)

        return median, std_dev[0][0]  # std_dev returns a 2D array, so we extract the single value
    
    def is_marker_visible(self, marker, median, std_dev, n):
        """Check if the marker's brightness is above the threshold."""
        threshold = median + n * std_dev
        check_size = 8  # Define a 4x4 region around the marker

        # Convert QPixmap to QImage to access pixel colors
        image = self.main_image.toImage()

        # Get marker coordinates in pixel space
        ra, dec = marker.get('ra'), marker.get('dec')
        if ra is not None and dec is not None:
            x, y = self.calculate_pixel_from_ra_dec(ra, dec)
            if x is None or y is None:
                return False  # Skip marker if it can't be converted to pixels
        else:
            return False

        # Calculate brightness in a 4x4 region around marker coordinates
        brightness_values = []
        for dx in range(-check_size // 2, check_size // 2):
            for dy in range(-check_size // 2, check_size // 2):
                px = x + dx
                py = y + dy
                if 0 <= px < image.width() and 0 <= py < image.height():
                    color = image.pixelColor(px, py)  # Get color from QImage
                    brightness = color.value() if color.isValid() else 0  # Adjust for grayscale
                    brightness_values.append(brightness)

        if brightness_values:
            average_brightness = sum(brightness_values) / len(brightness_values)
            return average_brightness > threshold
        else:
            return False



    def update_results_tree(self):
        """Refresh the TreeWidget to reflect current results."""
        self.results_tree.clear()
        for obj in self.results:
            item = QTreeWidgetItem([
                str(obj['ra']),
                str(obj['dec']),
                obj['name'],
                str(obj['diameter']),
                obj['short_type'],
                obj['long_type'],
                str(obj['redshift']),
                str(obj['comoving_distance'])
            ])
            self.results_tree.addTopLevelItem(item)

    def toggle_object_names(self, state):
        """Toggle the visibility of object names based on the checkbox state."""
        self.show_names = state == Qt.Checked
        self.main_preview.draw_query_results()  # Redraw to apply the change


    # Function to clear search results and remove markers
    def clear_search_results(self):
        """Clear the search results and remove all markers."""
        self.results_tree.clear()        # Clear the results from the tree
        self.results = []                # Clear the results list
        self.main_preview.results = []   # Clear results from the main preview
        self.main_preview.selected_object = None
        self.main_preview.draw_query_results()  # Redraw the main image without markers
        self.status_label.setText("Results cleared.")

    def on_tree_item_clicked(self, item):
        """Handle item click in the TreeWidget to highlight the associated object."""
        object_name = item.text(2)

        # Find the object in results
        selected_object = next(
            (obj for obj in self.results if obj.get("name") == object_name), None
        )

        if selected_object:
            # Set the selected object in MainWindow and update views
            self.selected_object = selected_object
            self.main_preview.select_object(selected_object)
            self.main_preview.draw_query_results()
            self.main_preview.update_mini_preview() 
            
            

    def on_tree_item_double_clicked(self, item):
        """Handle double-click event on a TreeWidget item to open SIMBAD or NED URL based on source."""
        object_name = item.text(2)  # Assuming 'Name' is in the third column
        ra = float(item.text(0).strip())  # Assuming RA is in the first column
        dec = float(item.text(1).strip())  # Assuming Dec is in the second column
        
        # Retrieve the entry directly from self.query_results
        entry = next((result for result in self.query_results if float(result['ra']) == ra and float(result['dec']) == dec), None)
        source = entry.get('source', 'Simbad') if entry else 'Simbad'  # Default to "Simbad" if entry not found

        if source == "Simbad" and object_name:
            # Open Simbad URL with encoded object name
            encoded_name = quote(object_name)
            simbad_url = f"https://simbad.cds.unistra.fr/simbad/sim-basic?Ident={encoded_name}&submit=SIMBAD+search"
            webbrowser.open(simbad_url)
        elif source == "Vizier":
            # Format the NED search URL with proper RA, Dec, and radius
            radius = 5 / 60  # Radius in arcminutes (5 arcseconds)
            dec_sign = "%2B" if dec >= 0 else "-"  # Determine sign for declination
            ned_url = f"http://ned.ipac.caltech.edu/conesearch?search_type=Near%20Position%20Search&ra={ra:.6f}d&dec={dec_sign}{abs(dec):.6f}d&radius={radius:.3f}&in_csys=Equatorial&in_equinox=J2000.0"
            webbrowser.open(ned_url)
        elif source == "Mast":
            # Open MAST URL using RA and Dec with a small radius for object lookup
            mast_url = f"https://mast.stsci.edu/portal/Mashup/Clients/Mast/Portal.html?searchQuery={ra}%2C{dec}%2Cradius%3D0.0006"
            webbrowser.open(mast_url)            

    def copy_ra_dec_to_clipboard(self):
        """Copy the currently displayed RA and Dec to the clipboard."""
        # Access the RA and Dec labels directly
        ra_text = self.ra_label.text()
        dec_text = self.dec_label.text()
        
        # Combine RA and Dec text for clipboard
        clipboard_text = f"{ra_text}, {dec_text}"
        
        clipboard = QApplication.instance().clipboard()
        clipboard.setText(clipboard_text)
        
        QMessageBox.information(self, "Copied", "Current RA/Dec copied to clipboard!")
    

    def open_image(self):
        self.image_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.jpeg *.tif *.tiff *.fit *.fits)")
        if self.image_path:
            img_array, original_header, bit_depth, is_mono = load_image(self.image_path)
            if img_array is not None:

                self.image_data = img_array
                self.original_header = original_header
                self.bit_depth = bit_depth
                self.is_mono = is_mono

                # Prepare image for display
                img = (img_array * 255).astype(np.uint8)
                height, width, _ = img.shape
                bytes_per_line = 3 * width
                qimg = QImage(img.tobytes(), width, height, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qimg)

                self.main_image = pixmap
                scaled_pixmap = pixmap.scaled(self.mini_preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.mini_preview.setPixmap(scaled_pixmap)

                self.main_scene.clear()
                self.main_scene.addPixmap(pixmap)
                self.main_preview.setSceneRect(QRectF(pixmap.rect()))
                self.zoom_level = 1.0
                self.main_preview.resetTransform()
                self.main_preview.centerOn(self.main_scene.sceneRect().center())
                self.update_green_box()

                # Initialize WCS from FITS header if it is a FITS file
                if self.image_path.lower().endswith(('.fits', '.fit')):
                    with fits.open(self.image_path) as hdul:
                        self.header = hdul[0].header
                        
                        try:
                            # Use only the first two dimensions for WCS
                            self.wcs = WCS(self.header, naxis=2, relax=True)
                            
                            # Calculate and set pixel scale
                            pixel_scale_matrix = self.wcs.pixel_scale_matrix
                            self.pixscale = np.sqrt(pixel_scale_matrix[0, 0]**2 + pixel_scale_matrix[1, 0]**2) * 3600  # arcsec/pixel
                            self.center_ra, self.center_dec = self.wcs.wcs.crval
                            self.wcs_header = self.wcs.to_header(relax=True)  # Store the full WCS header, including non-standard keywords
                            self.print_corner_coordinates()
                            
                            # Display WCS information
                            # Set orientation based on WCS data if available
                            if 'CROTA2' in self.header:
                                self.orientation = self.header['CROTA2']
                            else:
                                # Use calculate_orientation if CROTA2 is not present
                                self.orientation = calculate_orientation(self.header)
                                if self.orientation is None:
                                    print("Orientation: CD matrix elements not found in WCS header.")

                            # Update orientation label or print for debugging
                            if self.orientation is not None:
                                print(f"Orientation: {self.orientation:.2f}°")
                                self.orientation_label.setText(f"Orientation: {self.orientation:.2f}°")
                            else:
                                self.orientation_label.setText("Orientation: N/A")


                            print(f"WCS data loaded from FITS header: RA={self.center_ra}, Dec={self.center_dec}, "
                                f"Pixel Scale={self.pixscale} arcsec/px")
                            
                            
                        except ValueError as e:
                            print("Error initializing WCS:", e)
                            QMessageBox.warning(self, "WCS Error", "Failed to load WCS data from FITS header.")
                else:
                    # For non-FITS images (e.g., JPEG, PNG), prompt directly for a blind solve
                    self.prompt_blind_solve()


    def print_corner_coordinates(self):
        """Print the RA/Dec coordinates of the four corners of the image for debugging purposes."""
        if not hasattr(self, 'wcs'):
            print("WCS data is incomplete, cannot calculate corner coordinates.")
            return

        width = self.main_image.width()
        height = self.main_image.height()

        # Define the corner coordinates
        corners = {
            "Top-Left": (0, 0),
            "Top-Right": (width, 0),
            "Bottom-Left": (0, height),
            "Bottom-Right": (width, height)
        }

        print("Corner RA/Dec coordinates:")
        for corner_name, (x, y) in corners.items():
            ra, dec = self.calculate_ra_dec_from_pixel(x, y)
            ra_hms = self.convert_ra_to_hms(ra)
            dec_dms = self.convert_dec_to_dms(dec)
            print(f"{corner_name}: RA={ra_hms}, Dec={dec_dms}")

    def calculate_ra_dec_from_pixel(self, x, y):
        """Convert pixel coordinates (x, y) to RA/Dec using Astropy WCS."""
        if not hasattr(self, 'wcs'):
            print("WCS not initialized.")
            return None, None

        # Convert pixel coordinates to sky coordinates
        ra, dec = self.wcs.all_pix2world(x, y, 0)

        return ra, dec
                        


    def update_ra_dec_from_mouse(self, event):
        """Update RA and Dec based on mouse position over the main preview."""
        if self.main_image and self.wcs:
            pos = self.main_preview.mapToScene(event.pos())
            x, y = int(pos.x()), int(pos.y())
            
            if 0 <= x < self.main_image.width() and 0 <= y < self.main_image.height():
                ra, dec = self.calculate_ra_dec_from_pixel(x, y)
                ra_hms = self.convert_ra_to_hms(ra)
                dec_dms = self.convert_dec_to_dms(dec)

                # Update RA/Dec labels
                self.ra_label.setText(f"RA: {ra_hms}")
                self.dec_label.setText(f"Dec: {dec_dms}")
                
                # Update orientation label if available
                if self.orientation is not None:
                    self.orientation_label.setText(f"Orientation: {self.orientation:.2f}°")
                else:
                    self.orientation_label.setText("Orientation: N/A")
        else:
            self.ra_label.setText("RA: N/A")
            self.dec_label.setText("Dec: N/A")
            self.orientation_label.setText("Orientation: N/A")


    def convert_ra_to_hms(self, ra_deg):
        """Convert Right Ascension in degrees to Hours:Minutes:Seconds format."""
        ra_hours = ra_deg / 15.0  # Convert degrees to hours
        hours = int(ra_hours)
        minutes = int((ra_hours - hours) * 60)
        seconds = (ra_hours - hours - minutes / 60.0) * 3600
        return f"{hours:02d}h{minutes:02d}m{seconds:05.2f}s"

    def convert_dec_to_dms(self, dec_deg):
        """Convert Declination in degrees to Degrees:Minutes:Seconds format."""
        sign = "-" if dec_deg < 0 else "+"
        dec_deg = abs(dec_deg)
        degrees = int(dec_deg)
        minutes = int((dec_deg - degrees) * 60)
        seconds = (dec_deg - degrees - minutes / 60.0) * 3600
        degree_symbol = "\u00B0"
        return f"{sign}{degrees:02d}{degree_symbol}{minutes:02d}m{seconds:05.2f}s"                 

    def check_astrometry_data(self, header):
        return "CTYPE1" in header and "CTYPE2" in header

    def prompt_blind_solve(self):
        reply = QMessageBox.question(
            self, "Astrometry Data Missing",
            "No astrometry data found in the image. Would you like to perform a blind solve?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.perform_blind_solve()

    def perform_blind_solve(self):
        # Load or prompt for API key
        api_key = load_api_key()
        if not api_key:
            api_key, ok = QInputDialog.getText(self, "Enter API Key", "Please enter your Astrometry.net API key:")
            if ok and api_key:
                save_api_key(api_key)
            else:
                QMessageBox.warning(self, "API Key Required", "Blind solve cannot proceed without an API key.")
                return

        try:
            self.status_label.setText("Status: Logging in to Astrometry.net...")
            QApplication.processEvents()

            # Step 1: Login to Astrometry.net
            session_key = self.login_to_astrometry(api_key)

            self.status_label.setText("Status: Uploading image to Astrometry.net...")
            QApplication.processEvents()
            
            # Step 2: Upload the image and get submission ID
            subid = self.upload_image_to_astrometry(self.image_path, session_key)

            self.status_label.setText("Status: Waiting for job ID...")
            QApplication.processEvents()
            
            # Step 3: Poll for the job ID until it's available
            job_id = self.poll_submission_status(subid)
            if not job_id:
                raise TimeoutError("Failed to retrieve job ID from Astrometry.net after multiple attempts.")
            
            self.status_label.setText("Status: Job ID found, processing image...")
            QApplication.processEvents()

            # Step 4a: Poll for the calibration data, ensuring RA/Dec are available
            calibration_data = self.poll_calibration_data(job_id)
            if not calibration_data:
                raise TimeoutError("Calibration data did not complete in the expected timeframe.")
            
            # Set pixscale and other necessary attributes from calibration data
            self.pixscale = calibration_data.get('pixscale')

            self.status_label.setText("Status: Calibration complete, downloading WCS file...")
            QApplication.processEvents()

            # Step 4b: Download the WCS FITS file for complete calibration data
            wcs_header = self.retrieve_and_apply_wcs(job_id)
            if not wcs_header:
                raise TimeoutError("Failed to retrieve WCS FITS file from Astrometry.net.")

            self.status_label.setText("Status: Applying astrometric solution to the image...")
            QApplication.processEvents()

            # Apply calibration data to the WCS
            self.apply_wcs_header(wcs_header)
            self.status_label.setText("Status: Blind Solve Complete.")
            QMessageBox.information(self, "Blind Solve Complete", "Astrometric solution applied successfully.")
        except Exception as e:
            self.status_label.setText("Status: Blind Solve Failed.")
            QMessageBox.critical(self, "Blind Solve Failed", f"An error occurred: {str(e)}")


    def retrieve_and_apply_wcs(self, job_id):
        """Download the wcs.fits file from Astrometry.net, extract WCS header data, and apply it."""
        try:
            wcs_url = f"https://nova.astrometry.net/wcs_file/{job_id}"
            wcs_filepath = "wcs.fits"
            max_retries = 10
            delay = 10  # seconds
            
            for attempt in range(max_retries):
                # Attempt to download the file
                response = requests.get(wcs_url, stream=True)
                response.raise_for_status()

                # Save the WCS file locally
                with open(wcs_filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                # Check if the downloaded file is a valid FITS file
                try:
                    with fits.open(wcs_filepath, ignore_missing_simple=True, ignore_missing_end=True) as hdul:
                        # If it opens correctly, return the header
                        wcs_header = hdul[0].header
                        print("WCS header successfully retrieved.")
                        self.wcs = WCS(wcs_header)
                        return wcs_header
                except Exception as e:
                    print(f"Attempt {attempt + 1}: Failed to process WCS file - possibly HTML instead of FITS. Retrying in {delay} seconds...")
                    print(f"Error: {e}")
                    time.sleep(delay)  # Wait and retry
            
            print("Failed to download a valid WCS FITS file after multiple attempts.")
            return None

        except requests.exceptions.RequestException as e:
            print(f"Error downloading WCS file: {e}")
        except Exception as e:
            print(f"Error processing WCS file: {e}")
            
        return None



    def apply_wcs_header(self, wcs_header):
        """Apply WCS header to create a WCS object and set orientation."""
        self.wcs = WCS(wcs_header)  # Initialize WCS with header directly
        
        # Set orientation based on WCS data if available
        if 'CROTA2' in wcs_header:
            self.orientation = wcs_header['CROTA2']
        else:
            # Use calculate_orientation if CROTA2 is not present
            self.orientation = calculate_orientation(wcs_header)
            if self.orientation is None:
                print("Orientation: CD matrix elements not found in WCS header.")

        # Update orientation label
        if self.orientation is not None:
            self.orientation_label.setText(f"Orientation: {self.orientation:.2f}°")
        else:
            self.orientation_label.setText("Orientation: N/A")

        print("WCS applied successfully from header data.")


    def calculate_pixel_from_ra_dec(self, ra, dec):
        """Convert RA/Dec to pixel coordinates using the WCS data."""
        if not hasattr(self, 'wcs'):
            print("WCS not initialized.")
            return None, None

        # Convert RA and Dec to pixel coordinates using the WCS object
        sky_coord = SkyCoord(ra, dec, unit=(u.deg, u.deg), frame='icrs')
        x, y = self.wcs.world_to_pixel(sky_coord)
        
        return int(x), int(y)

    def login_to_astrometry(self, api_key):
        try:
            response = requests.post(
                ASTROMETRY_API_URL + "login",
                data={'request-json': json.dumps({"apikey": api_key})}
            )
            response_data = response.json()
            if response_data.get("status") == "success":
                return response_data["session"]
            else:
                raise ValueError("Login failed: " + response_data.get("error", "Unknown error"))
        except Exception as e:
            raise Exception("Login to Astrometry.net failed: " + str(e))

    def upload_image_to_astrometry(self, image_path, session_key):
        try:
            with open(image_path, 'rb') as image_file:
                files = {'file': image_file}
                data = {
                    'request-json': json.dumps({
                        "publicly_visible": "y",
                        "allow_modifications": "d",
                        "session": session_key,
                        "allow_commercial_use": "d"
                    })
                }
                response = requests.post(ASTROMETRY_API_URL + "upload", files=files, data=data)
                response_data = response.json()
                if response_data.get("status") == "success":
                    return response_data["subid"]
                else:
                    raise ValueError("Image upload failed: " + response_data.get("error", "Unknown error"))
        except Exception as e:
            raise Exception("Image upload to Astrometry.net failed: " + str(e))


    def poll_submission_status(self, subid):
        """Poll Astrometry.net to retrieve the job ID once the submission is processed."""
        max_retries = 90  # Adjust as necessary
        retries = 0
        while retries < max_retries:
            try:
                response = requests.get(ASTROMETRY_API_URL + f"submissions/{subid}")
                response_data = response.json()
                jobs = response_data.get("jobs", [])
                if jobs and jobs[0] is not None:
                    return jobs[0]
                else:
                    print(f"Polling attempt {retries + 1}: Job not ready yet.")
            except Exception as e:
                print(f"Error while polling submission status: {e}")
            
            retries += 1
            time.sleep(10)  # Wait 10 seconds between retries
        
        return None

    def poll_calibration_data(self, job_id):
        """Poll Astrometry.net to retrieve the calibration data once it's available."""
        max_retries = 90  # Retry for up to 15 minutes (90 * 10 seconds)
        retries = 0
        while retries < max_retries:
            try:
                response = requests.get(ASTROMETRY_API_URL + f"jobs/{job_id}/calibration/")
                response_data = response.json()
                if response_data and 'ra' in response_data and 'dec' in response_data:
                    print("Calibration data retrieved:", response_data)
                    return response_data  # Calibration data is complete
                else:
                    print(f"Calibration data not available yet (Attempt {retries + 1})")
            except Exception as e:
                print(f"Error retrieving calibration data: {e}")

            retries += 1
            time.sleep(10)  # Wait 10 seconds between retries

        return None


    #If originally a fits file update the header
    def update_fits_with_wcs(self, filepath, calibration_data):
        if not filepath.lower().endswith(('.fits', '.fit')):
            print("File is not a FITS file. Skipping WCS header update.")
            return

        print("Updating image with calibration data:", calibration_data)
        with fits.open(filepath, mode='update') as hdul:
            header = hdul[0].header
            header['CTYPE1'] = 'RA---TAN'
            header['CTYPE2'] = 'DEC--TAN'
            header['CRVAL1'] = calibration_data['ra']
            header['CRVAL2'] = calibration_data['dec']
            header['CRPIX1'] = hdul[0].data.shape[1] / 2
            header['CRPIX2'] = hdul[0].data.shape[0] / 2
            scale = calibration_data['pixscale'] / 3600
            orientation = np.radians(calibration_data['orientation'])
            header['CD1_1'] = -scale * np.cos(orientation)
            header['CD1_2'] = scale * np.sin(orientation)
            header['CD2_1'] = -scale * np.sin(orientation)
            header['CD2_2'] = -scale * np.cos(orientation)
            header['RADECSYS'] = 'ICRS'

    def on_mini_preview_press(self, event):
        # Set dragging flag and scroll the main preview to the position in the mini preview.
        self.dragging = True
        self.scroll_main_preview_to_mini_position(event)

    def on_mini_preview_drag(self, event):
        # Scroll to the new position while dragging in the mini preview.
        if self.dragging:
            self.scroll_main_preview_to_mini_position(event)

    def on_mini_preview_release(self, event):
        # Stop dragging
        self.dragging = False

    def scroll_main_preview_to_mini_position(self, event):
        """Scrolls the main preview to the corresponding position based on the mini preview click."""
        if self.main_image:
            # Get the click position in the mini preview
            click_x = event.pos().x()
            click_y = event.pos().y()
            
            # Calculate scale factors based on the difference in dimensions between main image and mini preview
            scale_factor_x = self.main_scene.sceneRect().width() / self.mini_preview.width()
            scale_factor_y = self.main_scene.sceneRect().height() / self.mini_preview.height()
            
            # Scale the click position to the main preview coordinates
            scaled_x = click_x * scale_factor_x
            scaled_y = click_y * scale_factor_y
            
            # Center the main preview on the calculated position
            self.main_preview.centerOn(scaled_x, scaled_y)
            
            # Update the green box after scrolling
            self.main_preview.update_mini_preview()

    def update_green_box(self):
        if self.main_image:
            factor_x = self.mini_preview.width() / self.main_image.width()
            factor_y = self.mini_preview.height() / self.main_image.height()
            
            # Get the current view rectangle in the main preview (in scene coordinates)
            view_rect = self.main_preview.mapToScene(self.main_preview.viewport().rect()).boundingRect()
            
            # Calculate the green box rectangle, shifted upward by half its height to center it
            green_box_rect = QRectF(
                view_rect.x() * factor_x,
                view_rect.y() * factor_y,
                view_rect.width() * factor_x,
                view_rect.height() * factor_y
            )
            
            # Scale the main image for the mini preview and draw the green box on it
            pixmap = self.main_image.scaled(self.mini_preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            painter = QPainter(pixmap)
            pen = QPen(QColor(0, 255, 0), 2)
            painter.setPen(pen)
            painter.drawRect(green_box_rect)
            painter.end()
            self.mini_preview.setPixmap(pixmap)

    @staticmethod
    def calculate_angular_distance(ra1, dec1, ra2, dec2):
        # Convert degrees to radians
        ra1, dec1, ra2, dec2 = map(math.radians, [ra1, dec1, ra2, dec2])

        # Haversine formula for angular distance
        delta_ra = ra2 - ra1
        delta_dec = dec2 - dec1
        a = (math.sin(delta_dec / 2) ** 2 +
            math.cos(dec1) * math.cos(dec2) * math.sin(delta_ra / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        angular_distance = math.degrees(c)
        return angular_distance
    
    @staticmethod
    def format_distance_as_dms(angle):
        degrees = int(angle)
        minutes = int((angle - degrees) * 60)
        seconds = (angle - degrees - minutes / 60) * 3600
        return f"{degrees}° {minutes}' {seconds:.2f}\""


    def wheel_zoom(self, event):
        if event.angleDelta().y() > 0:
            self.zoom_in()
        else:
            self.zoom_out()

    def zoom_in(self):
        self.zoom_level *= 1.2
        self.main_preview.setTransform(QTransform().scale(self.zoom_level, self.zoom_level))
        self.update_green_box()
        
    def zoom_out(self):
        self.zoom_level /= 1.2
        self.main_preview.setTransform(QTransform().scale(self.zoom_level, self.zoom_level))
        self.update_green_box()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_green_box()


    def update_circle_data(self):
        """Updates the status based on the circle's center and radius."""
        
        if self.circle_center and self.circle_radius > 0:
            if self.pixscale is None:
                print("Warning: Pixscale is None. Cannot calculate radius in arcminutes.")
                self.status_label.setText("No pixscale available for radius calculation.")
                return

            # Convert circle center to RA/Dec and radius to arcminutes
            ra, dec = self.calculate_ra_dec_from_pixel(self.circle_center.x(), self.circle_center.y())
            radius_arcmin = self.circle_radius * self.pixscale / 60.0  # Convert to arcminutes
            
            self.status_label.setText(
                f"Circle set at center RA={ra:.6f}, Dec={dec:.6f}, radius={radius_arcmin:.2f} arcmin"
            )
        else:
            self.status_label.setText("No search area defined.")



    def get_defined_radius(self):
        """Calculate radius in degrees for the defined region (circle radius)."""
        if self.circle_radius <= 0:
            return 0
        return float((self.circle_radius * self.pixscale) / 3600.0)


    def query_simbad(self, radius_deg, max_results=None):
        """Query Simbad based on the defined search circle using a single ADQL query, with filtering by selected types."""
            # If max_results is not provided, use the value from settings
        max_results = max_results if max_results is not None else self.max_results
        # Check if the circle center and radius are defined
        if not self.circle_center or self.circle_radius <= 0:
            QMessageBox.warning(self, "No Search Area", "Please define a search circle by Shift-clicking and dragging.")
            return

        # Calculate RA, Dec, and radius in degrees from pixel coordinates
        ra_center, dec_center = self.calculate_ra_dec_from_pixel(self.circle_center.x(), self.circle_center.y())
        if ra_center is None or dec_center is None:
            QMessageBox.warning(self, "Invalid Coordinates", "Could not determine the RA/Dec of the circle center.")
            return

        # Convert radius from arcseconds to degrees
        radius_deg = radius_deg

        # Get selected types from the tree widget
        selected_types = self.get_selected_object_types()
        if not selected_types:
            QMessageBox.warning(self, "No Object Types Selected", "Please select at least one object type.")
            return

        # Build ADQL query
        query = f"""
            SELECT TOP {max_results} ra, dec, main_id, rvz_redshift, otype, galdim_majaxis
            FROM basic
            WHERE CONTAINS(POINT('ICRS', basic.ra, basic.dec), CIRCLE('ICRS', {ra_center}, {dec_center}, {radius_deg})) = 1
        """

        try:
            # Execute the query using Simbad's TAP service
            result = Simbad.query_tap(query)

            # Clear previous results in the tree
            self.results_tree.clear()
            query_results = []

            if result is None or len(result) == 0:
                QMessageBox.information(self, "No Results", "No objects found in the specified area.")
                return

            # Process and display results, filtering by selected types
            for row in result:
                short_type = row["otype"]
                if short_type not in selected_types:
                    continue  # Skip items not in selected types

                # Retrieve other data fields
                ra = row["ra"]
                dec = row["dec"]
                main_id = row["main_id"]
                redshift = row["rvz_redshift"] if row["rvz_redshift"] is not None else "--"
                diameter = row.get("galdim_majaxis", "N/A")
                comoving_distance = calculate_comoving_distance(float(redshift)) if redshift != "--" else "N/A"

                # Map short type to long type
                long_type = otype_long_name_lookup.get(short_type, short_type)

                # Add to TreeWidget
                item = QTreeWidgetItem([
                    f"{ra:.6f}", f"{dec:.6f}", main_id, str(diameter), short_type, long_type, str(redshift), str(comoving_distance)
                ])
                self.results_tree.addTopLevelItem(item)

                # Append full details as a dictionary to query_results
                query_results.append({
                    'ra': ra,
                    'dec': dec,
                    'name': main_id,
                    'diameter': diameter,
                    'short_type': short_type,
                    'long_type': long_type,
                    'redshift': redshift,
                    'comoving_distance': comoving_distance,
                    'source' : "Simbad"
                })

            # Set query results in the CustomGraphicsView for display
            self.main_preview.set_query_results(query_results)
            self.query_results = query_results  # Keep a reference to results in MainWindow

        except Exception as e:
            QMessageBox.critical(self, "Query Failed", f"Failed to query Simbad: {str(e)}")

    def perform_deep_vizier_search(self):
        """Perform a Vizier catalog search and parse results based on catalog-specific fields, with duplicate handling."""
        if not self.circle_center or self.circle_radius <= 0:
            QMessageBox.warning(self, "No Search Area", "Please define a search circle by Shift-clicking and dragging.")
            return

        # Convert the center coordinates to RA/Dec
        ra_center, dec_center = self.calculate_ra_dec_from_pixel(self.circle_center.x(), self.circle_center.y())
        if ra_center is None or dec_center is None:
            QMessageBox.warning(self, "Invalid Coordinates", "Could not determine the RA/Dec of the circle center.")
            return

        # Convert radius from arcseconds to arcminutes
        radius_arcmin = float((self.circle_radius * self.pixscale) / 60.0)

        # List of Vizier catalogs
        catalog_ids = ["II/246", "I/350/gaiaedr3", "V/147/sdss12", "I/322A", "V/154"]

        coord = SkyCoord(ra_center, dec_center, unit="deg")
        all_results = []  # Collect all results for display in the main preview
        unique_entries = {}  # Dictionary to track unique entries by (RA, Dec) tuple

        try:
            for catalog_id in catalog_ids:
                # Query each catalog
                result = Vizier.query_region(coord, radius=radius_arcmin * u.arcmin, catalog=catalog_id)
                if result:
                    catalog_results = result[0]
                    for row in catalog_results:
                        # Map data to the columns in your tree view structure

                        # RA and Dec
                        ra = str(row.get("RAJ2000", row.get("RA_ICRS", "")))
                        dec = str(row.get("DEJ2000", row.get("DE_ICRS", "")))
                        if not ra or not dec:
                            
                            continue  # Skip this entry if RA or Dec is empty

                        # Create a unique key based on RA and Dec to track duplicates
                        unique_key = (ra, dec)

                        # Name (different columns based on catalog)
                        name = str(
                            row.get("_2MASS", "")
                            or row.get("Source", "")
                            or row.get("SDSS12", "")
                            or row.get("UCAC4", "")
                            or row.get("SDSS16", "")
                        )

                        # Diameter - store catalog ID as the diameter field to help with tracking
                        diameter = catalog_id

                        # Type (e.g., otype)
                        type_short = str(row.get("otype", "N/A"))

                        # Long Type (e.g., SpType)
                        long_type = str(row.get("SpType", "N/A"))

                        # Redshift or Parallax (zph for redshift or Plx for parallax)
                        redshift = row.get("zph", row.get("Plx", ""))
                        if redshift:
                            if "Plx" in row.colnames:
                                redshift = f"{redshift} (Parallax in mas)"
                                # Calculate the distance in light-years from parallax
                                try:
                                    parallax_value = float(row["Plx"])
                                    comoving_distance = f"{1000 / parallax_value * 3.2615637769:.2f} Ly"
                                except (ValueError, ZeroDivisionError):
                                    comoving_distance = "N/A"  # Handle invalid parallax values
                            else:
                                redshift = str(redshift)
                                # Calculate comoving distance for redshift if it's from zph
                                if "zph" in row.colnames and isinstance(row["zph"], (float, int)):
                                    comoving_distance = str(calculate_comoving_distance(float(row["zph"])))
                        else:
                            redshift = "N/A"
                            comoving_distance = "N/A"

                        # Handle duplicates: prioritize V/147/sdss12 over V/154 and only add unique entries
                        if unique_key not in unique_entries:
                            unique_entries[unique_key] = {
                                'ra': ra,
                                'dec': dec,
                                'name': name,
                                'diameter': diameter,
                                'short_type': type_short,
                                'long_type': long_type,
                                'redshift': redshift,
                                'comoving_distance': comoving_distance,
                                'source' : "Vizier"
                            }
                        else:
                            # Check if we should replace the existing entry
                            existing_entry = unique_entries[unique_key]
                            if (existing_entry['diameter'] == "V/154" and diameter == "V/147/sdss12"):
                                unique_entries[unique_key] = {
                                    'ra': ra,
                                    'dec': dec,
                                    'name': name,
                                    'diameter': diameter,
                                    'short_type': type_short,
                                    'long_type': long_type,
                                    'redshift': redshift,
                                    'comoving_distance': comoving_distance,
                                    'source' : "Vizier"
                                }

            # Convert unique entries to the main preview display
            for entry in unique_entries.values():
                item = QTreeWidgetItem([
                    entry['ra'], entry['dec'], entry['name'], entry['diameter'], entry['short_type'], entry['long_type'],
                    entry['redshift'], entry['comoving_distance']
                ])
                self.results_tree.addTopLevelItem(item)
                all_results.append(entry)

            # Update the main preview with the query results
            self.main_preview.set_query_results(all_results)
            self.query_results = all_results  # Keep a reference to results in MainWindow
            
        except Exception as e:
            QMessageBox.critical(self, "Vizier Search Failed", f"Failed to query Vizier: {str(e)}")

    def perform_mast_search(self):
        """Perform a MAST cone search in the user-defined region using astroquery."""
        if not self.circle_center or self.circle_radius <= 0:
            QMessageBox.warning(self, "No Search Area", "Please define a search circle by Shift-clicking and dragging.")
            return

        # Calculate RA and Dec for the center point
        ra_center, dec_center = self.calculate_ra_dec_from_pixel(self.circle_center.x(), self.circle_center.y())
        if ra_center is None or dec_center is None:
            QMessageBox.warning(self, "Invalid Coordinates", "Could not determine the RA/Dec of the circle center.")
            return

        # Convert radius from arcseconds to degrees (MAST uses degrees)
        search_radius_deg = float((self.circle_radius * self.pixscale) / 3600.0)  # Convert to degrees
        ra_center = float(ra_center)  # Ensure it's a regular float
        dec_center = float(dec_center)  # Ensure it's a regular float

        try:
            # Perform the MAST cone search using Mast.mast_query for the 'Mast.Caom.Cone' service
            observations = Mast.mast_query(
                'Mast.Caom.Cone',
                ra=ra_center,
                dec=dec_center,
                radius=search_radius_deg
            )

            # Limit the results to the first 100 rows
            limited_observations = observations[:100]

            if len(observations) == 0:
                QMessageBox.information(self, "No Results", "No objects found in the specified area on MAST.")
                return

            # Clear previous results
            self.results_tree.clear()
            query_results = []

            # Process each observation in the results
            for obj in limited_observations:

                def safe_get(value):
                    return "N/A" if np.ma.is_masked(value) else str(value)


                ra = safe_get(obj.get("s_ra", "N/A"))
                dec = safe_get(obj.get("s_dec", "N/A"))
                target_name = safe_get(obj.get("target_name", "N/A"))
                instrument = safe_get(obj.get("instrument_name", "N/A"))
                jpeg_url = safe_get(obj.get("dataURL", "N/A"))  # Adjust URL field as needed

                # Add to TreeWidget
                item = QTreeWidgetItem([
                    ra,
                    dec,
                    target_name,
                    instrument,
                    "N/A",  # Placeholder for observation date if needed
                    "N/A",  # Other placeholder
                    jpeg_url,  # URL in place of long type
                    "MAST"  # Source
                ])
                self.results_tree.addTopLevelItem(item)

                # Append full details as a dictionary to query_results
                query_results.append({
                    'ra': ra,
                    'dec': dec,
                    'name': target_name,
                    'diameter': instrument,
                    'short_type': "N/A",
                    'long_type': jpeg_url,
                    'redshift': "N/A",
                    'comoving_distance': "N/A",
                    'source': "Mast"
                })

            # Set query results in the CustomGraphicsView for display
            self.main_preview.set_query_results(query_results)
            self.query_results = query_results  # Keep a reference to results in MainWindow

        except Exception as e:
            QMessageBox.critical(self, "MAST Query Failed", f"Failed to query MAST: {str(e)}")

    def toggle_show_names(self, state):
        """Toggle showing/hiding names on the main image."""
        self.show_names = state == Qt.Checked
        self.main_preview.draw_query_results()  # Redraw with or without names

    def clear_results(self):
        """Clear the search results and remove markers from the main image."""
        self.results_tree.clear()
        self.main_preview.clear_query_results()
        self.status_label.setText("Results cleared.")

    def open_settings_dialog(self):
        """Open settings dialog to adjust max results and provide option to force a blind solve."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Settings")
        
        layout = QFormLayout(dialog)
        
        # Max Results setting
        max_results_spinbox = QSpinBox()
        max_results_spinbox.setRange(1, 25000)
        max_results_spinbox.setValue(self.max_results)
        layout.addRow("Max Results:", max_results_spinbox)
        
        # Force Blind Solve button
        force_blind_solve_button = QPushButton("Force Blind Solve")
        force_blind_solve_button.clicked.connect(lambda: self.force_blind_solve(dialog))
        layout.addWidget(force_blind_solve_button)
        
        # OK and Cancel buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(lambda: self.update_max_results(max_results_spinbox.value(), dialog))
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        
        dialog.setLayout(layout)
        dialog.exec_()

    def force_blind_solve(self, dialog):
        """Force a blind solve on the currently loaded image."""
        dialog.accept()  # Close the settings dialog
        self.prompt_blind_solve()  # Call the blind solve function

    def apply_dark_theme(self):
        """Apply the dark theme stylesheet."""
        self.setStyleSheet("""
            QWidget {
                background-color: #2E2E2E;
                color: #FFFFFF;
            }
            QPushButton {
                background-color: #3C3F41;
                color: #FFFFFF;
                border: 1px solid #5A5A5A;
                padding: 5px;
            }
            QPushButton:hover {
                background-color: #4C4F51;
            }
            QPushButton:pressed {
                background-color: #5C5F61;
            }
            QHeaderView::section {
                background-color: #3C3F41;
                color: #FFFFFF;
                padding: 4px;
                border: 1px solid #5A5A5A;
            }
            QTreeWidget::item {
                background-color: #2E2E2E;
                color: #FFFFFF;
            }
            QTreeWidget::item:selected {  /* Style for selected items */
                background-color: #505050;  /* Dark gray background for selected item */
                color: #FFFFFF;  /* White text color for selected item */
            }
            QTreeWidget::item:selected:active {  /* Style for selected items in active state */
                background-color: #707070;  /* Slightly lighter gray for active selected item */
                color: #FFFFFF;
            }
        """)
        self.theme_toggle_button.setText("Switch to Light Theme")
        self.title_label.setStyleSheet("font-size: 20px; color: lightgrey;")  # Set title color for dark theme

    def apply_light_theme(self):
        """Clear the stylesheet to revert to the default light theme."""
        self.setStyleSheet("")  # Clear the stylesheet
        self.theme_toggle_button.setText("Switch to Dark Theme")
        self.title_label.setStyleSheet("font-size: 20px; color: black;")  # Set title color for light theme

    def toggle_theme(self):
        """Toggle between dark and light themes."""
        if self.is_dark_mode:
            self.apply_light_theme()
        else:
            self.apply_dark_theme()
        self.is_dark_mode = not self.is_dark_mode

def extract_wcs_data(file_path):
    try:
        # Open the FITS file with minimal validation to ignore potential errors in non-essential parts
        with fits.open(file_path, ignore_missing_simple=True, ignore_missing_end=True) as hdul:
            header = hdul[0].header

            # Extract essential WCS parameters
            wcs_params = {}
            keys_to_extract = [
                'WCSAXES', 'CTYPE1', 'CTYPE2', 'EQUINOX', 'LONPOLE', 'LATPOLE',
                'CRVAL1', 'CRVAL2', 'CRPIX1', 'CRPIX2', 'CUNIT1', 'CUNIT2',
                'CD1_1', 'CD1_2', 'CD2_1', 'CD2_2', 'A_ORDER', 'A_0_0', 'A_0_1', 
                'A_0_2', 'A_1_0', 'A_1_1', 'A_2_0', 'B_ORDER', 'B_0_0', 'B_0_1', 
                'B_0_2', 'B_1_0', 'B_1_1', 'B_2_0', 'AP_ORDER', 'AP_0_0', 'AP_0_1', 
                'AP_0_2', 'AP_1_0', 'AP_1_1', 'AP_2_0', 'BP_ORDER', 'BP_0_0', 
                'BP_0_1', 'BP_0_2', 'BP_1_0', 'BP_1_1', 'BP_2_0'
            ]
            for key in keys_to_extract:
                if key in header:
                    wcs_params[key] = header[key]

            # Manually create a minimal header with WCS information
            wcs_header = fits.Header()
            for key, value in wcs_params.items():
                wcs_header[key] = value

            # Initialize WCS with this custom header
            wcs = WCS(wcs_header)
            print("WCS successfully initialized with minimal header.")
            return wcs

    except Exception as e:
        print(f"Error processing WCS file: {e}")
        return None

# Function to calculate comoving radial distance (in Gly)
def calculate_comoving_distance(z):
    z = abs(z)
    # Initialize variables
    WR = 4.165E-5 / ((H0 / 100) ** 2)  # Omega radiation
    WK = 1 - WM - WV - WR  # Omega curvature
    az = 1.0 / (1 + z)
    n = 1000  # number of points in integration

    # Comoving radial distance
    DCMR = 0.0
    for i in range(n):
        a = az + (1 - az) * (i + 0.5) / n
        adot = sqrt(WK + (WM / a) + (WR / (a ** 2)) + (WV * a ** 2))
        DCMR += 1 / (a * adot)
    
    DCMR = (1 - az) * DCMR / n
    DCMR_Gly = (c / H0) * DCMR * Mpc_to_Gly

    return round(DCMR_Gly, 3)  # Round to three decimal places for display

def calculate_orientation(header):
    """Calculate the orientation angle from the CD matrix if available."""
    # Extract CD matrix elements
    cd1_1 = header.get('CD1_1')
    cd1_2 = header.get('CD1_2')
    cd2_1 = header.get('CD2_1')
    cd2_2 = header.get('CD2_2')

    if cd1_1 is not None and cd1_2 is not None and cd2_1 is not None and cd2_2 is not None:
        # Calculate the orientation angle in degrees and adjust by adding 180 degrees
        orientation = (np.degrees(np.arctan2(cd1_2, cd1_1)) + 180) % 360
        return orientation
    else:
        print("CD matrix elements not found in the header.")
        return None



if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = AstroEditingSuite()
    window.show()
    sys.exit(app.exec_())
