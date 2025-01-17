# Standard library imports
import os
import sys
import time
import json
import csv
import math
from datetime import datetime
from decimal import getcontext
from urllib.parse import quote
import webbrowser
import warnings
import shutil
import subprocess
from xisf import XISF
import requests
import csv
import lz4.block
import zstandard
import base64
import ast
import platform
import glob
import time
from datetime import datetime
import pywt
from io import BytesIO



# Third-party library imports
import requests
import numpy as np
import pandas as pd
import cv2
from PIL import Image, ImageDraw, ImageFont


# Astropy and Astroquery imports
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, get_body, get_sun
import astropy.units as u
from astropy.wcs import WCS
from astroquery.simbad import Simbad
from astroquery.mast import Mast
from astroquery.vizier import Vizier
import tifffile as tiff
import pytz
from astropy.utils.data import conf
from scipy.interpolate import interp1d
from scipy.interpolate import PchipInterpolator

import rawpy

# PyQt5 imports
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QFileDialog, QGraphicsView, QGraphicsScene, QMessageBox, QInputDialog, QTreeWidget, 
    QTreeWidgetItem, QCheckBox, QDialog, QFormLayout, QSpinBox, QDialogButtonBox, QGridLayout,
    QGraphicsEllipseItem, QGraphicsLineItem, QGraphicsRectItem, QGraphicsPathItem, 
    QColorDialog, QFontDialog, QStyle, QSlider, QTabWidget, QScrollArea, QSizePolicy, QSpacerItem, QAbstractItemView,
    QGraphicsTextItem, QComboBox, QLineEdit, QRadioButton, QButtonGroup, QHeaderView, QStackedWidget, QSplitter, QMenu, QAction, QMenuBar, QTextEdit, QProgressBar, QGraphicsItem, QToolButton, QStatusBar
)
from PyQt5.QtGui import (
    QPixmap, QImage, QPainter, QPen, QColor, QTransform, QIcon, QPainterPath, QFont, QMovie, QCursor, QBrush
)
from PyQt5.QtCore import Qt, QRectF, QLineF, QPointF, QThread, pyqtSignal, QCoreApplication, QPoint, QTimer, QRect, QFileSystemWatcher, QEvent, pyqtSlot, QProcess, QSize, QObject

# Math functions
from math import sqrt


if hasattr(sys, '_MEIPASS'):
    # PyInstaller path
    icon_path = os.path.join(sys._MEIPASS, 'astrosuite.png')
else:
    # Development path
    icon_path = 'astrosuite.png'


class AstroEditingSuite(QMainWindow):
    def __init__(self):
        super().__init__()
        self.current_theme = "dark"  # Default theme
        self.image_manager = ImageManager(max_slots=5)  # Initialize ImageManager
        self.image_manager.image_changed.connect(self.update_file_name)
        self.initUI()

    def initUI(self):
        self.setWindowIcon(QIcon(icon_path))
        layout = QVBoxLayout()

        # Enable drag and drop
        self.setAcceptDrops(True)

        # Create a menu bar
        menubar = self.menuBar()  # Use the menu bar directly from QMainWindow

        # File Menu
        file_menu = menubar.addMenu("File")
        
        # Add actions for File menu
        open_action = QAction("Open Image", self)
        save_action = QAction("Save As", self)
        undo_action = QAction("Undo", self)
        redo_action = QAction("Redo", self)
        exit_action = QAction("Exit", self)
        
        open_action.triggered.connect(self.open_image)
        save_action.triggered.connect(self.save_image)
        undo_action.triggered.connect(self.undo_image)
        redo_action.triggered.connect(self.redo_image)
        exit_action.triggered.connect(self.close)  # Close the application

        # Add actions to the file menu
        file_menu.addAction(open_action)
        file_menu.addAction(save_action)
        file_menu.addAction(undo_action)
        file_menu.addAction(redo_action)
        file_menu.addAction(exit_action)

        # Themes Menu
        theme_menu = menubar.addMenu("Themes")
        light_theme_action = QAction("Light Theme", self)
        dark_theme_action = QAction("Dark Theme", self)

        light_theme_action.triggered.connect(lambda: self.apply_theme("light"))
        dark_theme_action.triggered.connect(lambda: self.apply_theme("dark"))

        theme_menu.addAction(light_theme_action)
        theme_menu.addAction(dark_theme_action)

        # Create the status bar
        self.statusBar = QStatusBar(self)
        self.setStatusBar(self.statusBar)

        # Create the file label to display the current file name in the status bar
        self.file_name_label = QLabel("No file selected")
        self.statusBar.addWidget(self.file_name_label, 1)  # Add label to status bar

        # Create the tab widget
        self.tabs = QTabWidget()
        # Add individual tabs for each tool
        self.tabs.addTab(XISFViewer(image_manager=self.image_manager), "XISF Liberator")
        self.tabs.addTab(BlinkTab(image_manager=self.image_manager), "Blink Comparator")
        self.tabs.addTab(CosmicClarityTab(image_manager=self.image_manager), "Cosmic Clarity Sharpen/Denoise")
        self.tabs.addTab(CosmicClaritySatelliteTab(), "Cosmic Clarity Satellite")
        self.tabs.addTab(StatisticalStretchTab(image_manager=self.image_manager), "Statistical Stretch")
        self.tabs.addTab(FullCurvesTab(image_manager=self.image_manager), "Curves Utility")
        self.tabs.addTab(PerfectPalettePickerTab(image_manager=self.image_manager), "Perfect Palette Picker")
        self.tabs.addTab(NBtoRGBstarsTab(image_manager=self.image_manager), "NB to RGB Stars")
        self.tabs.addTab(StarStretchTab(image_manager=self.image_manager), "Star Stretch")
        self.tabs.addTab(FrequencySeperationTab(image_manager=self.image_manager), "Frequency Separation")
        self.tabs.addTab(HaloBGonTab(image_manager=self.image_manager), "Halo-B-Gon")
        self.tabs.addTab(ContinuumSubtractTab(image_manager=self.image_manager), "Continuum Subtraction")
        self.tabs.addTab(MainWindow(), "What's In My Image")
        self.tabs.addTab(WhatsInMySky(), "What's In My Sky")

        # Set the layout for the main window
        central_widget = QWidget(self)  # Create a central widget
        layout = QVBoxLayout(central_widget)
        layout.addWidget(self.tabs)  # Add tabs to the central widget

        # Set the central widget of the main window
        self.setCentralWidget(central_widget)

        # Set the layout for the main window

        self.setWindowTitle('Seti Astro\'s Suite V2.4')

        # Populate the Quick Navigation menu with each tab name
        quicknav_menu = menubar.addMenu("Quick Navigation")
        for i in range(self.tabs.count()):
            tab_title = self.tabs.tabText(i)
            action = QAction(tab_title, self)
            action.triggered.connect(lambda checked, index=i: self.tabs.setCurrentIndex(index))
            quicknav_menu.addAction(action)

        # Apply the default theme
        self.apply_theme(self.current_theme)

    def dragEnterEvent(self, event):
        """Handle the drag enter event."""
        # Check if the dragged content is a file
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        """Handle the drop event."""
        # Get the file path from the dropped file
        file_path = event.mimeData().urls()[0].toLocalFile()
        
        # Check if the file is an image (you can customize this check as needed)
        if file_path.lower().endswith(('.png', '.tif', '.tiff', '.fits', '.xisf', '.fit', '.jpg', '.jpeg', '.cr2', '.nef', '.arw', '.dng', '.orf', '.rw2', '.pef')):
            try:
                # Load the image into ImageManager
                image, header, bit_depth, is_mono = load_image(file_path)
                metadata = {
                    'file_path': file_path,
                    'original_header': header,
                    'bit_depth': bit_depth,
                    'is_mono': is_mono
                }
                self.image_manager.add_image(self.image_manager.current_slot, image, metadata)  # Make sure to specify the slot here
                print(f"Image {file_path} loaded successfully via drag and drop.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load image: {e}")
        else:
            QMessageBox.warning(self, "Invalid File", "Only image files are supported.")

    def update_file_name(self, slot, image, metadata):
        """Update the file name in the status bar."""
        file_path = metadata.get('file_path', None)
        if file_path:
            self.file_name_label.setText(os.path.basename(file_path))  # Update the label with file name
        else:
            self.file_name_label.setText("No file selected")

    def apply_theme(self, theme):
        """Apply the selected theme to the application."""
        if theme == "light":
            self.current_theme = "light"
            light_stylesheet = """
            QWidget {
                background-color: #f0f0f0;
                color: #000000;
                font-family: Arial, sans-serif;
            }
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
                background-color: #ffffff;
                border: 1px solid #cccccc;
                color: #000000;
                padding: 2px;
            }
            QPushButton {
                background-color: #e0e0e0;
                border: 1px solid #cccccc;
                color: #000000;
                padding: 5px;
            }
            QPushButton:hover {
                background-color: #d0d0d0;
            }
            QScrollBar:vertical, QScrollBar:horizontal {
                background: #ffffff;
            }
            QTreeWidget {
                background-color: #ffffff;
                border: 1px solid #cccccc;
                color: #000000;
            }
            QHeaderView::section {
                background-color: #f0f0f0;
                color: #000000;
                padding: 5px;
            }
            QTabWidget::pane { 
                border: 1px solid #cccccc; 
                background-color: #f0f0f0;
            }
            QTabBar::tab {
                background: #e0e0e0;
                color: #000000;
                padding: 5px;
                border: 1px solid #cccccc;
                border-bottom: none;  /* Avoid double border at bottom */
            }
            QTabBar::tab:selected {
                background: #d0d0d0;  /* Highlight for the active tab */
                border-color: #000000;
            }
            QTabBar::tab:hover {
                background: #c0c0c0;
            }
            QTabBar::tab:!selected {
                margin-top: 2px;  /* Push unselected tabs down for better clarity */
            }
            QMenu {
                background-color: #f0f0f0;
                color: #000000;
            }
            QMenu::item:selected {
                background-color: #d0d0d0; 
                color: #000000;
            }            
            """
            self.setStyleSheet(light_stylesheet)

        elif theme == "dark":
            self.current_theme = "dark"
            dark_stylesheet = """
            QWidget {
                background-color: #2b2b2b;
                color: #dcdcdc;
                font-family: Arial, sans-serif;
            }
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
                background-color: #3c3f41;
                border: 1px solid #5c5c5c;
                color: #ffffff;
                padding: 2px;
            }
            QPushButton {
                background-color: #3c3f41;
                border: 1px solid #5c5c5c;
                color: #ffffff;
                padding: 5px;
            }
            QPushButton:hover {
                background-color: #4a4a4a;
            }
            QScrollBar:vertical, QScrollBar:horizontal {
                background: #3c3f41;
            }
            QTreeWidget {
                background-color: #3c3f41;
                border: 1px solid #5c5c5c;
                color: #ffffff;
            }
            QHeaderView::section {
                background-color: #3c3f41;
                color: #dcdcdc;
                padding: 5px;
            }
            QTabWidget::pane { 
                border: 1px solid #5c5c5c; 
                background-color: #2b2b2b;
            }
            QTabBar::tab {
                background: #3c3f41;
                color: #dcdcdc;
                padding: 5px;
                border: 1px solid #5c5c5c;
                border-bottom: none;  /* Avoid double border at bottom */
            }
            QTabBar::tab:selected {
                background: #4a4a4a;  /* Highlight for the active tab */
                border-color: #dcdcdc;
            }
            QTabBar::tab:hover {
                background: #505050;
            }
            QTabBar::tab:!selected {
                margin-top: 2px;  /* Push unselected tabs down for better clarity */
            }
            QMenu {
                background-color: #2b2b2b;
                color: #dcdcdc;
            }
            QMenu::item:selected {
                background-color: #3a75c4;  /* Blue background for selected items */
                color: #ffffff;  /* White text color */
            }       
            """          
            self.setStyleSheet(dark_stylesheet)

    def open_image(self):
        """Open an image and load it into the ImageManager."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", 
                                            "Images (*.png *.tif *.tiff *.fits *.xisf *.cr2 *.nef *.arw *.dng *.orf *.rw2 *.pef);;All Files (*)")

        if file_path:
            try:
                # Load the image into ImageManager
                image, header, bit_depth, is_mono = load_image(file_path)
                metadata = {
                    'file_path': file_path,
                    'original_header': header,
                    'bit_depth': bit_depth,
                    'is_mono': is_mono
                }
                self.image_manager.add_image(self.image_manager.current_slot, image, metadata)  # Make sure to specify the slot here
                print(f"Image {file_path} loaded successfully.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load image: {e}")


    def save_image(self):
        """Save the current image to a selected path."""
        if self.image_manager.image is not None:
            save_file, _ = QFileDialog.getSaveFileName(self, "Save As", "", "Images (*.png *.tif *.tiff *.fits *.fit);;All Files (*)")
            
            if save_file:
                # Prompt the user for bit depth
                bit_depth, ok = QInputDialog.getItem(
                    self,
                    "Select Bit Depth",
                    "Choose bit depth for saving:",
                    ["16-bit", "32-bit floating point"],
                    0,
                    False
                )
                if ok:
                    # Determine the user-selected format from the filename
                    _, ext = os.path.splitext(save_file)
                    selected_format = ext.lower().strip('.')

                    # Validate the selected format
                    valid_formats = ['png', 'tif', 'tiff', 'fits', 'fit']
                    if selected_format not in valid_formats:
                        QMessageBox.critical(
                            self,
                            "Error",
                            f"Unsupported file format: {selected_format}. Supported formats are: {', '.join(valid_formats)}"
                        )
                        return

                    try:
                        # Retrieve the image and metadata
                        image_data = self.image_manager.image
                        metadata = self.image_manager._metadata[self.image_manager.current_slot]
                        original_header = metadata.get('original_header', None)
                        is_mono = metadata.get('is_mono', False)

                        # Create a minimal header if the original header is missing
                        if original_header is None and selected_format in ['fits', 'fit']:
                            print("Creating a minimal FITS header for the data...")
                            original_header = self.create_minimal_fits_header(image_data, is_mono)

                        # Pass the image to the global save_image function
                        save_image(
                            img_array=image_data,
                            filename=save_file,
                            original_format=selected_format,
                            bit_depth=bit_depth,
                            original_header=original_header,
                            is_mono=is_mono
                        )
                        print(f"Image successfully saved to {save_file}.")
                        self.statusBar.showMessage(f"Image saved to: {save_file}", 5000)
                    except Exception as e:
                        QMessageBox.critical(self, "Error", f"Failed to save image: {e}")
                        print(f"Error saving image: {e}")
        else:
            QMessageBox.warning(self, "Warning", "No image loaded.")





    def create_minimal_fits_header(self, img_array, is_mono=False):
        """
        Creates a minimal FITS header when the original header is missing.
        """
        from astropy.io.fits import Header

        header = Header()
        header['SIMPLE'] = (True, 'Standard FITS file')
        header['BITPIX'] = -32  # 32-bit floating-point data
        header['NAXIS'] = 2 if is_mono else 3
        header['NAXIS1'] = img_array.shape[2] if img_array.ndim == 3 and not is_mono else img_array.shape[1]  # Image width
        header['NAXIS2'] = img_array.shape[1] if img_array.ndim == 3 and not is_mono else img_array.shape[0]  # Image height
        if not is_mono:
            header['NAXIS3'] = img_array.shape[0] if img_array.ndim == 3 else 1  # Number of color channels
        header['BZERO'] = 0.0  # No offset
        header['BSCALE'] = 1.0  # No scaling
        header.add_comment("Minimal FITS header generated by AstroEditingSuite.")

        return header
    
    def undo_image(self):
        """Undo the last action."""
        if self.image_manager.can_undo():
            self.image_manager.undo()
            print("Undo performed.")
        else:
            QMessageBox.information(self, "Undo", "No actions to undo.")

    def redo_image(self):
        """Redo the last undone action."""
        if self.image_manager.can_redo():
            self.image_manager.redo()
            print("Redo performed.")
        else:
            QMessageBox.information(self, "Redo", "No actions to redo.")            

class ImageManager(QObject):
    """
    Manages multiple image slots with associated metadata and supports undo/redo operations for each slot.
    Emits a signal whenever an image or its metadata changes.
    """
    
    # Signal emitted when an image or its metadata changes.
    # Parameters:
    # - slot (int): The slot number.
    # - image (np.ndarray): The new image data.
    # - metadata (dict): Associated metadata for the image.
    image_changed = pyqtSignal(int, np.ndarray, dict)

    def __init__(self, max_slots=5):
        """
        Initializes the ImageManager with a specified number of slots.
        
        :param max_slots: Maximum number of image slots to manage.
        """
        super().__init__()
        self.max_slots = max_slots
        self._images = {i: None for i in range(max_slots)}
        self._metadata = {i: {} for i in range(max_slots)}
        self._undo_stacks = {i: [] for i in range(max_slots)}
        self._redo_stacks = {i: [] for i in range(max_slots)}
        self.current_slot = 0  # Default to the first slot

    def set_current_slot(self, slot):
        """
        Sets the current active slot if the slot number is valid and has an image.
        
        :param slot: The slot number to activate.
        """
        if 0 <= slot < self.max_slots and self._images[slot] is not None:
            self.current_slot = slot
            self.image_changed.emit(slot, self._images[slot], self._metadata[slot])
            print(f"ImageManager: Current slot set to {slot}.")
        else:
            print(f"ImageManager: Slot {slot} is invalid or empty.")

    def add_image(self, slot, image, metadata):
        """
        Adds an image and its metadata to a specified slot.
        
        :param slot: The slot number where the image will be added.
        :param image: The image data (numpy array).
        :param metadata: A dictionary containing metadata for the image.
        """
        if 0 <= slot < self.max_slots:
            self._images[slot] = image
            self._metadata[slot] = metadata
            # Clear undo/redo stacks when a new image is added
            self._undo_stacks[slot].clear()
            self._redo_stacks[slot].clear()
            self.current_slot = slot
            self.image_changed.emit(slot, image, metadata)
            print(f"ImageManager: Image added to slot {slot} with metadata.")
        else:
            print(f"ImageManager: Slot {slot} is out of range. Max slots: {self.max_slots}")

    def set_image(self, new_image, metadata):
        """
        Sets a new image and metadata for the current slot, adding the previous state to the undo stack.
        
        :param new_image: The new image data (numpy array).
        :param metadata: A dictionary containing metadata for the new image.
        """
        slot = self.current_slot
        if self._images[slot] is not None:
            # Save current state to undo stack
            self._undo_stacks[slot].append((self._images[slot].copy(), self._metadata[slot].copy()))
            # Clear redo stack since new action invalidates the redo history
            self._redo_stacks[slot].clear()
            print(f"ImageManager: Previous image in slot {slot} pushed to undo stack.")
        else:
            print(f"ImageManager: No existing image in slot {slot} to push to undo stack.")
        self._images[slot] = new_image
        self._metadata[slot] = metadata
        self.image_changed.emit(slot, new_image, metadata)
        print(f"ImageManager: Image set for slot {slot} with new metadata.")

    @property
    def image(self):
        """
        Gets the image from the current slot.
        
        :return: The image data (numpy array) of the current slot.
        """
        return self._images[self.current_slot]

    @image.setter
    def image(self, new_image):
        """
        Sets a new image for the current slot, adding the previous state to the undo stack.
        
        :param new_image: The new image data (numpy array).
        """
        slot = self.current_slot
        if self._images[slot] is not None:
            # Save current state to undo stack
            self._undo_stacks[slot].append((self._images[slot].copy(), self._metadata[slot].copy()))
            # Clear redo stack since new action invalidates the redo history
            self._redo_stacks[slot].clear()
            print(f"ImageManager: Previous image in slot {slot} pushed to undo stack via property setter.")
        else:
            print(f"ImageManager: No existing image in slot {slot} to push to undo stack via property setter.")
        self._images[slot] = new_image
        self.image_changed.emit(slot, new_image, self._metadata[slot])
        print(f"ImageManager: Image set for slot {slot} via property setter.")

    def set_metadata(self, metadata):
        """
        Sets new metadata for the current slot, adding the previous state to the undo stack.
        
        :param metadata: A dictionary containing new metadata.
        """
        slot = self.current_slot
        if self._images[slot] is not None:
            # Save current state to undo stack
            self._undo_stacks[slot].append((self._images[slot].copy(), self._metadata[slot].copy()))
            # Clear redo stack since new action invalidates the redo history
            self._redo_stacks[slot].clear()
            print(f"ImageManager: Previous metadata in slot {slot} pushed to undo stack.")
        else:
            print(f"ImageManager: No existing image in slot {slot} to set metadata.")
        self._metadata[slot] = metadata
        self.image_changed.emit(slot, self._images[slot], metadata)
        print(f"ImageManager: Metadata set for slot {slot}.")

    def update_image(self, updated_image, metadata=None, slot=None):
        """
        Updates the image in the specified slot with an existing image, optionally updating metadata.
        Adds the previous state to the undo stack.
        
        :param updated_image: The updated image data (numpy array).
        :param metadata: (Optional) A dictionary containing metadata for the image.
        :param slot: (Optional) The slot number to update. If None, uses current_slot.
        """
        if slot is None:
            slot = self.current_slot

        if 0 <= slot < self.max_slots:
            if self._images[slot] is not None:
                # Save current state to undo stack
                self._undo_stacks[slot].append((self._images[slot].copy(), self._metadata[slot].copy()))
                # Clear redo stack since new action invalidates the redo history
                self._redo_stacks[slot].clear()
                print(f"ImageManager: Previous image and metadata in slot {slot} pushed to undo stack via update_image.")
            else:
                print(f"ImageManager: No existing image in slot {slot} to push to undo stack via update_image.")
            
            self._images[slot] = updated_image
            if metadata is not None:
                self._metadata[slot] = metadata
                print(f"ImageManager: Metadata updated for slot {slot} via update_image.")
            else:
                print(f"ImageManager: Metadata not provided; retaining existing metadata for slot {slot}.")
            
            self.image_changed.emit(slot, updated_image, self._metadata[slot])
            print(f"ImageManager: Image updated for slot {slot} via update_image.")
        else:
            print(f"ImageManager: Slot {slot} is out of range. Max slots: {self.max_slots}")

    def can_undo(self, slot=None):
        """
        Determines if there are actions available to undo for the specified slot.
        
        :param slot: (Optional) The slot number to check. If None, uses current_slot.
        :return: True if undo is possible, False otherwise.
        """
        if slot is None:
            slot = self.current_slot
        if 0 <= slot < self.max_slots:
            return len(self._undo_stacks[slot]) > 0
        else:
            print(f"ImageManager: Slot {slot} is out of range. Cannot check can_undo.")
            return False

    def can_redo(self, slot=None):
        """
        Determines if there are actions available to redo for the specified slot.
        
        :param slot: (Optional) The slot number to check. If None, uses current_slot.
        :return: True if redo is possible, False otherwise.
        """
        if slot is None:
            slot = self.current_slot
        if 0 <= slot < self.max_slots:
            return len(self._redo_stacks[slot]) > 0
        else:
            print(f"ImageManager: Slot {slot} is out of range. Cannot check can_redo.")
            return False

    def undo(self, slot=None):
        """
        Undoes the last change in the specified slot, restoring the previous image and metadata.
        
        :param slot: (Optional) The slot number to perform undo on. If None, uses current_slot.
        """
        if slot is None:
            slot = self.current_slot
        if 0 <= slot < self.max_slots:
            if self.can_undo(slot):
                # Save current state to redo stack
                self._redo_stacks[slot].append((self._images[slot].copy(), self._metadata[slot].copy()))
                # Restore the last state from undo stack
                self._images[slot], self._metadata[slot] = self._undo_stacks[slot].pop()
                self.image_changed.emit(slot, self._images[slot], self._metadata[slot])
                print(f"ImageManager: Undo performed on slot {slot}.")
            else:
                print(f"ImageManager: No actions to undo in slot {slot}.")
        else:
            print(f"ImageManager: Slot {slot} is out of range. Cannot perform undo.")

    def redo(self, slot=None):
        """
        Redoes the last undone change in the specified slot, restoring the image and metadata.
        
        :param slot: (Optional) The slot number to perform redo on. If None, uses current_slot.
        """
        if slot is None:
            slot = self.current_slot
        if 0 <= slot < self.max_slots:
            if self.can_redo(slot):
                # Save current state to undo stack
                self._undo_stacks[slot].append((self._images[slot].copy(), self._metadata[slot].copy()))
                # Restore the last state from redo stack
                self._images[slot], self._metadata[slot] = self._redo_stacks[slot].pop()
                self.image_changed.emit(slot, self._images[slot], self._metadata[slot])
                print(f"ImageManager: Redo performed on slot {slot}.")
            else:
                print(f"ImageManager: No actions to redo in slot {slot}.")
        else:
            print(f"ImageManager: Slot {slot} is out of range. Cannot perform redo.")


class XISFViewer(QWidget):
    def __init__(self, image_manager=None):
        super().__init__()
        self.image_manager = image_manager  # Reference to the ImageManager
        self.image_data = None
        self.file_meta = None
        self.image_meta = None
        self.is_mono = False
        self.bit_depth = None
        self.scale_factor = 1.0
        self.dragging = False
        self.drag_start_pos = QPoint()
        self.autostretch_enabled = False
        self.current_pixmap = None
        self.initUI()

        if self.image_manager:
            # Connect to ImageManager's image_changed signal
            self.image_manager.image_changed.connect(self.on_image_changed)
    
    def initUI(self):
        main_layout = QHBoxLayout()
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(5)



        # Set the window icon
        self.setWindowIcon(QIcon(icon_path))

        # Left side layout for image display and save button
        left_widget = QWidget()        
        left_layout = QVBoxLayout(left_widget)
        left_widget.setMinimumSize(600, 600)
        
        self.load_button = QPushButton("Load Image File")
        self.load_button.clicked.connect(self.load_xisf)
        left_layout.addWidget(self.load_button)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        
        # Add a scroll area to allow panning
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidget(self.image_label)
        self.scroll_area.setWidgetResizable(False)  # Keep it resizable
        self.scroll_area.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(self.scroll_area)

        self.toggle_button = QPushButton("Toggle Autostretch", self)
        self.toggle_button.setCheckable(True)
        self.toggle_button.clicked.connect(self.toggle_autostretch)
        left_layout.addWidget(self.toggle_button)        

        # Zoom buttons
        zoom_layout = QHBoxLayout()
        self.zoom_in_button = QPushButton("Zoom In")
        self.zoom_in_button.clicked.connect(self.zoom_in)
        self.zoom_out_button = QPushButton("Zoom Out")
        self.zoom_out_button.clicked.connect(self.zoom_out)
        zoom_layout.addWidget(self.zoom_in_button)
        zoom_layout.addWidget(self.zoom_out_button)
        left_layout.addLayout(zoom_layout)

        # Inside the initUI method, where the Save button is added
        self.save_button = QPushButton("Save As")
        self.save_button.clicked.connect(self.save_as)
        self.save_button.setEnabled(False)

        # Create the "Save Stretched Image" checkbox
        self.save_stretched_checkbox = QCheckBox("Save Stretched Image")
        self.save_stretched_checkbox.setChecked(False)  # Default is to save the original

        # Add the Save button and checkbox to a horizontal layout
        save_layout = QHBoxLayout()
        save_layout.addWidget(self.save_button)
        save_layout.addWidget(self.save_stretched_checkbox)
        left_layout.addLayout(save_layout)

        # Add a Batch Process button
        self.batch_process_button = QPushButton("XISF Converter Batch Process")
        self.batch_process_button.clicked.connect(self.open_batch_process_window)
        left_layout.addWidget(self.batch_process_button)


        footer_label = QLabel("""
            Written by Franklin Marek<br>
            <a href='http://www.setiastro.com'>www.setiastro.com</a>
        """)
        footer_label.setAlignment(Qt.AlignCenter)
        footer_label.setOpenExternalLinks(True)
        footer_label.setStyleSheet("font-size: 10px;")
        left_layout.addWidget(footer_label)

        self.load_logo()

        # Right side layout for metadata display
        right_widget = QWidget()
        right_widget.setMinimumWidth(300)
        right_layout = QVBoxLayout()
        self.metadata_tree = QTreeWidget()
        self.metadata_tree.setHeaderLabels(["Property", "Value"])
        self.metadata_tree.setColumnWidth(0, 150)
        right_layout.addWidget(self.metadata_tree)
        
        # Save Metadata button below metadata tree
        self.save_metadata_button = QPushButton("Save Metadata")
        self.save_metadata_button.clicked.connect(self.save_metadata)
        right_layout.addWidget(self.save_metadata_button)
        
        right_widget.setLayout(right_layout)

        # Add left widget and metadata tree to the splitter
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setSizes([800, 200])  # Initial sizes for the left (preview) and right (metadata) sections
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)

        main_layout.addWidget(splitter)
        self.setLayout(main_layout)
        self.setWindowTitle("XISF Liberator V1.2")

    def on_image_changed(self, slot, image, metadata):
        """
        This method is triggered when the image in ImageManager changes.
        It updates the UI with the new image.
        """
        if image is None:
            return

        # Clear the previous content before updating
        self.image_label.clear()
        self.metadata_tree.clear()  # Clear previous metadata display

        # Ensure the image is a numpy array if it is not already
        if not isinstance(image, np.ndarray):
            image = np.array(image)  # Convert to numpy array if needed

        # Update the image data and metadata
        self.image_data = image
        self.original_header = metadata.get('original_header', None)
        self.is_mono = metadata.get('is_mono', False)

        # Extract file path from metadata and pass to display_metadata
        file_path = metadata.get('file_path', None)
        if file_path:
            self.display_metadata(file_path)  # Pass the file path to display_metadata

        # Determine if the image is mono or color
        im_data = self.image_data
        if self.is_mono:
            # If the image is mono, skip squeezing as it should be 2D
            if len(im_data.shape) == 3 and im_data.shape[2] == 1:
                im_data = np.squeeze(im_data, axis=2)  # Remove the singleton channel dimension

        # Convert to the appropriate display format and update the display
        self.display_image()


    def load_logo(self):
        """
        Load and display the XISF Liberator logo before any image is loaded.
        """
        logo_path = resource_path("astrosuite.png")
        if not os.path.exists(logo_path):
            print(f"Logo image not found at path: {logo_path}")
            self.image_label.setText("XISF Liberator")
            return

        # Load the logo image
        logo_pixmap = QPixmap(logo_path)
        if logo_pixmap.isNull():
            print(f"Failed to load logo image from: {logo_path}")
            self.image_label.setText("XISF Liberator")
            return

        self.current_pixmap = logo_pixmap  # Store the logo pixmap
        scaled_pixmap = logo_pixmap.scaled(
            logo_pixmap.size() * self.scale_factor, 
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        )
        self.image_label.setPixmap(scaled_pixmap)
        self.image_label.resize(scaled_pixmap.size())

    def toggle_autostretch(self):
        self.autostretch_enabled = not self.autostretch_enabled
        if self.autostretch_enabled:
            self.apply_autostretch()
        else:
            self.stretched_image = self.image_data  # Reset to original image if stretch is disabled

        self.display_image()

    def apply_autostretch(self):
        # Determine if the image is mono or color
        if len(self.image_data.shape) == 2:  # Mono image
            self.stretched_image = stretch_mono_image(self.image_data, target_median=0.25, normalize=True)
        else:  # Color image
            self.stretched_image = stretch_color_image(self.image_data, target_median=0.25, linked=False, normalize=False)

    def open_batch_process_window(self):
        self.batch_dialog = BatchProcessDialog(self)
        self.batch_dialog.show()


    def load_xisf(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, 
            "Open Image File", 
            "", 
            "Image Files (*.png *.tif *.tiff *.fits *.fit *.xisf *.cr2 *.nef *.arw *.dng *.orf *.rw2 *.pef)"
        )

        if file_name:
            try:
                # Use the global load_image function to load the image and its metadata
                image, header, bit_depth, is_mono = load_image(file_name)
                
                # Apply debayering if needed (for non-mono images)
                if is_mono:  # Only debayer if the image is not mono
                    image, is_mono = self.debayer_image(image, file_name, header, is_mono)

                # Check if the image is mono or RGB
                self.is_mono = is_mono
                self.bit_depth = bit_depth
                self.image_data = image

                # Reset scale factor when a new image is loaded
                self.scale_factor = 0.25

                # If autostretch is enabled, apply stretch immediately after loading
                if self.autostretch_enabled:
                    self.apply_autostretch()

                # Display the image with scaling and normalization
                
                self.display_image()

                # Set image metadata (using header from load_image)
                self.file_meta = header  # Use the loaded header for metadata
                self.image_meta = None  # No separate image metadata for XISF in this example
                
                # Display metadata (using the global display_metadata method for appropriate file types)
                self.display_metadata(file_name)

                # Push the loaded image to ImageManager (only if image_manager exists)
                if hasattr(self, 'image_manager'):
                    metadata = {
                        'file_path': file_name,
                        'is_mono': self.is_mono,
                        'bit_depth': self.bit_depth,
                        'source': 'XISF'  # Or specify 'FITS' if applicable
                    }
                    # Push the numpy array to ImageManager (not memoryview)
                    self.image_manager.update_image(np.array(self.image_data), metadata, slot=0)  # Add image to slot 0 in ImageManager

                # Enable save button if the image is loaded successfully
                self.save_button.setEnabled(True)

            except Exception as e:
                self.image_label.setText(f"Failed to load XISF file: {e}")


    def debayer_image(self, image, file_path, header, is_mono):
        """Check if image is OSC (One-Shot Color) and debayer if required."""
        # Check for OSC (Bayer pattern in FITS or RAW data)
        if file_path.lower().endswith(('.fits', '.fit')):
            # Check if the FITS header contains BAYERPAT (Bayer pattern)
            bayer_pattern = header.get('BAYERPAT', None)
            if bayer_pattern:
                print(f"Debayering FITS image: {file_path} with Bayer pattern {bayer_pattern}")
                # Apply debayering logic for FITS
                is_mono = False
                image = self.debayer_fits(image, bayer_pattern)

            else:
                print(f"No Bayer pattern found in FITS header: {file_path}")
        elif file_path.lower().endswith(('.cr2', '.nef', '.arw', '.dng', '.orf', '.rw2', '.pef')):
            # If it's RAW (Bayer pattern detected), debayer it
            print(f"Debayering RAW image: {file_path}")
            # Apply debayering to the RAW image (assuming debayer_raw exists)
            is_mono = False
            image = self.debayer_raw(image)
        
        return image, is_mono

    def debayer_fits(self, image_data, bayer_pattern):
        """Debayer a FITS image using a basic Bayer pattern (2x2)."""
        if bayer_pattern == 'RGGB':
            # RGGB Bayer pattern
            r = image_data[::2, ::2]  # Red
            g1 = image_data[::2, 1::2]  # Green 1
            g2 = image_data[1::2, ::2]  # Green 2
            b = image_data[1::2, 1::2]  # Blue

            # Average green channels
            g = (g1 + g2) / 2
            return np.stack([r, g, b], axis=-1)

        elif bayer_pattern == 'BGGR':
            # BGGR Bayer pattern
            b = image_data[::2, ::2]  # Blue
            g1 = image_data[::2, 1::2]  # Green 1
            g2 = image_data[1::2, ::2]  # Green 2
            r = image_data[1::2, 1::2]  # Red

            # Average green channels
            g = (g1 + g2) / 2
            return np.stack([r, g, b], axis=-1)

        elif bayer_pattern == 'GRBG':
            # GRBG Bayer pattern
            g1 = image_data[::2, ::2]  # Green 1
            r = image_data[::2, 1::2]  # Red
            b = image_data[1::2, ::2]  # Blue
            g2 = image_data[1::2, 1::2]  # Green 2

            # Average green channels
            g = (g1 + g2) / 2
            return np.stack([r, g, b], axis=-1)

        elif bayer_pattern == 'GBRG':
            # GBRG Bayer pattern
            g1 = image_data[::2, ::2]  # Green 1
            b = image_data[::2, 1::2]  # Blue
            r = image_data[1::2, ::2]  # Red
            g2 = image_data[1::2, 1::2]  # Green 2

            # Average green channels
            g = (g1 + g2) / 2
            return np.stack([r, g, b], axis=-1)

        else:
            raise ValueError(f"Unsupported Bayer pattern: {bayer_pattern}")




    def debayer_raw(self, raw_image_data, bayer_pattern="RGGB"):
        """Debayer a RAW image based on the Bayer pattern."""
        if bayer_pattern == 'RGGB':
            # RGGB Bayer pattern (Debayering logic example)
            r = raw_image_data[::2, ::2]  # Red
            g1 = raw_image_data[::2, 1::2]  # Green 1
            g2 = raw_image_data[1::2, ::2]  # Green 2
            b = raw_image_data[1::2, 1::2]  # Blue

            # Average green channels
            g = (g1 + g2) / 2
            return np.stack([r, g, b], axis=-1)
        
        elif bayer_pattern == 'BGGR':
            # BGGR Bayer pattern
            b = raw_image_data[::2, ::2]  # Blue
            g1 = raw_image_data[::2, 1::2]  # Green 1
            g2 = raw_image_data[1::2, ::2]  # Green 2
            r = raw_image_data[1::2, 1::2]  # Red

            # Average green channels
            g = (g1 + g2) / 2
            return np.stack([r, g, b], axis=-1)

        elif bayer_pattern == 'GRBG':
            # GRBG Bayer pattern
            g1 = raw_image_data[::2, ::2]  # Green 1
            r = raw_image_data[::2, 1::2]  # Red
            b = raw_image_data[1::2, ::2]  # Blue
            g2 = raw_image_data[1::2, 1::2]  # Green 2

            # Average green channels
            g = (g1 + g2) / 2
            return np.stack([r, g, b], axis=-1)

        elif bayer_pattern == 'GBRG':
            # GBRG Bayer pattern
            g1 = raw_image_data[::2, ::2]  # Green 1
            b = raw_image_data[::2, 1::2]  # Blue
            r = raw_image_data[1::2, ::2]  # Red
            g2 = raw_image_data[1::2, 1::2]  # Green 2

            # Average green channels
            g = (g1 + g2) / 2
            return np.stack([r, g, b], axis=-1)

        else:
            raise ValueError(f"Unsupported Bayer pattern: {bayer_pattern}")


    def display_image(self):
        if self.image_data is None:
            return

        im_data = self.stretched_image if self.autostretch_enabled else self.image_data

        if self.is_mono:
            # For mono images, we expect either (height, width) or (height, width, 1) shapes.
            # If the image has 3 channels but is mono, collapse it to one channel
            if len(im_data.shape) == 3 and im_data.shape[2] == 3:
                im_data = np.mean(im_data, axis=-1)  # Convert to grayscale by averaging the channels
                print(f"Mono image with 3 channels collapsed to 1 channel: {im_data.shape}")

            elif len(im_data.shape) != 2:
                print(f"Unexpected mono image shape: {im_data.shape}")
                return  # Exit if the shape is not 2D or (height, width, 1)

            # Now im_data should be 2D (height, width) for mono images
            height, width = im_data.shape  # Unpacking 2D shape
            bytes_per_line = width

            if im_data.dtype == np.uint8:
                q_image = QImage(im_data.tobytes(), width, height, bytes_per_line, QImage.Format_Grayscale8)
            elif im_data.dtype == np.uint16:
                im_data = (im_data / 256).astype(np.uint8)
                q_image = QImage(im_data.tobytes(), width, height, bytes_per_line, QImage.Format_Grayscale8)
            elif im_data.dtype in [np.float32, np.float64]:
                im_data = np.clip((im_data - im_data.min()) / (im_data.max() - im_data.min()) * 255, 0, 255).astype(np.uint8)
                q_image = QImage(im_data.tobytes(), width, height, bytes_per_line, QImage.Format_Grayscale8)
            else:
                print(f"Unsupported mono image format: {im_data.dtype}")
                return
        else:
            # For color images, we expect (height, width, channels) shape
            height, width, channels = im_data.shape
            bytes_per_line = channels * width

            if im_data.dtype == np.uint8:
                q_image = QImage(im_data.tobytes(), width, height, bytes_per_line, QImage.Format_RGB888)
            elif im_data.dtype == np.uint16:
                im_data = (im_data / 256).astype(np.uint8)
                q_image = QImage(im_data.tobytes(), width, height, bytes_per_line, QImage.Format_RGB888)
            elif im_data.dtype in [np.float32, np.float64]:
                im_data = np.clip((im_data - im_data.min()) / (im_data.max() - im_data.min()) * 255, 0, 255).astype(np.uint8)
                q_image = QImage(im_data.tobytes(), width, height, bytes_per_line, QImage.Format_RGB888)
            else:
                print(f"Unsupported color image format: {im_data.dtype}")
                return

        # Calculate scaled dimensions
        scaled_width = int(q_image.width() * self.scale_factor)
        scaled_height = int(q_image.height() * self.scale_factor)

        # Apply scaling
        scaled_image = q_image.scaled(
            scaled_width,
            scaled_height,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        pixmap = QPixmap.fromImage(scaled_image)
        self.current_pixmap = pixmap  # **Store the current pixmap**
        self.image_label.setPixmap(pixmap)
        self.image_label.resize(scaled_image.size())


    def zoom_in(self):
        self.center_image_on_zoom(1.25)

    def zoom_out(self):
        self.center_image_on_zoom(1 / 1.25)

    def center_image_on_zoom(self, zoom_factor):
        # Get the current center point of the visible area
        current_center_x = self.scroll_area.horizontalScrollBar().value() + (self.scroll_area.viewport().width() / 2)
        current_center_y = self.scroll_area.verticalScrollBar().value() + (self.scroll_area.viewport().height() / 2)
        
        # Adjust the scale factor
        self.scale_factor *= zoom_factor
        
        # Display the image with the new scale factor
        self.display_image()
        
        # Calculate the new center point after zooming
        new_center_x = current_center_x * zoom_factor
        new_center_y = current_center_y * zoom_factor
        
        # Adjust scrollbars to keep the image centered
        self.scroll_area.horizontalScrollBar().setValue(int(new_center_x - self.scroll_area.viewport().width() / 2))
        self.scroll_area.verticalScrollBar().setValue(int(new_center_y - self.scroll_area.viewport().height() / 2))


    def wheelEvent(self, event):
        if event.angleDelta().y() > 0:
            self.zoom_in()
        else:
            self.zoom_out()


    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.dragging = True
            self.drag_start_pos = event.pos()

    def mouseMoveEvent(self, event):
        if self.dragging:
            delta = event.pos() - self.drag_start_pos
            self.scroll_area.horizontalScrollBar().setValue(
                self.scroll_area.horizontalScrollBar().value() - delta.x()
            )
            self.scroll_area.verticalScrollBar().setValue(
                self.scroll_area.verticalScrollBar().value() - delta.y()
            )
            self.drag_start_pos = event.pos()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.dragging = False

    def display_metadata(self, file_path):
        """
        Load and display metadata from the given file if the file is an XISF or FITS file.
        For other file types, simply skip without failing.
        """
        if file_path.lower().endswith('.xisf'):
            print("Loading metadata from XISF file.")
            # XISF handling (as before)
            try:
                # Load XISF file for metadata
                xisf = XISF(file_path)
                file_meta = xisf.get_file_metadata()
                image_meta = xisf.get_images_metadata()[0]

                self.metadata_tree.clear()  # Clear previous metadata
                
                # Add File Metadata
                file_meta_item = QTreeWidgetItem(["File Metadata"])
                self.metadata_tree.addTopLevelItem(file_meta_item)
                for key, value in file_meta.items():
                    item = QTreeWidgetItem([key, str(value.get('value', ''))])  # Ensure 'value' exists
                    file_meta_item.addChild(item)

                # Add Image Metadata
                image_meta_item = QTreeWidgetItem(["Image Metadata"])
                self.metadata_tree.addTopLevelItem(image_meta_item)
                for key, value in image_meta.items():
                    if key == 'FITSKeywords':
                        fits_item = QTreeWidgetItem(["FITS Keywords"])
                        image_meta_item.addChild(fits_item)
                        for kw, kw_values in value.items():
                            for kw_value in kw_values:
                                item = QTreeWidgetItem([kw, str(kw_value.get("value", ''))])
                                fits_item.addChild(item)
                    elif key == 'XISFProperties':
                        props_item = QTreeWidgetItem(["XISF Properties"])
                        image_meta_item.addChild(props_item)
                        for prop_name, prop in value.items():
                            item = QTreeWidgetItem([prop_name, str(prop.get("value", ''))])
                            props_item.addChild(item)
                    else:
                        item = QTreeWidgetItem([key, str(value)])
                        image_meta_item.addChild(item)

                self.metadata_tree.expandAll()  # Expand all metadata items
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load XISF metadata: {e}")

        elif file_path.lower().endswith(('.fits', '.fit')):
            print("Loading metadata from FITS file.")
            # FITS handling
            try:
                # Open the FITS file using Astropy
                hdul = fits.open(file_path)
                header = hdul[0].header  # Extract header from primary HDU
                hdul.close()

                self.metadata_tree.clear()  # Clear previous metadata

                # Add FITS Header Metadata
                fits_header_item = QTreeWidgetItem(["FITS Header"])
                self.metadata_tree.addTopLevelItem(fits_header_item)

                # Loop through the header and add each keyword
                for keyword, value in header.items():
                    item = QTreeWidgetItem([keyword, str(value)])
                    fits_header_item.addChild(item)

                self.metadata_tree.expandAll()  # Expand all metadata items
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load FITS metadata: {e}")

        # Handle Camera Raw files (e.g., .cr2, .nef, .arw, .dng)
        elif file_path.lower().endswith(('.cr2', '.nef', '.arw', '.dng', '.orf', '.rw2', '.pef')):
            print("Loading metadata from Camera RAW file.")
            try:
                # Use pyexiv2 to read RAW file metadata
                raw_meta_item = QTreeWidgetItem(["Camera RAW Metadata"])
                self.metadata_tree.addTopLevelItem(raw_meta_item)

                # Handle RAW file metadata using rawpy
                with rawpy.imread(file_path) as raw:
                    camera_info_item = QTreeWidgetItem(["Camera Info"])
                    raw_meta_item.addChild(camera_info_item)

                    # Camera-specific info (e.g., white balance, camera model)

                    camera_info_item.addChild(QTreeWidgetItem(["White Balance", str(raw.camera_whitebalance)]))

                    # Additional rawpy metadata
                    if raw.camera_white_level_per_channel is not None:
                        white_level_item = QTreeWidgetItem(["Camera White Level"])
                        raw_meta_item.addChild(white_level_item)
                        for i, level in enumerate(raw.camera_white_level_per_channel):
                            white_level_item.addChild(QTreeWidgetItem([f"Channel {i+1}", str(level)]))

                    # Add tone curve data if available
                    if raw.tone_curve is not None:
                        tone_curve_item = QTreeWidgetItem(["Tone Curve"])
                        raw_meta_item.addChild(tone_curve_item)
                        tone_curve_item.addChild(QTreeWidgetItem(["Tone Curve Length", str(len(raw.tone_curve))]))

                self.metadata_tree.expandAll()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load Camera RAW metadata: {e}")


        else:
            # If the file is not a FITS or XISF file, simply return without displaying metadata
            print(f"Skipping metadata for unsupported file type: {file_path}")


    def save_as(self):
        output_path, _ = QFileDialog.getSaveFileName(self, "Save Image As", "", "XISF (*.xisf);;FITS (*.fits);;TIFF (*.tif);;PNG (*.png)")
        
        if output_path:
            # Determine if we should save the stretched image or the original
            image_to_save = self.stretched_image if self.save_stretched_checkbox.isChecked() and self.stretched_image is not None else self.image_data
            _, ext = os.path.splitext(output_path)
            
            # Determine bit depth and color mode
            is_32bit_float = image_to_save.dtype == np.float32
            is_16bit = image_to_save.dtype == np.uint16
            is_8bit = image_to_save.dtype == np.uint8

            try:
                # Save as FITS file with FITS header only (no XISF properties)
                if ext.lower() in ['.fits', '.fit']:
                    header = fits.Header()
                    crval1, crval2 = None, None
                    
                    # Populate FITS header with FITS keywords and essential WCS keywords only
                    wcs_keywords = ["CTYPE1", "CTYPE2", "CRPIX1", "CRPIX2", "CRVAL1", "CRVAL2", "CDELT1", "CDELT2", "A_ORDER", "B_ORDER", "AP_ORDER", "BP_ORDER"]
                    
                    if 'FITSKeywords' in self.image_meta:
                        for keyword, values in self.image_meta['FITSKeywords'].items():
                            for entry in values:
                                if 'value' in entry:
                                    value = entry['value']
                                    if keyword in wcs_keywords:
                                        try:
                                            value = int(value)
                                        except ValueError:
                                            value = float(value)
                                    header[keyword] = value

                    # Manually add WCS information if missing
                    if 'CTYPE1' not in header:
                        header['CTYPE1'] = 'RA---TAN'
                    if 'CTYPE2' not in header:
                        header['CTYPE2'] = 'DEC--TAN'
                    
                    # Add the -SIP suffix if SIP coefficients are present
                    if any(key in header for key in ["A_ORDER", "B_ORDER", "AP_ORDER", "BP_ORDER"]):
                        header['CTYPE1'] = 'RA---TAN-SIP'
                        header['CTYPE2'] = 'DEC--TAN-SIP'

                    # Set default reference pixel (center of the image)
                    if 'CRPIX1' not in header:
                        header['CRPIX1'] = image_to_save.shape[1] / 2  # X center
                    if 'CRPIX2' not in header:
                        header['CRPIX2'] = image_to_save.shape[0] / 2  # Y center

                    # Retrieve RA and DEC values if available
                    if 'FITSKeywords' in self.image_meta:
                        if 'RA' in self.image_meta['FITSKeywords']:
                            crval1 = float(self.image_meta['FITSKeywords']['RA'][0]['value'])  # Reference RA
                        if 'DEC' in self.image_meta['FITSKeywords']:
                            crval2 = float(self.image_meta['FITSKeywords']['DEC'][0]['value'])  # Reference DEC

                    # Add CRVAL1 and CRVAL2 to the header if found
                    if crval1 is not None and crval2 is not None:
                        header['CRVAL1'] = crval1
                        header['CRVAL2'] = crval2
                    else:
                        print("RA and DEC values not found in FITS Keywords")

                    # Calculate pixel scale if focal length and pixel size are available
                    if 'FOCALLEN' in self.image_meta['FITSKeywords'] and 'XPIXSZ' in self.image_meta['FITSKeywords']:
                        focal_length = float(self.image_meta['FITSKeywords']['FOCALLEN'][0]['value'])  # in mm
                        pixel_size = float(self.image_meta['FITSKeywords']['XPIXSZ'][0]['value'])  # in μm
                        pixel_scale = (pixel_size * 206.265) / focal_length  # arcsec/pixel
                        header['CDELT1'] = -pixel_scale / 3600.0
                        header['CDELT2'] = pixel_scale / 3600.0
                    else:
                        header['CDELT1'] = -2.77778e-4  # ~1 arcsecond/pixel
                        header['CDELT2'] = 2.77778e-4

                    # Populate CD matrix using the XISF LinearTransformationMatrix if available
                    if 'XISFProperties' in self.image_meta and 'PCL:AstrometricSolution:LinearTransformationMatrix' in self.image_meta['XISFProperties']:
                        linear_transform = self.image_meta['XISFProperties']['PCL:AstrometricSolution:LinearTransformationMatrix']['value']
                        header['CD1_1'] = linear_transform[0][0]
                        header['CD1_2'] = linear_transform[0][1]
                        header['CD2_1'] = linear_transform[1][0]
                        header['CD2_2'] = linear_transform[1][1]
                    else:
                        header['CD1_1'] = header['CDELT1']
                        header['CD1_2'] = 0.0
                        header['CD2_1'] = 0.0
                        header['CD2_2'] = header['CDELT2']

                    # Duplicate the mono image to create a 3-channel image if it’s mono
                    if self.is_mono:
                        image_data_fits = np.stack([image_to_save[:, :, 0]] * 3, axis=-1)  # Create 3-channel from mono
                        image_data_fits = np.transpose(image_data_fits, (2, 0, 1))  # Reorder to (channels, height, width)
                        header['NAXIS'] = 3
                        header['NAXIS3'] = 3  # Channels (RGB)
                    else:
                        image_data_fits = np.transpose(image_to_save, (2, 0, 1))  # RGB images in (channels, height, width)
                        header['NAXIS'] = 3
                        header['NAXIS3'] = 3  # Channels (RGB)

                    hdu = fits.PrimaryHDU(image_data_fits, header=header)
                    hdu.writeto(output_path, overwrite=True)
                    print(f"Saved FITS image with metadata to: {output_path}")

                # Save as TIFF based on bit depth
                elif ext.lower() in ['.tif', '.tiff']:
                    if is_16bit:
                        self.save_tiff(output_path, bit_depth=16)
                    elif is_32bit_float:
                        self.save_tiff(output_path, bit_depth=32)
                    else:
                        self.save_tiff(output_path, bit_depth=8)
                    print(f"Saved TIFF image with {self.bit_depth} bit depth to: {output_path}")

                # Save as PNG
                elif ext.lower() == '.png':
                    # Convert mono images to RGB for PNG format
                    if self.is_mono:
                        image_8bit = (image_to_save[:, :, 0] * 255).astype(np.uint8) if not is_8bit else image_to_save[:, :, 0]
                        image_8bit_rgb = np.stack([image_8bit] * 3, axis=-1)  # Duplicate channel to create RGB
                    else:
                        image_8bit_rgb = (image_to_save * 255).astype(np.uint8) if not is_8bit else image_to_save
                    Image.fromarray(image_8bit_rgb).save(output_path)
                    print(f"Saved 8-bit PNG image to: {output_path}")

                # Save as XISF with metadata
                elif ext.lower() == '.xisf':
                    XISF.write(output_path, image_to_save, xisf_metadata=self.file_meta)
                    print(f"Saved XISF image with metadata to: {output_path}")

            except Exception as e:
                print(f"Error saving file: {e}")


    def process_batch(self, input_dir, output_dir, file_format, update_status_callback):
        import glob
        from pathlib import Path

        xisf_files = glob.glob(f"{input_dir}/*.xisf")
        if not xisf_files:
            QMessageBox.warning(self, "Error", "No XISF files found in the input directory.")
            update_status_callback("")
            return

        for i, xisf_file in enumerate(xisf_files, start=1):
            try:
                # Update progress
                update_status_callback(f"Processing file {i}/{len(xisf_files)}: {Path(xisf_file).name}")

                # Load the XISF file
                xisf = XISF(xisf_file)
                im_data = xisf.read_image(0)

                # Set metadata
                file_meta = xisf.get_file_metadata()
                image_meta = xisf.get_images_metadata()[0]
                is_mono = im_data.shape[2] == 1 if len(im_data.shape) == 3 else True

                # Determine output file path
                base_name = Path(xisf_file).stem
                output_file = Path(output_dir) / f"{base_name}{file_format}"

                # Save the file using save_direct
                self.save_direct(output_file, im_data, file_meta, image_meta, is_mono)

            except Exception as e:
                update_status_callback(f"Error processing file {Path(xisf_file).name}: {e}")
                continue  # Skip to the next file

        update_status_callback("Batch Processing Complete!")

    def save_direct(self, output_path, image_to_save, file_meta, image_meta, is_mono):
        """
        Save an image directly to the specified path with the given metadata.
        This function does not prompt the user and is suitable for batch processing.
        """
        _, ext = os.path.splitext(output_path)

        # Determine bit depth and color mode
        is_32bit_float = image_to_save.dtype == np.float32
        is_16bit = image_to_save.dtype == np.uint16
        is_8bit = image_to_save.dtype == np.uint8

        try:
            # Save as FITS file with metadata
            if ext.lower() in ['.fits', '.fit']:
                header = fits.Header()
                crval1, crval2 = None, None

                # Populate FITS header with FITS keywords and WCS keywords
                wcs_keywords = [
                    "CTYPE1", "CTYPE2", "CRPIX1", "CRPIX2", "CRVAL1", "CRVAL2", 
                    "CDELT1", "CDELT2", "A_ORDER", "B_ORDER", "AP_ORDER", "BP_ORDER"
                ]

                if 'FITSKeywords' in image_meta:
                    for keyword, values in image_meta['FITSKeywords'].items():
                        for entry in values:
                            if 'value' in entry:
                                value = entry['value']
                                # Convert only numerical values to float
                                if keyword in wcs_keywords and isinstance(value, (int, float)):
                                    value = float(value)
                                header[keyword] = value

                # Add default WCS information if missing
                if 'CTYPE1' not in header:
                    header['CTYPE1'] = 'RA---TAN'
                if 'CTYPE2' not in header:
                    header['CTYPE2'] = 'DEC--TAN'

                # Add the -SIP suffix for SIP coefficients
                if any(key in header for key in ["A_ORDER", "B_ORDER", "AP_ORDER", "BP_ORDER"]):
                    header['CTYPE1'] = 'RA---TAN-SIP'
                    header['CTYPE2'] = 'DEC--TAN-SIP'

                # Set default reference pixel if missing
                if 'CRPIX1' not in header:
                    header['CRPIX1'] = image_to_save.shape[1] / 2
                if 'CRPIX2' not in header:
                    header['CRPIX2'] = image_to_save.shape[0] / 2

                # Add CRVAL1 and CRVAL2 if available
                if 'RA' in image_meta.get('FITSKeywords', {}):
                    crval1 = float(image_meta['FITSKeywords']['RA'][0]['value'])
                if 'DEC' in image_meta.get('FITSKeywords', {}):
                    crval2 = float(image_meta['FITSKeywords']['DEC'][0]['value'])

                if crval1 is not None and crval2 is not None:
                    header['CRVAL1'] = crval1
                    header['CRVAL2'] = crval2

                # Add CD matrix if available
                if 'XISFProperties' in image_meta and 'PCL:AstrometricSolution:LinearTransformationMatrix' in image_meta['XISFProperties']:
                    linear_transform = image_meta['XISFProperties']['PCL:AstrometricSolution:LinearTransformationMatrix']['value']
                    header['CD1_1'] = linear_transform[0][0]
                    header['CD1_2'] = linear_transform[0][1]
                    header['CD2_1'] = linear_transform[1][0]
                    header['CD2_2'] = linear_transform[1][1]
                else:
                    header['CD1_1'] = header['CDELT1'] if 'CDELT1' in header else 0.0
                    header['CD1_2'] = 0.0
                    header['CD2_1'] = 0.0
                    header['CD2_2'] = header['CDELT2'] if 'CDELT2' in header else 0.0

                # Duplicate mono image to create 3-channel if necessary
                if is_mono:
                    image_data_fits = image_to_save[:, :, 0] if len(image_to_save.shape) == 3 else image_to_save
                    header['NAXIS'] = 2  # Mono images are 2-dimensional
                else:
                    image_data_fits = np.transpose(image_to_save, (2, 0, 1))
                    header['NAXIS'] = 3
                    header['NAXIS3'] = 3

                hdu = fits.PrimaryHDU(image_data_fits, header=header)
                hdu.writeto(output_path, overwrite=True)
                print(f"Saved FITS image to: {output_path}")


            # Save as TIFF
            elif ext.lower() in ['.tif', '.tiff']:
                if is_16bit:
                    tiff.imwrite(output_path, (image_to_save * 65535).astype(np.uint16))
                elif is_32bit_float:
                    tiff.imwrite(output_path, image_to_save.astype(np.float32))
                else:
                    tiff.imwrite(output_path, (image_to_save * 255).astype(np.uint8))
                print(f"Saved TIFF image to: {output_path}")

            # Save as PNG
            elif ext.lower() == '.png':
                if is_mono:
                    image_8bit = (image_to_save[:, :, 0] * 255).astype(np.uint8) if not is_8bit else image_to_save[:, :, 0]
                    image_8bit_rgb = np.stack([image_8bit] * 3, axis=-1)
                else:
                    image_8bit_rgb = (image_to_save * 255).astype(np.uint8) if not is_8bit else image_to_save
                Image.fromarray(image_8bit_rgb).save(output_path)
                print(f"Saved PNG image to: {output_path}")

            # Save as XISF
            elif ext.lower() == '.xisf':
                XISF.write(output_path, image_to_save, xisf_metadata=file_meta)
                print(f"Saved XISF image to: {output_path}")

            else:
                print(f"Unsupported file format: {ext}")

        except Exception as e:
            print(f"Error saving file {output_path}: {e}")


    def save_tiff(self, output_path, bit_depth):
        if bit_depth == 16:
            if self.is_mono:
                tiff.imwrite(output_path, (self.image_data[:, :, 0] * 65535).astype(np.uint16))
            else:
                tiff.imwrite(output_path, (self.image_data * 65535).astype(np.uint16))
        elif bit_depth == 32:
            if self.is_mono:
                tiff.imwrite(output_path, self.image_data[:, :, 0].astype(np.float32))
            else:
                tiff.imwrite(output_path, self.image_data.astype(np.float32))
        else:  # 8-bit
            image_8bit = (self.image_data * 255).astype(np.uint8)
            if self.is_mono:
                tiff.imwrite(output_path, image_8bit[:, :, 0])
            else:
                tiff.imwrite(output_path, image_8bit)

    def save_metadata(self):
        if not self.file_meta and not self.image_meta:
            QMessageBox.warning(self, "Warning", "No metadata to save.")
            return

        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Metadata", "", "CSV Files (*.csv);;All Files (*)", options=options)
        if file_path:
            try:
                # Flatten metadata function
                def flatten_metadata(data, parent_key=''):
                    items = []
                    for key, value in data.items():
                        new_key = f"{parent_key}.{key}" if parent_key else key
                        if isinstance(value, dict):
                            items.extend(flatten_metadata(value, new_key).items())
                        elif isinstance(value, list):
                            for i, list_item in enumerate(value):
                                list_key = f"{new_key}_{i}"
                                items.extend(flatten_metadata({list_key: list_item}).items())
                        else:
                            items.append((new_key, value if value is not None else ''))  # Replace None with an empty string
                    return dict(items)

                # Flatten both file_meta and image_meta
                flattened_file_meta = flatten_metadata(self.file_meta) if self.file_meta else {}
                flattened_image_meta = flatten_metadata(self.image_meta) if self.image_meta else {}

                # Combine both metadata into one dictionary for CSV
                combined_meta = {**flattened_file_meta, **flattened_image_meta}

                # Write to CSV
                with open(file_path, mode='w', newline='', encoding='utf-8') as file:
                    writer = csv.writer(file)
                    writer.writerow(["Key", "Value"])  # Header row
                    for key, value in combined_meta.items():
                        writer.writerow([key, value])

                QMessageBox.information(self, "Success", f"Metadata saved to {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save metadata: {e}")     
                
class BatchProcessDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Batch Process")
        self.setMinimumWidth(400)

        layout = QVBoxLayout()

        # Input directory
        self.input_dir_label = QLabel("Input Directory:")
        self.input_dir_button = QPushButton("Select Input Directory")
        self.input_dir_button.clicked.connect(self.select_input_directory)
        self.input_dir = QLineEdit()
        self.input_dir.setReadOnly(True)

        layout.addWidget(self.input_dir_label)
        layout.addWidget(self.input_dir)
        layout.addWidget(self.input_dir_button)

        # Output directory
        self.output_dir_label = QLabel("Output Directory:")
        self.output_dir_button = QPushButton("Select Output Directory")
        self.output_dir_button.clicked.connect(self.select_output_directory)
        self.output_dir = QLineEdit()
        self.output_dir.setReadOnly(True)

        layout.addWidget(self.output_dir_label)
        layout.addWidget(self.output_dir)
        layout.addWidget(self.output_dir_button)

        # File format
        self.format_label = QLabel("Select Output Format:")
        self.format_combo = QComboBox()
        self.format_combo.addItems([".png", ".fit", ".fits", ".tif", ".tiff"])

        layout.addWidget(self.format_label)
        layout.addWidget(self.format_combo)

        # Start Batch Processing button
        self.start_button = QPushButton("Start Batch Processing")
        self.start_button.clicked.connect(self.start_batch_processing)
        layout.addWidget(self.start_button)

        # Status label
        self.status_label = QLabel("")
        layout.addWidget(self.status_label)

        self.setLayout(layout)

    def select_input_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Input Directory")
        if directory:
            self.input_dir.setText(directory)

    def select_output_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if directory:
            self.output_dir.setText(directory)

    def start_batch_processing(self):
        input_dir = self.input_dir.text()
        output_dir = self.output_dir.text()
        file_format = self.format_combo.currentText()

        if not input_dir or not output_dir:
            QMessageBox.warning(self, "Error", "Please select both input and output directories.")
            return

        self.status_label.setText("Initializing batch processing...")
        QApplication.processEvents()  # Ensures UI updates immediately

        # Call the parent function to process files with progress updates
        self.parent().process_batch(input_dir, output_dir, file_format, self.update_status)

        self.status_label.setText("Batch Processing Complete!")

    def update_status(self, message):
        self.status_label.setText(message)
        QApplication.processEvents()  # Ensures UI updates immediately

class BlinkTab(QWidget):
    def __init__(self, image_manager=None):
        super().__init__()

        self.image_paths = []  # Store the file paths of loaded images
        self.loaded_images = []  # Store the image objects (as numpy arrays)
        self.image_labels = []  # Store corresponding file names for the TreeWidget
        self.image_manager = image_manager  # Reference to ImageManager
        self.zoom_level = 0.5  # Default zoom level
        self.dragging = False  # Track whether the mouse is dragging
        self.last_mouse_pos = None  # Store the last mouse position

        self.initUI()

    def initUI(self):
        main_layout = QHBoxLayout(self)

        # Create a QSplitter to allow resizing between left and right panels
        splitter = QSplitter(Qt.Horizontal, self)

        # Left Column for the file loading and TreeView
        left_widget = QWidget(self)
        left_layout = QVBoxLayout(left_widget)

        # File Selection Button
        self.fileButton = QPushButton('Select Images', self)
        self.fileButton.clicked.connect(self.openFileDialog)
        left_layout.addWidget(self.fileButton)

        # Playback controls (left arrow, play, pause, right arrow)
        playback_controls_layout = QHBoxLayout()

        # Left Arrow Button
        self.left_arrow_button = QPushButton(self)
        self.left_arrow_button.setIcon(self.style().standardIcon(QStyle.SP_ArrowLeft))
        self.left_arrow_button.clicked.connect(self.previous_item)
        playback_controls_layout.addWidget(self.left_arrow_button)

        # Play Button
        self.play_button = QPushButton(self)
        self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.play_button.clicked.connect(self.start_playback)
        playback_controls_layout.addWidget(self.play_button)

        # Pause Button
        self.pause_button = QPushButton(self)
        self.pause_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
        self.pause_button.clicked.connect(self.stop_playback)
        playback_controls_layout.addWidget(self.pause_button)

        # Right Arrow Button
        self.right_arrow_button = QPushButton(self)
        self.right_arrow_button.setIcon(self.style().standardIcon(QStyle.SP_ArrowRight))
        self.right_arrow_button.clicked.connect(self.next_item)
        playback_controls_layout.addWidget(self.right_arrow_button)

        left_layout.addLayout(playback_controls_layout)

        # Tree view for file names
        self.fileTree = QTreeWidget(self)
        self.fileTree.setColumnCount(1)
        self.fileTree.setHeaderLabels(["Image Files"])
        self.fileTree.setSelectionMode(QAbstractItemView.ExtendedSelection)  # Allow multiple selections
        self.fileTree.itemClicked.connect(self.on_item_clicked)
        self.fileTree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.fileTree.customContextMenuRequested.connect(self.on_right_click)
        self.fileTree.currentItemChanged.connect(self.on_current_item_changed) 
        self.fileTree.setStyleSheet("""
                QTreeWidget::item:selected {
                    background-color: #3a75c4;  /* Blue background for selected items */
                    color: #ffffff;  /* White text color */
                }
            """)
        left_layout.addWidget(self.fileTree)

        # Add progress bar
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 100)
        left_layout.addWidget(self.progress_bar)

        # Add loading message label
        self.loading_label = QLabel("Loading images...", self)
        left_layout.addWidget(self.loading_label)

        # Set the layout for the left widget
        left_widget.setLayout(left_layout)

        # Add the left widget to the splitter
        splitter.addWidget(left_widget)

        # Right Column for Image Preview
        right_widget = QWidget(self)
        right_layout = QVBoxLayout(right_widget)

        # Zoom controls: Add Zoom In and Zoom Out buttons
        zoom_controls_layout = QHBoxLayout()

        self.zoom_in_button = QPushButton("Zoom In")
        self.zoom_in_button.clicked.connect(self.zoom_in)
        zoom_controls_layout.addWidget(self.zoom_in_button)

        self.zoom_out_button = QPushButton("Zoom Out")
        self.zoom_out_button.clicked.connect(self.zoom_out)
        zoom_controls_layout.addWidget(self.zoom_out_button)

        self.fit_to_preview_button = QPushButton("Fit to Preview")
        self.fit_to_preview_button.clicked.connect(self.fit_to_preview)
        zoom_controls_layout.addWidget(self.fit_to_preview_button)

        right_layout.addLayout(zoom_controls_layout)

        # Scroll area for the preview
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.viewport().installEventFilter(self)

        # QLabel for the image preview
        self.preview_label = QLabel(self)
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.scroll_area.setWidget(self.preview_label)

        right_layout.addWidget(self.scroll_area)

        # Set the layout for the right widget
        right_widget.setLayout(right_layout)

        # Add the right widget to the splitter
        splitter.addWidget(right_widget)

        # Set initial splitter sizes
        splitter.setSizes([300, 700])  # Adjust proportions as needed

        # Add the splitter to the main layout
        main_layout.addWidget(splitter)

        # Set the main layout for the widget
        self.setLayout(main_layout)

        # Initialize playback timer
        self.playback_timer = QTimer(self)
        self.playback_timer.setInterval(200)  # Set the playback interval to 500ms
        self.playback_timer.timeout.connect(self.next_item)

        # Connect the selection change signal to update the preview when arrow keys are used
        self.fileTree.selectionModel().selectionChanged.connect(self.on_selection_changed)

    def on_current_item_changed(self, current, previous):
        """Ensure the selected item is visible by scrolling to it."""
        if current:
            self.fileTree.scrollToItem(current, QAbstractItemView.PositionAtCenter)

    def previous_item(self):
        """Select the previous item in the TreeWidget."""
        current_item = self.fileTree.currentItem()
        if current_item:
            all_items = self.get_all_leaf_items()
            current_index = all_items.index(current_item)
            if current_index > 0:
                previous_item = all_items[current_index - 1]
            else:
                previous_item = all_items[-1]  # Loop back to the last item
            self.fileTree.setCurrentItem(previous_item)
            self.on_item_clicked(previous_item, 0)  # Update the preview

    def next_item(self):
        """Select the next item in the TreeWidget, looping back to the first item if at the end."""
        current_item = self.fileTree.currentItem()
        if current_item:
            all_items = self.get_all_leaf_items()
            current_index = all_items.index(current_item)
            if current_index < len(all_items) - 1:
                next_item = all_items[current_index + 1]
            else:
                next_item = all_items[0]  # Loop back to the first item
            self.fileTree.setCurrentItem(next_item)
            self.on_item_clicked(next_item, 0)  # Update the preview

    def get_all_leaf_items(self):
        """Get a flat list of all leaf items (actual files) in the TreeWidget."""
        def recurse(parent):
            items = []
            for index in range(parent.childCount()):
                child = parent.child(index)
                if child.childCount() == 0:  # It's a leaf item
                    items.append(child)
                else:
                    items.extend(recurse(child))
            return items

        root = self.fileTree.invisibleRootItem()
        return recurse(root)

    def start_playback(self):
        """Start playing through the items in the TreeWidget."""
        if not self.playback_timer.isActive():
            self.playback_timer.start()

    def stop_playback(self):
        """Stop playing through the items."""
        if self.playback_timer.isActive():
            self.playback_timer.stop()


    def openFileDialog(self):
        """Allow users to select multiple images."""
        file_paths, _ = QFileDialog.getOpenFileNames(self, "Open Images", "", "Images (*.png *.tif *.tiff *.fits *.fit *.xisf *.cr2 *.nef *.arw *.dng *.orf *.rw2 *.pef);;All Files (*)")
        
        if file_paths:
            self.image_paths = file_paths
            self.fileTree.clear()  # Clear the existing tree items

            # Dictionary to store images grouped by filter and exposure time
            grouped_images = {}

            # Load the images into memory (storing both file path and image data)
            self.loaded_images = []
            total_files = len(file_paths)

            for index, file_path in enumerate(file_paths):
                image, header, bit_depth, is_mono = load_image(file_path)

                # Debayer the image if needed (for non-mono images)
                if is_mono:
                    image = self.debayer_image(image, file_path, header)

                # Stretch the image now while loading it
                target_median = 0.25
                if image.ndim == 2:  # Mono image
                    stretched_image = stretch_mono_image(image, target_median)
                else:  # Color image
                    stretched_image = stretch_color_image(image, target_median, linked=False)

                # Append the stretched image data
                self.loaded_images.append({
                    'file_path': file_path,
                    'image_data': stretched_image,
                    'header': header,
                    'bit_depth': bit_depth,
                    'is_mono': is_mono
                })

                # Extract filter and exposure time from FITS header
                object_name = header.get('OBJECT', 'Unknown')
                filter_name = header.get('FILTER', 'Unknown')
                exposure_time = header.get('EXPOSURE', 'Unknown')

                # Group images by filter and exposure time
                group_key = (object_name, filter_name, exposure_time)
                if group_key not in grouped_images:
                    grouped_images[group_key] = []
                grouped_images[group_key].append(file_path)

                # Update progress bar
                progress = int((index + 1) / total_files * 100)
                self.progress_bar.setValue(progress)
                QApplication.processEvents()  # Ensure the UI updates in real-time

            print(f"Loaded {len(self.loaded_images)} images into memory.")
            self.loading_label.setText(f"Loaded {len(self.loaded_images)} images.")

            # Optionally, reset the progress bar and loading message when done
            self.progress_bar.setValue(100)
            self.loading_label.setText("Loading complete.")

            # Display grouped images in the tree view
            grouped_by_object = {}

            # First, group by object_name
            for (object_name, filter_name, exposure_time), paths in grouped_images.items():
                if object_name not in grouped_by_object:
                    grouped_by_object[object_name] = {}
                if filter_name not in grouped_by_object[object_name]:
                    grouped_by_object[object_name][filter_name] = {}
                if exposure_time not in grouped_by_object[object_name][filter_name]:
                    grouped_by_object[object_name][filter_name][exposure_time] = []
                grouped_by_object[object_name][filter_name][exposure_time].extend(paths)

            # Now, create the tree structure
            for object_name, filters in grouped_by_object.items():
                object_item = QTreeWidgetItem([f"Object: {object_name}"])
                self.fileTree.addTopLevelItem(object_item)
                object_item.setExpanded(True)  # Expand the object item
                for filter_name, exposures in filters.items():
                    filter_item = QTreeWidgetItem([f"Filter: {filter_name}"])
                    object_item.addChild(filter_item)
                    filter_item.setExpanded(True)  # Expand the filter item
                    for exposure_time, paths in exposures.items():
                        exposure_item = QTreeWidgetItem([f"Exposure: {exposure_time}"])
                        filter_item.addChild(exposure_item)
                        exposure_item.setExpanded(True)  # Expand the exposure item
                        for file_path in paths:
                            file_name = os.path.basename(file_path)
                            item = QTreeWidgetItem([file_name])
                            exposure_item.addChild(item)


    def debayer_image(self, image, file_path, header):
        """Check if image is OSC (One-Shot Color) and debayer if required."""
        # Check for OSC (Bayer pattern in FITS or RAW data)
        if file_path.lower().endswith(('.fits', '.fit')):
            # Check if the FITS header contains BAYERPAT (Bayer pattern)
            bayer_pattern = header.get('BAYERPAT', None)
            if bayer_pattern:
                print(f"Debayering FITS image: {file_path} with Bayer pattern {bayer_pattern}")
                # Apply debayering logic for FITS
                image = self.debayer_fits(image, bayer_pattern)
            else:
                print(f"No Bayer pattern found in FITS header: {file_path}")
        elif file_path.lower().endswith(('.cr2', '.nef', '.arw', '.dng', '.orf', '.rw2', '.pef')):
            # If it's RAW (Bayer pattern detected), debayer it
            print(f"Debayering RAW image: {file_path}")
            # Apply debayering to the RAW image (assuming debayer_raw exists)
            image = self.debayer_raw(image)
        
        return image

    def debayer_fits(self, image_data, bayer_pattern):
        """Debayer a FITS image using a basic Bayer pattern (2x2)."""
        if bayer_pattern == 'RGGB':
            # RGGB Bayer pattern
            r = image_data[::2, ::2]  # Red
            g1 = image_data[::2, 1::2]  # Green 1
            g2 = image_data[1::2, ::2]  # Green 2
            b = image_data[1::2, 1::2]  # Blue

            # Average green channels
            g = (g1 + g2) / 2
            return np.stack([r, g, b], axis=-1)

        elif bayer_pattern == 'BGGR':
            # BGGR Bayer pattern
            b = image_data[::2, ::2]  # Blue
            g1 = image_data[::2, 1::2]  # Green 1
            g2 = image_data[1::2, ::2]  # Green 2
            r = image_data[1::2, 1::2]  # Red

            # Average green channels
            g = (g1 + g2) / 2
            return np.stack([r, g, b], axis=-1)

        elif bayer_pattern == 'GRBG':
            # GRBG Bayer pattern
            g1 = image_data[::2, ::2]  # Green 1
            r = image_data[::2, 1::2]  # Red
            b = image_data[1::2, ::2]  # Blue
            g2 = image_data[1::2, 1::2]  # Green 2

            # Average green channels
            g = (g1 + g2) / 2
            return np.stack([r, g, b], axis=-1)

        elif bayer_pattern == 'GBRG':
            # GBRG Bayer pattern
            g1 = image_data[::2, ::2]  # Green 1
            b = image_data[::2, 1::2]  # Blue
            r = image_data[1::2, ::2]  # Red
            g2 = image_data[1::2, 1::2]  # Green 2

            # Average green channels
            g = (g1 + g2) / 2
            return np.stack([r, g, b], axis=-1)

        else:
            raise ValueError(f"Unsupported Bayer pattern: {bayer_pattern}")




    def debayer_raw(self, raw_image_data, bayer_pattern="RGGB"):
        """Debayer a RAW image based on the Bayer pattern."""
        if bayer_pattern == 'RGGB':
            # RGGB Bayer pattern (Debayering logic example)
            r = raw_image_data[::2, ::2]  # Red
            g1 = raw_image_data[::2, 1::2]  # Green 1
            g2 = raw_image_data[1::2, ::2]  # Green 2
            b = raw_image_data[1::2, 1::2]  # Blue

            # Average green channels
            g = (g1 + g2) / 2
            return np.stack([r, g, b], axis=-1)
        
        elif bayer_pattern == 'BGGR':
            # BGGR Bayer pattern
            b = raw_image_data[::2, ::2]  # Blue
            g1 = raw_image_data[::2, 1::2]  # Green 1
            g2 = raw_image_data[1::2, ::2]  # Green 2
            r = raw_image_data[1::2, 1::2]  # Red

            # Average green channels
            g = (g1 + g2) / 2
            return np.stack([r, g, b], axis=-1)

        elif bayer_pattern == 'GRBG':
            # GRBG Bayer pattern
            g1 = raw_image_data[::2, ::2]  # Green 1
            r = raw_image_data[::2, 1::2]  # Red
            b = raw_image_data[1::2, ::2]  # Blue
            g2 = raw_image_data[1::2, 1::2]  # Green 2

            # Average green channels
            g = (g1 + g2) / 2
            return np.stack([r, g, b], axis=-1)

        elif bayer_pattern == 'GBRG':
            # GBRG Bayer pattern
            g1 = raw_image_data[::2, ::2]  # Green 1
            b = raw_image_data[::2, 1::2]  # Blue
            r = raw_image_data[1::2, ::2]  # Red
            g2 = raw_image_data[1::2, 1::2]  # Green 2

            # Average green channels
            g = (g1 + g2) / 2
            return np.stack([r, g, b], axis=-1)

        else:
            raise ValueError(f"Unsupported Bayer pattern: {bayer_pattern}")
    


    def on_item_clicked(self, item, column):
        """Handle click on a file name in the tree to preview the image."""
        file_name = item.text(0)
        file_path = next((path for path in self.image_paths if os.path.basename(path) == file_name), None)

        if file_path:
            # Get the index of the clicked image
            index = self.image_paths.index(file_path)

            # Retrieve the corresponding image data and header from the loaded images
            stretched_image = self.loaded_images[index]['image_data']

            # Convert to QImage and display
            qimage = self.convert_to_qimage(stretched_image)
            pixmap = QPixmap.fromImage(qimage)

            # Store the pixmap for zooming
            self.current_pixmap = pixmap

            # Apply zoom level
            self.apply_zoom()

    def apply_zoom(self):
        """Apply the current zoom level to the pixmap and update the display."""
        if self.current_pixmap:
            # Scale the pixmap based on the zoom level
            scaled_pixmap = self.current_pixmap.scaled(
                self.current_pixmap.size() * self.zoom_level,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )

            # Update the QLabel with the scaled pixmap
            self.preview_label.setPixmap(scaled_pixmap)
            self.preview_label.resize(scaled_pixmap.size())

            # Adjust scroll position to center the view
            self.scroll_area.horizontalScrollBar().setValue(
                (self.preview_label.width() - self.scroll_area.viewport().width()) // 2
            )
            self.scroll_area.verticalScrollBar().setValue(
                (self.preview_label.height() - self.scroll_area.viewport().height()) // 2
            )



    def zoom_in(self):
        """Increase the zoom level and refresh the image."""
        self.zoom_level = min(self.zoom_level * 1.2, 3.0)  # Cap at 3x
        self.apply_zoom()

    def zoom_out(self):
        """Decrease the zoom level and refresh the image."""
        self.zoom_level = max(self.zoom_level / 1.2, 0.05)  # Cap at 0.2x
        self.apply_zoom()


    def fit_to_preview(self):
        """Adjust the zoom level so the image fits within the QScrollArea viewport."""
        if self.current_pixmap:
            # Get the size of the QScrollArea's viewport
            viewport_size = self.scroll_area.viewport().size()
            pixmap_size = self.current_pixmap.size()

            # Calculate the zoom level required to fit the pixmap in the QScrollArea viewport
            width_ratio = viewport_size.width() / pixmap_size.width()
            height_ratio = viewport_size.height() / pixmap_size.height()
            self.zoom_level = min(width_ratio, height_ratio)

            # Apply the zoom level
            self.apply_zoom()
        else:
            print("No image loaded. Cannot fit to preview.")
            QMessageBox.warning(self, "Warning", "No image loaded. Cannot fit to preview.")





    def on_right_click(self, pos):
        """Allow renaming, moving, and deleting an image file from the list."""
        item = self.fileTree.itemAt(pos)
        if item:
            menu = QMenu(self)

            # Add action to push image to ImageManager
            push_action = QAction("Push Image for Processing", self)
            push_action.triggered.connect(lambda: self.push_image_to_manager(item))
            menu.addAction(push_action)

            # Add action to rename the image
            rename_action = QAction("Rename", self)
            rename_action.triggered.connect(lambda: self.rename_item(item))
            menu.addAction(rename_action)


            # Add action to batch rename items
            batch_rename_action = QAction("Batch Flag Items", self)
            batch_rename_action.triggered.connect(lambda: self.batch_rename_items())
            menu.addAction(batch_rename_action)

            # Add action to move the image
            move_action = QAction("Move Selected Items", self)
            move_action.triggered.connect(lambda: self.move_items())
            menu.addAction(move_action)

            # Add action to delete image from the list
            delete_action = QAction("Delete Selected Items", self)
            delete_action.triggered.connect(lambda: self.delete_items())
            menu.addAction(delete_action)

            menu.exec_(self.fileTree.mapToGlobal(pos))

    def rename_item(self, item):
        """Allow the user to rename the selected image."""
        current_name = item.text(0)
        new_name, ok = QInputDialog.getText(self, "Rename Image", "Enter new name:", text=current_name)

        if ok and new_name:
            file_path = next((path for path in self.image_paths if os.path.basename(path) == current_name), None)
            if file_path:
                # Get the new file path with the new name
                new_file_path = os.path.join(os.path.dirname(file_path), new_name)

                try:
                    # Rename the file
                    os.rename(file_path, new_file_path)
                    print(f"File renamed from {current_name} to {new_name}")
                    
                    # Update the image paths and tree view
                    self.image_paths[self.image_paths.index(file_path)] = new_file_path
                    item.setText(0, new_name)
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Failed to rename the file: {e}")

    def batch_rename_items(self):
        """Batch rename selected items by adding a prefix or suffix."""
        selected_items = self.fileTree.selectedItems()

        if not selected_items:
            QMessageBox.warning(self, "Warning", "No items selected for renaming.")
            return

        # Create a custom dialog for entering the prefix and suffix
        dialog = QDialog(self)
        dialog.setWindowTitle("Batch Rename")
        dialog_layout = QVBoxLayout(dialog)

        instruction_label = QLabel("Enter a prefix or suffix to rename selected files:")
        dialog_layout.addWidget(instruction_label)

        # Create fields for prefix and suffix
        form_layout = QHBoxLayout()

        prefix_field = QLineEdit(dialog)
        prefix_field.setPlaceholderText("Prefix")
        form_layout.addWidget(prefix_field)

        current_filename_label = QLabel("currentfilename", dialog)
        form_layout.addWidget(current_filename_label)

        suffix_field = QLineEdit(dialog)
        suffix_field.setPlaceholderText("Suffix")
        form_layout.addWidget(suffix_field)

        dialog_layout.addLayout(form_layout)

        # Add OK and Cancel buttons
        button_layout = QHBoxLayout()
        ok_button = QPushButton("OK", dialog)
        ok_button.clicked.connect(dialog.accept)
        button_layout.addWidget(ok_button)

        cancel_button = QPushButton("Cancel", dialog)
        cancel_button.clicked.connect(dialog.reject)
        button_layout.addWidget(cancel_button)

        dialog_layout.addLayout(button_layout)

        # Show the dialog and handle user input
        if dialog.exec_() == QDialog.Accepted:
            prefix = prefix_field.text().strip()
            suffix = suffix_field.text().strip()

            # Rename each selected file
            for item in selected_items:
                current_name = item.text(0)
                file_path = next((path for path in self.image_paths if os.path.basename(path) == current_name), None)

                if file_path:
                    # Construct the new filename
                    directory = os.path.dirname(file_path)
                    new_name = f"{prefix}{current_name}{suffix}"
                    new_file_path = os.path.join(directory, new_name)

                    try:
                        # Rename the file
                        os.rename(file_path, new_file_path)
                        print(f"File renamed from {file_path} to {new_file_path}")

                        # Update the paths and tree view
                        self.image_paths[self.image_paths.index(file_path)] = new_file_path
                        item.setText(0, new_name)

                    except Exception as e:
                        print(f"Failed to rename {file_path}: {e}")
                        QMessageBox.critical(self, "Error", f"Failed to rename the file: {e}")

            print(f"Batch renamed {len(selected_items)} items.")


    def move_items(self):
        """Allow the user to move selected images to a different directory."""
        selected_items = self.fileTree.selectedItems()
        
        if not selected_items:
            QMessageBox.warning(self, "Warning", "No items selected for moving.")
            return

        # Open file dialog to select a new directory
        new_directory = QFileDialog.getExistingDirectory(self, "Select Destination Folder", "")
        if not new_directory:
            return  # User canceled the directory selection

        for item in selected_items:
            current_name = item.text(0)
            file_path = next((path for path in self.image_paths if os.path.basename(path) == current_name), None)

            if file_path:
                new_file_path = os.path.join(new_directory, current_name)

                try:
                    # Move the file
                    os.rename(file_path, new_file_path)
                    print(f"File moved from {file_path} to {new_file_path}")
                    
                    # Update the image paths
                    self.image_paths[self.image_paths.index(file_path)] = new_file_path
                    item.setText(0, current_name)  # Update the tree view item's text (if needed)

                except Exception as e:
                    print(f"Failed to move {file_path}: {e}")
                    QMessageBox.critical(self, "Error", f"Failed to move the file: {e}")

        # Update the tree view to reflect the moved items
        self.fileTree.clear()
        for file_path in self.image_paths:
            file_name = os.path.basename(file_path)
            item = QTreeWidgetItem([file_name])
            self.fileTree.addTopLevelItem(item)

        print(f"Moved {len(selected_items)} items.")


    def push_image_to_manager(self, item):
        """Push the selected image to the ImageManager."""
        file_name = item.text(0)
        file_path = next((path for path in self.image_paths if os.path.basename(path) == file_name), None)

        if file_path and self.image_manager:
            # Load the image into ImageManager
            image, header, bit_depth, is_mono = load_image(file_path)

            # Check for Bayer pattern or RAW image type (For FITS and RAW images)
            if file_path.lower().endswith(('.fits', '.fit')):
                # For FITS, check the header for Bayer pattern
                bayer_pattern = header.get('BAYERPAT', None) if header else None
                if bayer_pattern:
                    print(f"Bayer pattern detected in FITS image: {bayer_pattern}")
                    # Debayer the FITS image based on the Bayer pattern
                    image = self.debayer_fits(image, bayer_pattern)
                    is_mono = False  # After debayering, the image is no longer mono

            elif file_path.lower().endswith(('.cr2', '.nef', '.arw', '.dng', '.orf', '.rw2', '.pef')):
                # For RAW images, debayer directly using the raw image data
                print(f"Debayering RAW image: {file_path}")
                # We assume `header` contains the Bayer pattern info from rawpy
                bayer_pattern = header.get('BAYERPAT', None) if header else None
                if bayer_pattern:
                    # Debayer the RAW image based on the Bayer pattern
                    image = self.debayer_raw(image, bayer_pattern)
                    is_mono = False  # After debayering, the image is no longer mono
                else:
                    # If no Bayer pattern in the header, default to RGGB for debayering
                    print("No Bayer pattern found in RAW header. Defaulting to RGGB.")
                    image = self.debayer_raw(image, 'RGGB')
                    is_mono = False  # After debayering, the image is no longer mono

            # Create metadata for the image
            metadata = {
                'file_path': file_path,
                'original_header': header,
                'bit_depth': bit_depth,
                'is_mono': is_mono
            }

            # Add the debayered image to ImageManager (use the current slot)
            self.image_manager.add_image(self.image_manager.current_slot, image, metadata)
            print(f"Image {file_path} pushed to ImageManager for processing.")




    def delete_items(self):
        """Delete the selected items from the tree, the loaded images list, and the file system."""
        selected_items = self.fileTree.selectedItems()

        if not selected_items:
            QMessageBox.warning(self, "Warning", "No items selected for deletion.")
            return

        # Confirmation dialog
        reply = QMessageBox.question(
            self,
            'Confirm Deletion',
            f"Are you sure you want to permanently delete {len(selected_items)} selected images? This action is irreversible.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            for item in selected_items:
                file_name = item.text(0)
                file_path = next((path for path in self.image_paths if os.path.basename(path) == file_name), None)

                if file_path:
                    try:
                        # Remove the image from image_paths
                        if file_path in self.image_paths:
                            self.image_paths.remove(file_path)
                            print(f"Image path {file_path} removed from image_paths.")
                        else:
                            print(f"Image path {file_path} not found in image_paths.")

                        # Remove the corresponding image from loaded_images
                        matching_image_data = next((entry for entry in self.loaded_images if entry['file_path'] == file_path), None)
                        if matching_image_data:
                            self.loaded_images.remove(matching_image_data)
                            print(f"Image {file_name} removed from loaded_images.")
                        else:
                            print(f"Image {file_name} not found in loaded_images.")

                        # Delete the file from the filesystem
                        os.remove(file_path)
                        print(f"File {file_path} deleted successfully.")

                    except Exception as e:
                        print(f"Failed to delete {file_path}: {e}")
                        QMessageBox.critical(self, "Error", f"Failed to delete the image file: {e}")

            # Remove the selected items from the TreeWidget
            for item in selected_items:
                parent = item.parent()
                if parent:
                    parent.removeChild(item)
                else:
                    index = self.fileTree.indexOfTopLevelItem(item)
                    if index != -1:
                        self.fileTree.takeTopLevelItem(index)

            print(f"Deleted {len(selected_items)} items.")
            
            # Clear the preview if the deleted items include the currently displayed image
            self.preview_label.clear()
            self.preview_label.setText('No image selected.')

            self.current_image = None


    def eventFilter(self, source, event):
        """Handle mouse events for dragging."""
        if source == self.scroll_area.viewport():
            if event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                # Start dragging
                self.dragging = True
                self.last_mouse_pos = event.pos()
                return True
            elif event.type() == QEvent.MouseMove and self.dragging:
                # Handle dragging
                delta = event.pos() - self.last_mouse_pos
                self.scroll_area.horizontalScrollBar().setValue(
                    self.scroll_area.horizontalScrollBar().value() - delta.x()
                )
                self.scroll_area.verticalScrollBar().setValue(
                    self.scroll_area.verticalScrollBar().value() - delta.y()
                )
                self.last_mouse_pos = event.pos()
                return True
            elif event.type() == QEvent.MouseButtonRelease and event.button() == Qt.LeftButton:
                # Stop dragging
                self.dragging = False
                return True
        return super().eventFilter(source, event)

    def on_selection_changed(self, selected, deselected):
        """Handle the selection change event."""
        # Get the selected item from the TreeView
        selected_items = self.fileTree.selectedItems()
        if selected_items:
            item = selected_items[0]  # Get the first selected item (assuming single selection)
            self.on_item_clicked(item, 0)  # Update the preview with the selected image

    def convert_to_qimage(self, img_array):
        """Convert numpy image array to QImage."""
        img_array = (img_array * 255).astype(np.uint8)  # Ensure image is in uint8
        h, w = img_array.shape[:2]

        # Convert the image data to a byte buffer
        img_data = img_array.tobytes()  # This converts the image to a byte buffer

        if img_array.ndim == 3:  # RGB Image
            return QImage(img_data, w, h, 3 * w, QImage.Format_RGB888)
        else:  # Grayscale Image
            return QImage(img_data, w, h, w, QImage.Format_Grayscale8)


class CosmicClarityTab(QWidget):
    def __init__(self, image_manager=None):
        super().__init__()
        self.image_manager = image_manager  # Reference to the ImageManager
        self.loaded_image_path = None
        self.original_header = None
        self.bit_depth = None
        self.is_mono = False
        self.settings_file = "cosmic_clarity_folder.txt"  # Path to save the folder location
        self.zoom_factor = 1  # Zoom level
        self.drag_start_position = QPoint()  # Starting point for drag
        self.is_dragging = False  # Flag to indicate if dragging
        self.scroll_position = QPoint(0, 0)  # Initialize scroll position
        self.original_image = None  # Image before processing
        self.processed_image = None  # Most recent processed image    
        self.is_selecting_preview = False  # Initialize preview selection attribute
        self.preview_start_position = None
        self.preview_end_position = None
        self.preview_rect = None  # Stores the preview selection rectangle
        self.autostretch_enabled = False  # Track autostretch status

        self.initUI()

        self.load_cosmic_clarity_folder()

        if self.image_manager:
            # Connect to ImageManager's image_changed signal
            self.image_manager.image_changed.connect(self.on_image_changed)

    def initUI(self):
        main_layout = QHBoxLayout()

        # Left panel for controls
        left_layout = QVBoxLayout()

        

        # Load button to load an image
        self.load_button = QPushButton("Load Image")
        self.load_button.clicked.connect(self.load_image)
        left_layout.addWidget(self.load_button)

        # AutoStretch toggle button
        self.auto_stretch_button = QPushButton("AutoStretch (Off)")
        self.auto_stretch_button.setCheckable(True)
        self.auto_stretch_button.toggled.connect(self.toggle_auto_stretch)
        left_layout.addWidget(self.auto_stretch_button)

        # Radio buttons to switch between Sharpen and Denoise
        self.sharpen_radio = QRadioButton("Sharpen")
        self.denoise_radio = QRadioButton("Denoise")
        self.sharpen_radio.setChecked(True)  # Default to Sharpen
        self.sharpen_radio.toggled.connect(self.update_ui_for_mode)
        left_layout.addWidget(self.sharpen_radio)
        left_layout.addWidget(self.denoise_radio)

        # GPU Acceleration dropdown
        self.gpu_label = QLabel("Use GPU Acceleration:")
        left_layout.addWidget(self.gpu_label)
        self.gpu_dropdown = QComboBox()
        self.gpu_dropdown.addItems(["Yes", "No"])
        left_layout.addWidget(self.gpu_dropdown)

        # Add Sharpening specific controls
        self.sharpen_mode_label = QLabel("Sharpening Mode:")
        self.sharpen_mode_dropdown = QComboBox()
        self.sharpen_mode_dropdown.addItems(["Both", "Stellar Only", "Non-Stellar Only"])
        left_layout.addWidget(self.sharpen_mode_label)
        left_layout.addWidget(self.sharpen_mode_dropdown)

        # Dropdown for Sharpen Channels Separately option
        self.sharpen_channels_label = QLabel("Sharpen RGB Channels Separately:")
        self.sharpen_channels_dropdown = QComboBox()
        self.sharpen_channels_dropdown.addItems(["No", "Yes"])  # "No" means don't separate, "Yes" means separate
        left_layout.addWidget(self.sharpen_channels_label)
        left_layout.addWidget(self.sharpen_channels_dropdown)

        # Non-Stellar Sharpening PSF Slider
        self.psf_slider_label = QLabel("Non-Stellar Sharpening PSF (1-8): 3")
        self.psf_slider = QSlider(Qt.Horizontal)
        self.psf_slider.setMinimum(10)
        self.psf_slider.setMaximum(80)
        self.psf_slider.setValue(30)
        self.psf_slider.valueChanged.connect(self.update_psf_slider_label)
        left_layout.addWidget(self.psf_slider_label)
        left_layout.addWidget(self.psf_slider)

        # Stellar Amount Slider
        self.stellar_amount_label = QLabel("Stellar Sharpening Amount (0-1): 0.50")
        self.stellar_amount_slider = QSlider(Qt.Horizontal)
        self.stellar_amount_slider.setMinimum(0)
        self.stellar_amount_slider.setMaximum(100)
        self.stellar_amount_slider.setValue(50)
        self.stellar_amount_slider.valueChanged.connect(self.update_stellar_amount_label)
        left_layout.addWidget(self.stellar_amount_label)
        left_layout.addWidget(self.stellar_amount_slider)

        # Non-Stellar Amount Slider
        self.nonstellar_amount_label = QLabel("Non-Stellar Sharpening Amount (0-1): 0.50")
        self.nonstellar_amount_slider = QSlider(Qt.Horizontal)
        self.nonstellar_amount_slider.setMinimum(0)
        self.nonstellar_amount_slider.setMaximum(100)
        self.nonstellar_amount_slider.setValue(50)
        self.nonstellar_amount_slider.valueChanged.connect(self.update_nonstellar_amount_label)
        left_layout.addWidget(self.nonstellar_amount_label)
        left_layout.addWidget(self.nonstellar_amount_slider)

        # Denoise Strength Slider
        self.denoise_strength_label = QLabel("Denoise Strength (0-1): 0.50")
        self.denoise_strength_slider = QSlider(Qt.Horizontal)
        self.denoise_strength_slider.setMinimum(0)
        self.denoise_strength_slider.setMaximum(100)
        self.denoise_strength_slider.setValue(50)
        self.denoise_strength_slider.valueChanged.connect(self.update_denoise_strength_label)
        left_layout.addWidget(self.denoise_strength_label)
        left_layout.addWidget(self.denoise_strength_slider)

        # Denoise Mode dropdown
        self.denoise_mode_label = QLabel("Denoise Mode:")
        self.denoise_mode_dropdown = QComboBox()
        self.denoise_mode_dropdown.addItems(["luminance", "full"])  # 'luminance' for luminance-only, 'full' for full YCbCr denoising
        left_layout.addWidget(self.denoise_mode_label)
        left_layout.addWidget(self.denoise_mode_dropdown)

        # Execute button
        self.execute_button = QPushButton("Execute")
        self.execute_button.clicked.connect(self.run_cosmic_clarity)
        left_layout.addWidget(self.execute_button)

        # Undo and Redo buttons
        self.undo_button = QPushButton("Original")
        self.undo_button.setIcon(QApplication.style().standardIcon(QStyle.SP_ArrowLeft))
        self.undo_button.clicked.connect(self.undo)
        self.undo_button.setEnabled(False)  # Disabled initially
        left_layout.addWidget(self.undo_button)

        self.redo_button = QPushButton("Current")
        self.redo_button.setIcon(QApplication.style().standardIcon(QStyle.SP_ArrowRight))
        self.redo_button.clicked.connect(self.redo)
        self.redo_button.setEnabled(False)  # Disabled initially
        left_layout.addWidget(self.redo_button)        

        # Save button to save the processed image
        self.save_button = QPushButton("Save Image")
        self.save_button.clicked.connect(self.save_processed_image_to_disk)
        left_layout.addWidget(self.save_button)  

        # Spacer to push the wrench button to the bottom
        left_layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

        # Cosmic Clarity folder path label
        self.cosmic_clarity_folder_label = QLabel("No folder selected")
        left_layout.addWidget(self.cosmic_clarity_folder_label)

        # Wrench button to select Cosmic Clarity folder
        self.wrench_button = QPushButton()

        # Set the path for the wrench icon
        if hasattr(sys, '_MEIPASS'):
            wrench_path = os.path.join(sys._MEIPASS, "wrench_icon.png")
        else:
            wrench_path = "wrench_icon.png"

        self.wrench_button.setIcon(QIcon(wrench_path))  # Set the wrench icon with the dynamic path
        self.wrench_button.clicked.connect(self.select_cosmic_clarity_folder)
        left_layout.addWidget(self.wrench_button)  

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

        # Right panel for image preview with zoom controls
        right_layout = QVBoxLayout()

        # Zoom controls

        zoom_layout = QHBoxLayout()
        zoom_in_button = QPushButton("Zoom In")
        zoom_in_button.clicked.connect(self.zoom_in)
        zoom_layout.addWidget(zoom_in_button)

        zoom_out_button = QPushButton("Zoom Out")
        zoom_out_button.clicked.connect(self.zoom_out)
        zoom_layout.addWidget(zoom_out_button)

        # **Add "Fit to Preview" Button**
        self.fitToPreviewButton = QPushButton("Fit to Preview")
        self.fitToPreviewButton.clicked.connect(self.fit_to_preview)
        zoom_layout.addWidget(self.fitToPreviewButton)

        right_layout.addLayout(zoom_layout)

        # Scroll area for image preview with click-and-drag functionality
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.scroll_area.setWidget(self.image_label)
        right_layout.addWidget(self.scroll_area)

        # Button to open the preview area selection dialog
        self.select_preview_button = QPushButton("Select Preview Area")
        self.select_preview_button.clicked.connect(self.open_preview_dialog)
        right_layout.addWidget(self.select_preview_button)        

        left_widget = QWidget()
        left_widget.setLayout(left_layout)
        left_widget.setFixedWidth(400)

        # Add left and right layouts to the main layout
        main_layout.addWidget(left_widget)
        main_layout.addLayout(right_layout)

        self.setLayout(main_layout)
        self.update_ui_for_mode()

    def update_psf_slider_label(self):
        """Update the label text to display the current value of the PSF slider as a non-integer."""
        psf_value = self.psf_slider.value() / 10  # Convert to a float in the range 1.0 - 8.0
        self.psf_slider_label.setText(f"Non-Stellar Sharpening PSF (1.0-8.0): {psf_value:.1f}")

    def update_stellar_amount_label(self):
        self.stellar_amount_label.setText(f"Stellar Sharpening Amount (0-1): {self.stellar_amount_slider.value() / 100:.2f}")

    def update_nonstellar_amount_label(self):
        self.nonstellar_amount_label.setText(f"Non-Stellar Sharpening Amount (0-1): {self.nonstellar_amount_slider.value() / 100:.2f}")

    def update_denoise_strength_label(self):
        self.denoise_strength_label.setText(f"Denoise Strength (0-1): {self.denoise_strength_slider.value() / 100:.2f}")

    def mousePressEvent(self, event):
        """Handle the start of the drag action or selection of a preview area."""
        if event.button() == Qt.LeftButton:
            self.is_dragging = True
            self.drag_start_position = event.pos()              
                

    def mouseMoveEvent(self, event):
        """Handle dragging or adjusting the preview selection area."""
        if self.is_dragging:
            # Handle image panning
            delta = event.pos() - self.drag_start_position
            self.scroll_area.horizontalScrollBar().setValue(self.scroll_area.horizontalScrollBar().value() - delta.x())
            self.scroll_area.verticalScrollBar().setValue(self.scroll_area.verticalScrollBar().value() - delta.y())
            self.drag_start_position = event.pos()


    def mouseReleaseEvent(self, event):
        """End the drag action or finalize the preview selection area."""
        if event.button() == Qt.LeftButton:
            if self.is_dragging:
                self.is_dragging = False


    def open_preview_dialog(self):
        """Open a preview dialog to select a 640x480 area of the image at 100% scale."""
        if self.image is not None:
            # Pass the 32-bit numpy image directly to maintain bit depth
            self.preview_dialog = PreviewDialog(self.image, parent_tab=self, is_mono=self.is_mono)
            self.preview_dialog.show()
        else:
            print("No image loaded. Please load an image first.")



    def convert_numpy_to_qimage(self, np_img):
        """Convert a numpy array to QImage."""
        # Ensure image is in 8-bit format for QImage compatibility
        if np_img.dtype == np.float32:
            np_img = (np_img * 255).astype(np.uint8)  # Convert normalized float32 to uint8 [0, 255]
        
        if np_img.dtype == np.uint8:
            if len(np_img.shape) == 2:
                # Grayscale image
                height, width = np_img.shape
                bytes_per_line = width
                return QImage(np_img.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
            elif len(np_img.shape) == 3 and np_img.shape[2] == 3:
                # RGB image
                height, width, channels = np_img.shape
                bytes_per_line = 3 * width
                return QImage(np_img.data, width, height, bytes_per_line, QImage.Format_RGB888)
        else:
            print("Image format not supported for conversion to QImage.")
            return None



    def select_cosmic_clarity_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Cosmic Clarity Folder")
        if folder:
            self.cosmic_clarity_folder = folder
            self.save_cosmic_clarity_folder(folder)
            self.cosmic_clarity_folder_label.setText(f"Folder: {folder}")
            print(f"Selected Cosmic Clarity folder: {folder}")

    def zoom_in(self):
        """Zoom in on the image and update the display."""
        self.zoom_factor *= 1.2
        self.apply_zoom()  # Use apply_zoom to handle zoom correctly

    def zoom_out(self):
        """Zoom out on the image and update the display."""
        self.zoom_factor /= 1.2
        self.apply_zoom()  # Use apply_zoom to handle zoom correctly

    def fit_to_preview(self):
        """Adjust the zoom factor so that the image's width fits within the preview area's width."""
        if self.image is not None:
            # Get the width of the scroll area's viewport (preview area)
            preview_width = self.scroll_area.viewport().width()
            
            # Get the original image width from the numpy array
            # Assuming self.image has shape (height, width, channels) or (height, width) for grayscale
            if self.image.ndim == 3:
                image_width = self.image.shape[1]
            elif self.image.ndim == 2:
                image_width = self.image.shape[1]
            else:
                print("Unexpected image dimensions!")
                self.statusLabel.setText("Cannot fit image to preview due to unexpected dimensions.")
                return
            
            # Calculate the required zoom factor to fit the image's width into the preview area
            new_zoom_factor = preview_width / image_width
            
            # Update the zoom factor without enforcing any limits
            self.zoom_factor = new_zoom_factor
            
            # Apply the new zoom factor to update the display
            self.apply_zoom()
            
            # Update the status label to reflect the new zoom level
        else:
            print("No image loaded. Cannot fit to preview.")

      

    def apply_zoom(self):
        """Apply the current zoom level to the image."""
        self.update_image_display()  # Call without extra arguments; it will calculate dimensions based on zoom factor
    

    def undo(self):
        """Undo to the original image without changing zoom and scroll position."""
        self.image_manager.undo()
        self.redo_button.setEnabled(True)
        self.undo_button.setEnabled(False)

    def redo(self):
        """Redo to the processed image without changing zoom and scroll position."""
        self.image_manager.redo()
        self.redo_button.setEnabled(False)
        self.undo_button.setEnabled(True)


    def restore_image(self, image_array):
        """Display a given image array, preserving the current zoom level and scroll position."""
        # Save the current zoom level and scroll position
        current_zoom = self.zoom_factor
        current_scroll_position = (
            self.scroll_area.horizontalScrollBar().value(),
            self.scroll_area.verticalScrollBar().value()
        )

        # Display the image
        self.show_image(image_array)

        # Restore the zoom level and scroll position
        self.zoom_factor = current_zoom
        self.update_image_display()  # Refresh display with the preserved zoom level

        self.scroll_area.horizontalScrollBar().setValue(current_scroll_position[0])
        self.scroll_area.verticalScrollBar().setValue(current_scroll_position[1])


    def save_cosmic_clarity_folder(self, folder):
        """Save the Cosmic Clarity folder path to a text file."""
        with open(self.settings_file, 'w') as file:
            file.write(folder)
        print(f"Saved Cosmic Clarity folder to {self.settings_file}")

    def load_cosmic_clarity_folder(self):
        """Load the saved Cosmic Clarity folder path from a text file."""
        if os.path.exists(self.settings_file):
            with open(self.settings_file, 'r') as file:
                folder = file.read().strip()
                if folder:
                    self.cosmic_clarity_folder = folder
                    self.cosmic_clarity_folder_label.setText(f"Folder: {folder}")
                    print(f"Loaded Cosmic Clarity folder from {self.settings_file}: {folder}")
                else:
                    print("Cosmic Clarity folder path in file is empty.")
        else:
            print("No saved Cosmic Clarity folder found.")

    def on_image_changed(self, slot, image, metadata):
        """
        Slot to handle image changes from ImageManager.
        Updates the display if the current slot is affected.
        """
        if slot == self.image_manager.current_slot:
            self.loaded_image_path = metadata.get('file_path', None)  # Assuming metadata contains file path
            self.original_header = metadata.get('original_header', None)
            self.bit_depth = metadata.get('bit_depth', None)
            self.is_mono = metadata.get('is_mono', False)

            # Ensure image is in numpy array format
            if not isinstance(image, np.ndarray):
                image = np.array(image)  # Convert to numpy array if not already

            self.image = image

            # Handle mono images by checking the dimensions before passing to display functions
            if self.is_mono:
                # For mono images, ensure the image is 2D (height, width) rather than (height, width, 1)
                if len(image.shape) == 3 and image.shape[2] == 1:
                    image = np.squeeze(image, axis=2)  # Remove singleton channel
                # If already 2D (height, width), we keep it as is
            else:
                # For color images, ensure the image has 3 channels (RGB)
                if len(image.shape) == 2:
                    raise ValueError("Unexpected image format! Image must be either RGB or Grayscale.")

            # Show the image using the show_image method
            self.show_image(image)

            # Update the image display (it will account for zoom and other parameters)
            self.update_image_display()
            
            print(f"CosmicClarityTab: Image updated from ImageManager slot {slot}.")




    def load_image(self):
        """Load an image and set it as the current and original image."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Select Image", 
            "", 
            "Image Files (*.png *.jpg *.tif *.tiff *.fits *.fit *.jpeg *.xisf)"
        )
        if file_path:
            print(f"Loading file: {file_path}")

            # Load the image and store it as the original image
            image, original_header, bit_depth, is_mono = load_image(file_path)
            
            # Check if the image was loaded successfully
            if image is None:
                print("Error: Failed to load the image data.")
                QMessageBox.critical(self, "Error", "Failed to load the image. Please try a different file.")
                return

            print(f"Image loaded successfully. Shape: {image.shape}, Dtype: {image.dtype}")

            # Make a copy of the original image for reference
            try:
                self.original_image = image.copy()
                print("Original image copied successfully.")
            except Exception as e:
                print(f"Error copying original image: {e}")
                QMessageBox.critical(self, "Error", "Failed to copy the original image.")
                return

            # Clear any existing processed image
            self.processed_image = None

            # Attempt to display the loaded image in the preview
            try:
                self.show_image(image)  # Ensure this function can handle 32-bit float images
                print("Image displayed successfully.")
            except Exception as e:
                print(f"Error displaying image: {e}")
                QMessageBox.critical(self, "Error", "Failed to display the image.")
                return

            # Enable or disable buttons as necessary
            self.undo_button.setEnabled(False)
            self.redo_button.setEnabled(False)

            # Center scrollbars after a short delay
            try:
                QTimer.singleShot(50, self.center_scrollbars)  # Delay of 50 ms for centering scrollbars
                print("Scrollbars centered.")
            except Exception as e:
                print(f"Error centering scrollbars: {e}")

            # Update the display after another short delay to ensure scrollbars are centered first
            try:
                QTimer.singleShot(100, self.update_image_display)  # Delay of 100 ms for display update
                print("Image display updated.")
            except Exception as e:
                print(f"Error updating image display: {e}")

            # Update ImageManager with the new image
            metadata = {
                'file_path': file_path,
                'original_header': original_header,
                'bit_depth': bit_depth,
                'is_mono': is_mono
            }
            self.image_manager.add_image(slot=self.image_manager.current_slot, image=image, metadata=metadata)

        else:
            print("No file selected.")



    def center_scrollbars(self):
        """Centers the scrollbars to start in the middle of the image."""
        h_scroll = self.scroll_area.horizontalScrollBar()
        v_scroll = self.scroll_area.verticalScrollBar()
        h_scroll.setValue((h_scroll.maximum() + h_scroll.minimum()) // 2)
        v_scroll.setValue((v_scroll.maximum() + v_scroll.minimum()) // 2)

    def show_image(self, image=None):
        """Display the loaded image or a specified image, preserving zoom and scroll position."""

        if image is None:
            image = self.image  # Use the current image from ImageManager


        if image is None:
            print("[ERROR] No image to display.")
            QMessageBox.warning(self, "No Image", "No image data available to display.")
            return False  # Indicate failure

        if not isinstance(image, np.ndarray):
            print(f"[ERROR] Invalid image data. Expected a NumPy array, got {type(image)}.")
            QMessageBox.critical(self, "Error", "Invalid image data. Cannot display the image.")
            return False  # Indicate failure



        # Save the current scroll position if it exists
        current_scroll_position = (
            self.scroll_area.horizontalScrollBar().value(),
            self.scroll_area.verticalScrollBar().value()
        )


        # Stretch and display the image
        display_image = image.copy()
        target_median = 0.25

        # Determine if the image is mono based on dimensions
        is_mono = display_image.ndim == 2 or (display_image.ndim == 3 and display_image.shape[2] == 1)


        if self.auto_stretch_button.isChecked():

            if is_mono:
                if display_image.ndim == 2:
                    stretched_mono = stretch_mono_image(display_image, target_median=0.25)

                else:
                    stretched_mono = stretch_mono_image(display_image[:, :, 0], target_median=0.25)

                # Convert to RGB by stacking
                display_image = np.stack([stretched_mono] * 3, axis=-1)

            else:
                display_image = stretch_color_image(display_image, target_median=0.25, linked=False)

        else:
            print("AutoStretch is disabled.")

        # Convert to QImage for display
        try:
            display_image_uint8 = (display_image * 255).astype(np.uint8)

        except Exception as e:
            print(f"[ERROR] Error converting image to uint8: {e}")
            QMessageBox.critical(self, "Error", f"Error processing image for display:\n{e}")
            return False  # Indicate failure

        if display_image_uint8.ndim == 3 and display_image_uint8.shape[2] == 3:  # RGB image
            height, width, _ = display_image_uint8.shape
            bytes_per_line = 3 * width
            qimage = QImage(
                display_image_uint8.data, 
                width, 
                height, 
                bytes_per_line, 
                QImage.Format_RGB888
            )

        elif display_image_uint8.ndim == 2:  # Grayscale image
            height, width = display_image_uint8.shape
            bytes_per_line = width
            qimage = QImage(
                display_image_uint8.data, 
                width, 
                height, 
                bytes_per_line, 
                QImage.Format_Grayscale8
            )

        else:
            print("[ERROR] Unexpected image format! Image must be either RGB or Grayscale.")
            QMessageBox.critical(self, "Error", "Unexpected image format! Image must be either RGB or Grayscale.")
            return False  # Indicate failure

        # Create QPixmap from QImage
        pixmap = QPixmap.fromImage(qimage)


        if pixmap.isNull():
            print("[ERROR] Failed to convert QImage to QPixmap.")
            QMessageBox.critical(self, "Error", "Failed to display the image.")
            return False  # Indicate failure

        # Set pixmap without applying additional scaling (keep the original zoom level)
        self.image_label.setPixmap(pixmap)


        # Force the label to update
        self.image_label.repaint()
        self.image_label.update()


        # Restore the previous scroll position
        self.scroll_area.horizontalScrollBar().setValue(current_scroll_position[0])
        self.scroll_area.verticalScrollBar().setValue(current_scroll_position[1])

        return True  # Indicate success




    def update_image_display(self, display_width=None, display_height=None):
        """Update the displayed image according to the current zoom level and autostretch setting."""
        if self.image is None:
            print("No image to display.")
            return

        # Get the current center point of the visible area
        current_center_x = self.scroll_area.horizontalScrollBar().value() + (self.scroll_area.viewport().width() / 2)
        current_center_y = self.scroll_area.verticalScrollBar().value() + (self.scroll_area.viewport().height() / 2)

        # Apply autostretch if enabled
        display_image = self.image.copy()
        if self.auto_stretch_button.isChecked():
            target_median = 0.25
            if self.is_mono:
                print("Autostretch enabled for mono image.")
                stretched_mono = stretch_mono_image(display_image if display_image.ndim == 2 else display_image[:, :, 0], target_median)
                display_image = np.stack([stretched_mono] * 3, axis=-1)  # Convert mono to RGB for display
            else:
                print("Autostretch enabled for color image.")
                display_image = stretch_color_image(display_image, target_median, linked=False)

        # Convert to QImage for display (Ensure the data is in 8-bit for QImage)
        print(f"Image dtype before conversion: {display_image.dtype}")
        display_image_uint8 = (display_image * 255).astype(np.uint8)

        # Debugging the shape of the image
        print(f"Image shape after conversion to uint8: {display_image_uint8.shape}")

        # Handle mono and RGB images differently
        if display_image_uint8.ndim == 3 and display_image_uint8.shape[2] == 3:
            print("Detected RGB image.")
            # RGB image
            height, width, _ = display_image_uint8.shape
            bytes_per_line = 3 * width
            qimage = QImage(display_image_uint8.tobytes(), width, height, bytes_per_line, QImage.Format_RGB888)
        elif display_image_uint8.ndim == 2:  # Grayscale image
            print("Detected Grayscale image.")
            height, width = display_image_uint8.shape
            bytes_per_line = width
            qimage = QImage(display_image_uint8.tobytes(), width, height, bytes_per_line, QImage.Format_Grayscale8)
        else:
            print("Unexpected image format!")
            print(f"Image dimensions: {display_image_uint8.ndim}")
            print(f"Image shape: {display_image_uint8.shape}")
            return

        # Calculate the new dimensions based on the zoom factor
        if display_width is None or display_height is None:
            display_width = int(width * self.zoom_factor)
            display_height = int(height * self.zoom_factor)

        # Scale QPixmap and set it on the image label
        pixmap = QPixmap.fromImage(qimage).scaled(display_width, display_height, Qt.KeepAspectRatio)
        self.image_label.setPixmap(pixmap)

        # Calculate the new center point after zooming
        new_center_x = current_center_x * self.zoom_factor
        new_center_y = current_center_y * self.zoom_factor

        # Adjust scroll bars to keep the view centered on the same area
        self.scroll_area.horizontalScrollBar().setValue(int(new_center_x - self.scroll_area.viewport().width() / 2))
        self.scroll_area.verticalScrollBar().setValue(int(new_center_y - self.scroll_area.viewport().height() / 2))


    def store_processed_image(self, processed_image):
        """Store the processed image and update the ImageManager."""
        if processed_image is not None:
            # Store a copy of the processed image
            self.processed_image = processed_image.copy()
            
            # Enable the undo button and reset the redo button
            self.undo_button.setEnabled(True)
            self.redo_button.setEnabled(False)  # Reset redo button for new process
            
            # Prepare metadata for the ImageManager
            metadata = {
                'file_path': self.loaded_image_path,      # Ensure this is correctly set elsewhere
                'original_header': self.original_header,  # Ensure this is correctly set elsewhere
                'bit_depth': self.bit_depth,              # Ensure this is correctly set elsewhere
                'is_mono': self.is_mono                   # Ensure this is correctly set elsewhere
            }
            
            # Update the ImageManager with the new image and metadata
            if self.image_manager:
                try:
                    self.image_manager.update_image(updated_image=self.processed_image, metadata=metadata)
                    print("FullCurvesTab: Processed image stored in ImageManager.")
                except Exception as e:
                    # Handle potential errors during the update
                    QMessageBox.critical(self, "Error", f"Failed to update ImageManager:\n{e}")
                    print(f"Error updating ImageManager: {e}")
            else:
                print("ImageManager is not initialized.")
                QMessageBox.warning(self, "Warning", "ImageManager is not initialized. Cannot store the processed image.")
        else:
            print("No processed image available to store.")
            QMessageBox.warning(self, "Warning", "No processed image available to store.")


    def toggle_auto_stretch(self, checked):
        """Toggle autostretch and apply it to the current image display."""
        self.autostretch_enabled = checked
        self.auto_stretch_button.setText("AutoStretch (On)" if checked else "AutoStretch (Off)")
        self.update_image_display()  # Redraw with autostretch if enabled

    def save_input_image(self, file_path):
        """Save the current image to the specified path in TIF format."""
        if self.image is not None:
            try:
                from tifffile import imwrite
                # Force saving as `.tif` format
                if not file_path.endswith(".tif"):
                    file_path += ".tif"
                imwrite(file_path, self.image.astype(np.float32))
                print(f"Image saved as TIFF to {file_path}")  # Debug print
            except Exception as e:
                print(f"Error saving input image: {e}")
                QMessageBox.critical(self, "Error", f"Failed to save input image:\n{e}")
        else:
            QMessageBox.warning(self, "Warning", "No image to save.")




    def update_ui_for_mode(self):
        # Show/hide sharpening controls based on mode
        if self.sharpen_radio.isChecked():
            self.sharpen_mode_label.show()
            self.sharpen_mode_dropdown.show()
            self.psf_slider_label.show()
            self.psf_slider.show()
            self.stellar_amount_label.show()
            self.stellar_amount_slider.show()
            self.nonstellar_amount_label.show()
            self.nonstellar_amount_slider.show()
            self.sharpen_channels_label.show()  # Show the label for RGB sharpening
            self.sharpen_channels_dropdown.show()  # Show the dropdown for RGB sharpening
            # Hide denoise controls
            self.denoise_strength_label.hide()
            self.denoise_strength_slider.hide()
            self.denoise_mode_label.hide()
            self.denoise_mode_dropdown.hide()
        else:
            # Show denoise controls
            self.denoise_strength_label.show()
            self.denoise_strength_slider.show()
            self.denoise_mode_label.show()
            self.denoise_mode_dropdown.show()
            self.sharpen_mode_label.hide()
            self.sharpen_mode_dropdown.hide()
            self.psf_slider_label.hide()
            self.psf_slider.hide()
            self.stellar_amount_label.hide()
            self.stellar_amount_slider.hide()
            self.nonstellar_amount_label.hide()
            self.nonstellar_amount_slider.hide()
            self.sharpen_channels_label.hide()  # Hide the label for RGB sharpening
            self.sharpen_channels_dropdown.hide()  # Hide the dropdown for RGB sharpening

    def get_psf_value(self):
        """Convert the slider value to a float in the range 1.0 - 8.0."""
        return self.psf_slider.value() / 10.0
    
    def run_cosmic_clarity(self, input_file_path=None):
        """Run Cosmic Clarity with the current parameters."""
        psf_value = self.get_psf_value()
        if not self.cosmic_clarity_folder:
            QMessageBox.warning(self, "Warning", "Please select the Cosmic Clarity folder.")
            return
        if self.image is None:  # Ensure an image is currently displayed
            QMessageBox.warning(self, "Warning", "Please load an image first.")
            return

        # Check the current autostretch state
        was_autostretch_enabled = self.auto_stretch_button.isChecked()

        # Disable autostretch if it was enabled
        if was_autostretch_enabled:
            self.auto_stretch_button.setChecked(False)

        # Determine mode from the radio buttons
        if self.sharpen_radio.isChecked():
            mode = "sharpen"
            output_suffix = "_sharpened"
        else:
            mode = "denoise"
            output_suffix = "_denoised"

        # Determine the correct executable name based on platform and mode
        if os.name == 'nt':
            # Windows
            if mode == "sharpen":
                exe_name = "SetiAstroCosmicClarity.exe"
            else:
                exe_name = "SetiAstroCosmicClarity_denoise.exe"
        else:
            # macOS or Linux (posix)
            if sys.platform == "darwin":
                # macOS
                if mode == "sharpen":
                    exe_name = "SetiAstroCosmicClaritymac"
                else:
                    exe_name = "SetiAstroCosmicClarity_denoisemac"
            else:
                # Linux
                if mode == "sharpen":
                    exe_name = "SetiAstroCosmicClarity"
                else:
                    exe_name = "SetiAstroCosmicClarity_denoise"

        # Define paths for input and output
        input_folder = os.path.join(self.cosmic_clarity_folder, "input")
        output_folder = os.path.join(self.cosmic_clarity_folder, "output")

        # Construct the base filename from the loaded image path
        base_filename = os.path.splitext(os.path.basename(self.loaded_image_path))[0]
        print(f"Base filename before saving: {base_filename}")  # Debug print

        # Save the current previewed image directly to the input folder
        input_file_path = os.path.join(input_folder, f"{base_filename}.tif")
        self.save_input_image(input_file_path)  # Save as `.tif`
        self.current_input_file_path = input_file_path

        # Construct the expected output file glob
        output_file_glob = os.path.join(output_folder, f"{base_filename}{output_suffix}.tif")
        print(f"Waiting for output file matching: {output_file_glob}")  # Debug print

        # Check if the executable exists
        exe_path = os.path.join(self.cosmic_clarity_folder, exe_name)
        if not os.path.exists(exe_path):
            QMessageBox.critical(self, "Error", f"Executable not found: {exe_path}. Please use the wrench icon to select the correct folder.")
            return

        cmd = self.build_command_args(exe_name, mode)
        exe_path = cmd[0]
        args = cmd[1:]  # Separate the executable from its arguments
        print(f"Running command: {exe_path} {' '.join(args)}")  # Debug print

        # Use QProcess instead of subprocess
        self.process_q = QProcess(self)
        self.process_q.setProcessChannelMode(QProcess.MergedChannels)  # Combine stdout/stderr

        # Connect signals
        self.process_q.readyReadStandardOutput.connect(self.qprocess_output)
        self.process_q.finished.connect(self.qprocess_finished)

        # Start the process
        self.process_q.setProgram(exe_path)
        self.process_q.setArguments(args)
        self.process_q.start()

        if not self.process_q.waitForStarted(3000):
            QMessageBox.critical(self, "Error", "Failed to start the Cosmic Clarity process.")
            return

        # Set up file waiting worker and wait dialog as before
        self.wait_thread = WaitForFileWorker(output_file_glob, timeout=3000)
        self.wait_thread.fileFound.connect(self.on_file_found)
        self.wait_thread.error.connect(self.on_file_error)
        self.wait_thread.cancelled.connect(self.on_file_cancelled)

        self.wait_dialog = WaitDialog(self)
        self.wait_dialog.cancelled.connect(self.on_wait_cancelled)
        self.wait_dialog.setWindowModality(Qt.NonModal)
        self.wait_dialog.show()

        self.wait_thread.start()

        # Once the dialog is closed (either by file found, error, or cancellation), restore autostretch if needed
        if was_autostretch_enabled:
            self.auto_stretch_button.setChecked(True)



    ########################################
    # Below are the new helper slots (methods) to handle signals from worker and dialog.
    ########################################

    def qprocess_output(self):
        if not hasattr(self, 'process_q') or self.process_q is None:
            return
        output = self.process_q.readAllStandardOutput().data().decode("utf-8", errors="replace")
        for line in output.splitlines():
            line = line.strip()
            if not line:
                continue

            if line.startswith("Progress:"):
                # Extract the percentage and update the progress bar
                parts = line.split()
                percentage_str = parts[1].replace("%", "")
                try:
                    percentage = float(percentage_str)
                    self.wait_dialog.progress_bar.setValue(int(percentage))
                except ValueError:
                    pass
            else:
                # Append all other lines to the text box
                self.wait_dialog.append_output(line)




    def qprocess_finished(self, exitCode, exitStatus):
        """Slot called when the QProcess finishes."""
        pass  # Handle cleanup logic if needed

    def read_process_output(self):
        """Read output from the process and display it in the wait_dialog's text edit."""
        if self.process is None:
            return

        # Read all available lines from stdout
        while True:
            line = self.process.stdout.readline()
            if not line:
                break
            line = line.strip()
            if line:
                # Append the line to the wait_dialog's output text
                self.wait_dialog.append_output(line)

        # Check if process has finished
        if self.process.poll() is not None:
            # Process ended
            self.output_timer.stop()
            # You can handle any cleanup here if needed

    def on_file_found(self, output_file_path):
        print(f"File found: {output_file_path}")
        self.wait_dialog.close()
        self.wait_thread = None

        if getattr(self, 'is_cropped_mode', False):

            # Cropped image logic
            processed_image, _, _, _ = load_image(output_file_path)
            if processed_image is None:
                print(f"[ERROR] Failed to load cropped image from {output_file_path}")
                QMessageBox.critical(self, "Error", f"Failed to load cropped image from {output_file_path}.")
                return


            # Apply autostretch if requested
            if getattr(self, 'cropped_apply_autostretch', False):

                if self.is_mono:
                    stretched_mono = stretch_mono_image(processed_image[:, :, 0], target_median=0.25)

                    processed_image = np.stack([stretched_mono] * 3, axis=-1)

                else:
                    processed_image = stretch_color_image(processed_image, target_median=0.25, linked=False)


            # Update the preview dialog
            try:
                self.preview_dialog.display_qimage(processed_image)

            except Exception as e:
                print(f"[ERROR] Failed to update preview dialog: {e}")
                QMessageBox.critical(self, "Error", f"Failed to update preview dialog:\n{e}")
                return

            # Cleanup with known paths
            input_file_path = os.path.join(self.cosmic_clarity_folder, "input", "cropped_preview_image.tiff")
            self.cleanup_files(input_file_path, output_file_path)


            # Reset cropped mode
            self.is_cropped_mode = False

        else:

            # Normal mode logic
            processed_image_path = output_file_path
            self.loaded_image_path = processed_image_path


            # Attempt to load the image with retries
            processed_image, original_header, bit_depth, is_mono = self.load_image_with_retry(processed_image_path)
            if processed_image is None:
                QMessageBox.critical(self, "Error", f"Failed to load image from {processed_image_path} after multiple attempts.")
                print(f"[ERROR] Failed to load image from {processed_image_path} after multiple attempts.")
                return


            # Show the processed image by passing the image data
            try:
                self.show_image(processed_image)

            except Exception as e:
                print(f"[ERROR] Exception occurred while showing image: {e}")
                QMessageBox.critical(self, "Error", f"Exception occurred while showing image:\n{e}")
                return

            # Store the image in memory
            try:
                self.store_processed_image(processed_image)

            except Exception as e:
                print(f"[ERROR] Failed to store processed image: {e}")
                QMessageBox.critical(self, "Error", f"Failed to store processed image:\n{e}")
                return

            # Use the stored input file path
            input_file_path = self.current_input_file_path


            # Cleanup input and output files
            self.cleanup_files(input_file_path, processed_image_path)


            # Update the image display
            try:
                self.update_image_display()

            except Exception as e:
                print(f"[ERROR] Failed to update image display: {e}")
                QMessageBox.critical(self, "Error", f"Failed to update image display:\n{e}")


    def on_file_error(self, msg):
        # File not found in time
        self.wait_dialog.close()
        self.wait_thread = None
        QMessageBox.critical(self, "Error", msg)


    def on_file_cancelled(self):
        # The worker was stopped before finding a file
        self.wait_dialog.close()
        self.wait_thread = None
        QMessageBox.information(self, "Cancelled", "File waiting was cancelled.")


    def on_wait_cancelled(self):
        # User clicked cancel in the wait dialog
        if self.wait_thread and self.wait_thread.isRunning():
            self.wait_thread.stop()

        # If we have a QProcess reference, terminate it
        if hasattr(self, 'process_q') and self.process_q is not None:
            self.process_q.kill()  # or self.process_q.terminate()

        QMessageBox.information(self, "Cancelled", "Operation was cancelled by the user.")




    def run_cosmic_clarity_on_cropped(self, cropped_image, apply_autostretch=False):
        """Run Cosmic Clarity on a cropped image, with an option to autostretch upon receipt."""
        psf_value = self.get_psf_value()
        if not self.cosmic_clarity_folder:
            QMessageBox.warning(self, "Warning", "Please select the Cosmic Clarity folder.")
            return
        if cropped_image is None:  # Ensure a cropped image is provided
            QMessageBox.warning(self, "Warning", "No cropped image provided.")
            return

        # Convert the cropped image to 32-bit floating point format
        cropped_image_32bit = cropped_image.astype(np.float32) / np.max(cropped_image)  # Normalize if needed

        # Determine mode and suffix
        if self.sharpen_radio.isChecked():
            mode = "sharpen"
            output_suffix = "_sharpened"
        else:
            mode = "denoise"
            output_suffix = "_denoised"

        # Determine the correct executable name based on platform and mode
        if os.name == 'nt':
            # Windows
            if mode == "sharpen":
                exe_name = "SetiAstroCosmicClarity.exe"
            else:
                exe_name = "SetiAstroCosmicClarity_denoise.exe"
        else:
            # macOS or Linux (posix)
            if sys.platform == "darwin":
                # macOS
                if mode == "sharpen":
                    exe_name = "SetiAstroCosmicClaritymac"
                else:
                    exe_name = "SetiAstroCosmicClarity_denoisemac"
            else:
                # Linux
                if mode == "sharpen":
                    exe_name = "SetiAstroCosmicClarity"
                else:
                    exe_name = "SetiAstroCosmicClarity_denoise"

        # Define paths for input and output
        input_folder = os.path.join(self.cosmic_clarity_folder, "input")
        output_folder = os.path.join(self.cosmic_clarity_folder, "output")
        input_file_path = os.path.join(input_folder, "cropped_preview_image.tiff")

        # Save the 32-bit floating-point cropped image to the input folder
        save_image(cropped_image_32bit, input_file_path, "tiff", "32-bit floating point", self.original_header, self.is_mono)

        # Build command args (no batch script)
        cmd = self.build_command_args(exe_name, mode)

        # Set cropped mode and store parameters needed after file is found
        self.is_cropped_mode = True
        self.cropped_apply_autostretch = apply_autostretch
        self.cropped_output_suffix = output_suffix

        # Use QProcess (already defined in run_cosmic_clarity)
        self.process_q = QProcess(self)
        self.process_q.setProcessChannelMode(QProcess.MergedChannels)
        self.process_q.readyReadStandardOutput.connect(self.qprocess_output)
        self.process_q.finished.connect(self.qprocess_finished)

        exe_path = cmd[0]
        args = cmd[1:]
        self.process_q.setProgram(exe_path)
        self.process_q.setArguments(args)
        self.process_q.start()

        if not self.process_q.waitForStarted(3000):
            QMessageBox.critical(self, "Error", "Failed to start the Cosmic Clarity process.")
            return

        # Set up wait thread for cropped file
        output_file_glob = os.path.join(output_folder, "cropped_preview_image" + output_suffix + ".*")
        self.wait_thread = WaitForFileWorker(output_file_glob, timeout=1800)
        self.wait_thread.fileFound.connect(self.on_file_found)
        self.wait_thread.error.connect(self.on_file_error)
        self.wait_thread.cancelled.connect(self.on_file_cancelled)

        # Use the same WaitDialog
        self.wait_dialog = WaitDialog(self)
        self.wait_dialog.cancelled.connect(self.on_wait_cancelled)
        self.wait_dialog.setWindowModality(Qt.NonModal)
        self.wait_dialog.show()

        self.wait_thread.start()
        
    def build_command_args(self, exe_name, mode):
        """Build the command line arguments for Cosmic Clarity without using a batch file."""
        # exe_name is now fully resolved (including .exe on Windows if needed)
        exe_path = os.path.join(self.cosmic_clarity_folder, exe_name)
        cmd = [exe_path]

        # Add sharpening or denoising arguments
        if mode == "sharpen":
            psf_value = self.get_psf_value()
            cmd += [
                "--sharpening_mode", self.sharpen_mode_dropdown.currentText(),
                "--stellar_amount", f"{self.stellar_amount_slider.value() / 100:.2f}",
                "--nonstellar_strength", f"{psf_value:.1f}",
                "--nonstellar_amount", f"{self.nonstellar_amount_slider.value() / 100:.2f}"
            ]
            if self.sharpen_channels_dropdown.currentText() == "Yes":
                cmd.append("--sharpen_channels_separately")
        elif mode == "denoise":
            cmd += [
                "--denoise_strength", f"{self.denoise_strength_slider.value() / 100:.2f}",
                "--denoise_mode", self.denoise_mode_dropdown.currentText()
            ]

        # GPU option
        if self.gpu_dropdown.currentText() == "No":
            cmd.append("--disable_gpu")

        return cmd

    def save_processed_image(self):
        """Save the current displayed image as the processed image."""
        self.processed_image = self.image.copy()
        self.undo_button.setEnabled(True)
        self.redo_button.setEnabled(False)  # Reset redo

    def save_processed_image_to_disk(self):
        """Save the processed image to disk, using the correct format, bit depth, and header information."""
        if self.processed_image is None:
            QMessageBox.warning(self, "Warning", "No processed image to save.")
            return

        # Prompt user for the file path and format
        options = QFileDialog.Options()
        save_path, _ = QFileDialog.getSaveFileName(
            self, "Save Processed Image", "", 
            "TIFF Files (*.tif *.tiff);;PNG Files (*.png);;FITS Files (*.fits *.fit)", 
            options=options
        )
        
        if not save_path:
            return  # User cancelled the save dialog

        # Determine the format based on file extension
        _, file_extension = os.path.splitext(save_path)
        file_extension = file_extension.lower().lstrip('.')
        original_format = file_extension if file_extension in ['tiff', 'tif', 'png', 'fits', 'fit'] else 'tiff'

        # Call the save_image function with the necessary parameters
        try:
            save_image(
                img_array=self.processed_image,
                filename=save_path,
                original_format=original_format,
                bit_depth=self.bit_depth,
                original_header=self.original_header,
                is_mono=self.is_mono
            )
            QMessageBox.information(self, "Success", f"Image saved successfully at: {save_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save image: {e}")

    def load_image_with_retry(self, file_path, retries=5, delay=2):
        """
        Attempts to load an image multiple times with delays between attempts.

        :param file_path: Path to the image file.
        :param retries: Number of retry attempts.
        :param delay: Delay between retries in seconds.
        :return: Tuple of (image_array, original_header, bit_depth, is_mono) or (None, None, None, None) if failed.
        """

        for attempt in range(1, retries + 1):
            image, original_header, bit_depth, is_mono = load_image(file_path)
            if image is not None:

                return image, original_header, bit_depth, is_mono
            else:
                print(f"[WARNING] Attempt {attempt} failed to load image. Retrying in {delay} seconds...")
                time.sleep(delay)
        print("[ERROR] All attempts to load the image failed.")
        return None, None, None, None


    def wait_for_output_file(self, output_file_glob, timeout=3000, check_interval=1, stable_checks=3):
        """
        Wait for the output file with any extension within the specified timeout.
        Ensures the file size remains constant over a series of checks to confirm it's fully written.

        :param output_file_glob: Glob pattern to match the output file.
        :param timeout: Maximum time to wait in seconds.
        :param check_interval: Time between size checks in seconds.
        :param stable_checks: Number of consecutive checks with the same size.
        :return: Path to the output file or None if not found.
        """
        start_time = time.time()
        last_size = -1
        stable_count = 0

        while time.time() - start_time < timeout:
            matching_files = glob.glob(output_file_glob)
            if matching_files:
                current_size = os.path.getsize(matching_files[0])
                if current_size == last_size:
                    stable_count += 1
                    if stable_count >= stable_checks:
                        print(f"Output file found and stable: {matching_files[0]}")
                        return matching_files[0]
                else:
                    stable_count = 0
                    last_size = current_size
            time.sleep(check_interval)
        
        print("Timeout reached. Output file not found or not stable.")
        return None

    def display_image(self, file_path):
        """Load and display the output image."""
        self.image, self.original_header, self.bit_depth, self.is_mono = load_image(file_path)
        self.display_image()  # Update display with the new image

    def cleanup_files(self, input_file_path, output_file_path):
        """Delete input and output files after processing."""
        try:
            if input_file_path and os.path.exists(input_file_path):
                os.remove(input_file_path)
                print(f"Deleted input file: {input_file_path}")
            else:
                print(f"")

            if output_file_path and os.path.exists(output_file_path):
                os.remove(output_file_path)
                print(f"Deleted output file: {output_file_path}")
            else:
                print(f"")
        except Exception as e:
            print(f"Failed to delete files: {e}")

class PreviewDialog(QDialog):
    def __init__(self, np_image, parent_tab=None, is_mono=False):
        super().__init__(parent=parent_tab)
        self.setWindowTitle("Select Preview Area")
        self.setWindowFlags(self.windowFlags() | Qt.WindowContextHelpButtonHint | Qt.MSWindowsFixedSizeDialogHint)
        self.setFixedSize(640, 480)  # Fix the size to 640x480
        self.autostretch_enabled = False  # Autostretch toggle for preview
        self.is_mono = is_mono  # Store is_mono flag

        # Store the 32-bit numpy image for reference
        self.np_image = np_image
        self.original_np_image = np_image.copy()  # Copy to allow undo
        self.parent_tab = parent_tab
        # Track saved scroll positions for Undo
        self.saved_h_scroll = 0
        self.saved_v_scroll = 0        

        # Set up the layout and the scroll area
        layout = QVBoxLayout(self)

        # Autostretch button
        self.autostretch_button = QPushButton("AutoStretch (Off)")
        self.autostretch_button.setCheckable(True)
        self.autostretch_button.toggled.connect(self.toggle_autostretch)
        layout.addWidget(self.autostretch_button)

        # Scroll area for displaying the image
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        layout.addWidget(self.scroll_area)

        # Set up the QLabel to display the image
        self.image_label = QLabel()
        self.display_qimage(self.np_image)  # Display the image with the initial numpy array
        self.scroll_area.setWidget(self.image_label)

        # Add the Process Visible Area and Undo buttons
        button_layout = QHBoxLayout()
        
        self.process_button = QPushButton("Process Visible Area")
        self.process_button.clicked.connect(self.process_visible_area)
        button_layout.addWidget(self.process_button)

        self.undo_button = QPushButton("Undo")
        self.undo_button.clicked.connect(self.undo_last_process)
        self.undo_button.setIcon(QApplication.style().standardIcon(QStyle.SP_ArrowLeft))
        button_layout.addWidget(self.undo_button)

        layout.addLayout(button_layout)

        # Set up mouse dragging
        self.dragging = False
        self.drag_start_pos = QPoint()

        # Center the scroll area on initialization
        QTimer.singleShot(0, self.center_scrollbars)  # Delay to ensure layout is set
                
        # Enable What's This functionality
        self.setWhatsThis(
            "Instructions:\n\n"
            "1. Use the scroll bars to center on the area of the image you want to preview.\n"
            "2. Click and drag to move around the image.\n"
            "3. When ready, click the 'Process Visible Area' button to process the selected section."
        )

    def display_qimage(self, np_img):
        """Convert a numpy array to QImage and display it at 100% scale."""
        # Ensure the numpy array is scaled to [0, 255] and converted to uint8
        display_image_uint8 = (np.clip(np_img, 0, 1) * 255).astype(np.uint8)
        
        if len(display_image_uint8.shape) == 3 and display_image_uint8.shape[2] == 3:
            # RGB image
            height, width, channels = display_image_uint8.shape
            bytes_per_line = 3 * width
            qimage = QImage(display_image_uint8.tobytes(), width, height, bytes_per_line, QImage.Format_RGB888)
        elif len(display_image_uint8.shape) == 2:
            # Grayscale image
            height, width = display_image_uint8.shape
            bytes_per_line = width
            qimage = QImage(display_image_uint8.tobytes(), width, height, bytes_per_line, QImage.Format_Grayscale8)
        else:
            raise ValueError(f"Unexpected image shape: {display_image_uint8.shape}")

        # Display the QImage at 100% scale in QLabel
        self.image_label.setPixmap(QPixmap.fromImage(qimage))
        self.image_label.adjustSize()


    def toggle_autostretch(self, checked):
        self.autostretch_enabled = checked
        self.autostretch_button.setText("AutoStretch (On)" if checked else "AutoStretch (Off)")
        self.apply_autostretch()

    def apply_autostretch(self):
        """Apply or remove autostretch while maintaining 32-bit precision."""
        target_median = 0.25  # Target median for stretching

        if self.autostretch_enabled:
            if self.is_mono:  # Apply mono stretch
                # Directly use the 2D array for mono images
                stretched_mono = stretch_mono_image(self.np_image, target_median)  # Mono image is 2D
                display_image = np.stack([stretched_mono] * 3, axis=-1)  # Convert to RGB for display
            else:  # Apply color stretch
                display_image = stretch_color_image(self.np_image, target_median, linked=False)
        else:
            display_image = self.np_image  # Use original image if autostretch is off

        # Convert and display the QImage
        self.display_qimage(display_image)


    def undo_last_process(self):
        """Revert to the original image in the preview, respecting the autostretch setting."""
        print("Undo last process")
        
        # Reset to the original image
        self.np_image = self.original_np_image.copy()
        
        # Apply autostretch if it is enabled
        if self.autostretch_enabled:
            print("Applying autostretch on undo")
            self.apply_autostretch()
        else:
            # Display the original image without autostretch
            self.display_qimage(self.np_image)
        
        # Restore saved scroll positions with a slight delay
        QTimer.singleShot(0, self.restore_scrollbars)
        print("Scrollbars will be restored to saved positions")


    def restore_scrollbars(self):
        """Restore the scrollbars to the saved positions after a delay."""
        self.scroll_area.horizontalScrollBar().setValue(self.saved_h_scroll)
        self.scroll_area.verticalScrollBar().setValue(self.saved_v_scroll)
        print("Scrollbars restored to saved positions")
   
    def center_scrollbars(self):
        """Centers the scrollbars to start in the middle of the image."""
        # Set the horizontal and vertical scrollbar positions to center
        h_scroll = self.scroll_area.horizontalScrollBar()
        v_scroll = self.scroll_area.verticalScrollBar()
        h_scroll.setValue((h_scroll.maximum() + h_scroll.minimum()) // 2)
        v_scroll.setValue((v_scroll.maximum() + v_scroll.minimum()) // 2)

    def mousePressEvent(self, event):
        """Start dragging if the left mouse button is pressed."""
        if event.button() == Qt.LeftButton:
            self.dragging = True
            self.drag_start_pos = event.pos()

    def mouseMoveEvent(self, event):
        """Handle dragging to move the scroll area."""
        if self.dragging:
            delta = event.pos() - self.drag_start_pos
            self.scroll_area.horizontalScrollBar().setValue(
                self.scroll_area.horizontalScrollBar().value() - delta.x()
            )
            self.scroll_area.verticalScrollBar().setValue(
                self.scroll_area.verticalScrollBar().value() - delta.y()
            )
            self.drag_start_pos = event.pos()

    def mouseReleaseEvent(self, event):
        """Stop dragging when the left mouse button is released."""
        if event.button() == Qt.LeftButton:
            self.dragging = False

    def process_visible_area(self):
        print("Process Visible Area button pressed")  # Initial debug print to confirm button press

        """Crop the image to the visible area and send it to CosmicClarityTab for processing."""

        self.saved_h_scroll = self.scroll_area.horizontalScrollBar().value()
        self.saved_v_scroll = self.scroll_area.verticalScrollBar().value()

        # Calculate the visible area in the original image coordinates
        h_scroll = self.scroll_area.horizontalScrollBar().value()
        v_scroll = self.scroll_area.verticalScrollBar().value()
        visible_rect = QRect(h_scroll, v_scroll, 640, 480)  # 640x480 fixed size
        print(f"Visible area rectangle: {visible_rect}")  # Debug print to confirm visible area coordinates

        # Crop the numpy image array directly using slicing
        if len(self.np_image.shape) == 2:  # Mono image (2D array)
            cropped_np_image = self.np_image[
                v_scroll : v_scroll + visible_rect.height(),
                h_scroll : h_scroll + visible_rect.width(),
            ]
            # Convert cropped mono image to RGB for consistent handling
            cropped_np_image = np.stack([cropped_np_image] * 3, axis=-1)
        elif len(self.np_image.shape) == 3:  # Color image (3D array)
            cropped_np_image = self.np_image[
                v_scroll : v_scroll + visible_rect.height(),
                h_scroll : h_scroll + visible_rect.width(),
                :
            ]
        else:
            print("Error: Unsupported image format")
            return

        if cropped_np_image is None:
            print("Error: Failed to crop numpy image")  # Debug if cropping failed
        else:
            print("Image cropped successfully")  # Debug print to confirm cropping

        # Pass the cropped image to CosmicClarityTab for processing
        if self.parent_tab:
            print("Sending to parent class for processing")  # Debug print before sending to parent
            self.parent_tab.run_cosmic_clarity_on_cropped(cropped_np_image, apply_autostretch=self.autostretch_enabled)
        else:
            print("Error: Failed to send to parent class")  # Debug if parent reference is missing


    def convert_qimage_to_numpy(self, qimage):
        """Convert QImage to a 32-bit float numpy array, preserving the 32-bit precision."""
        qimage = qimage.convertToFormat(QImage.Format_RGB888)
        
        width = qimage.width()
        height = qimage.height()
        ptr = qimage.bits()
        ptr.setsize(height * width * 3)
        
        arr = np.frombuffer(ptr, dtype=np.uint8).reshape((height, width, 3)).astype(np.float32) / 255.0
        return arr

    def closeEvent(self, event):
        """Handle dialog close event if any cleanup is necessary."""
        self.dragging = False
        event.accept()

class WaitDialog(QDialog):
    cancelled = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Processing...")
        
        self.layout = QVBoxLayout()
        
        self.label = QLabel("Processing, please wait...")
        self.layout.addWidget(self.label)
        
        # Add a QTextEdit to show process output
        self.output_text_edit = QTextEdit()
        self.output_text_edit.setReadOnly(True)
        self.layout.addWidget(self.output_text_edit)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.layout.addWidget(self.progress_bar)

        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.cancelled.emit)
        self.layout.addWidget(cancel_button)
        
        self.setLayout(self.layout)

    def append_output(self, text):
        self.output_text_edit.append(text)


class WaitForFileWorker(QThread):
    fileFound = pyqtSignal(str)
    cancelled = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, output_file_glob, timeout=1800, parent=None):
        super().__init__(parent)
        self.output_file_glob = output_file_glob
        self.timeout = timeout
        self._running = True

    def run(self):
        start_time = time.time()
        while self._running and (time.time() - start_time < self.timeout):
            matching_files = glob.glob(self.output_file_glob)
            if matching_files:
                self.fileFound.emit(matching_files[0])
                return
            time.sleep(1)
        if self._running:
            self.error.emit("Output file not found within timeout.")
        else:
            self.cancelled.emit()

    def stop(self):
        self._running = False

class CosmicClaritySatelliteTab(QWidget):
    def __init__(self):
        super().__init__()
        self.cosmic_clarity_folder = None
        self.input_folder = None
        self.output_folder = None
        self.settings_file = "cosmic_clarity_satellite_folder.txt"
        self.file_watcher = QFileSystemWatcher()  # Watcher for input and output folders
        self.file_watcher.directoryChanged.connect(self.on_folder_changed)  # Connect signal
        self.sensitivity = 0.1
        self.initUI()
        self.load_cosmic_clarity_folder()

    def initUI(self):
        # Main horizontal layout
        main_layout = QHBoxLayout()

        # Left layout for controls and settings
        left_layout = QVBoxLayout()

        # Input/Output Folder Selection in a Horizontal Sizer
        folder_layout = QHBoxLayout()
        self.input_folder_button = QPushButton("Select Input Folder")
        self.input_folder_button.clicked.connect(self.select_input_folder)
        self.output_folder_button = QPushButton("Select Output Folder")
        self.output_folder_button.clicked.connect(self.select_output_folder)
        folder_layout.addWidget(self.input_folder_button)
        folder_layout.addWidget(self.output_folder_button)
        left_layout.addLayout(folder_layout)

        # GPU Acceleration
        self.gpu_label = QLabel("Use GPU Acceleration:")
        left_layout.addWidget(self.gpu_label)
        self.gpu_dropdown = QComboBox()
        self.gpu_dropdown.addItems(["Yes", "No"])
        left_layout.addWidget(self.gpu_dropdown)

        # Removal Mode
        self.mode_label = QLabel("Satellite Removal Mode:")
        left_layout.addWidget(self.mode_label)
        self.mode_dropdown = QComboBox()
        self.mode_dropdown.addItems(["Full", "Luminance"])
        left_layout.addWidget(self.mode_dropdown)

        # Clip Trail
        self.clip_trail_checkbox = QCheckBox("Clip Satellite Trail to 0.000")
        self.clip_trail_checkbox.setChecked(True)
        left_layout.addWidget(self.clip_trail_checkbox)

        # **Add Sensitivity Slider**
        sensitivity_layout = QHBoxLayout()
        sensitivity_label = QLabel("Clipping Sensitivity (Lower Values more Aggressive Clipping):")
        sensitivity_layout.addWidget(sensitivity_label)

        self.sensitivity_slider = QSlider(Qt.Horizontal)
        self.sensitivity_slider.setMinimum(1)    # Represents 0.01
        self.sensitivity_slider.setMaximum(50)   # Represents 0.5
        self.sensitivity_slider.setValue(int(self.sensitivity * 100))  # e.g., 0.1 * 100 = 10
        self.sensitivity_slider.setTickInterval(1)
        self.sensitivity_slider.setTickPosition(QSlider.TicksBelow)
        self.sensitivity_slider.valueChanged.connect(self.update_sensitivity)
        sensitivity_layout.addWidget(self.sensitivity_slider)

        # Label to display current sensitivity value
        self.sensitivity_value_label = QLabel(f"{self.sensitivity:.2f}")
        sensitivity_layout.addWidget(self.sensitivity_value_label)

        left_layout.addLayout(sensitivity_layout)        

        # Skip Save
        self.skip_save_checkbox = QCheckBox("Skip Save if No Satellite Trail Detected")
        self.skip_save_checkbox.setChecked(False)
        left_layout.addWidget(self.skip_save_checkbox)

        # Process Single Image and Batch Process in a Horizontal Sizer
        process_layout = QHBoxLayout()
        self.process_single_button = QPushButton("Process Single Image")
        self.process_single_button.clicked.connect(self.process_single_image)
        process_layout.addWidget(self.process_single_button)

        self.batch_process_button = QPushButton("Batch Process Input Folder")
        self.batch_process_button.clicked.connect(self.batch_process_folder)
        process_layout.addWidget(self.batch_process_button)
        left_layout.addLayout(process_layout)

        # Live Monitor
        self.live_monitor_button = QPushButton("Live Monitor Input Folder")
        self.live_monitor_button.clicked.connect(self.live_monitor_folder)
        left_layout.addWidget(self.live_monitor_button)

        # Folder Selection
        self.folder_label = QLabel("No folder selected")
        left_layout.addWidget(self.folder_label)
        self.wrench_button = QPushButton()
        self.wrench_button.setIcon(QIcon(wrench_path))  # Ensure the icon is available
        self.wrench_button.clicked.connect(self.select_cosmic_clarity_folder)
        left_layout.addWidget(self.wrench_button)

        # Footer
        footer_label = QLabel("""
            Written by Franklin Marek<br>
            <a href='http://www.setiastro.com'>www.setiastro.com</a>
        """)
        footer_label.setAlignment(Qt.AlignCenter)
        footer_label.setOpenExternalLinks(True)
        footer_label.setStyleSheet("font-size: 10px;")
        left_layout.addWidget(footer_label)

        # Right layout for TreeBoxes
        right_layout = QVBoxLayout()

        # Input Files TreeBox
        input_files_label = QLabel("Input Folder Files:")
        right_layout.addWidget(input_files_label)
        self.input_files_tree = QTreeWidget()
        self.input_files_tree.setHeaderLabels(["Filename"])
        self.input_files_tree.itemDoubleClicked.connect(lambda: self.preview_image(self.input_files_tree))
        self.input_files_tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.input_files_tree.customContextMenuRequested.connect(lambda pos: self.show_context_menu(self.input_files_tree, pos))
        right_layout.addWidget(self.input_files_tree)

        # Output Files TreeBox
        output_files_label = QLabel("Output Folder Files:")
        right_layout.addWidget(output_files_label)
        self.output_files_tree = QTreeWidget()
        self.output_files_tree.setHeaderLabels(["Filename"])
        self.output_files_tree.itemDoubleClicked.connect(lambda: self.preview_image(self.output_files_tree))
        self.output_files_tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.output_files_tree.customContextMenuRequested.connect(lambda pos: self.show_context_menu(self.output_files_tree, pos))
        right_layout.addWidget(self.output_files_tree)


        # Add the left and right layouts to the main layout
        main_layout.addLayout(left_layout, stretch=2)  # More space for the left layout
        main_layout.addLayout(right_layout, stretch=1)  # Less space for the right layout

        self.setLayout(main_layout)

    def update_sensitivity(self, value):
        """
        Update the sensitivity value based on the slider's position.
        """
        self.sensitivity = value / 100.0  # Convert from integer to float (0.01 to 0.5)
        self.sensitivity_value_label.setText(f"{self.sensitivity:.2f}")  # Update label





    def preview_image(self, treebox):
        """Preview the selected image."""
        selected_item = treebox.currentItem()
        if selected_item:
            file_path = os.path.join(self.input_folder if treebox == self.input_files_tree else self.output_folder, selected_item.text(0))
            if os.path.isfile(file_path):
                try:
                    image, _, _, is_mono = load_image(file_path)
                    if image is not None:
                        self.current_preview_dialog = ImagePreviewDialog(image, is_mono=is_mono)  # Store reference
                        self.current_preview_dialog.setAttribute(Qt.WA_DeleteOnClose)  # Ensure cleanup on close
                        self.current_preview_dialog.show()  # Open non-blocking dialog
                    else:
                        QMessageBox.critical(self, "Error", "Failed to load image for preview.")
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Failed to preview image: {e}")


    def open_preview_dialog(self, image, is_mono):
        """Open the preview dialog."""
        preview_dialog = ImagePreviewDialog(image, is_mono=is_mono)
        preview_dialog.setAttribute(Qt.WA_DeleteOnClose)  # Ensure proper cleanup when closed
        preview_dialog.show()  # Open the dialog without blocking the main UI





    def show_context_menu(self, treebox, pos):
        """Show context menu for the treebox."""
        menu = QMenu()
        delete_action = QAction("Delete File")
        rename_action = QAction("Rename File")
        delete_action.triggered.connect(lambda: self.delete_file(treebox))
        rename_action.triggered.connect(lambda: self.rename_file(treebox))
        menu.addAction(delete_action)
        menu.addAction(rename_action)
        menu.exec_(treebox.viewport().mapToGlobal(pos))

    def delete_file(self, treebox):
        """Delete the selected file."""
        selected_item = treebox.currentItem()
        if selected_item:
            folder = self.input_folder if treebox == self.input_files_tree else self.output_folder
            file_path = os.path.join(folder, selected_item.text(0))
            if os.path.exists(file_path):
                reply = QMessageBox.question(self, "Confirm Delete", f"Are you sure you want to delete {selected_item.text(0)}?",
                                             QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                if reply == QMessageBox.Yes:
                    os.remove(file_path)
                    self.refresh_input_files() if treebox == self.input_files_tree else self.refresh_output_files()

    def rename_file(self, treebox):
        """Rename the selected file."""
        selected_item = treebox.currentItem()
        if selected_item:
            folder = self.input_folder if treebox == self.input_files_tree else self.output_folder
            file_path = os.path.join(folder, selected_item.text(0))
            new_name, ok = QInputDialog.getText(self, "Rename File", "Enter new name:", text=selected_item.text(0))
            if ok and new_name:
                new_path = os.path.join(folder, new_name)
                os.rename(file_path, new_path)
                self.refresh_input_files() if treebox == self.input_files_tree else self.refresh_output_files()

    def refresh_input_files(self):
        """Populate the input TreeBox with files from the input folder."""
        self.input_files_tree.clear()
        if not self.input_folder:
            return
        for file_name in os.listdir(self.input_folder):
            if file_name.lower().endswith(('.png', '.tif', '.tiff', '.fit', '.fits', '.xisf', '.cr2', '.nef', '.arw', '.dng', '.orf', '.rw2', '.pef')):
                QTreeWidgetItem(self.input_files_tree, [file_name])

    def refresh_output_files(self):
        """Populate the output TreeBox with files from the output folder."""
        self.output_files_tree.clear()
        if not self.output_folder:
            return
        for file_name in os.listdir(self.output_folder):
            if file_name.lower().endswith(('.png', '.tif', '.tiff', '.fit', '.fits', '.xisf', '.cr2', '.nef', '.arw', '.dng', '.orf', '.rw2', '.pef')):
                QTreeWidgetItem(self.output_files_tree, [file_name])



    def select_input_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Input Folder")
        if folder:
            self.input_folder = folder
            self.input_folder_button.setText(f"Input Folder: {os.path.basename(folder)}")
            self.file_watcher.addPath(folder)  # Add folder to watcher
            self.refresh_input_files()

    def select_output_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self.output_folder = folder
            self.output_folder_button.setText(f"Output Folder: {os.path.basename(folder)}")
            self.file_watcher.addPath(folder)  # Add folder to watcher
            self.refresh_output_files()

    def on_folder_changed(self, path):
        """Refresh the TreeBox when files are added or removed from the watched folder."""
        if path == self.input_folder:
            self.refresh_input_files()
        elif path == self.output_folder:
            self.refresh_output_files()


    def select_cosmic_clarity_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Cosmic Clarity Folder")
        if folder:
            self.cosmic_clarity_folder = folder
            self.folder_label.setText(f"Folder: {folder}")
            with open(self.settings_file, 'w') as f:
                f.write(folder)

    def load_cosmic_clarity_folder(self):
        if os.path.exists(self.settings_file):
            with open(self.settings_file, 'r') as f:
                folder = f.read().strip()
                if folder:
                    self.cosmic_clarity_folder = folder
                    self.folder_label.setText(f"Folder: {folder}")

    def process_single_image(self):
        # Step 1: Open File Dialog to Select Image
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Select Image", 
            "", 
            "Image Files (*.png *.tif *.tiff *.fit *.fits *.xisf *.cr2 *.nef *.arw *.dng *.orf *.rw2 *.pef)"
        )
        if not file_path:
            QMessageBox.warning(self, "Warning", "No file selected.")
            return

        # Create temp input and output folders
        temp_input = self.create_temp_folder()
        temp_output = self.create_temp_folder()

        # Copy the selected file to the temp input folder
        shutil.copy(file_path, temp_input)

        # Run Cosmic Clarity Satellite Removal Tool
        try:
            self.run_cosmic_clarity_satellite(temp_input, temp_output)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error processing image: {e}")
            return

        # Locate the processed file in the temp output folder
        processed_file = glob.glob(os.path.join(temp_output, "*_satellited.*"))
        if processed_file:
            # Move the processed file back to the original folder
            original_folder = os.path.dirname(file_path)
            destination_path = os.path.join(original_folder, os.path.basename(processed_file[0]))
            shutil.move(processed_file[0], destination_path)

            # Inform the user
            QMessageBox.information(self, "Success", f"Processed image saved to: {destination_path}")
        else:
            QMessageBox.warning(self, "Warning", "No output file found.")

        # Cleanup temporary folders
        if os.path.exists(temp_input):
            shutil.rmtree(temp_input)
        if os.path.exists(temp_output):
            shutil.rmtree(temp_output)

    def batch_process_folder(self):
        if not self.input_folder or not self.output_folder:
            QMessageBox.warning(self, "Warning", "Please select both input and output folders.")
            return

        exe_name = "setiastrocosmicclarity_satellite"
        exe_path = os.path.join(self.cosmic_clarity_folder, f"{exe_name}.exe") if os.name == 'nt' else os.path.join(self.cosmic_clarity_folder, exe_name)

        if not os.path.exists(exe_path):
            QMessageBox.critical(self, "Error", f"Executable not found: {exe_path}")
            return

        # Construct the command
        command = [
            exe_path,
            "--input", self.input_folder,
            "--output", self.output_folder,
            "--mode", self.mode_dropdown.currentText().lower(),
            "--batch"
        ]
        if self.gpu_dropdown.currentText() == "Yes":
            command.append("--use-gpu")
        if self.clip_trail_checkbox.isChecked():
            command.append("--clip-trail")
            print("--clip-trail argument added.")
        else:
            command.append("--no-clip-trail")
            print("--no-clip-trail argument added.")
        if self.skip_save_checkbox.isChecked():
            command.append("--skip-save")

        # **Add Sensitivity Argument**
        command.extend(["--sensitivity", str(self.sensitivity)])            

        # Run the command in a separate thread
        self.satellite_thread = SatelliteProcessingThread(command)
        self.satellite_thread.finished.connect(lambda: QMessageBox.information(self, "Success", "Batch processing finished."))
        self.satellite_thread.start()

    def live_monitor_folder(self):
        if not self.input_folder or not self.output_folder:
            QMessageBox.warning(self, "Warning", "Please select both input and output folders.")
            return

        exe_name = "setiastrocosmicclarity_satellite"
        exe_path = os.path.join(self.cosmic_clarity_folder, f"{exe_name}.exe") if os.name == 'nt' else os.path.join(self.cosmic_clarity_folder, exe_name)

        if not os.path.exists(exe_path):
            QMessageBox.critical(self, "Error", f"Executable not found: {exe_path}")
            return

        # Construct the command
        command = [
            exe_path,
            "--input", self.input_folder,
            "--output", self.output_folder,
            "--mode", self.mode_dropdown.currentText().lower(),
            "--monitor"
        ]
        if self.gpu_dropdown.currentText() == "Yes":
            command.append("--use-gpu")
        if self.clip_trail_checkbox.isChecked():
            command.append("--clip-trail")
            print("--clip-trail argument added.")
        else:
            command.append("--no-clip-trail")
            print("--no-clip-trail argument added.")
        if self.skip_save_checkbox.isChecked():
            command.append("--skip-save")

        # **Add Sensitivity Argument**
        command.extend(["--sensitivity", str(self.sensitivity)])            

        # Run the command in a separate thread
        self.sensitivity_slider.setEnabled(False)
        self.satellite_thread = SatelliteProcessingThread(command)
        self.satellite_thread.finished.connect(lambda: QMessageBox.information(self, "Success", "Live monitoring stopped."))
        self.satellite_thread.finished.connect(lambda:self.sensitivity_slider.setEnabled(True))
        self.satellite_thread.start()

        # **Disable the sensitivity slider**
        


    def on_live_monitor_finished(self):
        """
        Slot to handle actions after live monitoring has finished.
        """
        QMessageBox.information(self, "Live Monitoring", "Live monitoring has been stopped.")
        self.sensitivity_slider.setEnabled(True)

        self.live_monitor_button.setEnabled(True)
        self.stop_monitor_button.setEnabled(False)
        
    @staticmethod
    def create_temp_folder(base_folder="~"):
        """
        Create a temporary folder for processing in the user's directory.
        :param base_folder: Base folder to create the temp directory in (default is the user's home directory).
        :return: Path to the created temporary folder.
        """
        user_dir = os.path.expanduser(base_folder)
        temp_folder = os.path.join(user_dir, "CosmicClarityTemp")
        os.makedirs(temp_folder, exist_ok=True)  # Create the folder if it doesn't exist
        return temp_folder


    def run_cosmic_clarity_satellite(self, input_dir, output_dir, live_monitor=False):
        if not self.cosmic_clarity_folder:
            QMessageBox.warning(self, "Warning", "Please select the Cosmic Clarity folder.")
            return

        exe_name = "setiastrocosmicclarity_satellite"
        exe_path = os.path.join(self.cosmic_clarity_folder, f"{exe_name}.exe") if os.name == 'nt' else os.path.join(self.cosmic_clarity_folder, exe_name)

        # Check if the executable exists
        if not os.path.exists(exe_path):
            QMessageBox.critical(self, "Error", f"Executable not found: {exe_path}")
            return

        # Construct command arguments
        command = [
            exe_path,
            "--input", input_dir,
            "--output", output_dir,
            "--mode", self.mode_dropdown.currentText().lower(),
        ]
        if self.gpu_dropdown.currentText() == "Yes":
            command.append("--use-gpu")
        if self.clip_trail_checkbox.isChecked():
            command.append("--clip-trail")
            print("--clip-trail argument added.")
        else:
            command.append("--no-clip-trail")
            print("--no-clip-trail argument added.")
        if self.skip_save_checkbox.isChecked():
            command.append("--skip-save")
        if live_monitor:
            command.append("--monitor")
        else:
            command.append("--batch")

        # **Add Sensitivity Argument**
        command.extend(["--sensitivity", str(self.sensitivity)])

        # Debugging: Print the command to verify
        print(f"Running command: {' '.join(command)}")

        # Execute the command
        try:
            subprocess.run(command, check=True)
            QMessageBox.information(self, "Success", "Processing complete.")
        except subprocess.CalledProcessError as e:
            QMessageBox.critical(self, "Error", f"Processing failed: {e}")

    def execute_script(self, script_path):
        """Execute the batch or shell script."""
        if os.name == 'nt':  # Windows
            subprocess.Popen(["cmd.exe", "/c", script_path], shell=True)
        else:  # macOS/Linux
            subprocess.Popen(["/bin/sh", script_path], shell=True)

    def wait_for_output_files(self, output_file_glob, timeout=1800):
        """Wait for output files matching the glob pattern within a timeout."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            matching_files = glob.glob(output_file_glob)
            if matching_files:
                time.sleep(2)
                return matching_files
            time.sleep(1)
        return None

class ImagePreviewDialog(QDialog):
    def __init__(self, np_image, is_mono=False):
        super().__init__()
        self.setWindowTitle("Image Preview")
        self.resize(640, 480)  # Set initial size
        self.autostretch_enabled = False  # Autostretch toggle for preview
        self.is_mono = is_mono  # Store is_mono flag
        self.zoom_factor = 1.0  # Track the zoom level

        # Store the 32-bit numpy image for reference
        self.np_image = np_image

        # Set up the layout and the scroll area
        layout = QVBoxLayout(self)

        # Autostretch and Zoom Buttons
        button_layout = QHBoxLayout()
        self.autostretch_button = QPushButton("AutoStretch (Off)")
        self.autostretch_button.setCheckable(True)
        self.autostretch_button.toggled.connect(self.toggle_autostretch)
        button_layout.addWidget(self.autostretch_button)

        self.zoom_in_button = QPushButton("Zoom In")
        self.zoom_in_button.clicked.connect(self.zoom_in)
        button_layout.addWidget(self.zoom_in_button)

        self.zoom_out_button = QPushButton("Zoom Out")
        self.zoom_out_button.clicked.connect(self.zoom_out)
        button_layout.addWidget(self.zoom_out_button)

        layout.addLayout(button_layout)

        # Scroll area for displaying the image
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        layout.addWidget(self.scroll_area)

        # Set up the QLabel to display the image
        self.image_label = QLabel()
        self.display_qimage(self.np_image)  # Display the image with the initial numpy array
        self.scroll_area.setWidget(self.image_label)

        # Set up mouse dragging
        self.dragging = False
        self.drag_start_pos = QPoint()

        # Enable mouse wheel for zooming
        self.image_label.installEventFilter(self)

        # Center the scroll area on initialization
        QTimer.singleShot(0, self.center_scrollbars)  # Delay to ensure layout is set

    def display_qimage(self, np_img):
        """Convert a numpy array to QImage and display it at the current zoom level."""
        display_image_uint8 = (np.clip(np_img, 0, 1) * 255).astype(np.uint8)

        if len(display_image_uint8.shape) == 3 and display_image_uint8.shape[2] == 3:
            # RGB image
            height, width, channels = display_image_uint8.shape
            bytes_per_line = 3 * width
            qimage = QImage(display_image_uint8.tobytes(), width, height, bytes_per_line, QImage.Format_RGB888)
        elif len(display_image_uint8.shape) == 2:
            # Grayscale image
            height, width = display_image_uint8.shape
            bytes_per_line = width
            qimage = QImage(display_image_uint8.tobytes(), width, height, bytes_per_line, QImage.Format_Grayscale8)
        else:
            raise ValueError(f"Unexpected image shape: {display_image_uint8.shape}")

        # Apply zoom
        pixmap = QPixmap.fromImage(qimage)
        scaled_width = int(pixmap.width() * self.zoom_factor)  # Convert to integer
        scaled_height = int(pixmap.height() * self.zoom_factor)  # Convert to integer
        scaled_pixmap = pixmap.scaled(scaled_width, scaled_height, Qt.KeepAspectRatio)
        self.image_label.setPixmap(scaled_pixmap)
        self.image_label.adjustSize()


    def toggle_autostretch(self, checked):
        self.autostretch_enabled = checked
        self.autostretch_button.setText("AutoStretch (On)" if checked else "AutoStretch (Off)")
        self.apply_autostretch()

    def apply_autostretch(self):
        """Apply or remove autostretch while maintaining 32-bit precision."""
        target_median = 0.25  # Target median for stretching

        if self.autostretch_enabled:
            if self.is_mono:  # Apply mono stretch
                if self.np_image.ndim == 2:  # Ensure single-channel mono
                    stretched_mono = stretch_mono_image(self.np_image, target_median)
                    display_image = np.stack([stretched_mono] * 3, axis=-1)  # Convert to RGB for display
                else:
                    raise ValueError(f"Unexpected mono image shape: {self.np_image.shape}")
            else:  # Apply color stretch
                display_image = stretch_color_image(self.np_image, target_median, linked=False)
        else:
            if self.is_mono and self.np_image.ndim == 2:
                display_image = np.stack([self.np_image] * 3, axis=-1)  # Convert to RGB for display
            else:
                display_image = self.np_image  # Use original image if autostretch is off

        print(f"Debug: Display image shape before QImage conversion: {display_image.shape}")
        self.display_qimage(display_image)



    def zoom_in(self):
        """Increase the zoom factor and refresh the display."""
        self.zoom_factor *= 1.2  # Increase zoom by 20%
        self.display_qimage(self.np_image)

    def zoom_out(self):
        """Decrease the zoom factor and refresh the display."""
        self.zoom_factor /= 1.2  # Decrease zoom by 20%
        self.display_qimage(self.np_image)

    def eventFilter(self, source, event):
        """Handle mouse wheel events for zooming."""
        if source == self.image_label and event.type() == QEvent.Wheel:
            if event.angleDelta().y() > 0:
                self.zoom_in()
            else:
                self.zoom_out()
            return True
        return super().eventFilter(source, event)

    def mousePressEvent(self, event):
        """Start dragging if the left mouse button is pressed."""
        if event.button() == Qt.LeftButton:
            self.dragging = True
            self.drag_start_pos = event.pos()

    def mouseMoveEvent(self, event):
        """Handle dragging to move the scroll area."""
        if self.dragging:
            delta = event.pos() - self.drag_start_pos
            self.scroll_area.horizontalScrollBar().setValue(
                self.scroll_area.horizontalScrollBar().value() - delta.x()
            )
            self.scroll_area.verticalScrollBar().setValue(
                self.scroll_area.verticalScrollBar().value() - delta.y()
            )
            self.drag_start_pos = event.pos()

    def mouseReleaseEvent(self, event):
        """Stop dragging when the left mouse button is released."""
        if event.button() == Qt.LeftButton:
            self.dragging = False

    def center_scrollbars(self):
        """Centers the scrollbars to start in the middle of the image."""
        h_scroll = self.scroll_area.horizontalScrollBar()
        v_scroll = self.scroll_area.verticalScrollBar()
        h_scroll.setValue((h_scroll.maximum() + h_scroll.minimum()) // 2)
        v_scroll.setValue((v_scroll.maximum() + v_scroll.minimum()) // 2)

    def resizeEvent(self, event):
        """Handle resizing of the dialog."""
        super().resizeEvent(event)
        self.display_qimage(self.np_image)



class SatelliteProcessingThread(QThread):
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()

    def __init__(self, command):
        super().__init__()
        self.command = command

    def run(self):
        try:
            self.log_signal.emit(f"Running command: {' '.join(self.command)}")
            subprocess.run(self.command, check=True)
            self.log_signal.emit("Processing complete.")
        except subprocess.CalledProcessError as e:
            self.log_signal.emit(f"Processing failed: {e}")
        except Exception as e:
            self.log_signal.emit(f"Unexpected error: {e}")
        finally:
            self.finished_signal.emit()  # Emit the finished signal            


class StatisticalStretchTab(QWidget):
    def __init__(self, image_manager=None):
        super().__init__()
        self.image_manager = image_manager  # Reference to the ImageManager
        self.loaded_image_path = None
        self.original_header = None
        self.bit_depth = None
        self.is_mono = False
        self.zoom_factor = 1.0
        self.image = None  # Current image (from ImageManager)
        self.stretched_image = None  # Processed image
        self.initUI()

        if self.image_manager:
            # Connect to ImageManager's image_changed signal
            self.image_manager.image_changed.connect(self.on_image_changed)


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

        self.fileLabel = QLabel('', self)
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

        # Buttons (Undo and Preview Stretch)
        button_layout = QHBoxLayout()

        self.previewButton = QPushButton('Preview Stretch', self)
        self.previewButton.clicked.connect(self.previewStretch)
        button_layout.addWidget(self.previewButton)

        self.undoButton = QPushButton('Undo', self)
        undo_icon = self.style().standardIcon(QStyle.SP_ArrowBack)  # Standard left arrow icon
        self.undoButton.setIcon(undo_icon)
        self.undoButton.clicked.connect(self.undo_image)
        button_layout.addWidget(self.undoButton)


        left_layout.addLayout(button_layout)

        # **Remove Zoom Buttons from Left Panel**
        # Commented out to move to the right panel
        # zoom_layout = QHBoxLayout()
        # self.zoomInButton = QPushButton('Zoom In', self)
        # self.zoomInButton.clicked.connect(self.zoom_in)
        # zoom_layout.addWidget(self.zoomInButton)

        # self.zoomOutButton = QPushButton('Zoom Out', self)
        # self.zoomOutButton.clicked.connect(self.zoom_out)
        # zoom_layout.addWidget(self.zoomOutButton)

        # left_layout.addLayout(zoom_layout)

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

        # **Create Right Panel Layout**
        right_widget = QWidget(self)
        right_layout = QVBoxLayout(right_widget)

        # Zoom controls

        zoom_layout = QHBoxLayout()
        zoom_in_button = QPushButton("Zoom In")
        zoom_in_button.clicked.connect(self.zoom_in)
        zoom_layout.addWidget(zoom_in_button)

        zoom_out_button = QPushButton("Zoom Out")
        zoom_out_button.clicked.connect(self.zoom_out)
        zoom_layout.addWidget(zoom_out_button)

        # **Add "Fit to Preview" Button**
        self.fitToPreviewButton = QPushButton("Fit to Preview")
        self.fitToPreviewButton.clicked.connect(self.fit_to_preview)
        zoom_layout.addWidget(self.fitToPreviewButton)

        right_layout.addLayout(zoom_layout)

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

        right_layout.addWidget(self.scrollArea)

        # Add the right widget to the main layout
        main_layout.addWidget(right_widget)

        self.setLayout(main_layout)
        self.zoom_factor = 0.25
        self.scrollArea.viewport().setMouseTracking(True)
        self.scrollArea.viewport().installEventFilter(self)
        self.dragging = False
        self.last_pos = QPoint()

    def on_image_changed(self, slot, image, metadata):
        """
        Slot to handle image changes from ImageManager.
        Updates the display if the current slot is affected.
        """
        if slot == self.image_manager.current_slot:
            # Ensure the image is a numpy array before proceeding
            if not isinstance(image, np.ndarray):
                image = np.array(image)  # Convert to numpy array if necessary
            
            self.image = image  # Set the original image
            self.preview_image = None  # Reset the preview image
            self.original_header = metadata.get('original_header', None)
            self.is_mono = metadata.get('is_mono', False)
            self.filename = metadata.get('file_path', self.fileLabel)

            # Update the image display
            self.updateImageDisplay()

            print(f"Statistical Stretch: Image updated from ImageManager slot {slot}.")

    def updateImageDisplay(self):
        if self.image is not None:
            # Prepare the image for display by normalizing and converting to uint8
            display_image = (self.image * 255).astype(np.uint8)
            h, w = display_image.shape[:2]

            if display_image.ndim == 3:  # RGB Image
                # Convert the image to QImage format
                q_image = QImage(display_image.tobytes(), w, h, 3 * w, QImage.Format_RGB888)
            else:  # Grayscale Image
                q_image = QImage(display_image.tobytes(), w, h, w, QImage.Format_Grayscale8)

            # Create a QPixmap from QImage
            pixmap = QPixmap.fromImage(q_image)
            self.current_pixmap = pixmap  # Store the original pixmap for future reference

            # Scale the pixmap based on the zoom factor
            scaled_pixmap = pixmap.scaled(pixmap.size() * self.zoom_factor, Qt.KeepAspectRatio, Qt.SmoothTransformation)

            # Set the pixmap on the image label
            self.imageLabel.setPixmap(scaled_pixmap)
            self.imageLabel.resize(scaled_pixmap.size())  # Resize the label to fit the image
        else:
            # If no image is available, clear the label and show a message
            self.imageLabel.clear()
            self.imageLabel.setText('No image loaded.')

    def updatePreview(self, stretched_image):
        # Store the stretched image for saving
        self.preview_image = stretched_image

        # Update the ImageManager with the new stretched image
        metadata = {
            'file_path': self.filename if self.filename else "Stretched Image",
            'original_header': self.original_header if self.original_header else {},
            'bit_depth': "Unknown",  # Update if bit_depth is available
            'is_mono': self.is_mono,
            'processing_timestamp': datetime.now().isoformat(),
            'source_images': {
                'Original': self.filename if self.filename else "Not Provided"
            }
        }

        # Update ImageManager with the new processed image
        if self.image_manager:
            try:
                self.image_manager.update_image(updated_image=self.preview_image, metadata=metadata)
                print("StarStretchTab: Processed image stored in ImageManager.")
            except Exception as e:
                print(f"Error updating ImageManager: {e}")
                QMessageBox.critical(self, "Error", f"Failed to update ImageManager:\n{e}")
        else:
            print("ImageManager is not initialized.")
            QMessageBox.warning(self, "Warning", "ImageManager is not initialized. Cannot store the processed image.")

        # Update the preview once the processing thread emits the result
        preview_image = (stretched_image * 255).astype(np.uint8)
        h, w = preview_image.shape[:2]
        if preview_image.ndim == 3:
            q_image = QImage(preview_image.data, w, h, 3 * w, QImage.Format_RGB888)
        else:
            q_image = QImage(preview_image.data, w, h, w, QImage.Format_Grayscale8)

        pixmap = QPixmap.fromImage(q_image)
        self.current_pixmap = pixmap  # **Store the original pixmap**
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


    def openFileDialog(self):
        if not self.image_manager:
            QMessageBox.warning(self, "Warning", "ImageManager not initialized.")
            return

        self.filename, _ = QFileDialog.getOpenFileName(self, "Open Image", "", 
                                            "Images (*.png *.tif *.tiff *.fits *.xisf *.cr2 *.nef *.arw *.dng *.orf *.rw2 *.pef);;All Files (*)")
        if self.filename:
            self.fileLabel.setText(self.filename)

            # Load the image using ImageManager
            image, original_header, bit_depth, is_mono = load_image(self.filename)

            if image is None:
                QMessageBox.critical(self, "Error", "Failed to load the image. Please try a different file.")
                return

            # Update ImageManager with the new image
            metadata = {
                'file_path': self.filename,
                'original_header': original_header,
                'bit_depth': bit_depth,
                'is_mono': is_mono
            }
            self.image_manager.add_image(slot=self.image_manager.current_slot, image=image, metadata=metadata)

            print("Image added to ImageManager.")

    def undo_image(self):
        """Undo the last action."""
        if self.image_manager.can_undo():
            self.image_manager.undo()  # Reverts to the previous image
            self.updateImageDisplay()  # Update the display with the reverted image
            print("Undo performed.")
        else:
            QMessageBox.information(self, "Undo", "No actions to undo.")

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

        # Create QPixmap from QImage
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(pixmap.size() * self.zoom_factor, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.imageLabel.setPixmap(scaled_pixmap)
        self.imageLabel.resize(scaled_pixmap.size())

        # Hide the spinner after processing is done
        self.hideSpinner()

        # Prepare metadata with safeguards
        metadata = {
            'file_path': self.filename if self.filename else "Processed Image",
            'original_header': self.original_header if self.original_header else {},
            'bit_depth': "Unknown",  # Update if bit_depth is available
            'is_mono': self.is_mono,
            'processing_parameters': {
                'target_median': self.medianSlider.value() / 100.0,
                'linked_stretch': self.linkedCheckBox.isChecked(),
                'normalize_image': self.normalizeCheckBox.isChecked(),
                'curves_adjustment': self.curvesCheckBox.isChecked(),
                'curves_boost': self.curvesBoostSlider.value() / 100.0
            },
            'processing_timestamp': datetime.now().isoformat(),
            'source_images': {
                'Original': self.filename if self.filename else "Not Provided"
            }
        }

        # Update ImageManager with the new processed image
        if self.image_manager:
            try:
                self.image_manager.update_image(updated_image=self.stretched_image, metadata=metadata)
                print("StatisticalStretchTab: Processed image stored in ImageManager.")
            except Exception as e:
                print(f"Error updating ImageManager: {e}")
                QMessageBox.critical(self, "Error", f"Failed to update ImageManager:\n{e}")
        else:
            print("ImageManager is not initialized.")
            QMessageBox.warning(self, "Warning", "ImageManager is not initialized. Cannot store the processed image.")


    def showSpinner(self):
        self.spinnerLabel.show()
        self.spinnerMovie.start()

    def hideSpinner(self):
        self.spinnerLabel.hide()
        self.spinnerMovie.stop()    

    def zoom_in(self):
        if self.current_pixmap is not None:
            self.zoom_factor *= 1.2
            self.apply_zoom()
        else:
            print("No image available to zoom in.")
            QMessageBox.warning(self, "Warning", "No image available to zoom in.")

    def zoom_out(self):
        if self.current_pixmap is not None:
            self.zoom_factor /= 1.2
            self.apply_zoom()
        else:
            print("No image available to zoom out.")
            QMessageBox.warning(self, "Warning", "No image available to zoom out.")

    def fit_to_preview(self):
        """Adjust the zoom factor so that the image's width fits within the preview area's width."""
        if self.current_pixmap is not None:
            # Get the width of the scroll area's viewport (preview area)
            preview_width = self.scrollArea.viewport().width()
            
            # Get the original image width from the pixmap
            image_width = self.current_pixmap.width()
            
            # Calculate the required zoom factor to fit the image's width into the preview area
            new_zoom_factor = preview_width / image_width
            
            # Update the zoom factor
            self.zoom_factor = new_zoom_factor
            
            # Apply the new zoom factor to update the display
            self.apply_zoom()
        else:
            print("No image loaded. Cannot fit to preview.")
            QMessageBox.warning(self, "Warning", "No image loaded. Cannot fit to preview.")

    def apply_zoom(self):
        """Apply the current zoom level to the stored pixmap and update the display."""
        if self.current_pixmap is not None:
            scaled_pixmap = self.current_pixmap.scaled(
                self.current_pixmap.size() * self.zoom_factor, 
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            )
            self.imageLabel.setPixmap(scaled_pixmap)
            self.imageLabel.resize(scaled_pixmap.size())
        else:
            print("No pixmap available to apply zoom.")
            QMessageBox.warning(self, "Warning", "No pixmap available to apply zoom.")

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
    def __init__(self, image_manager):
        super().__init__()
        self.image_manager = image_manager  # Store the ImageManager instance
        self.initUI()
        
        # Connect to ImageManager's image_changed signal
        self.image_manager.image_changed.connect(self.on_image_changed)
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
        self.current_pixmap = None  # **New Attribute**

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

        self.fileLabel = QLabel('', self)
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

        # **Create a horizontal layout for Refresh Preview, Undo, and Redo buttons**
        action_buttons_layout = QHBoxLayout()

        # Refresh Preview button
        self.refreshButton = QPushButton("Refresh Preview", self)
        self.refreshButton.clicked.connect(self.generatePreview)
        action_buttons_layout.addWidget(self.refreshButton)

        # Undo button with left arrow icon
        self.undoButton = QPushButton("Undo", self)
        undo_icon = self.style().standardIcon(QStyle.SP_ArrowBack)  # Standard left arrow icon
        self.undoButton.setIcon(undo_icon)
        self.undoButton.clicked.connect(self.undoAction)
        self.undoButton.setEnabled(False)  # Disabled by default
        action_buttons_layout.addWidget(self.undoButton)

        # Redo button with right arrow icon
        self.redoButton = QPushButton("Redo", self)
        redo_icon = self.style().standardIcon(QStyle.SP_ArrowForward)  # Standard right arrow icon
        self.redoButton.setIcon(redo_icon)
        self.redoButton.clicked.connect(self.redoAction)
        self.redoButton.setEnabled(False)  # Disabled by default
        action_buttons_layout.addWidget(self.redoButton)

        # Add the horizontal layout to the left layout
        left_layout.addLayout(action_buttons_layout)

        # Progress indicator (spinner) label
        self.spinnerLabel = QLabel(self)
        self.spinnerLabel.setAlignment(Qt.AlignCenter)
        # Use the resource path function to access the GIF
        self.spinnerMovie = QMovie(resource_path("spinner.gif"))  # Updated path
        self.spinnerLabel.setMovie(self.spinnerMovie)
        self.spinnerLabel.hide()  # Hide spinner by default
        left_layout.addWidget(self.spinnerLabel)

        # **Remove Zoom Buttons from Left Panel**
        # Comment out or remove the existing zoom buttons in the left panel
        # zoom_layout = QHBoxLayout()
        # self.zoomInButton = QPushButton('Zoom In', self)
        # self.zoomInButton.clicked.connect(self.zoom_in)
        # zoom_layout.addWidget(self.zoomInButton)
        #
        # self.zoomOutButton = QPushButton('Zoom Out', self)
        # self.zoomOutButton.clicked.connect(self.zoom_out)
        # zoom_layout.addWidget(self.zoomOutButton)
        # left_layout.addLayout(zoom_layout)

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
        main_layout.addWidget(left_widget)

        # **Create Right Panel Layout**
        right_widget = QWidget(self)
        right_layout = QVBoxLayout(right_widget)

        # Zoom controls

        zoom_layout = QHBoxLayout()
        zoom_in_button = QPushButton("Zoom In")
        zoom_in_button.clicked.connect(self.zoom_in)
        zoom_layout.addWidget(zoom_in_button)

        zoom_out_button = QPushButton("Zoom Out")
        zoom_out_button.clicked.connect(self.zoom_out)
        zoom_layout.addWidget(zoom_out_button)

        # **Add "Fit to Preview" Button**
        self.fitToPreviewButton = QPushButton("Fit to Preview")
        self.fitToPreviewButton.clicked.connect(self.fit_to_preview)
        zoom_layout.addWidget(self.fitToPreviewButton)

        right_layout.addLayout(zoom_layout)

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

        right_layout.addWidget(self.scrollArea)

        # Add the right widget to the main layout
        main_layout.addWidget(right_widget)

        self.setLayout(main_layout)
        self.scrollArea.viewport().setMouseTracking(True)
        self.scrollArea.viewport().installEventFilter(self)

    def saveImage(self):
        # Use the processed/stretched image for saving
        if self.preview_image is not None:
            # Pre-populate the save dialog with the original image name
            base_name = os.path.basename(self.filename) if self.filename else "stretched_image"
            default_save_name = os.path.splitext(base_name)[0] + '_stretched.tif'
            original_dir = os.path.dirname(self.filename) if self.filename else os.getcwd()

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
                        save_image(
                            self.preview_image, 
                            save_filename, 
                            original_format, 
                            bit_depth, 
                            self.original_header, 
                            self.is_mono
                        )
                        self.fileLabel.setText(f'Image saved as: {save_filename}')
                    else:
                        self.fileLabel.setText('Save canceled.')
                else:
                    # For non-TIFF/FITS formats, save directly without bit depth selection
                    save_image(
                        self.preview_image, 
                        save_filename, 
                        original_format
                    )
                    self.fileLabel.setText(f'Image saved as: {save_filename}')
            else:
                self.fileLabel.setText('Save canceled.')
        else:
            self.fileLabel.setText('No stretched image to save. Please generate a preview first.')


    def undoAction(self):
        if self.image_manager and self.image_manager.can_undo():
            try:
                # Perform the undo operation
                self.image_manager.undo()
                print("StarStretchTab: Undo performed.")
            except Exception as e:
                print(f"Error performing undo: {e}")
                QMessageBox.critical(self, "Error", f"Failed to perform undo:\n{e}")
        else:
            QMessageBox.information(self, "Info", "Nothing to undo.")
            print("StarStretchTab: No actions to undo.")

        # Update the state of the Undo and Redo buttons
        if self.image_manager:
            self.undoButton.setEnabled(self.image_manager.can_undo())
            self.redoButton.setEnabled(self.image_manager.can_redo())

    def redoAction(self):
        if self.image_manager and self.image_manager.can_redo():
            try:
                # Perform the redo operation
                self.image_manager.redo()
                print("StarStretchTab: Redo performed.")
            except Exception as e:
                print(f"Error performing redo: {e}")
                QMessageBox.critical(self, "Error", f"Failed to perform redo:\n{e}")
        else:
            QMessageBox.information(self, "Info", "Nothing to redo.")
            print("StarStretchTab: No actions to redo.")

        # Update the state of the Undo and Redo buttons
        if self.image_manager:
            self.undoButton.setEnabled(self.image_manager.can_undo())
            self.redoButton.setEnabled(self.image_manager.can_redo())

    def on_image_changed(self, slot, image, metadata):
        """
        Slot to handle image changes from ImageManager.
        Updates the display if the current slot is affected.
        """
        if slot == self.image_manager.current_slot:
            # Ensure the image is a numpy array before proceeding
            if not isinstance(image, np.ndarray):
                image = np.array(image)  # Convert to numpy array if necessary
            
            self.image = image  # Set the original image
            self.preview_image = None  # Reset the preview image
            self.original_header = metadata.get('original_header', None)
            self.is_mono = metadata.get('is_mono', False)
            self.filename = metadata.get('file_path', self.filename)

            # Update the image display
            self.updateImageDisplay()

            print(f"StarStretchTab: Image updated from ImageManager slot {slot}.")

            # **Update Undo and Redo Button States**
            if self.image_manager:
                self.undoButton.setEnabled(self.image_manager.can_undo())
                self.redoButton.setEnabled(self.image_manager.can_redo())



    def updateImageDisplay(self):
        if self.image is not None:
            # Prepare the image for display by normalizing and converting to uint8
            display_image = (self.image * 255).astype(np.uint8)
            h, w = display_image.shape[:2]

            if display_image.ndim == 3:  # RGB Image
                # Convert the image to QImage format
                q_image = QImage(display_image.tobytes(), w, h, 3 * w, QImage.Format_RGB888)
            else:  # Grayscale Image
                q_image = QImage(display_image.tobytes(), w, h, w, QImage.Format_Grayscale8)

            # Create a QPixmap from QImage
            pixmap = QPixmap.fromImage(q_image)
            self.current_pixmap = pixmap  # Store the original pixmap for future reference

            # Scale the pixmap based on the zoom factor
            scaled_pixmap = pixmap.scaled(pixmap.size() * self.zoom_factor, Qt.KeepAspectRatio, Qt.SmoothTransformation)

            # Set the pixmap on the image label
            self.imageLabel.setPixmap(scaled_pixmap)
            self.imageLabel.resize(scaled_pixmap.size())  # Resize the label to fit the image
        else:
            # If no image is available, clear the label and show a message
            self.imageLabel.clear()
            self.imageLabel.setText('No image loaded.')


    def selectImage(self):
        selected_file, _ = QFileDialog.getOpenFileName(self, "Select Stars Only Image", "", "Images (*.png *.tif *.tiff *.fits *.fit *.xisf)")
        if selected_file:
            try:
                # Load image with header
                self.image, self.original_header, _, self.is_mono = load_image(selected_file)
                self.filename = selected_file  # Store the selected file path
                self.fileLabel.setText(os.path.basename(selected_file))

                # Push the loaded image to ImageManager so it can be tracked for undo/redo
                metadata = {
                    'file_path': self.filename,
                    'original_header': self.original_header,
                    'bit_depth': 'Unknown',  # You can update this if needed
                    'is_mono': self.is_mono
                }
                self.image_manager.add_image(self.image_manager.current_slot, self.image, metadata)
                print(f"Image {self.filename} pushed to ImageManager.")

                # Update the display with the loaded image (before applying any stretch)
                self.updateImageDisplay()

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
        self.preview_image = stretched_image

        # Update the ImageManager with the new stretched image
        metadata = {
            'file_path': self.filename if self.filename else "Stretched Image",
            'original_header': self.original_header if self.original_header else {},
            'bit_depth': "Unknown",  # Update if bit_depth is available
            'is_mono': self.is_mono,
            'processing_parameters': {
                'stretch_factor': self.stretch_factor,
                'color_boost': self.sat_amount,
                'remove_green': self.scnrCheckBox.isChecked()
            },
            'processing_timestamp': datetime.now().isoformat(),
            'source_images': {
                'Original': self.filename if self.filename else "Not Provided"
            }
        }

        # Update ImageManager with the new processed image
        if self.image_manager:
            try:
                self.image_manager.update_image(updated_image=self.preview_image, metadata=metadata)
                print("StarStretchTab: Processed image stored in ImageManager.")
            except Exception as e:
                print(f"Error updating ImageManager: {e}")
                QMessageBox.critical(self, "Error", f"Failed to update ImageManager:\n{e}")
        else:
            print("ImageManager is not initialized.")
            QMessageBox.warning(self, "Warning", "ImageManager is not initialized. Cannot store the processed image.")

        # Update the preview once the processing thread emits the result
        preview_image = (stretched_image * 255).astype(np.uint8)
        h, w = preview_image.shape[:2]
        if preview_image.ndim == 3:
            q_image = QImage(preview_image.data, w, h, 3 * w, QImage.Format_RGB888)
        else:
            q_image = QImage(preview_image.data, w, h, w, QImage.Format_Grayscale8)

        pixmap = QPixmap.fromImage(q_image)
        self.current_pixmap = pixmap  # **Store the original pixmap**
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
        if self.current_pixmap is not None:
            self.zoom_factor *= 1.2
            self.apply_zoom()
        else:
            print("No image available to zoom in.")
            QMessageBox.warning(self, "Warning", "No image available to zoom in.")

    def zoom_out(self):
        if self.current_pixmap is not None:
            self.zoom_factor /= 1.2
            self.apply_zoom()
        else:
            print("No image available to zoom out.")
            QMessageBox.warning(self, "Warning", "No image available to zoom out.")

    def fit_to_preview(self):
        """Adjust the zoom factor so that the image's width fits within the preview area's width."""
        if self.current_pixmap is not None:
            # Get the width of the scroll area's viewport (preview area)
            preview_width = self.scrollArea.viewport().width()
            
            # Get the original image width from the pixmap
            image_width = self.current_pixmap.width()
            
            # Calculate the required zoom factor to fit the image's width into the preview area
            new_zoom_factor = preview_width / image_width
            
            # Update the zoom factor
            self.zoom_factor = new_zoom_factor
            
            # Apply the new zoom factor to update the display
            self.apply_zoom()
        else:
            print("No image loaded. Cannot fit to preview.")
            QMessageBox.warning(self, "Warning", "No image loaded. Cannot fit to preview.")

    def apply_zoom(self):
        """Apply the current zoom level to the stored pixmap and update the display."""
        if self.current_pixmap is not None:
            scaled_pixmap = self.current_pixmap.scaled(
                self.current_pixmap.size() * self.zoom_factor, 
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            )
            self.imageLabel.setPixmap(scaled_pixmap)
            self.imageLabel.resize(scaled_pixmap.size())
        else:
            print("No pixmap available to apply zoom.")
            QMessageBox.warning(self, "Warning", "No pixmap available to apply zoom.")


    def applyStretch(self):
        if self.image is not None and self.image.size > 0:
            print(f"Applying stretch: {self.stretch_factor}, Color Boost: {self.sat_amount:.2f}, SCNR: {self.scnrCheckBox.isChecked()}")
            self.generatePreview()

class FullCurvesTab(QWidget):
    def __init__(self, image_manager=None):
        super().__init__()
        self.initUI()
        self.image = None
        self.image_manager = image_manager
        self.filename = None
        self.original_image = None  # Reference to the original image
        self.preview_image = None   # Reference to the preview image        
        self.zoom_factor = 1.0
        self.original_header = None
        self.bit_depth = None
        self.is_mono = None
        self.curve_mode = "K (Brightness)"  # Default curve mode
        self.current_lut = np.linspace(0, 255, 256, dtype=np.uint8)  # Initialize with identity LUT

        # Initialize the Undo stack with a limited size
        self.undo_stack = []
        self.max_undo = 10  # Maximum number of undo steps        

        # Precompute transformation matrices
        self.M = np.array([
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041]
        ], dtype=np.float32)

        self.M_inv = np.array([
            [ 3.2404542, -1.5371385, -0.4985314],
            [-0.9692660,  1.8760108,  0.0415560],
            [ 0.0556434, -0.2040259,  1.0572252]
        ], dtype=np.float32)   

        if self.image_manager:
            # Connect to ImageManager's image_changed signal
            self.image_manager.image_changed.connect(self.on_image_changed)             

    def initUI(self):
        main_layout = QHBoxLayout()

        # Left column for controls
        left_widget = QWidget(self)
        left_layout = QVBoxLayout(left_widget)
        left_widget.setFixedWidth(400)

        # Load button
        self.fileButton = QPushButton('Load Image', self)
        self.fileButton.setIcon(self.style().standardIcon(QStyle.SP_DirOpenIcon))
        self.fileButton.clicked.connect(self.openFileDialog)
        left_layout.addWidget(self.fileButton)

        # File label
        self.fileLabel = QLabel('', self)
        left_layout.addWidget(self.fileLabel)

        # Curve Mode Selection
        self.curveModeLabel = QLabel('Select Curve Mode:', self)
        left_layout.addWidget(self.curveModeLabel)

        self.curveModeGroup = QButtonGroup(self)
        curve_modes = [
            ('K (Brightness)', 0, 0),  # Text, row, column
            ('R', 1, 0),
            ('G', 2, 0),
            ('B', 3, 0),
            ('L*', 0, 1),
            ('a*', 1, 1),
            ('b*', 2, 1),
            ('Chroma', 0, 2),
            ('Saturation', 1, 2)
        ]

        curve_mode_layout = QGridLayout()

        # Connect all buttons to set_curve_mode
        for mode, row, col in curve_modes:
            button = QRadioButton(mode, self)
            if mode == "K (Brightness)":
                button.setChecked(True)  # Default selection
            button.toggled.connect(self.set_curve_mode)  # Update curve_mode on toggle
            self.curveModeGroup.addButton(button)
            curve_mode_layout.addWidget(button, row, col)

        left_layout.addLayout(curve_mode_layout)
        self.set_curve_mode()

        # Curve editor placeholder
        self.curveEditor = CurveEditor(self)
        left_layout.addWidget(self.curveEditor)

        # Connect the CurveEditor preview callback
        self.curveEditor.setPreviewCallback(lambda lut: self.updatePreviewLUT(lut, self.curve_mode))

        self.statusLabel = QLabel('X:0 Y:0', self)
        left_layout.addWidget(self.statusLabel)

        self.applySourceGroup = QButtonGroup(self)

        self.applyOriginalRadio = QRadioButton("Original", self)
        self.applyOriginalRadio.setChecked(True)
        self.applySourceGroup.addButton(self.applyOriginalRadio)

        self.applyCurrentRadio = QRadioButton("Current", self)
        self.applySourceGroup.addButton(self.applyCurrentRadio)

        source_layout = QHBoxLayout()
        source_layout.addWidget(QLabel("Apply to:", self))
        source_layout.addWidget(self.applyOriginalRadio)
        source_layout.addWidget(self.applyCurrentRadio)
        left_layout.addLayout(source_layout)

        # Horizontal layout for Apply, Undo, and Reset buttons
        button_layout = QHBoxLayout()

        # Apply Curve Button
        self.applyButton = QPushButton('Apply Curve', self)
        self.applyButton.setIcon(self.style().standardIcon(QStyle.SP_DialogApplyButton))
        self.applyButton.clicked.connect(self.startProcessing)
        button_layout.addWidget(self.applyButton)

        # Undo Curve Button
        self.undoButton = QPushButton('Undo', self)
        self.undoButton.setIcon(self.style().standardIcon(QStyle.SP_ArrowBack))
        self.undoButton.setEnabled(False)  # Initially disabled
        self.undoButton.clicked.connect(self.undo)
        button_layout.addWidget(self.undoButton)

        # Reset Curve Button as a small tool button with an icon
        self.resetCurveButton = QToolButton(self)
        self.resetCurveButton.setIcon(self.style().standardIcon(QStyle.SP_BrowserReload))  # Provide a suitable icon
        self.resetCurveButton.setToolTip("Reset Curve")
        # Set a small icon size if needed
        self.resetCurveButton.setIconSize(QSize(16,16))

        # Optionally, if you want the reset button even smaller, you can also adjust its size:
        # self.resetCurveButton.setFixedSize(24, 24)

        # Connect the clicked signal to the resetCurve method
        self.resetCurveButton.clicked.connect(self.resetCurve)
        button_layout.addWidget(self.resetCurveButton)

        # Add the horizontal layout with buttons to the main left layout
        left_layout.addLayout(button_layout)

        # **Remove Zoom Buttons from Left Panel**
        # Commented out to move to the right panel
        # zoom_layout = QHBoxLayout()
        # self.zoomInButton = QPushButton('Zoom In', self)
        # self.zoomInButton.clicked.connect(self.zoom_in)
        # zoom_layout.addWidget(self.zoomInButton)

        # self.zoomOutButton = QPushButton('Zoom Out', self)
        # self.zoomOutButton.clicked.connect(self.zoom_out)
        # zoom_layout.addWidget(self.zoomOutButton)

        # left_layout.addLayout(zoom_layout)


        # **Add Spinner Label**
        self.spinnerLabel = QLabel(self)
        self.spinnerLabel.setAlignment(Qt.AlignCenter)
        self.spinnerMovie = QMovie("spinner.gif")  # Ensure spinner.gif exists in your project directory
        self.spinnerLabel.setMovie(self.spinnerMovie)
        self.spinnerLabel.hide()  # Initially hidden
        left_layout.addWidget(self.spinnerLabel)

        # Spacer
        left_layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

        # Save button
        self.saveButton = QPushButton('Save Image', self)
        self.saveButton.setIcon(self.style().standardIcon(QStyle.SP_DialogSaveButton))
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

        # Add the left widget to the main layout
        main_layout.addWidget(left_widget)

        # **Create Right Panel Layout**
        right_widget = QWidget(self)
        right_layout = QVBoxLayout(right_widget)

        # Zoom controls

        zoom_layout = QHBoxLayout()
        zoom_in_button = QPushButton("Zoom In")
        zoom_in_button.clicked.connect(self.zoom_in)
        zoom_layout.addWidget(zoom_in_button)

        zoom_out_button = QPushButton("Zoom Out")
        zoom_out_button.clicked.connect(self.zoom_out)
        zoom_layout.addWidget(zoom_out_button)

        # **Add "Fit to Preview" Button**
        self.fitToPreviewButton = QPushButton("Fit to Preview")
        self.fitToPreviewButton.clicked.connect(self.fit_to_preview)
        zoom_layout.addWidget(self.fitToPreviewButton)

        right_layout.addLayout(zoom_layout)

        # Right side for the preview inside a QScrollArea
        self.scrollArea = QScrollArea(self)
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scrollArea.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        # QLabel for the image preview
        self.imageLabel = ImageLabel(self)
        self.imageLabel.setAlignment(Qt.AlignCenter)
        self.scrollArea.setWidget(self.imageLabel)
        self.scrollArea.setMinimumSize(400, 400)
        self.scrollArea.setWidgetResizable(True)
        self.imageLabel.mouseMoved.connect(self.handleImageMouseMove)

        right_layout.addWidget(self.scrollArea)

        # Add the right widget to the main layout
        main_layout.addWidget(right_widget)

        self.setLayout(main_layout)
        self.zoom_factor = 1.0
        self.scrollArea.viewport().setMouseTracking(True)
        self.scrollArea.viewport().installEventFilter(self)
        self.dragging = False
        self.last_pos = QPoint()

    # -----------------------------
    # Spinner Control Methods
    # -----------------------------
    def showSpinner(self):
        """Show the spinner animation."""
        self.spinnerLabel.show()
        self.spinnerMovie.start()

    def hideSpinner(self):
        """Hide the spinner animation."""
        self.spinnerLabel.hide()
        self.spinnerMovie.stop()

    def set_curve_mode(self):
        selected_button = self.curveModeGroup.checkedButton()
        if selected_button:
            self.curve_mode = selected_button.text()
            # Assuming you have the current LUT, update the preview
            if hasattr(self, 'current_lut'):
                self.updatePreviewLUT(self.current_lut, self.curve_mode)

    def get_visible_region(self):
        """Retrieve the coordinates of the visible region in the image."""
        viewport = self.scrollArea.viewport()
        # Top-left corner of the visible area
        x = self.scrollArea.horizontalScrollBar().value()
        y = self.scrollArea.verticalScrollBar().value()
        # Size of the visible area
        w = viewport.width()
        h = viewport.height()
        return x, y, w, h


    def updatePreviewLUT(self, lut, curve_mode):
        """Apply the 8-bit LUT to the preview image for real-time updates."""
        if self.original_image is None:
            return

        # Determine the visible region
        x, y, w, h = self.get_visible_region()

        # Ensure the region is within image bounds
        x_end = min(x + w, self.original_image.shape[1])
        y_end = min(y + h, self.original_image.shape[0])

        # Determine the base image based on apply mode
        if self.applyOriginalRadio.isChecked():
            base_image = self.original_image
        else:
            base_image = self.image

        # Extract the visible region from the base image
        visible_region = base_image[y:y_end, x:x_end]

        # Create an 8-bit version of the visible region for faster processing
        image_8bit = (visible_region * 255).astype(np.uint8)

        if image_8bit.ndim == 3:  # RGB image
            adjusted_image = image_8bit.copy()

            if curve_mode == "K (Brightness)":
                # Apply LUT to all channels equally (Brightness)
                for channel in range(3):
                    adjusted_image[:, :, channel] = np.take(lut, image_8bit[:, :, channel])

            elif curve_mode in ["R", "G", "B"]:
                # Apply LUT to a single channel
                channel_index = {"R": 0, "G": 1, "B": 2}[curve_mode]
                adjusted_image[:, :, channel_index] = np.take(lut, image_8bit[:, :, channel_index])

            elif curve_mode in ["L*", "a*", "b*"]:
                # Manual RGB to Lab Conversion
                # Use precomputed transformation matrices
                M = self.M
                M_inv = self.M_inv

                # Normalize RGB to [0,1]
                rgb = image_8bit.astype(np.float32) / 255.0

                # Convert RGB to XYZ
                xyz = np.dot(rgb.reshape(-1, 3), M.T).reshape(rgb.shape)

                # Reference white point (D65)
                Xn, Yn, Zn = 0.95047, 1.00000, 1.08883

                # Normalize XYZ
                X = xyz[:, :, 0] / Xn
                Y = xyz[:, :, 1] / Yn
                Z = xyz[:, :, 2] / Zn

                # Define the f(t) function
                delta = 6 / 29
                def f(t):
                    return np.where(t > delta**3, np.cbrt(t), (t / (3 * delta**2)) + (4 / 29))

                fx = f(X)
                fy = f(Y)
                fz = f(Z)

                # Compute L*, a*, b*
                L = 116 * fy - 16
                a = 500 * (fx - fy)
                b = 200 * (fy - fz)

                # Apply LUT to the respective channel
                if curve_mode == "L*":
                    # L* typically ranges from 0 to 100
                    L_normalized = np.clip(L / 100.0, 0, 1)  # Normalize to [0,1]
                    L_lut_indices = (L_normalized * 255).astype(np.uint8)
                    L_adjusted = lut[L_lut_indices].astype(np.float32) * 100.0 / 255.0  # Scale back to [0,100]
                    L = L_adjusted

                elif curve_mode == "a*":
                    # a* typically ranges from -128 to +127
                    a_normalized = np.clip((a + 128.0) / 255.0, 0, 1)  # Normalize to [0,1]
                    a_lut_indices = (a_normalized * 255).astype(np.uint8)
                    a_adjusted = lut[a_lut_indices].astype(np.float32) - 128.0  # Scale back to [-128,127]
                    a = a_adjusted

                elif curve_mode == "b*":
                    # b* typically ranges from -128 to +127
                    b_normalized = np.clip((b + 128.0) / 255.0, 0, 1)  # Normalize to [0,1]
                    b_lut_indices = (b_normalized * 255).astype(np.uint8)
                    b_adjusted = lut[b_lut_indices].astype(np.float32) - 128.0  # Scale back to [-128,127]
                    b = b_adjusted

                # Update Lab channels
                lab_new = np.stack([L, a, b], axis=2)

                # Convert Lab back to XYZ
                fy_new = (lab_new[:, :, 0] + 16) / 116
                fx_new = fy_new + lab_new[:, :, 1] / 500
                fz_new = fy_new - lab_new[:, :, 2] / 200

                def f_inv(ft):
                    return np.where(ft > delta, ft**3, 3 * delta**2 * (ft - 4 / 29))

                X_new = f_inv(fx_new) * Xn
                Y_new = f_inv(fy_new) * Yn
                Z_new = f_inv(fz_new) * Zn

                # Stack XYZ channels
                xyz_new = np.stack([X_new, Y_new, Z_new], axis=2)

                # Convert XYZ back to RGB
                rgb_new = np.dot(xyz_new.reshape(-1, 3), M_inv.T).reshape(xyz_new.shape)

                # Clip RGB to [0,1]
                rgb_new = np.clip(rgb_new, 0, 1)

                # Convert back to 8-bit
                adjusted_image = (rgb_new * 255).astype(np.uint8)

            elif curve_mode == "Chroma":
                # === Manual RGB to Lab Conversion ===
                # Use precomputed transformation matrices
                M = self.M
                M_inv = self.M_inv

                # Normalize RGB to [0,1]
                rgb = image_8bit.astype(np.float32) / 255.0

                # Convert RGB to XYZ
                xyz = np.dot(rgb.reshape(-1, 3), M.T).reshape(rgb.shape)

                # Reference white point (D65)
                Xn, Yn, Zn = 0.95047, 1.00000, 1.08883

                # Normalize XYZ
                X = xyz[:, :, 0] / Xn
                Y = xyz[:, :, 1] / Yn
                Z = xyz[:, :, 2] / Zn

                # Define the f(t) function
                delta = 6 / 29
                def f(t):
                    return np.where(t > delta**3, np.cbrt(t), (t / (3 * delta**2)) + (4 / 29))

                fx = f(X)
                fy = f(Y)
                fz = f(Z)

                # Compute L*, a*, b*
                L = 116 * fy - 16
                a = 500 * (fx - fy)
                b = 200 * (fy - fz)

                # Compute Chroma
                chroma = np.sqrt(a**2 + b**2)

                # Define a fixed maximum Chroma for normalization to prevent over-scaling
                fixed_max_chroma = 200.0  # Adjust this value as needed

                # Normalize Chroma to [0,1] using fixed_max_chroma
                chroma_norm = np.clip(chroma / fixed_max_chroma, 0, 1)

                # Apply LUT to Chroma
                chroma_lut_indices = (chroma_norm * 255).astype(np.uint8)
                chroma_adjusted = lut[chroma_lut_indices].astype(np.float32)  # Ensure float32

                # Compute scaling factor, avoiding division by zero
                scale = np.ones_like(chroma_adjusted, dtype=np.float32)
                mask = chroma > 0
                scale[mask] = chroma_adjusted[mask] / chroma[mask]

                # Scale a* and b* channels
                a_new = a * scale
                b_new = b * scale

                # Update Lab channels
                lab_new = np.stack([L, a_new, b_new], axis=2)

                # Convert Lab back to XYZ
                fy_new = (lab_new[:, :, 0] + 16) / 116
                fx_new = fy_new + lab_new[:, :, 1] / 500
                fz_new = fy_new - lab_new[:, :, 2] / 200

                def f_inv(ft):
                    return np.where(ft > delta, ft**3, 3 * delta**2 * (ft - 4 / 29))

                X_new = f_inv(fx_new) * Xn
                Y_new = f_inv(fy_new) * Yn
                Z_new = f_inv(fz_new) * Zn

                # Stack XYZ channels
                xyz_new = np.stack([X_new, Y_new, Z_new], axis=2)

                # Convert XYZ back to RGB
                rgb_new = np.dot(xyz_new.reshape(-1, 3), M_inv.T).reshape(xyz_new.shape)

                # Clip RGB to [0,1]
                rgb_new = np.clip(rgb_new, 0, 1)

                # Convert back to 8-bit
                adjusted_image = (rgb_new * 255).astype(np.uint8)

            elif curve_mode == "Saturation":
                # === Manual RGB to HSV Conversion ===
                # Normalize RGB to [0,1]
                rgb = image_8bit.astype(np.float32) / 255.0

                # Split channels
                R, G, B = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

                # Compute Cmax, Cmin, Delta
                Cmax = np.maximum(np.maximum(R, G), B)
                Cmin = np.minimum(np.minimum(R, G), B)
                Delta = Cmax - Cmin

                # Initialize Hue (H), Saturation (S), and Value (V)
                H = np.zeros_like(Cmax)
                S = np.zeros_like(Cmax)
                V = Cmax.copy()

                # Compute Hue (H)
                mask = Delta != 0
                # Avoid division by zero
                H[mask & (Cmax == R)] = ((G[mask & (Cmax == R)] - B[mask & (Cmax == R)]) / Delta[mask & (Cmax == R)]) % 6
                H[mask & (Cmax == G)] = ((B[mask & (Cmax == G)] - R[mask & (Cmax == G)]) / Delta[mask & (Cmax == G)]) + 2
                H[mask & (Cmax == B)] = ((R[mask & (Cmax == B)] - G[mask & (Cmax == B)]) / Delta[mask & (Cmax == B)]) + 4
                H = H / 6.0  # Normalize Hue to [0,1]

                # Compute Saturation (S)
                S[Cmax != 0] = Delta[Cmax != 0] / Cmax[Cmax != 0]

                # Apply LUT to Saturation (S) channel
                S_normalized = np.clip(S, 0, 1)  # Ensure S is within [0,1]
                S_lut_indices = (S_normalized * 255).astype(np.uint8)
                S_adjusted = lut[S_lut_indices].astype(np.float32) / 255.0  # Normalize back to [0,1]
                S = S_adjusted

                # Convert HSV back to RGB
                C = V * S
                X = C * (1 - np.abs((H * 6) % 2 - 1))
                m = V - C

                # Initialize RGB channels
                R_new = np.zeros_like(R)
                G_new = np.zeros_like(G)
                B_new = np.zeros_like(B)

                # Define masks for different sectors of Hue
                mask0 = (H >= 0) & (H < 1/6)
                mask1 = (H >= 1/6) & (H < 2/6)
                mask2 = (H >= 2/6) & (H < 3/6)
                mask3 = (H >= 3/6) & (H < 4/6)
                mask4 = (H >= 4/6) & (H < 5/6)
                mask5 = (H >= 5/6) & (H < 1)

                # Assign RGB values based on the sector of Hue
                R_new[mask0] = C[mask0]
                G_new[mask0] = X[mask0]
                B_new[mask0] = 0

                R_new[mask1] = X[mask1]
                G_new[mask1] = C[mask1]
                B_new[mask1] = 0

                R_new[mask2] = 0
                G_new[mask2] = C[mask2]
                B_new[mask2] = X[mask2]

                R_new[mask3] = 0
                G_new[mask3] = X[mask3]
                B_new[mask3] = C[mask3]

                R_new[mask4] = X[mask4]
                G_new[mask4] = 0
                B_new[mask4] = C[mask4]

                R_new[mask5] = C[mask5]
                G_new[mask5] = 0
                B_new[mask5] = X[mask5]

                # Add m to match the Value (V)
                R_new += m
                G_new += m
                B_new += m

                # Stack the channels back together
                rgb_new = np.stack([R_new, G_new, B_new], axis=2)

                # Clip RGB to [0,1] to maintain valid color ranges
                rgb_new = np.clip(rgb_new, 0, 1)

                # Convert back to 8-bit
                adjusted_image = (rgb_new * 255).astype(np.uint8)

            else:
                # Unsupported curve mode
                print(f"Unsupported curve mode: {curve_mode}")
                return

        else:  # Grayscale image
            # For grayscale images, apply LUT directly
            adjusted_image = np.take(lut, image_8bit)

        # Copy the adjusted region back to the full image
        full_adjusted_image = self.image.copy()
        full_adjusted_image[y:y_end, x:x_end] = adjusted_image / 255.0  # Assuming self.image is float [0,1]

        # Convert the full adjusted image to 8-bit for display
        full_adjusted_8bit = (full_adjusted_image * 255).astype(np.uint8)

        # Convert to QImage
        if full_adjusted_8bit.ndim == 3:
            bytes_per_line = 3 * full_adjusted_8bit.shape[1]
            q_image_full = QImage(full_adjusted_8bit.tobytes(), full_adjusted_8bit.shape[1], full_adjusted_8bit.shape[0], bytes_per_line, QImage.Format_RGB888)
        else:
            bytes_per_line = full_adjusted_8bit.shape[1]
            q_image_full = QImage(full_adjusted_8bit.tobytes(), full_adjusted_8bit.shape[1], full_adjusted_8bit.shape[0], bytes_per_line, QImage.Format_Grayscale8)

        # Create QPixmap from QImage
        pixmap_full = QPixmap.fromImage(q_image_full)

        # Correctly scale the pixmap based on zoom_factor
        scaled_width = int(pixmap_full.width() * self.zoom_factor)
        scaled_height = int(pixmap_full.height() * self.zoom_factor)
        scaled_pixmap = pixmap_full.scaled(
            scaled_width,
            scaled_height,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        # Set the scaled pixmap to the imageLabel
        self.imageLabel.setPixmap(scaled_pixmap)
        self.imageLabel.resize(scaled_pixmap.size())


    def handleImageMouseMove(self, x, y):
        if self.image is None:
            return

        # Convert from scaled coordinates to original image coords
        # scaled_pixmap was created with: pixmap.scaled(orig_size * zoom_factor)
        # So original coordinate = x/zoom_factor, y/zoom_factor
        # Make sure to also consider the imageLabel size and whether the image is centered.

        # If you have the pixmap stored, you can find original width/height from self.image shape:
        h, w = self.image.shape[:2]

        # Convert mouse coords to image coords
        img_x = int(x / self.zoom_factor)
        img_y = int(y / self.zoom_factor)

        # Ensure within bounds
        if 0 <= img_x < w and 0 <= img_y < h:
            pixel_value = self.image[img_y, img_x]
            if self.image.ndim == 3:
                # RGB pixel
                r, g, b = pixel_value
                text = f"X:{img_x} Y:{img_y} R:{r:.3f} G:{g:.3f} B:{b:.3f}"
            else:
                # Grayscale pixel
                text = f"X:{img_x} Y:{img_y} Val:{pixel_value:.3f}"
            # Update a status label or print it
            self.statusLabel.setText(text)  # For example, reuse fileLabel or add a dedicated status label.

    def startProcessing(self):
        if self.original_image is None:
            QMessageBox.warning(self, "Warning", "No image loaded to apply curve.")
            return

        curve_mode = self.curveModeGroup.checkedButton().text()
        curve_func = self.curveEditor.getCurveFunction()

        # Determine source image based on user choice
        if self.applyOriginalRadio.isChecked():
            source_image = self.original_image.copy()
        else:
            source_image = self.image.copy()

        # Push the current image to the undo stack before modifying
        self.pushUndo(self.image.copy())

        # Show the spinner before starting processing
        self.showSpinner()

        # Initialize and start the processing thread
        self.processing_thread = FullCurvesProcessingThread(source_image, curve_mode, curve_func)
        self.processing_thread.result_ready.connect(self.finishProcessing)
        self.processing_thread.start()
        print("Started FullCurvesProcessingThread.")

    def finishProcessing(self, adjusted_image):
        # Hide the spinner after processing is done
        self.hideSpinner()

        # Update the image state based on apply mode without modifying original_image
        self.image = adjusted_image.copy()

        # Update the preview to reflect the applied changes
        self.preview_image = self.image.copy()
        self.updatePreview(self.preview_image)

        # Optionally, emit a signal to ImageManager if needed
        if self.image_manager:
            metadata = {
                'file_path': self.loaded_image_path,  # Update as needed
                'original_header': self.original_header,
                'bit_depth': self.bit_depth,
                'is_mono': self.is_mono
            }
            self.image_manager.update_image(updated_image=self.image, metadata=metadata)
            print("FullCurvesTab: Image updated in ImageManager after processing.")

    def pushUndo(self, image_state):
        """Push the current image state onto the undo stack."""
        if len(self.undo_stack) >= self.max_undo:
            # Remove the oldest state to maintain the stack size
            self.undo_stack.pop(0)
        self.undo_stack.append(image_state)
        self.updateUndoButtonState()

    def updateUndoButtonState(self):
        """Enable or disable the Undo button based on the undo stack."""
        if hasattr(self, 'undoButton'):
            self.undoButton.setEnabled(len(self.undo_stack) > 0)

    def undo(self):
        """Revert the image to the last state in the undo stack."""
        if not self.undo_stack:
            QMessageBox.information(self, "Undo", "No actions to undo.")
            return

        # Pop the last state from the stack
        last_state = self.undo_stack.pop()

        # Update ImageManager with the previous image state
        if self.image_manager:
            metadata = {
                'file_path': self.loaded_image_path,  # Update as needed
                'original_header': self.original_header,
                'bit_depth': self.bit_depth,
                'is_mono': self.is_mono
            }
            self.image_manager.update_image(updated_image=last_state)
            print("Undo: Image reverted in ImageManager.")

        # Update the Undo button state
        self.updateUndoButtonState()


    def resetCurve(self):
        # Reset the curve in the curve editor
        self.curveEditor.initCurve()

        # Reset the image to the original image
        if self.original_image is not None:
            # Push the current image to the undo stack before resetting
            self.pushUndo(self.image.copy())

            # Update ImageManager with the original image
            if self.image_manager:
                metadata = {
                    'file_path': self.loaded_image_path,  # Update as needed
                    'original_header': self.original_header,
                    'bit_depth': self.bit_depth,
                    'is_mono': self.is_mono
                }
                self.image_manager.update_image(updated_image=self.original_image.copy())
                print("Curve reset: Original image restored in ImageManager.")
        else:
            QMessageBox.warning(self, "Warning", "Original image not loaded.")
            print("Reset Curve called, but original image not loaded.")

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
        try:
            self.loaded_image_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", 
                                            "Images (*.png *.tif *.tiff *.fits *.xisf *.cr2 *.nef *.arw *.dng *.orf *.rw2 *.pef);;All Files (*)")
            if self.loaded_image_path:
                self.fileLabel.setText(self.loaded_image_path)
                self.image, self.original_header, self.bit_depth, self.is_mono = load_image(self.loaded_image_path)
                # Keep a copy of the original image
                self.original_image = self.image.copy()
                # Initialize the preview image as a copy of the original
                self.stretched_image = self.original_image.copy()
                self.updatePreview(self.stretched_image)
                self.applyButton.setEnabled(True)
                self.saveButton.setEnabled(True)
                print(f"Image loaded successfully from {self.loaded_image_path}.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load image: {e}")
            print(f"Failed to load image: {e}")

    def updatePreview(self, preview_image=None):
        """Display the preview_image on the imageLabel."""
        if preview_image is None:
            preview_image = self.preview_image

        if preview_image is not None:
            img = (preview_image * 255).astype(np.uint8)
            h, w = img.shape[:2]

            if img.ndim == 3:
                bytes_per_line = 3 * w
                q_image = QImage(img.tobytes(), w, h, bytes_per_line, QImage.Format_RGB888)
            else:
                bytes_per_line = w
                q_image = QImage(img.tobytes(), w, h, bytes_per_line, QImage.Format_Grayscale8)

            pixmap = QPixmap.fromImage(q_image)
            # Correctly scale the pixmap based on zoom_factor
            scaled_width = int(pixmap.width() * self.zoom_factor)
            scaled_height = int(pixmap.height() * self.zoom_factor)
            scaled_pixmap = pixmap.scaled(
                scaled_width,
                scaled_height,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.imageLabel.setPixmap(scaled_pixmap)
            self.imageLabel.resize(scaled_pixmap.size())

    def on_image_changed(self, slot, image, metadata):
        """
        Slot to handle image changes from ImageManager.
        Updates the display if the current slot is affected.
        """
        if slot == self.image_manager.current_slot:
            # Ensure the image is a numpy array before proceeding
            if not isinstance(image, np.ndarray):
                image = np.array(image)  # Convert to numpy array if necessary
                
            # Set the image and store a copy for later use
            self.loaded_image_path = metadata.get('file_path', None)
            self.image = image
            self.original_image = image.copy()  # Store a copy of the original image
            self.original_header = metadata.get('original_header', None)
            self.bit_depth = metadata.get('bit_depth', None)
            self.is_mono = metadata.get('is_mono', False)
            
            # Save the previous scroll position
            self.previous_scroll_pos = (
                self.scrollArea.horizontalScrollBar().value(),
                self.scrollArea.verticalScrollBar().value()
            )
            
            # Make a copy of the image for stretching
            self.stretched_image = image.copy()

            # Update the UI elements (buttons, etc.)
            self.show_image(image)
            self.update_image_display()

            # Enable or disable buttons based on image processing state
            self.applyButton.setEnabled(True)
            self.saveButton.setEnabled(True)
            self.undoButton.setEnabled(len(self.undo_stack) > 0)

            print(f"FullCurvesTab: Image updated from ImageManager slot {slot}.")




    def show_image(self, image):
        """
        Display the loaded image in the imageLabel.
        """
        try:
            # Normalize image to 0-255 and convert to uint8
            display_image = (image * 255).astype(np.uint8)

            if display_image.ndim == 3 and display_image.shape[2] == 3:
                # RGB Image
                height, width, channels = display_image.shape
                bytes_per_line = 3 * width
                q_image = QImage(display_image.tobytes(), width, height, bytes_per_line, QImage.Format_RGB888)
            elif display_image.ndim == 2:
                # Grayscale Image
                height, width = display_image.shape
                bytes_per_line = width
                q_image = QImage(display_image.tobytes(), width, height, bytes_per_line, QImage.Format_Grayscale8)
            else:
                print("Unsupported image format for display.")
                QMessageBox.critical(self, "Error", "Unsupported image format for display.")
                return

            pixmap = QPixmap.fromImage(q_image)
            scaled_pixmap = pixmap.scaled(
                pixmap.size() * self.zoom_factor,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.imageLabel.setPixmap(scaled_pixmap)
            self.imageLabel.resize(scaled_pixmap.size())

            # Restore the previous scroll position
            if hasattr(self, 'previous_scroll_pos'):
                self.scrollArea.horizontalScrollBar().setValue(self.previous_scroll_pos[0])
                self.scrollArea.verticalScrollBar().setValue(self.previous_scroll_pos[1])

            print("Image displayed successfully.")
        except Exception as e:
            print(f"Error displaying image: {e}")
            QMessageBox.critical(self, "Error", f"Failed to display the image: {e}")


    def update_image_display(self):
        if self.image is not None:
            # Prepare the image for display by normalizing and converting to uint8
            display_image = (self.image * 255).astype(np.uint8)
            h, w = display_image.shape[:2]

            if display_image.ndim == 3:  # RGB Image
                # Convert the image to QImage format
                q_image = QImage(display_image.tobytes(), w, h, 3 * w, QImage.Format_RGB888)
            else:  # Grayscale Image
                q_image = QImage(display_image.tobytes(), w, h, w, QImage.Format_Grayscale8)

            # Create a QPixmap from QImage
            pixmap = QPixmap.fromImage(q_image)
            self.current_pixmap = pixmap  # Store the original pixmap for future reference

            # Scale the pixmap based on the zoom factor
            scaled_pixmap = pixmap.scaled(pixmap.size() * self.zoom_factor, Qt.KeepAspectRatio, Qt.SmoothTransformation)

            # Set the pixmap on the image label
            self.imageLabel.setPixmap(scaled_pixmap)
            self.imageLabel.resize(scaled_pixmap.size())  # Resize the label to fit the image
        else:
            # If no image is available, clear the label and show a message
            self.imageLabel.clear()
            self.imageLabel.setText('No image loaded.')

    def updatePreview(self, preview_image=None):
        # Store the stretched image for saving
        self.preview_image = preview_image
        # Update the ImageManager with the new stretched image
        metadata = {
            'file_path': self.filename if self.filename else "Stretched Image",
            'original_header': self.original_header if self.original_header else {},
            'bit_depth': "Unknown",  # Update if bit_depth is available
            'is_mono': self.is_mono,
            'processing_timestamp': datetime.now().isoformat(),
            'source_images': {
                'Original': self.filename if self.filename else "Not Provided"
            }
        }

        # Update ImageManager with the new processed image
        if self.image_manager:
            try:
                self.image_manager.update_image(updated_image=self.preview_image, metadata=metadata)
                print("StarStretchTab: Processed image stored in ImageManager.")
            except Exception as e:
                print(f"Error updating ImageManager: {e}")
                QMessageBox.critical(self, "Error", f"Failed to update ImageManager:\n{e}")
        else:
            print("ImageManager is not initialized.")
            QMessageBox.warning(self, "Warning", "ImageManager is not initialized. Cannot store the processed image.")

        # Update the preview once the processing thread emits the result
        preview_image = (preview_image * 255).astype(np.uint8)
        h, w = preview_image.shape[:2]
        if preview_image.ndim == 3:
            q_image = QImage(preview_image.data, w, h, 3 * w, QImage.Format_RGB888)
        else:
            q_image = QImage(preview_image.data, w, h, w, QImage.Format_Grayscale8)

        pixmap = QPixmap.fromImage(q_image)
        self.current_pixmap = pixmap  # **Store the original pixmap**
        scaled_pixmap = pixmap.scaled(pixmap.size() * self.zoom_factor, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.imageLabel.setPixmap(scaled_pixmap)
        self.imageLabel.resize(scaled_pixmap.size())

        # Hide the spinner after processing is done
        self.hideSpinner()

    def zoom_in(self):
        """
        Zoom into the image by increasing the zoom factor.
        """
        if self.stretched_image is not None:
            self.zoom_factor *= 1.2
            self.refresh_display()
            print(f"Zoomed in. New zoom factor: {self.zoom_factor:.2f}")
        else:
            print("No stretched image to zoom in.")
            QMessageBox.warning(self, "Warning", "No stretched image to zoom in.")

    def zoom_out(self):
        """
        Zoom out of the image by decreasing the zoom factor.
        """
        if self.stretched_image is not None:
            self.zoom_factor /= 1.2
            self.refresh_display()
            print(f"Zoomed out. New zoom factor: {self.zoom_factor:.2f}")
        else:
            print("No stretched image to zoom out.")
            QMessageBox.warning(self, "Warning", "No stretched image to zoom out.")

    def fit_to_preview(self):
        """Adjust the zoom factor so that the image's width fits within the preview area's width."""
        if self.image is not None:
            # Get the width of the scroll area's viewport (preview area)
            preview_width = self.scrollArea.viewport().width()
            
            # Get the original image width from the numpy array
            # Assuming self.image has shape (height, width, channels) or (height, width) for grayscale
            if self.image.ndim == 3:
                image_width = self.image.shape[1]
            elif self.image.ndim == 2:
                image_width = self.image.shape[1]
            else:
                print("Unexpected image dimensions!")
                QMessageBox.warning(self, "Warning", "Cannot fit image to preview due to unexpected dimensions.")
                return
            
            # Calculate the required zoom factor to fit the image's width into the preview area
            new_zoom_factor = preview_width / image_width
            
            # Update the zoom factor without enforcing any limits
            self.zoom_factor = new_zoom_factor
            
            # Apply the new zoom factor to update the display
            self.refresh_display()
            
            print(f"Fit to preview applied. New zoom factor: {self.zoom_factor:.2f}")
        else:
            print("No image loaded. Cannot fit to preview.")
            QMessageBox.warning(self, "Warning", "No image loaded. Cannot fit to preview.")

    def refresh_display(self):
        """
        Refresh the image display based on the current zoom factor.
        """
        if self.stretched_image is None:
            print("No stretched image to display.")
            return

        try:
            # Normalize and convert to uint8 for display
            img = (self.stretched_image * 255).astype(np.uint8)
            h, w = img.shape[:2]

            if img.ndim == 3 and img.shape[2] == 3:
                bytes_per_line = 3 * w
                q_image = QImage(img.tobytes(), w, h, bytes_per_line, QImage.Format_RGB888)
            elif img.ndim == 2:
                bytes_per_line = w
                q_image = QImage(img.tobytes(), w, h, bytes_per_line, QImage.Format_Grayscale8)
            else:
                raise ValueError("Unsupported image format for display.")

            pixmap = QPixmap.fromImage(q_image)
            scaled_pixmap = pixmap.scaled(
                pixmap.size() * self.zoom_factor,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.imageLabel.setPixmap(scaled_pixmap)
            self.imageLabel.resize(scaled_pixmap.size())

            print("Display refreshed successfully.")
        except Exception as e:
            print(f"Error refreshing display: {e}")
            QMessageBox.critical(self, "Error", f"Failed to refresh display: {e}")

    def apply_zoom(self):
        """Apply the current zoom level to the image."""
        self.updatePreview()  # Call without extra arguments; it will calculate dimensions based on zoom factor            

    def saveImage(self):
        if self.image is not None:
            # Open the file save dialog
            save_filename, _ = QFileDialog.getSaveFileName(
                self, 'Save Image As', '', 
                'Images (*.tiff *.tif *.png *.fit *.fits *.xisf);;All Files (*)'
            )
            
            if save_filename:
                # Extract the file extension from the user-provided filename
                file_extension = save_filename.split('.')[-1].lower()

                # Map the extension to the format expected by save_image
                if file_extension in ['tif', 'tiff']:
                    file_format = 'tiff'
                elif file_extension == 'png':
                    file_format = 'png'
                elif file_extension in ['fit', 'fits']:
                    file_format = 'fits'
                elif file_extension == 'xisf':
                    file_format = 'xisf'
                else:
                    QMessageBox.warning(self, "Error", f"Unsupported file format: .{file_extension}")
                    return
                
                try:
                    # Initialize metadata if not already set (e.g., for PNG)
                    if not hasattr(self, 'image_meta') or self.image_meta is None:
                        self.image_meta = [{
                            'geometry': (self.image.shape[1], self.image.shape[0], self.image.shape[2] if not self.is_mono else 1),
                            'colorSpace': 'Gray' if self.is_mono else 'RGB'
                        }]

                    if not hasattr(self, 'file_meta') or self.file_meta is None:
                        self.file_meta = {}

                    # Initialize a default header for FITS if none exists
                    if not hasattr(self, 'original_header') or self.original_header is None:
                        print("Creating default FITS header...")
                        self.original_header = {
                            'SIMPLE': True,
                            'BITPIX': -32 if self.bit_depth == "32-bit floating point" else 16,
                            'NAXIS': 2 if self.is_mono else 3,
                            'NAXIS1': self.image.shape[1],
                            'NAXIS2': self.image.shape[0],
                            'NAXIS3': 1 if self.is_mono else self.image.shape[2],
                            'BZERO': 0.0,
                            'BSCALE': 1.0,
                            'COMMENT': "Default header created by Seti Astro Suite"
                        }

                    # Call save_image with the appropriate arguments
                    save_image(
                        self.image,
                        save_filename,
                        file_format,  # Use the user-specified format
                        self.bit_depth,
                        self.original_header,
                        self.is_mono,
                        self.image_meta,
                        self.file_meta
                    )
                    print(f"Image saved successfully to {save_filename}")
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Failed to save image: {e}")



class DraggablePoint(QGraphicsEllipseItem):
    def __init__(self, curve_editor, x, y, color=Qt.green, lock_axis=None, position_type=None):
        super().__init__(-5, -5, 10, 10)
        self.curve_editor = curve_editor
        self.lock_axis = lock_axis
        self.position_type = position_type
        self.setBrush(QBrush(color))
        self.setFlags(QGraphicsEllipseItem.ItemIsMovable | QGraphicsEllipseItem.ItemSendsScenePositionChanges)
        self.setCursor(Qt.OpenHandCursor)
        self.setAcceptedMouseButtons(Qt.LeftButton | Qt.RightButton)
        self.setPos(x, y)

    def mousePressEvent(self, event):
        if event.button() == Qt.RightButton:
            if self in self.curve_editor.control_points:
                self.curve_editor.control_points.remove(self)
                self.curve_editor.scene.removeItem(self)
                self.curve_editor.updateCurve()
            return
        super().mousePressEvent(event)

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionHasChanged:
            new_pos = value
            x = new_pos.x()
            y = new_pos.y()

            if self.position_type == 'top_right':
                dist_to_top = abs(y-0)
                dist_to_right = abs(x-360)
                if dist_to_right<dist_to_top:
                    nx=360
                    ny=min(max(y,0),360)
                else:
                    ny=0
                    nx=min(max(x,0),360)
                x,y=nx,ny
            elif self.position_type=='bottom_left':
                dist_to_left=abs(x-0)
                dist_to_bottom=abs(y-360)
                if dist_to_left<dist_to_bottom:
                    nx=0
                    ny=min(max(y,0),360)
                else:
                    ny=360
                    nx=min(max(x,0),360)
                x,y=nx,ny

            all_points=self.curve_editor.end_points+self.curve_editor.control_points
            other_points=[p for p in all_points if p is not self]
            other_points_sorted=sorted(other_points,key=lambda p:p.scenePos().x())

            insert_index=0
            for i,p in enumerate(other_points_sorted):
                if p.scenePos().x()<x:
                    insert_index=i+1
                else:
                    break

            if insert_index>0:
                left_p=other_points_sorted[insert_index-1]
                left_x=left_p.scenePos().x()
                if x<=left_x:
                    x=left_x+0.0001

            if insert_index<len(other_points_sorted):
                right_p=other_points_sorted[insert_index]
                right_x=right_p.scenePos().x()
                if x>=right_x:
                    x=right_x-0.0001

            x=max(0,min(x,360))
            y=max(0,min(y,360))

            super().setPos(x,y)
            self.curve_editor.updateCurve()

        return super().itemChange(change, value)

class ImageLabel(QLabel):
    mouseMoved = pyqtSignal(float, float)
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
    def mouseMoveEvent(self, event):
        self.mouseMoved.emit(event.x(), event.y())
        super().mouseMoveEvent(event)

class CurveEditor(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.setRenderHint(QPainter.Antialiasing)
        self.setFixedSize(380, 425)
        self.preview_callback = None  # To trigger real-time updates

        # Initialize control points and curve path
        self.end_points = []  # Start and end points with axis constraints
        self.control_points = []  # Dynamically added control points
        self.curve_path = QPainterPath()
        self.curve_item = None  # Stores the curve line

        # Set scene rectangle
        self.scene.setSceneRect(0, 0, 360, 360)

        self.initGrid()
        self.initCurve()

    def initGrid(self):
        pen = QPen(Qt.gray)
        pen.setStyle(Qt.DashLine)
        for i in range(0, 361, 45):  # Grid lines at 0,45,...,360
            self.scene.addLine(i, 0, i, 360, pen)  # Vertical lines
            self.scene.addLine(0, i, 360, i, pen)  # Horizontal lines

        # Add X-axis labels
        # Each line corresponds to i/360.0
        for i in range(0, 361, 45):
            val = i/360.0
            label = QGraphicsTextItem(f"{val:.3f}")
            # Position label slightly below the x-axis (360 is bottom)
            # For X-axis, put them near bottom at y=365 for example
            label.setPos(i-5, 365) 
            self.scene.addItem(label)

        # Optionally add Y-axis labels if needed
        # Similar approach for the Y-axis if you want

    def initCurve(self):
        # Remove existing items from the scene
        # First remove control points
        for p in self.control_points:
            self.scene.removeItem(p)
        # Remove end points
        for p in self.end_points:
            self.scene.removeItem(p)
        # Remove the curve item if any
        if self.curve_item:
            self.scene.removeItem(self.curve_item)
            self.curve_item = None

        # Clear existing point lists
        self.end_points = []
        self.control_points = []

        # Add the default endpoints again
        self.addEndPoint(0, 360, lock_axis=None, position_type='bottom_left', color=Qt.black)
        self.addEndPoint(360, 0, lock_axis=None, position_type='top_right', color=Qt.white)

        # Redraw the initial line
        self.updateCurve()

    def addEndPoint(self, x, y, lock_axis=None, position_type=None, color=Qt.red):
        point = DraggablePoint(self, x, y, color=color, lock_axis=lock_axis, position_type=position_type)
        self.scene.addItem(point)
        self.end_points.append(point)

    def addControlPoint(self, x, y, lock_axis=None):

        point = DraggablePoint(self, x, y, color=Qt.green, lock_axis=lock_axis, position_type=None)
        self.scene.addItem(point)
        self.control_points.append(point)
        self.updateCurve()

    def catmull_rom_spline(self, p0, p1, p2, p3, t):
        """
        Compute a point on a Catmull-Rom spline segment at parameter t (0<=t<=1).
        Each p is a QPointF.
        """
        t2 = t * t
        t3 = t2 * t

        x = 0.5 * (2*p1.x() + (-p0.x() + p2.x()) * t +
                    (2*p0.x() - 5*p1.x() + 4*p2.x() - p3.x()) * t2 +
                    (-p0.x() + 3*p1.x() - 3*p2.x() + p3.x()) * t3)
        y = 0.5 * (2*p1.y() + (-p0.y() + p2.y()) * t +
                    (2*p0.y() - 5*p1.y() + 4*p2.y() - p3.y()) * t2 +
                    (-p0.y() + 3*p1.y() - 3*p2.y() + p3.y()) * t3)

        # Clamp to bounding box
        x = max(0, min(360, x))
        y = max(0, min(360, y))

        return QPointF(x, y)

    def generateSmoothCurvePoints(self, points):
        """
        Given a sorted list of QGraphicsItems (endpoints + control points),
        generate a list of smooth points approximating a Catmull-Rom spline
        through these points.
        """
        if len(points) < 2:
            return []
        if len(points) == 2:
            # Just a straight line between two points
            p0 = points[0].scenePos()
            p1 = points[1].scenePos()
            return [p0, p1]

        # Extract scene positions
        pts = [p.scenePos() for p in points]

        # For Catmull-Rom, we need points before the first and after the last
        # We'll duplicate the first and last points.
        extended_pts = [pts[0]] + pts + [pts[-1]]

        smooth_points = []
        steps_per_segment = 20  # increase for smoother curve
        for i in range(len(pts) - 1):
            p0 = extended_pts[i]
            p1 = extended_pts[i+1]
            p2 = extended_pts[i+2]
            p3 = extended_pts[i+3]

            # Sample the spline segment between p1 and p2
            for step in range(steps_per_segment+1):
                t = step / steps_per_segment
                pos = self.catmull_rom_spline(p0, p1, p2, p3, t)
                smooth_points.append(pos)

        return smooth_points

    # Add a callback for the preview
    def setPreviewCallback(self, callback):
        self.preview_callback = callback

    def get8bitLUT(self):
        import numpy as np

        # 8-bit LUT size
        lut_size = 256

        curve_pts = self.getCurvePoints()
        if len(curve_pts) == 0:
            # No curve points, return a linear LUT
            lut = np.linspace(0, 255, lut_size, dtype=np.uint8)
            return lut

        curve_array = np.array(curve_pts, dtype=np.float64)
        xs = curve_array[:, 0]   # X from 0 to 360
        ys = curve_array[:, 1]   # Y from 0 to 360

        ys_for_lut = 360.0 - ys

        # Input positions for interpolation (0..255 mapped to 0..360)
        input_positions = np.linspace(0, 360, lut_size, dtype=np.float64)

        # Interpolate using the inverted Y
        output_values = np.interp(input_positions, xs, ys_for_lut)

        # Map 0..360 to 0..255
        output_values = (output_values / 360.0) * 255.0
        output_values = np.clip(output_values, 0, 255).astype(np.uint8)

        return output_values

    def updateCurve(self):
        """Update the curve by redrawing based on endpoints and control points."""


        all_points = self.end_points + self.control_points
        if not all_points:
            # No points, no curve
            if self.curve_item:
                self.scene.removeItem(self.curve_item)
                self.curve_item = None
            return

        # Sort points by X coordinate
        sorted_points = sorted(all_points, key=lambda p: p.scenePos().x())

        # Extract arrays of X and Y
        xs = [p.scenePos().x() for p in sorted_points]
        ys = [p.scenePos().y() for p in sorted_points]

        # If there's only one point or none, we can't interpolate
        if len(xs) < 2:
            # If there's a single point, just draw a dot or do nothing
            if self.curve_item:
                self.scene.removeItem(self.curve_item)
                self.curve_item = None

            if len(xs) == 1:
                # Optionally draw a single point
                single_path = QPainterPath()
                single_path.addEllipse(xs[0]-2, ys[0]-2, 4, 4)
                pen = QPen(Qt.white)
                pen.setWidth(3)
                self.curve_item = self.scene.addPath(single_path, pen)
            return

        # Create a PCHIP interpolator
        interpolator = PchipInterpolator(xs, ys)
        self.curve_function = interpolator

        # Sample the curve
        sample_xs = np.linspace(xs[0], xs[-1], 361)
        sample_ys = interpolator(sample_xs)



        curve_points = [QPointF(float(x), float(y)) for x, y in zip(sample_xs, sample_ys)]
        self.curve_points = curve_points

        if not curve_points:
            if self.curve_item:
                self.scene.removeItem(self.curve_item)
                self.curve_item = None
            return

        self.curve_path = QPainterPath()
        self.curve_path.moveTo(curve_points[0])
        for pt in curve_points[1:]:
            self.curve_path.lineTo(pt)

        if self.curve_item:
            self.scene.removeItem(self.curve_item)
        pen = QPen(Qt.white)
        pen.setWidth(3)
        self.curve_item = self.scene.addPath(self.curve_path, pen)

        # Trigger the preview callback
        if hasattr(self, 'preview_callback') and self.preview_callback:
            # Generate the 8-bit LUT and pass it to the callback
            lut = self.get8bitLUT()
            self.preview_callback(lut)  # Pass curve_mode      


    def getCurveFunction(self):
        return self.curve_function

    def getCurvePoints(self):
        if not hasattr(self, 'curve_points') or not self.curve_points:
            return []
        return [(pt.x(), pt.y()) for pt in self.curve_points]

    def getLUT(self):
        import numpy as np

        # 16-bit LUT size
        lut_size = 65536

        curve_pts = self.getCurvePoints()
        if len(curve_pts) == 0:
            # No curve points, return a linear LUT
            lut = np.linspace(0, 65535, lut_size, dtype=np.uint16)
            return lut

        curve_array = np.array(curve_pts, dtype=np.float64)
        xs = curve_array[:,0]   # X from 0 to 360
        ys = curve_array[:,1]   # Y from 0 to 360

        ys_for_lut = 360.0 - ys


        # Input positions for interpolation (0..65535 mapped to 0..360)
        input_positions = np.linspace(0, 360, lut_size, dtype=np.float64)

        # Interpolate using the inverted Y
        output_values = np.interp(input_positions, xs, ys_for_lut)

        # Map 0..360 to 0..65535
        output_values = (output_values / 360.0) * 65535.0
        output_values = np.clip(output_values, 0, 65535).astype(np.uint16)

        return output_values


    def mouseDoubleClickEvent(self, event):
        """
        Handle double-click events to add a new control point.
        """
        scene_pos = self.mapToScene(event.pos())

        self.addControlPoint(scene_pos.x(), scene_pos.y())
        super().mouseDoubleClickEvent(event)

    def keyPressEvent(self, event):
        """Remove selected points on Delete key press."""
        if event.key() == Qt.Key_Delete:
            for point in self.control_points[:]:
                if point.isSelected():
                    self.scene.removeItem(point)
                    self.control_points.remove(point)
            self.updateCurve()
        super().keyPressEvent(event)


class FullCurvesProcessingThread(QThread):
    result_ready = pyqtSignal(np.ndarray)

    def __init__(self, image, curve_mode, curve_func):
        super().__init__()
        self.image = image
        self.curve_mode = curve_mode
        self.curve_func = curve_func

    def run(self):
        adjusted_image = self.process_curve(self.image, self.curve_mode, self.curve_func)
        self.result_ready.emit(adjusted_image)

    @staticmethod
    def apply_curve_direct(value, curve_func):
        # value in [0..1]
        # Evaluate curve at value*360 (X), get Y in [0..360]
        # Invert it: out = 360 - curve_func(X)
        # Map back to [0..1]: out/360
        out = curve_func(value*360.0)
        out = 360.0 - out
        return np.clip(out/360.0, 0, 1).astype(np.float32)

    @staticmethod
    def process_curve(image, curve_mode, curve_func):
        if image is None:
            return image

        if curve_func is None:
            # No curve defined, identity
            return image

        if image.dtype != np.float32:
            image = image.astype(np.float32, copy=False)

        is_gray = (image.ndim == 2 or image.shape[2] == 1)
        if is_gray:
            image = image.reshape(image.shape[0], image.shape[1], 1)

        mode = curve_mode.lower()

        # Helper functions for color modes
        def apply_to_all_channels(img):
            for c in range(img.shape[2]):
                img[:,:,c] = FullCurvesProcessingThread.apply_curve_direct(img[:,:,c], curve_func)
            return img

        def apply_to_channel(img, ch):
            img[:,:,ch] = FullCurvesProcessingThread.apply_curve_direct(img[:,:,ch], curve_func)
            return img

        if mode == 'r':
            if image.shape[2] == 3:
                image = apply_to_channel(image, 0)

        elif mode == 'g':
            if image.shape[2] == 3:
                image = apply_to_channel(image, 1)

        elif mode == 'b':
            if image.shape[2] == 3:
                image = apply_to_channel(image, 2)

        elif mode == 'k (brightness)':
            image = apply_to_all_channels(image)

        elif mode == 'l*':
            # Convert to Lab, apply curve to L
            # L in [0..100], normalize to [0..1], apply curve, then *100
            from_color = FullCurvesProcessingThread.rgb_to_xyz
            to_color = FullCurvesProcessingThread.xyz_to_rgb
            xyz = from_color(image)
            lab = FullCurvesProcessingThread.xyz_to_lab(xyz)

            L_norm = np.clip(lab[:,:,0]/100.0, 0, 1)
            L_new = FullCurvesProcessingThread.apply_curve_direct(L_norm, curve_func)*100.0
            lab[:,:,0] = L_new

            xyz_new = FullCurvesProcessingThread.lab_to_xyz(lab)
            image = to_color(xyz_new)
            
        elif mode == 'a*':
            # Convert to Lab, apply curve to a*
            from_color = FullCurvesProcessingThread.rgb_to_xyz
            to_color = FullCurvesProcessingThread.xyz_to_rgb
            xyz = from_color(image)
            lab = FullCurvesProcessingThread.xyz_to_lab(xyz)

            # a* in [-128..127], shift/scale to [0..1]
            a_norm = np.clip((lab[:,:,1] + 128.0)/255.0, 0, 1)
            a_new = FullCurvesProcessingThread.apply_curve_direct(a_norm, curve_func)*255.0 - 128.0
            lab[:,:,1] = a_new

            xyz_new = FullCurvesProcessingThread.lab_to_xyz(lab)
            image = to_color(xyz_new)

        elif mode == 'b*':
            # Convert to Lab, apply curve to b*
            from_color = FullCurvesProcessingThread.rgb_to_xyz
            to_color = FullCurvesProcessingThread.xyz_to_rgb
            xyz = from_color(image)
            lab = FullCurvesProcessingThread.xyz_to_lab(xyz)

            b_norm = np.clip((lab[:,:,2] + 128.0)/255.0, 0, 1)
            b_new = FullCurvesProcessingThread.apply_curve_direct(b_norm, curve_func)*255.0 - 128.0
            lab[:,:,2] = b_new

            xyz_new = FullCurvesProcessingThread.lab_to_xyz(lab)
            image = to_color(xyz_new)

        elif mode == 'chroma':
            # Convert to Lab, apply curve to Chroma
            from_color = FullCurvesProcessingThread.rgb_to_xyz
            to_color = FullCurvesProcessingThread.xyz_to_rgb
            xyz = from_color(image)
            lab = FullCurvesProcessingThread.xyz_to_lab(xyz)

            a_ = lab[:,:,1]
            b_ = lab[:,:,2]
            C = np.sqrt(a_*a_ + b_*b_)
            C_norm = np.clip(C/200.0, 0, 1)
            C_new = FullCurvesProcessingThread.apply_curve_direct(C_norm, curve_func)*200.0

            ratio = np.divide(C_new, C, out=np.zeros_like(C), where=(C!=0))
            lab[:,:,1] = a_*ratio
            lab[:,:,2] = b_*ratio

            xyz_new = FullCurvesProcessingThread.lab_to_xyz(lab)
            image = to_color(xyz_new)

        elif mode == 'saturation':
            # Convert to HSV, apply curve to S
            hsv = FullCurvesProcessingThread.rgb_to_hsv(image)
            S = hsv[:,:,1]
            S_new = FullCurvesProcessingThread.apply_curve_direct(S, curve_func)
            hsv[:,:,1] = S_new
            image = FullCurvesProcessingThread.hsv_to_rgb(hsv)

        if is_gray:
            image = image[:,:,0]

        return image

    @staticmethod
    def rgb_to_xyz(rgb):
        M = np.array([[0.4124564, 0.3575761, 0.1804375],
                      [0.2126729, 0.7151522, 0.0721750],
                      [0.0193339, 0.1191920, 0.9503041]], dtype=np.float32)
        shape = rgb.shape
        out = rgb.reshape(-1,3) @ M.T
        return out.reshape(shape)

    @staticmethod
    def xyz_to_rgb(xyz):
        M_inv = np.array([[ 3.2404542, -1.5371385, -0.4985314],
                          [-0.9692660,  1.8760108,  0.0415560],
                          [ 0.0556434, -0.2040259,  1.0572252]], dtype=np.float32)
        shape = xyz.shape
        out = xyz.reshape(-1,3) @ M_inv.T
        out = np.clip(out, 0, 1)
        return out.reshape(shape)

    @staticmethod
    def f_lab(t):
        delta = 6/29
        mask = t > delta**3
        f = np.zeros_like(t)
        f[mask] = np.cbrt(t[mask])
        f[~mask] = t[~mask]/(3*delta*delta)+4/29
        return f

    @staticmethod
    def xyz_to_lab(xyz):
        Xn, Yn, Zn = 0.95047, 1.00000, 1.08883
        X = xyz[:,:,0]/Xn
        Y = xyz[:,:,1]/Yn
        Z = xyz[:,:,2]/Zn

        fx = FullCurvesProcessingThread.f_lab(X)
        fy = FullCurvesProcessingThread.f_lab(Y)
        fz = FullCurvesProcessingThread.f_lab(Z)

        L = (116 * fy - 16)
        a = 500*(fx - fy)
        b = 200*(fy - fz)
        return np.dstack([L, a, b]).astype(np.float32)

    @staticmethod
    def lab_to_xyz(lab):
        L = lab[:,:,0]
        a = lab[:,:,1]
        b = lab[:,:,2]

        delta = 6/29
        fy = (L+16)/116
        fx = fy + a/500
        fz = fy - b/200

        def f_inv(ft):
            return np.where(ft > delta, ft**3, 3*delta*delta*(ft - 4/29))

        Xn, Yn, Zn = 0.95047, 1.00000, 1.08883
        X = Xn*f_inv(fx)
        Y = Yn*f_inv(fy)
        Z = Zn*f_inv(fz)
        return np.dstack([X, Y, Z]).astype(np.float32)

    @staticmethod
    def rgb_to_hsv(rgb):
        cmax = rgb.max(axis=2)
        cmin = rgb.min(axis=2)
        delta = cmax - cmin

        H = np.zeros_like(cmax)
        S = np.zeros_like(cmax)
        V = cmax

        mask = delta != 0
        r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
        H[mask & (cmax==r)] = 60*(((g[mask&(cmax==r)]-b[mask&(cmax==r)])/delta[mask&(cmax==r)])%6)
        H[mask & (cmax==g)] = 60*(((b[mask&(cmax==g)]-r[mask&(cmax==g)])/delta[mask&(cmax==g)])+2)
        H[mask & (cmax==b)] = 60*(((r[mask&(cmax==b)]-g[mask&(cmax==b)])/delta[mask&(cmax==b)])+4)

        S[cmax>0] = delta[cmax>0]/cmax[cmax>0]
        return np.dstack([H,S,V]).astype(np.float32)

    @staticmethod
    def hsv_to_rgb(hsv):
        H, S, V = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
        C = V*S
        X = C*(1-np.abs((H/60.0)%2-1))
        m = V-C

        R = np.zeros_like(H)
        G = np.zeros_like(H)
        B = np.zeros_like(H)

        cond0 = (H<60)
        cond1 = (H>=60)&(H<120)
        cond2 = (H>=120)&(H<180)
        cond3 = (H>=180)&(H<240)
        cond4 = (H>=240)&(H<300)
        cond5 = (H>=300)

        R[cond0]=C[cond0]; G[cond0]=X[cond0]; B[cond0]=0
        R[cond1]=X[cond1]; G[cond1]=C[cond1]; B[cond1]=0
        R[cond2]=0; G[cond2]=C[cond2]; B[cond2]=X[cond2]
        R[cond3]=0; G[cond3]=X[cond3]; B[cond3]=C[cond3]
        R[cond4]=X[cond4]; G[cond4]=0; B[cond4]=C[cond4]
        R[cond5]=C[cond5]; G[cond5]=0; B[cond5]=X[cond5]

        rgb = np.dstack([R+m, G+m, B+m])
        rgb = np.clip(rgb, 0, 1)
        return rgb

class FrequencySeperationTab(QWidget):
    def __init__(self, image_manager=None):
        super().__init__()
        self.image_manager = image_manager  # Reference to the shared ImageManager
        self.filename = None
        self.image = None  # Original input image
        self.low_freq_image = None
        self.high_freq_image = None
        self.original_header = None
        self.is_mono = False
        self.processing_thread = None
        self.hfEnhancementThread = None
        self.hf_history = []

        # Default parameters
        self.method = 'Gaussian'
        self.radius = 25
        self.mirror = False
        self.tolerance = 50  # new tolerance param

        # Zoom/pan control
        self.zoom_factor = 1.0
        self.dragging = False
        self.last_mouse_pos = QPoint()

        # For the preview
        self.spinnerLabel = None
        self.spinnerMovie = None

        # A guard variable to avoid infinite scroll loops
        self.syncing_scroll = False

        self.initUI()

        # Connect to ImageManager's image_changed signal if available
        if self.image_manager:
            self.image_manager.image_changed.connect(self.on_image_changed)
            # Load the existing image from ImageManager, if any
            if self.image_manager.image is not None:
                self.on_image_changed(
                    slot=self.image_manager.current_slot,
                    image=self.image_manager.image,
                    metadata=self.image_manager.current_metadata
                )

    def initUI(self):
        """
        Set up the GUI layout:
          - Left panel with controls (Load, Method, Radius, Mirror, Tolerance, Apply, Save, etc.)
          - Right panel with two scroll areas for HF/LF previews
        """
        main_layout = QHBoxLayout(self)
        self.setLayout(main_layout)

        # -----------------------------
        # Left side: Controls
        # -----------------------------
        left_widget = QWidget(self)
        left_widget.setFixedWidth(250)
        left_layout = QVBoxLayout(left_widget)

        # 1) Load image
        self.loadButton = QPushButton("Load Image", self)
        self.loadButton.setIcon(self.style().standardIcon(QStyle.SP_DirOpenIcon))
        self.loadButton.clicked.connect(self.selectImage)
        left_layout.addWidget(self.loadButton)

        self.fileLabel = QLabel("", self)
        left_layout.addWidget(self.fileLabel)

        # Method Combo
        self.method_combo = QComboBox(self)
        self.method_combo.addItems(['Gaussian', 'Median', 'Bilateral'])
        self.method_combo.currentTextChanged.connect(self.on_method_changed)
        left_layout.addWidget(QLabel("Method:", self))
        left_layout.addWidget(self.method_combo)

        # Radius Slider + Label
        self.radiusSlider = QSlider(Qt.Horizontal, self)
        self.radiusSlider.setRange(1, 100)
        self.radiusSlider.setValue(10)  # or whatever integer in [1..100] you want
        self.radiusSlider.valueChanged.connect(self.on_radius_changed)

        self.radiusLabel = QLabel("Radius:", self)
        left_layout.addWidget(self.radiusLabel)   
        left_layout.addWidget(self.radiusSlider)

        # Now force an initial update so label is correct from the start
        self.on_radius_changed(self.radiusSlider.value())

        # Tolerance Slider + Label
        self.toleranceSlider = QSlider(Qt.Horizontal, self)
        self.toleranceSlider.setRange(0, 100)
        self.toleranceSlider.setValue(self.tolerance)
        self.toleranceSlider.valueChanged.connect(self.on_tolerance_changed)
        self.toleranceLabel = QLabel(f"Tolerance: {self.tolerance}%", self)
        self.toleranceSlider.setEnabled(False)
        self.toleranceLabel.setEnabled(False)
        left_layout.addWidget(self.toleranceLabel)
        left_layout.addWidget(self.toleranceSlider)

        # Apply button
        self.applyButton = QPushButton("Apply - Split HF and LF", self)
        self.applyButton.clicked.connect(self.apply_frequency_separation)
        left_layout.addWidget(self.applyButton)        

        # -----------------------------------
        # *** New Sharpening Controls ***
        # -----------------------------------
        # 1) Checkbox for "Enable Sharpen Scale"
        self.sharpenScaleCheckBox = QCheckBox("Enable Sharpen Scale", self)
        self.sharpenScaleCheckBox.setChecked(True)  # or False by default
        left_layout.addWidget(self.sharpenScaleCheckBox)

        # Sharpen Scale Label + Slider
        self.sharpenScaleLabel = QLabel("Sharpen Scale: 1.00", self)
        left_layout.addWidget(self.sharpenScaleLabel)

        self.sharpenScaleSlider = QSlider(Qt.Horizontal, self)
        self.sharpenScaleSlider.setRange(10, 300)  # => 0.1..3.0
        self.sharpenScaleSlider.setValue(100)      # 1.00 initially
        self.sharpenScaleSlider.valueChanged.connect(self.onSharpenScaleChanged)
        left_layout.addWidget(self.sharpenScaleSlider)

        # 2) Checkbox for "Enable Wavelet Sharpening"
        self.waveletCheckBox = QCheckBox("Enable Wavelet Sharpening", self)
        self.waveletCheckBox.setChecked(True)  # or False by default
        left_layout.addWidget(self.waveletCheckBox)

        # Wavelet Sharpening Sliders
        wavelet_title = QLabel("<b>Wavelet Sharpening:</b>", self)
        left_layout.addWidget(wavelet_title)

        self.waveletLevelLabel = QLabel("Wavelet Level: 2", self)
        left_layout.addWidget(self.waveletLevelLabel)

        self.waveletLevelSlider = QSlider(Qt.Horizontal, self)
        self.waveletLevelSlider.setRange(1, 5)
        self.waveletLevelSlider.setValue(2)
        self.waveletLevelSlider.valueChanged.connect(self.onWaveletLevelChanged)
        left_layout.addWidget(self.waveletLevelSlider)

        self.waveletBoostLabel = QLabel("Wavelet Boost: 1.20", self)
        left_layout.addWidget(self.waveletBoostLabel)

        self.waveletBoostSlider = QSlider(Qt.Horizontal, self)
        self.waveletBoostSlider.setRange(50, 300)  # => 0.5..3.0
        self.waveletBoostSlider.setValue(120)      # 1.20 initially
        self.waveletBoostSlider.valueChanged.connect(self.onWaveletBoostChanged)
        left_layout.addWidget(self.waveletBoostSlider)

        self.enableDenoiseCheckBox = QCheckBox("Enable HF Denoise", self)
        self.enableDenoiseCheckBox.setChecked(False)  # default off or on, your choice
        left_layout.addWidget(self.enableDenoiseCheckBox)

        # Label + Slider for denoise strength
        self.denoiseStrengthLabel = QLabel("Denoise Strength: 3.00", self)
        left_layout.addWidget(self.denoiseStrengthLabel)

        self.denoiseStrengthSlider = QSlider(Qt.Horizontal, self)
        self.denoiseStrengthSlider.setRange(0, 50)  # Example range -> 1..50 => 1.0..50.0
        self.denoiseStrengthSlider.setValue(3)      # default 3
        self.denoiseStrengthSlider.valueChanged.connect(self.onDenoiseStrengthChanged)
        left_layout.addWidget(self.denoiseStrengthSlider)
        self.onDenoiseStrengthChanged(self.denoiseStrengthSlider.value())

        # Create a horizontal layout for HF Enhancements and Undo
        hfEnhance_hlayout = QHBoxLayout()

        # Apply HF Enhancements button
        self.applyHFEnhancementsButton = QPushButton("Apply HF Enhancements", self)
        self.applyHFEnhancementsButton.setIcon(self.style().standardIcon(QStyle.SP_DialogApplyButton))
        self.applyHFEnhancementsButton.clicked.connect(self.applyHFEnhancements)
        hfEnhance_hlayout.addWidget(self.applyHFEnhancementsButton)

        # Undo button (tool button with back arrow icon)
        self.undoHFButton = QToolButton(self)
        self.undoHFButton.setIcon(self.style().standardIcon(QStyle.SP_ArrowBack))
        self.undoHFButton.setToolTip("Undo last HF enhancement")
        self.undoHFButton.clicked.connect(self.undoHFEnhancement)
        self.undoHFButton.setEnabled(False)  # Initially disabled
        hfEnhance_hlayout.addWidget(self.undoHFButton)

        # Now add this horizontal layout to the main left_layout
        left_layout.addLayout(hfEnhance_hlayout)

        # ------------------------------------
        # Save HF / LF - in a horizontal layout
        # ------------------------------------
        save_hlayout = QHBoxLayout()

        self.saveHFButton = QPushButton("Save HF", self)
        self.saveHFButton.clicked.connect(self.save_high_frequency)
        save_hlayout.addWidget(self.saveHFButton)

        self.saveLFButton = QPushButton("Save LF", self)
        self.saveLFButton.clicked.connect(self.save_low_frequency)
        save_hlayout.addWidget(self.saveLFButton)

        left_layout.addLayout(save_hlayout)

        # ------------------------------------
        # Import HF / LF - in a separate horizontal layout
        # ------------------------------------
        load_hlayout = QHBoxLayout()

        self.importHFButton = QPushButton("Load HF", self)
        self.importHFButton.clicked.connect(self.loadHF)
        load_hlayout.addWidget(self.importHFButton)

        self.importLFButton = QPushButton("Load LF", self)
        self.importLFButton.clicked.connect(self.loadLF)
        load_hlayout.addWidget(self.importLFButton)

        left_layout.addLayout(load_hlayout)

        # Combine HF + LF
        self.combineButton = QPushButton("Combine HF + LF", self)
        self.combineButton.setIcon(self.style().standardIcon(QStyle.SP_DialogYesButton))
        self.combineButton.clicked.connect(self.combineHFandLF)
        left_layout.addWidget(self.combineButton)

        # Spinner for background processing
        self.spinnerLabel = QLabel(self)
        self.spinnerLabel.setAlignment(Qt.AlignCenter)
        self.spinnerMovie = QMovie(resource_path("spinner.gif"))  # Provide your spinner path
        self.spinnerLabel.setMovie(self.spinnerMovie)
        self.spinnerLabel.hide()
        left_layout.addWidget(self.spinnerLabel)

        # Spacer
        left_layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))
        main_layout.addWidget(left_widget)

        # -----------------------------
        # Right Panel (vertical layout)
        # -----------------------------
        right_widget = QWidget(self)
        right_vbox = QVBoxLayout(right_widget)

        # 1) Zoom Buttons row (top)
        zoom_hbox = QHBoxLayout()
        self.zoom_in_btn = QPushButton("Zoom In")
        self.zoom_in_btn.clicked.connect(self.zoom_in)
        zoom_hbox.addWidget(self.zoom_in_btn)

        self.zoom_out_btn = QPushButton("Zoom Out")
        self.zoom_out_btn.clicked.connect(self.zoom_out)
        zoom_hbox.addWidget(self.zoom_out_btn)

        right_vbox.addLayout(zoom_hbox)

        # 2) HF / LF previews row (below)
        scroll_hbox = QHBoxLayout()

        self.scrollHF = QScrollArea(self)
        self.scrollHF.setWidgetResizable(False)
        self.labelHF = QLabel("High Frequency", self)
        self.labelHF.setAlignment(Qt.AlignCenter)
        self.labelHF.setStyleSheet("background-color: #333; color: #CCC;")
        self.scrollHF.setWidget(self.labelHF)

        self.scrollLF = QScrollArea(self)
        self.scrollLF.setWidgetResizable(False)
        self.labelLF = QLabel("Low Frequency", self)
        self.labelLF.setAlignment(Qt.AlignCenter)
        self.labelLF.setStyleSheet("background-color: #333; color: #CCC;")
        self.scrollLF.setWidget(self.labelLF)

        scroll_hbox.addWidget(self.scrollHF, stretch=1)
        scroll_hbox.addWidget(self.scrollLF, stretch=1)

        right_vbox.addLayout(scroll_hbox, stretch=1)
        main_layout.addWidget(right_widget, stretch=1)

        # Sync scrollbars
        self.scrollHF.horizontalScrollBar().valueChanged.connect(self.syncHFHScroll)
        self.scrollHF.verticalScrollBar().valueChanged.connect(self.syncHFVScroll)
        self.scrollLF.horizontalScrollBar().valueChanged.connect(self.syncLFHScroll)
        self.scrollLF.verticalScrollBar().valueChanged.connect(self.syncLFVScroll)

        # Mouse drag panning
        self.scrollHF.viewport().installEventFilter(self)
        self.scrollLF.viewport().installEventFilter(self)

        # Force initial label update
        self.on_radius_changed(self.radiusSlider.value())
        self.on_tolerance_changed(self.toleranceSlider.value())

    # -----------------------------
    # Image Manager Integration
    # -----------------------------
    def on_image_changed(self, slot, image, metadata):
        """
        Slot to handle image changes from ImageManager.
        Updates the FrequencySeperationTab if the change is relevant.
        """
        if slot == self.image_manager.current_slot:
            # Ensure the image is a numpy array
            if not isinstance(image, np.ndarray):
                image = np.array(image)  # Convert to numpy array if needed

            # Update internal state with the new image and metadata
            self.loaded_image_path = metadata.get('file_path', None)
            self.image = image
            self.original_header = metadata.get('original_header', None)
            self.is_mono = metadata.get('is_mono', False)
            self.filename = self.loaded_image_path

            # Reset HF / LF placeholders
            self.low_freq_image = None
            self.high_freq_image = None

            # Update UI label to show the file name or indicate no file
            # Update the fileLabel in the Frequency Separation Tab (or any other tab)
            if self.image_manager.image is not None:
                # Retrieve the file path from the metadata in ImageManager
                file_path = self.image_manager._metadata[self.image_manager.current_slot].get('file_path', None)
                # Update the file label with the basename of the file path
                self.fileLabel.setText(os.path.basename(file_path) if file_path else "No file selected")
            else:
                self.fileLabel.setText("No file selected")


            # Automatically apply frequency separation
            self.apply_frequency_separation()

            print(f"FrequencySeperationTab: Image updated from ImageManager slot {slot}.")


    def map_slider_to_radius(self, slider_pos):
        """
        Convert a slider position (0..100) into a non-linear float radius.
        Segment A: [0..10]   -> [0.1..1.0]
        Segment B: [10..50]  -> [1.0..10.0]
        Segment C: [50..100] -> [10.0..100.0]
        """
        if slider_pos <= 10:
            # Scale 0..10 -> 0.1..1.0
            t = slider_pos / 10.0           # t in [0..1]
            radius = 0.1 + t*(1.0 - 0.1)    # 0.1 -> 1.0
        elif slider_pos <= 50:
            # Scale 10..50 -> 1.0..10.0
            t = (slider_pos - 10) / 40.0    # t in [0..1]
            radius = 1.0 + t*(10.0 - 1.0)   # 1.0 -> 10.0
        else:
            # Scale 50..100 -> 10.0..100.0
            t = (slider_pos - 50) / 50.0    # t in [0..1]
            radius = 10.0 + t*(100.0 - 10.0)  # 10.0 -> 100.0
        
        return radius

    def onSharpenScaleChanged(self, val):
        scale = val / 100.0  # 10..300 => 0.1..3.0
        self.sharpenScaleLabel.setText(f"Sharpen Scale: {scale:.2f}")

    def onWaveletLevelChanged(self, val):
        self.waveletLevelLabel.setText(f"Wavelet Level: {val}")

    def onWaveletBoostChanged(self, val):
        boost = val / 100.0  # e.g. 50..300 => 0.50..3.00
        self.waveletBoostLabel.setText(f"Wavelet Boost: {boost:.2f}")

    def onDenoiseStrengthChanged(self, val):
        # Map 0..50 => 0..5.0 by dividing by 10
        denoise_strength = val / 10.0
        self.denoiseStrengthLabel.setText(f"Denoise Strength: {denoise_strength:.2f}")

    # -------------------------------------------------
    # Event Filter for Drag Panning
    # -------------------------------------------------
    def eventFilter(self, obj, event):
        if obj in (self.scrollHF.viewport(), self.scrollLF.viewport()):
            if event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                self.dragging = True
                self.last_mouse_pos = event.pos()
                return True
            elif event.type() == QEvent.MouseMove and self.dragging:
                delta = event.pos() - self.last_mouse_pos
                self.last_mouse_pos = event.pos()

                if obj == self.scrollHF.viewport():
                    # Move HF scrollbars
                    self.syncing_scroll = True
                    try:
                        self.scrollHF.horizontalScrollBar().setValue(
                            self.scrollHF.horizontalScrollBar().value() - delta.x()
                        )
                        self.scrollHF.verticalScrollBar().setValue(
                            self.scrollHF.verticalScrollBar().value() - delta.y()
                        )
                        # Sync LF
                        self.scrollLF.horizontalScrollBar().setValue(
                            self.scrollHF.horizontalScrollBar().value()
                        )
                        self.scrollLF.verticalScrollBar().setValue(
                            self.scrollHF.verticalScrollBar().value()
                        )
                    finally:
                        self.syncing_scroll = False
                else:
                    # Move LF scrollbars
                    self.syncing_scroll = True
                    try:
                        self.scrollLF.horizontalScrollBar().setValue(
                            self.scrollLF.horizontalScrollBar().value() - delta.x()
                        )
                        self.scrollLF.verticalScrollBar().setValue(
                            self.scrollLF.verticalScrollBar().value() - delta.y()
                        )
                        # Sync HF
                        self.scrollHF.horizontalScrollBar().setValue(
                            self.scrollLF.horizontalScrollBar().value()
                        )
                        self.scrollHF.verticalScrollBar().setValue(
                            self.scrollLF.verticalScrollBar().value()
                        )
                    finally:
                        self.syncing_scroll = False
                return True
            elif event.type() == QEvent.MouseButtonRelease and event.button() == Qt.LeftButton:
                self.dragging = False
                return True
        return super().eventFilter(obj, event)

    # -----------------------------
    # Scrolling Sync
    # -----------------------------
    def syncHFHScroll(self, value):
        if not self.syncing_scroll:
            self.syncing_scroll = True
            self.scrollLF.horizontalScrollBar().setValue(value)
            self.syncing_scroll = False

    def syncHFVScroll(self, value):
        if not self.syncing_scroll:
            self.syncing_scroll = True
            self.scrollLF.verticalScrollBar().setValue(value)
            self.syncing_scroll = False

    def syncLFHScroll(self, value):
        if not self.syncing_scroll:
            self.syncing_scroll = True
            self.scrollHF.horizontalScrollBar().setValue(value)
            self.syncing_scroll = False

    def syncLFVScroll(self, value):
        if not self.syncing_scroll:
            self.syncing_scroll = True
            self.scrollHF.verticalScrollBar().setValue(value)
            self.syncing_scroll = False

    # -----------------------------
    # Zooming
    # -----------------------------
    def zoom_in(self):
        self.zoom_factor *= 1.25
        self.update_previews()

    def zoom_out(self):
        self.zoom_factor /= 1.25
        self.update_previews()

    # -----------------------------
    # Control Handlers
    # -----------------------------
    def on_method_changed(self, text):
        """
        Called whenever the method dropdown changes (Gaussian, Median, Bilateral).
        Enable the tolerance slider only for 'Bilateral'.
        """
        self.method = text
        if self.method == 'Bilateral':
            self.toleranceSlider.setEnabled(True)
            self.toleranceLabel.setEnabled(True)
        else:
            self.toleranceSlider.setEnabled(False)
            self.toleranceLabel.setEnabled(False)

    def on_radius_changed(self, value):
        new_radius = self.map_slider_to_radius(value)  # use self.
        self.radius = new_radius
        self.radiusLabel.setText(f"Radius: {new_radius:.2f}")


    def on_tolerance_changed(self, value):
        self.tolerance = value
        self.toleranceLabel.setText(f"Tolerance: {value}%")  # Update label

    def undoHFEnhancement(self):
        """
        Revert HF to the last state from hf_history, if available.
        Disable Undo if no more history is left.
        """
        if len(self.hf_history) == 0:
            return  # No history to revert
        
        # Pop the last saved HF
        old_hf = self.hf_history.pop()

        # Restore it
        self.high_freq_image = old_hf
        self.update_previews()
        self.fileLabel.setText("Undid last HF enhancement.")

        # If no more states are left, disable the Undo button again
        if len(self.hf_history) == 0:
            self.undoHFButton.setEnabled(False)


    def applyHFEnhancements(self):
        if self.high_freq_image is None:
            self.fileLabel.setText("No HF image to enhance.")
            return
        
        self.hf_history.append(self.high_freq_image.copy())

        # Enable the Undo button because now we have at least one state
        self.undoHFButton.setEnabled(True)

        self.showSpinner()

        # If a previous thread is running, kill it safely
        if self.hfEnhancementThread and self.hfEnhancementThread.isRunning():
            self.hfEnhancementThread.quit()
            self.hfEnhancementThread.wait()

        # Check Sharpen Scale
        enable_scale = self.sharpenScaleCheckBox.isChecked()
        sharpen_scale = self.sharpenScaleSlider.value() / 100.0

        # Wavelet
        enable_wavelet = self.waveletCheckBox.isChecked()
        wavelet_level = self.waveletLevelSlider.value()
        wavelet_boost = self.waveletBoostSlider.value() / 100.0

        # Denoise
        enable_denoise = self.enableDenoiseCheckBox.isChecked()
        denoise_strength = float(self.denoiseStrengthSlider.value()/10.0)  # or do /10 if you want finer steps

        # Instantiate HFEnhancementThread with denoise params
        self.hfEnhancementThread = HFEnhancementThread(
            hf_image=self.high_freq_image,
            enable_scale=enable_scale,
            sharpen_scale=sharpen_scale,
            enable_wavelet=enable_wavelet,
            wavelet_level=wavelet_level,
            wavelet_boost=wavelet_boost,
            wavelet_name='db2',
            enable_denoise=enable_denoise,
            denoise_strength=denoise_strength
        )
        self.hfEnhancementThread.enhancement_done.connect(self.onHFEnhancementDone)
        self.hfEnhancementThread.error_signal.connect(self.onHFEnhancementError)
        self.hfEnhancementThread.start()


    def onHFEnhancementDone(self, newHF):
        self.hideSpinner()
        self.high_freq_image = newHF  # updated HF
        self.update_previews()
        self.fileLabel.setText("HF enhancements applied (thread).")

    def onHFEnhancementError(self, msg):
        self.hideSpinner()
        self.fileLabel.setText(f"HF enhancement error: {msg}")

    # -----------------------------
    # Image Selection and Preview Methods
    # -----------------------------
    def selectImage(self):
        if not self.image_manager:
            QMessageBox.warning(self, "Warning", "ImageManager not initialized.")
            return

        selected_file, _ = QFileDialog.getOpenFileName(self, "Open Image", "", 
                                            "Images (*.png *.tif *.tiff *.fits *.xisf *.cr2 *.nef *.arw *.dng *.orf *.rw2 *.pef);;All Files (*)")
        if selected_file:
            try:
                img, header, bit_depth, is_mono = load_image(selected_file)
                if img is None:
                    QMessageBox.critical(self, "Error", "Failed to load the image. Please try a different file.")
                    return

                print(f"FrequencySeperationTab: Image loaded successfully. Shape: {img.shape}, Dtype: {img.dtype}")

                self.image = img
                self.original_header = header
                self.is_mono = is_mono
                self.filename = selected_file
                self.fileLabel.setText(os.path.basename(selected_file))

                # Reset HF / LF placeholders
                self.low_freq_image = None
                self.high_freq_image = None

                # Update ImageManager with the new image
                metadata = {
                    'file_path': self.filename,
                    'original_header': self.original_header,
                    'bit_depth': bit_depth,
                    'is_mono': self.is_mono
                }
                self.image_manager.set_current_image(image=img, metadata=metadata)
                print("FrequencySeperationTab: Image updated in ImageManager.")

            except Exception as e:
                self.fileLabel.setText(f"Error: {str(e)}")
                print(f"FrequencySeperationTab: Error loading image: {e}")

    def save_high_frequency(self):
        if self.high_freq_image is None:
            self.fileLabel.setText("No high-frequency image to save.")
            return
        self._save_image_with_dialog(self.high_freq_image, suffix="_HF")

    def save_low_frequency(self):
        if self.low_freq_image is None:
            self.fileLabel.setText("No low-frequency image to save.")
            return
        self._save_image_with_dialog(self.low_freq_image, suffix="_LF")

    def _save_image_with_dialog(self, image_to_save, suffix=""):
        """
        Always save HF in 32-bit floating point, either .tif or .fits.
        """
        if self.filename:
            base_name = os.path.basename(self.filename)
            default_save_name = os.path.splitext(base_name)[0] + suffix + '.tif'
            original_dir = os.path.dirname(self.filename)
        else:
            default_save_name = "untitled" + suffix + '.tif'
            original_dir = os.getcwd()

        # Restrict the file dialog to TIF/FITS by default,
        # but let's keep .png, etc., in case user tries to pick it.
        # We'll override if they do.
        save_filename, _ = QFileDialog.getSaveFileName(
            self,
            'Save HF Image as 32-bit Float',
            os.path.join(original_dir, default_save_name),
            'TIFF or FITS (*.tif *.tiff *.fits *.fit);;All Files (*)'
        )
        if save_filename:
            # Identify extension
            file_ext = os.path.splitext(save_filename)[1].lower().strip('.')  # e.g. 'tif', 'fits', etc.

            # If user picks something else (png/jpg), override to .tif
            if file_ext not in ['tif', 'tiff', 'fit', 'fits']:
                file_ext = 'tif'
                # Force the filename to end with .tif
                save_filename = os.path.splitext(save_filename)[0] + '.tif'

            # We skip prompting for bit depth since we always want 32-bit float
            bit_depth = "32-bit floating point"

            # Force original_format to the extension we ended up with
            save_image(
                image_to_save,
                save_filename,
                original_format=file_ext,     # e.g. 'tif' or 'fits'
                bit_depth=bit_depth,
                original_header=self.original_header,
                is_mono=self.is_mono
            )
            self.fileLabel.setText(f"Saved 32-bit float HF: {os.path.basename(save_filename)}")


    def loadHF(self):
        selected_file, _ = QFileDialog.getOpenFileName(
            self, "Load High Frequency Image", "", "Images (*.png *.tif *.tiff *.fits *.fit *.xisf)"
        )
        if selected_file:
            try:
                hf, _, _, _ = load_image(selected_file)
                self.high_freq_image = hf
                self.update_previews()
            except Exception as e:
                self.fileLabel.setText(f"Error loading HF: {str(e)}")

    def loadLF(self):
        selected_file, _ = QFileDialog.getOpenFileName(
            self, "Load Low Frequency Image", "", "Images (*.png *.tif *.tiff *.fits *.fit *.xisf)"
        )
        if selected_file:
            try:
                lf, _, _, _ = load_image(selected_file)
                self.low_freq_image = lf
                self.update_previews()
            except Exception as e:
                self.fileLabel.setText(f"Error loading LF: {str(e)}")

    def combineHFandLF(self):
        if self.low_freq_image is None or self.high_freq_image is None:
            self.fileLabel.setText("Cannot combine; LF or HF is missing.")
            return

        # Check shape
        if self.low_freq_image.shape != self.high_freq_image.shape:
            self.fileLabel.setText("Error: LF and HF dimensions do not match.")
            return

        # Combine
        combined = self.low_freq_image + self.high_freq_image
        combined = np.clip(combined, 0, 1)  # float32 in [0,1]

        # Create a new preview window (non-modal)
        self.combined_window = CombinedPreviewWindow(
            combined, 
            image_manager=self.image_manager,
            original_header=self.original_header,
            is_mono=self.is_mono
        )
        # Show it. Because we use `show()`, it won't block the main UI
        self.combined_window.show()


    # -----------------------------
    # Applying Frequency Separation (background thread)
    # -----------------------------
    def apply_frequency_separation(self):
        if self.image is None:
            self.fileLabel.setText("No input image loaded.")
            return

        self.showSpinner()

        if self.processing_thread and self.processing_thread.isRunning():
            self.processing_thread.quit()
            self.processing_thread.wait()

        # pass in 'tolerance' too
        self.processing_thread = FrequencySeperationThread(
            image=self.image,
            method=self.method,
            radius=self.radius,
            tolerance=self.tolerance
        )
        self.processing_thread.separation_done.connect(self.onSeparationDone)
        self.processing_thread.error_signal.connect(self.onSeparationError)
        self.processing_thread.start()

    def onSeparationDone(self, lf, hf):
        self.hideSpinner()
        self.low_freq_image = lf
        self.high_freq_image = hf
        self.update_previews()

    def onSeparationError(self, msg):
        self.hideSpinner()
        self.fileLabel.setText(f"Error during separation: {msg}")

    # -----------------------------
    # Spinner control
    # -----------------------------
    def showSpinner(self):
        self.spinnerLabel.show()
        self.spinnerMovie.start()

    def hideSpinner(self):
        self.spinnerLabel.hide()
        self.spinnerMovie.stop()

    # -----------------------------
    # Preview
    # -----------------------------
    def update_previews(self):
        """
        Render HF/LF images with current zoom_factor.
        HF gets an offset of +0.5 for display.
        """
        # Low Frequency
        if self.low_freq_image is not None:
            lf_disp = np.clip(self.low_freq_image, 0, 1)
            pixmap_lf = self._numpy_to_qpixmap(lf_disp)
            # Scale by zoom_factor (cast to int)
            scaled_lf = pixmap_lf.scaled(
                int(pixmap_lf.width() * self.zoom_factor),
                int(pixmap_lf.height() * self.zoom_factor),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.labelLF.setPixmap(scaled_lf)
            self.labelLF.resize(scaled_lf.size())
        else:
            self.labelLF.setText("Low Frequency")
            self.labelLF.resize(self.labelLF.sizeHint())

        # High Frequency
        if self.high_freq_image is not None:
            hf_disp = self.high_freq_image + 0.5
            hf_disp = np.clip(hf_disp, 0, 1)
            pixmap_hf = self._numpy_to_qpixmap(hf_disp)
            scaled_hf = pixmap_hf.scaled(
                int(pixmap_hf.width() * self.zoom_factor),
                int(pixmap_hf.height() * self.zoom_factor),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.labelHF.setPixmap(scaled_hf)
            self.labelHF.resize(scaled_hf.size())
        else:
            self.labelHF.setText("High Frequency")
            self.labelHF.resize(self.labelHF.sizeHint())

    def _numpy_to_qpixmap(self, img_float32):
        """
        Convert float32 [0,1] array (H,W) or (H,W,3) to a QPixmap for display.
        """
        if img_float32.ndim == 2:
            img_float32 = np.stack([img_float32]*3, axis=-1)

        img_ubyte = (img_float32 * 255).astype(np.uint8)
        h, w, ch = img_ubyte.shape
        bytes_per_line = ch * w
        q_img = QImage(img_ubyte.data, w, h, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(q_img)

class CombinedPreviewWindow(QWidget):
    """
    A pop-out window that shows the combined HF+LF image in a scrollable, zoomable preview.
    """
    def __init__(self, combined_image, image_manager, original_header=None, is_mono=False, parent=None):
        """
        :param combined_image: Float32 numpy array in [0,1], shape = (H,W) or (H,W,3).
        :param original_header: Optional metadata (for saving as FITS, etc.).
        :param is_mono: Boolean indicating grayscale vs. color.
        """
        super().__init__(parent)
        self.setWindowTitle("Combined HF + LF Preview")
        self.combined_image = combined_image
        self.image_manager = image_manager  # Reference to ImageManage
        self.original_header = original_header
        self.is_mono = is_mono

        # Zoom/panning
        self.zoom_factor = 1.0
        self.dragging = False
        self.last_mouse_pos = QPoint()

        self.initUI()
        # Render the combined image initially
        self.updatePreview()

    def initUI(self):
        main_layout = QVBoxLayout(self)
        self.setLayout(main_layout)

        # --- Top: Zoom / Fit / Save Buttons ---
        top_btn_layout = QHBoxLayout()
        self.zoom_in_btn = QPushButton("Zoom In", self)
        self.zoom_in_btn.clicked.connect(self.zoom_in)
        top_btn_layout.addWidget(self.zoom_in_btn)

        self.zoom_out_btn = QPushButton("Zoom Out", self)
        self.zoom_out_btn.clicked.connect(self.zoom_out)
        top_btn_layout.addWidget(self.zoom_out_btn)

        self.fit_btn = QPushButton("Fit to Preview", self)
        self.fit_btn.clicked.connect(self.fit_to_preview)
        top_btn_layout.addWidget(self.fit_btn)

        self.save_btn = QPushButton("Save Combined", self)
        self.save_btn.clicked.connect(self.save_combined_image)
        top_btn_layout.addWidget(self.save_btn)

        main_layout.addLayout(top_btn_layout)

        # --- Scroll Area with a QLabel for image ---
        self.scrollArea = QScrollArea(self)
        self.scrollArea.setWidgetResizable(False)
        self.imageLabel = QLabel(self)
        self.imageLabel.setAlignment(Qt.AlignCenter)

        # Put the label inside the scroll area
        self.scrollArea.setWidget(self.imageLabel)
        main_layout.addWidget(self.scrollArea)

        # Enable mouse-drag panning
        self.scrollArea.viewport().installEventFilter(self)

        # Provide a decent default window size
        self.resize(1000, 600)

    def eventFilter(self, source, event):
        if source == self.scrollArea.viewport():
            if event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                self.dragging = True
                self.last_mouse_pos = event.pos()
                return True
            elif event.type() == QEvent.MouseMove and self.dragging:
                delta = event.pos() - self.last_mouse_pos
                self.last_mouse_pos = event.pos()
                # Adjust scrollbars
                self.scrollArea.horizontalScrollBar().setValue(
                    self.scrollArea.horizontalScrollBar().value() - delta.x()
                )
                self.scrollArea.verticalScrollBar().setValue(
                    self.scrollArea.verticalScrollBar().value() - delta.y()
                )
                return True
            elif event.type() == QEvent.MouseButtonRelease and event.button() == Qt.LeftButton:
                self.dragging = False
                return True
        return super().eventFilter(source, event)

    def updatePreview(self):
        """
        Render the combined image into self.imageLabel at the current zoom_factor.
        """
        if self.combined_image is None:
            self.imageLabel.setText("No combined image.")
            return

        # Convert float32 [0,1] -> QPixmap
        pixmap = self.numpy_to_qpixmap(self.combined_image)
        # Scale by zoom_factor
        new_width = int(pixmap.width() * self.zoom_factor)
        new_height = int(pixmap.height() * self.zoom_factor)
        scaled = pixmap.scaled(new_width, new_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        # Update label
        self.imageLabel.setPixmap(scaled)
        self.imageLabel.resize(scaled.size())

    def numpy_to_qpixmap(self, img_float32):
        """
        Convert float32 [0,1] array (H,W) or (H,W,3) to QPixmap.
        """
        if img_float32.ndim == 2:
            # grayscale
            img_float32 = np.stack([img_float32]*3, axis=-1)
        img_ubyte = (np.clip(img_float32, 0, 1) * 255).astype(np.uint8)
        h, w, ch = img_ubyte.shape
        bytes_per_line = ch * w
        q_image = QImage(img_ubyte.data, w, h, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(q_image)

    # -----------------------------
    # Zoom
    # -----------------------------
    def zoom_in(self):
        self.zoom_factor *= 1.2
        self.updatePreview()

    def zoom_out(self):
        self.zoom_factor /= 1.2
        self.updatePreview()

    def fit_to_preview(self):
        """
        Adjust zoom_factor so the combined image width fits in the scrollArea width.
        """
        if self.combined_image is None:
            return

        # Get the actual image size
        h, w = self.combined_image.shape[:2]
        # The scrollArea's viewport is how much space we have to show it
        viewport_width = self.scrollArea.viewport().width()

        # Estimate new zoom factor so image fits horizontally
        # (You could also consider fitting by height or whichever is smaller.)
        # Must convert w from image to display pixel scale.
        # We'll guess the "base" is 1.0 => original width => we guess that is w pixels wide
        # So new_zoom = viewport_width / (w in original scale).
        new_zoom = viewport_width / float(w)
        if new_zoom < 0.01:
            new_zoom = 0.01

        self.zoom_factor = new_zoom
        self.updatePreview()

    # -----------------------------
    # Save
    # -----------------------------
    def save_combined_image(self):
        """
        Let the user save the combined image (float32 [0,1]) with a typical "Save As" dialog.
        - TIF/TIFF, FIT/FITS, XISF: prompt for 16-bit or 32-bit.
        - PNG/JPG/JPEG: automatically save as 8-bit (no prompt).
        - Otherwise, default to 8-bit.
        """
        if self.combined_image is None:
            return

        options = "Images (*.tif *.tiff *.fits *.fit *.png *.xisf);;All Files (*)"
        default_filename = "combined_image.tif"
        save_filename, _ = QFileDialog.getSaveFileName(
            self, 
            "Save Combined Image", 
            default_filename, 
            options
        )
        if not save_filename:
            return  # user canceled

        file_ext = os.path.splitext(save_filename)[1].lower().strip('.')  # e.g., 'tif', 'fits', 'png', etc.
        
        # Decide bit depth
        if file_ext in ['tif', 'tiff', 'fit', 'fits', 'xisf']:
            # Prompt user for bit depth
            bit_depth_options = ["16-bit", "32-bit floating point"]
            bit_depth, ok = QInputDialog.getItem(
                self,
                "Select Bit Depth",
                "Choose bit depth for saving:",
                bit_depth_options,
                1,  # default index = "32-bit floating point"
                False
            )
            if not ok:
                return  # user canceled the bit-depth dialog
        elif file_ext in ['png']:
            # Force 8-bit
            bit_depth = "8-bit"
        else:
            # Some other extension: default to "8-bit" (or you can ask user, or raise an error)
            bit_depth = "8-bit"

        # Now call your global save_image
        save_image(
            self.combined_image,         # float32 in [0,1]
            save_filename,
            original_format=file_ext,    # 'tif', 'fits', 'png', etc.
            bit_depth=bit_depth,
            original_header=self.original_header,
            is_mono=self.is_mono
        )

        QMessageBox.information(
            self,
            "Save Complete",
            f"Saved {bit_depth} {file_ext.upper()} image to:\n{os.path.basename(save_filename)}"
        )

        # Update ImageManager with the new combined image
        metadata = {
            'file_path': save_filename,
            'original_header': self.original_header,
            'bit_depth': bit_depth,
            'is_mono': self.is_mono
        }

        # Set the combined image in ImageManager's current slot
        self.image_manager.set_image(self.combined_image, metadata)

        # Optionally, update any labels or statuses if needed
        # For example, you might want to update the title or notify the user
        self.close()  # Close the preview window after saving, if desired        

class HFEnhancementThread(QThread):
    """
    A QThread that can:
      1) Scale HF by 'sharpen_scale' (if enabled)
      2) Wavelet-sharpen HF (if enabled)
      3) Denoise HF (if enabled)
    """
    enhancement_done = pyqtSignal(np.ndarray)
    error_signal = pyqtSignal(str)

    def __init__(
        self, 
        hf_image, 
        enable_scale=True,
        sharpen_scale=1.0, 
        enable_wavelet=True,
        wavelet_level=2, 
        wavelet_boost=1.2, 
        wavelet_name='db2',
        enable_denoise=False,
        denoise_strength=3.0,
        parent=None
    ):
        super().__init__(parent)
        self.hf_image = hf_image
        self.enable_scale = enable_scale
        self.sharpen_scale = sharpen_scale
        self.enable_wavelet = enable_wavelet
        self.wavelet_level = wavelet_level
        self.wavelet_boost = wavelet_boost
        self.wavelet_name = wavelet_name
        self.enable_denoise = enable_denoise
        self.denoise_strength = denoise_strength

    def run(self):
        try:
            # Make a copy so we don't mutate the original
            enhanced_hf = self.hf_image.copy()

            # 1) Sharpen Scale
            if self.enable_scale:
                enhanced_hf *= self.sharpen_scale

            # 2) Wavelet Sharpen
            if self.enable_wavelet:
                enhanced_hf = self.wavelet_sharpen(
                    enhanced_hf,
                    wavelet=self.wavelet_name,
                    level=self.wavelet_level,
                    boost=self.wavelet_boost
                )

            # 3) Denoise
            if self.enable_denoise:
                enhanced_hf = self.denoise_hf(enhanced_hf, self.denoise_strength)

            self.enhancement_done.emit(enhanced_hf.astype(np.float32))
        except Exception as e:
            self.error_signal.emit(str(e))

    # -------------------------------------
    # Wavelet Sharpen Methods
    # -------------------------------------
    def wavelet_sharpen(self, hf, wavelet='db2', level=2, boost=1.2):
        """
        Apply wavelet sharpening to the HF image.
        Handles both color and monochrome images.
        """
        # Check if the image is color or mono
        if hf.ndim == 3 and hf.shape[2] == 3:
            # Color image: process each channel separately
            channels = []
            for c in range(3):
                c_data = hf[..., c]
                c_sharp = self.wavelet_sharpen_mono(c_data, wavelet, level, boost)
                channels.append(c_sharp)
            # Stack the channels back into a color image
            return np.stack(channels, axis=-1)
        else:
            # Monochrome image
            return self.wavelet_sharpen_mono(hf, wavelet, level, boost)

    def wavelet_sharpen_mono(self, mono_hf, wavelet, level, boost):
        """
        Apply wavelet sharpening to a single-channel (monochrome) HF image.
        Ensures that the output image has the same dimensions as the input.
        """
        # Perform wavelet decomposition with 'periodization' mode to preserve dimensions
        coeffs = pywt.wavedec2(mono_hf, wavelet=wavelet, level=level, mode='periodization')

        # Boost the detail coefficients
        new_coeffs = [coeffs[0]]  # Approximation coefficients remain unchanged
        for detail in coeffs[1:]:
            cH, cV, cD = detail
            cH *= boost
            cV *= boost
            cD *= boost
            new_coeffs.append((cH, cV, cD))

        # Reconstruct the image with 'periodization' mode
        result = pywt.waverec2(new_coeffs, wavelet=wavelet, mode='periodization')

        # Ensure the reconstructed image has the same shape as the original
        original_shape = mono_hf.shape
        reconstructed_shape = result.shape

        if reconstructed_shape != original_shape:
            # Calculate the difference in dimensions                                            
            delta_h = reconstructed_shape[0] - original_shape[0]
            delta_w = reconstructed_shape[1] - original_shape[1]

            # Crop the excess pixels if the reconstructed image is larger
            if delta_h > 0 or delta_w > 0:
                result = result[:original_shape[0], :original_shape[1]]
            # Pad the image with zeros if it's smaller (rare, but for robustness)
            elif delta_h < 0 or delta_w < 0:
                pad_h = max(-delta_h, 0)
                pad_w = max(-delta_w, 0)
                result = np.pad(result, 
                               ((0, pad_h), (0, pad_w)), 
                               mode='constant', 
                               constant_values=0)

        return result

    # -------------------------------------
    # Denoise HF
    # -------------------------------------
    def denoise_hf(self, hf, strength=3.0):
        """
        Use OpenCV's fastNlMeansDenoisingColored or fastNlMeansDenoising for HF.
        Because HF can be negative, we offset +0.5 -> [0..1], scale -> [0..255].
        """
        # If color
        if hf.ndim == 3 and hf.shape[2] == 3:
            bgr = cv2.cvtColor(hf, cv2.COLOR_RGB2BGR)
            tmp = np.clip(bgr + 0.5, 0, 1)
            tmp8 = (tmp * 255).astype(np.uint8)
            # fastNlMeansDenoisingColored(src, None, hColor, hLuminance, templateWindowSize, searchWindowSize)
            denoised8 = cv2.fastNlMeansDenoisingColored(tmp8, None, strength, strength, 7, 21)
            denoised_f32 = denoised8.astype(np.float32) / 255.0 - 0.5
            denoised_rgb = cv2.cvtColor(denoised_f32, cv2.COLOR_BGR2RGB)
            return denoised_rgb
        else:
            # Mono
            tmp = np.clip(hf + 0.5, 0, 1)
            tmp8 = (tmp * 255).astype(np.uint8)
            denoised8 = cv2.fastNlMeansDenoising(tmp8, None, strength, 7, 21)
            denoised_f32 = denoised8.astype(np.float32) / 255.0 - 0.5
            return denoised_f32

class FrequencySeperationThread(QThread):
    """
    A QThread that performs frequency separation on a float32 [0,1] image array.
    This keeps the GUI responsive while processing.

    Signals:
        separation_done(np.ndarray, np.ndarray):
            Emitted with (low_freq, high_freq) images when finished.
        error_signal(str):
            Emitted if an error or exception occurs.
    """

    # Signal emitted when separation is complete. 
    # The arguments are low-frequency (LF) and high-frequency (HF) images.
    separation_done = pyqtSignal(np.ndarray, np.ndarray)

    # Signal emitted if there's an error during processing
    error_signal = pyqtSignal(str)

    def __init__(self, image, method='Gaussian', radius=5, tolerance=50, parent=None):
        """
        :param image: Float32 NumPy array in [0,1], shape = (H,W) or (H,W,3).
        :param method: 'Gaussian', 'Median', or 'Bilateral' (default: 'Gaussian').
        :param radius: Numeric value controlling the filter's strength (e.g., Gaussian sigma).
        :param mirror: Boolean to indicate if border handling is mirrored (optional example param).
        """
        super().__init__(parent)
        self.image = image
        self.method = method
        self.radius = radius
        self.tolerance = tolerance

    def run(self):
        try:
            # Convert the input image from RGB to BGR if it's 3-channel
            if self.image.ndim == 3 and self.image.shape[2] == 3:
                bgr = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)
            else:
                # If mono, just use it as is
                bgr = self.image.copy()

            # Choose the filter based on self.method
            if self.method == 'Gaussian':
                # For Gaussian, interpret radius as sigma
                low_bgr = cv2.GaussianBlur(bgr, (0, 0), self.radius)
            elif self.method == 'Median':
                # For Median, the radius is the kernel size (must be odd)
                ksize = max(1, int(self.radius) // 2 * 2 + 1)
                low_bgr = cv2.medianBlur(bgr, ksize)
            elif self.method == 'Bilateral':
                # Example usage: interpret "tolerance" as a fraction of the default 50
                # so if tolerance=50 => sigmaColor=50*(50/100)=25, sigmaSpace=25
                # Or do your own logic for how tolerance modifies Bilateral
                sigma = 50 * (self.tolerance / 100.0)
                d = int(self.radius)
                low_bgr = cv2.bilateralFilter(bgr, d, sigma, sigma)
            else:
                # Fallback to Gaussian if unknown
                low_bgr = cv2.GaussianBlur(bgr, (0, 0), self.radius)

            # Convert low frequency image back to RGB if it's 3-channel
            if low_bgr.ndim == 3 and low_bgr.shape[2] == 3:
                low_rgb = cv2.cvtColor(low_bgr, cv2.COLOR_BGR2RGB)
            else:
                low_rgb = low_bgr

            # Calculate the high frequency
            # (note: keep in float32 to preserve negative/positive values)
            high_rgb = self.image - low_rgb

            # Emit the results
            self.separation_done.emit(low_rgb, high_rgb)

        except Exception as e:
            # Any error gets reported via the error_signal
            self.error_signal.emit(str(e))

class PalettePickerProcessingThread(QThread):
    """
    Thread for processing images to prevent UI freezing.
    """
    preview_generated = pyqtSignal(np.ndarray)

    def __init__(self, ha_image, oiii_image, sii_image, osc1_image, osc2_image, ha_to_oii_ratio, enable_star_stretch, stretch_factor):
        super().__init__()
        self.ha_image = ha_image
        self.oiii_image = oiii_image
        self.sii_image = sii_image
        self.osc1_image = osc1_image  # Added for OSC1
        self.osc2_image = osc2_image  # Added for OSC2
        self.ha_to_oii_ratio = ha_to_oii_ratio
        self.enable_star_stretch = enable_star_stretch
        self.stretch_factor = stretch_factor

    def run(self):
        """
        Perform image processing to generate a combined preview.
        """
        try:
            combined_ha = self.ha_image.copy() if self.ha_image is not None else None
            combined_oiii = self.oiii_image.copy() if self.oiii_image is not None else None

            # Process OSC1 if available
            if self.osc1_image is not None:
                # Extract synthetic Ha and OIII from OSC1
                ha_osc1 = self.osc1_image[:, :, 0]  # Red channel -> Ha
                oiii_osc1 = np.mean(self.osc1_image[:, :, 1:3], axis=2)  # Average of green and blue channels -> OIII

                # Apply stretching if enabled
                if self.enable_star_stretch:
                    ha_osc1 = stretch_mono_image(ha_osc1, target_median=self.stretch_factor)
                    oiii_osc1 = stretch_mono_image(oiii_osc1, target_median=self.stretch_factor)

                # Combine with existing Ha and OIII
                if combined_ha is not None:
                    combined_ha = (combined_ha * 0.5) + (ha_osc1 * 0.5)
                else:
                    combined_ha = ha_osc1

                if combined_oiii is not None:
                    combined_oiii = (combined_oiii * 0.5) + (oiii_osc1 * 0.5)
                else:
                    combined_oiii = oiii_osc1

            # Process OSC2 if available
            if self.osc2_image is not None:
                # Extract synthetic Ha and OIII from OSC2
                ha_osc2 = self.osc2_image[:, :, 0]  # Red channel -> Ha
                oiii_osc2 = np.mean(self.osc2_image[:, :, 1:3], axis=2)  # Average of green and blue channels -> OIII

                # Apply stretching if enabled
                if self.enable_star_stretch:
                    ha_osc2 = stretch_mono_image(ha_osc2, target_median=self.stretch_factor)
                    oiii_osc2 = stretch_mono_image(oiii_osc2, target_median=self.stretch_factor)

                # Combine with existing Ha and OIII
                if combined_ha is not None:
                    combined_ha = (combined_ha * 0.5) + (ha_osc2 * 0.5)
                else:
                    combined_ha = ha_osc2

                if combined_oiii is not None:
                    combined_oiii = (combined_oiii * 0.5) + (oiii_osc2 * 0.5)
                else:
                    combined_oiii = oiii_osc2

            # Ensure that combined Ha and OIII are present
            if combined_ha is not None and combined_oiii is not None:
                # Combine Ha and OIII based on the specified ratio
                combined = (combined_ha * self.ha_to_oii_ratio) + (combined_oiii * (1 - self.ha_to_oii_ratio))

                # Apply stretching if enabled
                if self.enable_star_stretch:
                    combined = stretch_mono_image(combined, target_median=self.stretch_factor)

                # Incorporate SII channel if available
                if self.sii_image is not None:
                    combined = combined + self.sii_image
                    # Normalize to prevent overflow
                    combined = self.normalize_image(combined)

                self.preview_generated.emit(combined)
            else:
                # If required channels are missing, emit a dummy image or handle accordingly
                combined = np.zeros((100, 100, 3))  # Dummy image
                self.preview_generated.emit(combined)
        except Exception as e:
            print(f"Error in PalettePickerProcessingThread: {e}")
            self.preview_generated.emit(None)

    @staticmethod
    def normalize_image(image):
        return image


class PerfectPalettePickerTab(QWidget):
    """
    Perfect Palette Picker Tab for Seti Astro Suite.
    Creates 12 popular NB palettes from Ha/OIII/SII or OSC channels.
    """
    def __init__(self, image_manager=None):
        super().__init__()
        self.image_manager = image_manager  # Reference to the ImageManager
        self.initUI()
        self.ha_image = None
        self.oiii_image = None
        self.sii_image = None
        self.osc1_image = None  # Added for OSC1
        self.osc2_image = None  # Added for OSC2
        self.combined_image = None
        self.is_mono = False
        # Filenames
        self.ha_filename = None
        self.oiii_filename = None
        self.sii_filename = None
        self.osc1_filename = None  # Added for OSC1
        self.osc2_filename = None  # Added for OSC2      
        self.filename = None  # Store the selected file path
        self.zoom_factor = 1.0  # Initialize to 1.0 for normal size
        self.processing_thread = None
        self.original_header = None
        self.original_pixmap = None  # To store the original QPixmap for zooming
        self.bit_depth = "Unknown"
        self.dragging = False
        self.last_mouse_position = None
        self.selected_palette_button = None
        self.selected_palette = None  # To track the currently selected palette
        
        # Preview scale factor
        self.preview_scale = 1  # Start at no scaling

        if self.image_manager:
            # Connect to ImageManager's image_changed signal if needed
            # self.image_manager.image_changed.connect(self.on_image_changed)
            pass

    def initUI(self):
        main_layout = QHBoxLayout()

        # Left column for controls
        left_widget = QWidget(self)
        left_layout = QVBoxLayout(left_widget)
        left_widget.setFixedWidth(300)

        # Title label
        title_label = QLabel("Perfect Palette Picker v1.0", self)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setFont(QFont("Helvetica", 14, QFont.Bold))
        left_layout.addWidget(title_label)

        # Instruction label
        instruction_label = QLabel(self)
        instruction_label.setText(
            "Instructions:\n"
            "1. Add narrowband images or an OSC camera image.\n"
            "2. Check the 'Linear Input Data' checkbox if the images are linear.\n"
            "3. Click 'Create Palettes' to generate the palettes.\n"
            "4. Use the Zoom buttons to zoom in and out.\n"
            "5. Resize the UI by dragging the lower right corner.\n"
            "6. Click on a palette from the preview selection to generate that palette.\n\n"
            "Multiple palettes can be generated."
        )
        instruction_label.setWordWrap(True)
        instruction_label.setAlignment(Qt.AlignLeft)
        instruction_label.setStyleSheet(
            "font-size: 8pt; padding: 10px;"
        )
        instruction_label.setFixedHeight(200)
        left_layout.addWidget(instruction_label)

        # "Linear Input Data" checkbox
        self.linear_checkbox = QCheckBox("Linear Input Data", self)
        self.linear_checkbox.setChecked(True)
        self.linear_checkbox.setToolTip(
            "When checked, we apply the 0.25 stretch for previews/final images."
        )
        left_layout.addWidget(self.linear_checkbox)

        # Load buttons for Ha, OIII, SII, OSC
        self.load_ha_button = QPushButton("Load Ha Image", self)
        self.load_ha_button.clicked.connect(lambda: self.load_image('Ha'))
        left_layout.addWidget(self.load_ha_button)

        self.ha_label = QLabel("No Ha image loaded.", self)
        self.ha_label.setWordWrap(True)
        left_layout.addWidget(self.ha_label)

        self.load_oiii_button = QPushButton("Load OIII Image", self)
        self.load_oiii_button.clicked.connect(lambda: self.load_image('OIII'))
        left_layout.addWidget(self.load_oiii_button)

        self.oiii_label = QLabel("No OIII image loaded.", self)
        self.oiii_label.setWordWrap(True)
        left_layout.addWidget(self.oiii_label)

        self.load_sii_button = QPushButton("Load SII Image", self)
        self.load_sii_button.clicked.connect(lambda: self.load_image('SII'))
        left_layout.addWidget(self.load_sii_button)

        self.sii_label = QLabel("No SII image loaded.", self)
        self.sii_label.setWordWrap(True)
        left_layout.addWidget(self.sii_label)

        # **Add OSC1 Load Button and Label**
        self.load_osc1_button = QPushButton("Load OSC HaO3 Image", self)
        self.load_osc1_button.clicked.connect(lambda: self.load_image('OSC1'))
        left_layout.addWidget(self.load_osc1_button)

        self.osc1_label = QLabel("No OSC HaO3 image loaded.", self)
        self.osc1_label.setWordWrap(True)
        left_layout.addWidget(self.osc1_label)

        # **Add OSC2 Load Button and Label**
        self.load_osc2_button = QPushButton("Load OSC S2O3 Image", self)
        self.load_osc2_button.clicked.connect(lambda: self.load_image('OSC2'))
        left_layout.addWidget(self.load_osc2_button)

        self.osc2_label = QLabel("No OSC S2O3 image loaded.", self)
        self.osc2_label.setWordWrap(True)
        left_layout.addWidget(self.osc2_label)

        # "Create Palettes" button
        create_palettes_button = QPushButton("Create Palettes", self)
        create_palettes_button.clicked.connect(self.prepare_preview_palettes)
        left_layout.addWidget(create_palettes_button)

        # Spacer
        left_layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

        self.push_palette_button = QPushButton("Push Final Palette for Further Processing")
        self.push_palette_button.clicked.connect(self.push_final_palette_to_image_manager)
        left_layout.addWidget(self.push_palette_button)

        # Spacer
        left_layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

        # Add a "Clear All Images" button
        self.clear_all_button = QPushButton("Clear All Images", self)
        self.clear_all_button.clicked.connect(self.clear_all_images)
        left_layout.addWidget(self.clear_all_button)


        # Footer
        footer_label = QLabel("""
            Written by Franklin Marek<br>
            <a href='http://www.setiastro.com'>www.setiastro.com</a>
        """)
        footer_label.setAlignment(Qt.AlignCenter)
        footer_label.setOpenExternalLinks(True)
        footer_label.setFont(QFont("Helvetica", 8))
        left_layout.addWidget(footer_label)

        # Add the left widget to the main layout
        main_layout.addWidget(left_widget)

        # Right column for previews and controls
        right_widget = QWidget(self)
        right_layout = QVBoxLayout(right_widget)

        # Zoom controls
        zoom_layout = QHBoxLayout()
        zoom_in_button = QPushButton("Zoom In", self)
        zoom_in_button.clicked.connect(self.zoom_in)
        zoom_layout.addWidget(zoom_in_button)

        zoom_out_button = QPushButton("Zoom Out", self)
        zoom_out_button.clicked.connect(self.zoom_out)
        zoom_layout.addWidget(zoom_out_button)

        fit_to_preview_button = QPushButton("Fit to Preview", self)
        fit_to_preview_button.clicked.connect(self.fit_to_preview)
        zoom_layout.addWidget(fit_to_preview_button)

        right_layout.addLayout(zoom_layout)

        # Scroll area for image preview
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.installEventFilter(self)
        self.image_label.setMouseTracking(True)

        self.scroll_area.setWidget(self.image_label)
        self.scroll_area.setMinimumSize(400, 250)
        right_layout.addWidget(self.scroll_area, stretch=1)


        # Preview thumbnails grid
        self.thumbs_grid = QGridLayout()
        self.palette_names = [
            "SHO", "HOO", "HSO", "HOS",
            "OSS", "OHH", "OSH", "OHS",
            "HSS", "Realistic1", "Realistic2", "Foraxx"
        ]
        self.thumbnail_buttons = []
        row = 0
        col = 0

        for palette in self.palette_names:
            button = QPushButton(palette, self)
            button.setMinimumSize(200, 100)  # Minimum size for buttons
            button.setMaximumHeight(100)  # Fixed height for buttons
            button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)  # Expand width, fixed height
            button.setIcon(QIcon())  # Placeholder, will be set later
            button.clicked.connect(lambda checked, p=palette: self.generate_final_palette_image(p))
            button.setIconSize(QSize(200, 100))
            button.setIcon(QIcon())  # Placeholder, will be set later
            self.thumbnail_buttons.append(button)
            self.thumbs_grid.addWidget(button, row, col)
            col += 1
            if col >= 4:
                col = 0
                row += 1

        # Wrap the grid in a QWidget for better layout handling
        thumbs_widget = QWidget()
        thumbs_widget.setLayout(self.thumbs_grid)
        thumbs_widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

        # Add the thumbnails widget to the layout
        right_layout.addWidget(thumbs_widget, stretch=0)

        # Save button
        save_button = QPushButton("Save Combined Image", self)
        save_button.clicked.connect(self.save_image)
        right_layout.addWidget(save_button)

        # Status label
        self.status_label = QLabel("", self)
        self.status_label.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(self.status_label)

        # Add the right widget to the main layout
        main_layout.addWidget(right_widget)

        self.setLayout(main_layout)
        self.setWindowTitle("Perfect Palette Picker v1.0")

    def clear_all_images(self):
        """
        Clears all loaded images (Ha, OIII, SII, OSC1, OSC2).
        """
        # Clear Ha image and reset filename and label
        self.ha_image = None
        self.ha_filename = None
        self.ha_label.setText("No Ha image loaded.")

        # Clear OIII image and reset filename and label
        self.oiii_image = None
        self.oiii_filename = None
        self.oiii_label.setText("No OIII image loaded.")

        # Clear SII image and reset filename and label
        self.sii_image = None
        self.sii_filename = None
        self.sii_label.setText("No SII image loaded.")

        # Clear OSC1 image and reset filename and label
        self.osc1_image = None
        self.osc1_filename = None
        self.osc1_label.setText("No OSC HaO3 image loaded.")

        # Clear OSC2 image and reset filename and label
        self.osc2_image = None
        self.osc2_filename = None
        self.osc2_label.setText("No OSC S2O3 image loaded.")

        # Clean up preview windows
        self.cleanup_preview_windows()        

        # Update the status label
        self.status_label.setText("All images cleared.")


    def load_image(self, image_type):
        """
        Opens a file dialog to load an image based on the image type.
        """
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_filter = "Images (*.png *.tif *.tiff *.fits *.fit *.xisf)"
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            f"Select {image_type} Image",
            "",
            file_filter,
            options=options
        )
        if file_path:
            image, original_header, bit_depth, is_mono = load_image(file_path)
            if image is None:
                QMessageBox.critical(self, "Error", f"Failed to load {image_type} image.")
                return
            if image_type == 'Ha':
                self.ha_image = image
                self.ha_filename = file_path
                self.original_header = original_header
                self.bit_depth = bit_depth
                self.is_mono = is_mono
                self.ha_label.setText(f"Loaded: {os.path.basename(file_path)}")
            elif image_type == 'OIII':
                self.oiii_image = image
                self.oiii_filename = file_path
                self.original_header = original_header
                self.bit_depth = bit_depth
                self.is_mono = is_mono
                self.oiii_label.setText(f"Loaded: {os.path.basename(file_path)}")
            elif image_type == 'SII':
                self.sii_image = image
                self.sii_filename = file_path
                self.original_header = original_header
                self.bit_depth = bit_depth
                self.is_mono = is_mono
                self.sii_label.setText(f"Loaded: {os.path.basename(file_path)}")
            elif image_type == 'OSC1':
                self.osc1_image = image
                self.osc1_filename = file_path
                self.original_header = original_header
                self.bit_depth = bit_depth
                self.is_mono = is_mono
                self.osc1_label.setText(f"Loaded: {os.path.basename(file_path)}")
            elif image_type == 'OSC2':
                self.osc2_image = image
                self.osc2_filename = file_path
                self.original_header = original_header
                self.bit_depth = bit_depth
                self.is_mono = is_mono
                self.osc2_label.setText(f"Loaded: {os.path.basename(file_path)}")

            # Apply stretching if linear input is checked and image is mono
            if self.linear_checkbox.isChecked() and is_mono:
                if image_type == 'Ha':
                    self.ha_image = stretch_mono_image(self.ha_image, target_median=0.25)
                elif image_type == 'OIII':
                    self.oiii_image = stretch_mono_image(self.oiii_image, target_median=0.25)
                elif image_type == 'SII':
                    self.sii_image = stretch_mono_image(self.sii_image, target_median=0.25)
                elif image_type in ['OSC1', 'OSC2']:
                    # Assuming OSC has multiple channels; stretching would be handled during processing
                    pass

            self.status_label.setText(f"{image_type} image loaded successfully.")
        else:
            self.status_label.setText(f"{image_type} image loading canceled.")

    def prepare_preview_palettes(self):
        """
        
        Prepares the preview thumbnails for each palette based on selected images.
        """
        have_ha = self.ha_image is not None
        have_oiii = self.oiii_image is not None
        have_sii = self.sii_image is not None
        have_osc1 = self.osc1_image is not None
        have_osc2 = self.osc2_image is not None

        print(f"prepare_preview_palettes() => Ha: {have_ha} | OIII: {have_oiii} | SII: {have_sii} | OSC1: {have_osc1} | OSC2: {have_osc2}")



        # Initialize combined channels
        combined_ha = self.ha_image.copy() if self.ha_image is not None else None
        combined_oiii = self.oiii_image.copy() if self.oiii_image is not None else None
        combined_sii = self.sii_image.copy() if self.sii_image is not None else None  # Initialize combined SII

        # Process OSC1 if available
        if have_osc1:
            # Extract synthetic Ha and OIII from OSC1
            ha_osc1 = self.osc1_image[:, :, 0]  # Red channel -> Ha
            oiii_osc1 = np.mean(self.osc1_image[:, :, 1:3], axis=2)  # Average of green and blue channels -> OIII

            # Apply stretching if enabled
            if self.linear_checkbox.isChecked():
                ha_osc1 = stretch_mono_image(ha_osc1, target_median=0.25)
                oiii_osc1 = stretch_mono_image(oiii_osc1, target_median=0.25)

            # Combine with existing Ha and OIII
            if combined_ha is not None:
                combined_ha = (combined_ha * 0.5) + (ha_osc1 * 0.5)
            else:
                combined_ha = ha_osc1

            if combined_oiii is not None:
                combined_oiii = (combined_oiii * 0.5) + (oiii_osc1 * 0.5)
            else:
                combined_oiii = oiii_osc1

        # Process OSC2 if available
        if have_osc2:
            # Extract synthetic SII from OSC2 red channel
            sii_osc2 = self.osc2_image[:, :, 0]  # Red channel -> SII
            oiii_osc2 = np.mean(self.osc2_image[:, :, 1:3], axis=2)  # Average of green and blue channels -> OIII

            # Apply stretching if enabled
            if self.linear_checkbox.isChecked():
                sii_osc2 = stretch_mono_image(sii_osc2, target_median=0.25)
                oiii_osc2 = stretch_mono_image(oiii_osc2, target_median=0.25)

            # Combine with existing SII
            if combined_sii is not None:
                combined_sii = (combined_sii * 0.5) + (sii_osc2 * 0.5)
            else:
                combined_sii = sii_osc2

            if combined_oiii is not None:
                combined_oiii = (combined_oiii * 0.5) + (oiii_osc2 * 0.5)
            else:
                combined_oiii = oiii_osc2    

        # Assign combined images back to self.ha_image, self.oiii_image, and self.sii_image
        self.ha_image = combined_ha
        self.oiii_image = combined_oiii
        self.sii_image = combined_sii  # Updated SII image

        # Ensure images are single-channel
        def ensure_single_channel(image, image_type):
            if image is not None:
                if image.ndim == 3:
                    if image.shape[2] == 1:
                        image = image[:, :, 0]
                        print(f"Converted {image_type} image to single channel: {image.shape}")
                    else:
                        # If image has multiple channels, retain the first channel
                        image = image[:, :, 0]
                        print(f"Extracted first channel from multi-channel {image_type} image: {image.shape}")
                return image
            return None

        self.ha_image = ensure_single_channel(self.ha_image, 'Ha')
        self.oiii_image = ensure_single_channel(self.oiii_image, 'OIII')
        self.sii_image = ensure_single_channel(self.sii_image, 'SII')

        print(f"Combined Ha image shape: {self.ha_image.shape if self.ha_image is not None else 'None'}")
        print(f"Combined OIII image shape: {self.oiii_image.shape if self.oiii_image is not None else 'None'}")
        print(f"Combined SII image shape: {self.sii_image.shape if self.sii_image is not None else 'None'}")

        # Validate required channels
        # Allow if (Ha and OIII) or (SII and OIII) are present
        if not ((self.ha_image is not None and self.oiii_image is not None) or
                (self.sii_image is not None and self.oiii_image is not None)):
            QMessageBox.warning(
                self,
                "Warning",
                "Please load at least Ha and OIII images or SII and OIII images to create palettes."
            )
            self.status_label.setText("Insufficient images loaded.")
            return

        # Start processing thread to generate previews
        ha_to_oii_ratio = 0.3  # Example ratio; adjust as needed
        enable_star_stretch = self.linear_checkbox.isChecked()
        stretch_factor = 0.25  # Example stretch factor; adjust as needed

        self.processing_thread = PalettePickerProcessingThread(
            ha_image=self.ha_image,
            oiii_image=self.oiii_image,
            sii_image=self.sii_image,
            osc1_image=None,  # OSC1 is already processed
            osc2_image=None,  # OSC2 is already processed
            ha_to_oii_ratio=ha_to_oii_ratio,
            enable_star_stretch=enable_star_stretch,
            stretch_factor=stretch_factor
        )
        self.processing_thread.preview_generated.connect(self.update_preview_thumbnails)
        self.processing_thread.start()

        self.status_label.setText("Generating preview palettes...")



    def update_preview_thumbnails(self, combined_preview):
        """
        Updates the preview thumbnails with the generated combined preview.
        Downsamples the images for efficient processing of mini-previews.
        """
        if combined_preview is None:
            # Only update the text overlays
            for i, palette in enumerate(self.palette_names):
                pixmap = self.thumbnail_buttons[i].icon().pixmap(self.thumbnail_buttons[i].iconSize())
                if pixmap.isNull():
                    print(f"Failed to retrieve pixmap for palette '{palette}'. Skipping.")
                    continue
                text_color = Qt.green if self.selected_palette == palette else Qt.white
                painter = QPainter(pixmap)
                painter.setRenderHint(QPainter.Antialiasing)
                painter.setPen(QPen(text_color))
                painter.setFont(QFont("Helvetica", 8))
                painter.drawText(pixmap.rect(), Qt.AlignCenter, palette)
                painter.end()
                self.thumbnail_buttons[i].setIcon(QIcon(pixmap))
                QApplication.processEvents()

            return

        def downsample_image(image, factor=8):
            """
            Downsample the image by an integer factor using cv2.resize.
            """
            if image is not None:
                height, width = image.shape[:2]
                new_size = (max(1, width // factor), max(1, height // factor))  # Ensure size is at least 1x1
                return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
            return image

        # Downsample images
        ha = downsample_image(self.ha_image)
        oiii = downsample_image(self.oiii_image)
        sii = downsample_image(self.sii_image)

        # Helper function to extract single channel
        def extract_channel(image):
            return image if image is not None and image.ndim == 2 else (image[:, :, 0] if image is not None else None)

        # Helper function for channel substitution
        def get_channel(preferred, substitute):
            return preferred if preferred is not None else substitute

        for i, palette in enumerate(self.palette_names):
            text_color = Qt.green if self.selected_palette == palette else Qt.white

            # Determine availability
            ha_available = self.ha_image is not None
            sii_available = self.sii_image is not None

            # Define substitution channels
            substituted_ha = sii if not ha_available and sii_available else ha
            substituted_sii = ha if not sii_available and ha_available else sii

            # Map channels based on palette
            if palette == "SHO":
                r = get_channel(extract_channel(sii), substituted_ha)
                g = get_channel(extract_channel(ha), substituted_sii)
                b = extract_channel(oiii)
            elif palette == "HOO":
                r = get_channel(extract_channel(ha), substituted_sii)
                g = extract_channel(oiii)
                b = extract_channel(oiii)
            elif palette == "HSO":
                r = get_channel(extract_channel(ha), substituted_sii)
                g = get_channel(extract_channel(sii), substituted_ha)
                b = extract_channel(oiii)
            elif palette == "HOS":
                r = get_channel(extract_channel(ha), substituted_sii)
                g = extract_channel(oiii)
                b = get_channel(extract_channel(sii), substituted_ha)
            elif palette == "OSS":
                r = extract_channel(oiii)
                g = get_channel(extract_channel(sii), substituted_ha)
                b = get_channel(extract_channel(sii), substituted_ha)
            elif palette == "OHH":
                r = extract_channel(oiii)
                g = get_channel(extract_channel(ha), substituted_sii)
                b = get_channel(extract_channel(ha), substituted_sii)
            elif palette == "OSH":
                r = extract_channel(oiii)
                g = get_channel(extract_channel(sii), substituted_ha)
                b = get_channel(extract_channel(ha), substituted_sii)
            elif palette == "OHS":
                r = extract_channel(oiii)
                g = get_channel(extract_channel(ha), substituted_sii)
                b = get_channel(extract_channel(sii), substituted_ha)
            elif palette == "HSS":
                r = get_channel(extract_channel(ha), substituted_sii)
                g = get_channel(extract_channel(sii), substituted_ha)
                b = get_channel(extract_channel(sii), substituted_ha)
            elif palette in ["Realistic1", "Realistic2", "Foraxx"]:
                r, g, b = self.map_special_palettes(palette, ha, oiii, sii)
            else:
                # Fallback to SHO
                r, g, b = self.map_channels("SHO", ha, oiii, sii)

            # Replace NaNs and clip to [0, 1]
            r = np.clip(np.nan_to_num(r, nan=0.0, posinf=1.0, neginf=0.0), 0, 1) if r is not None else None
            g = np.clip(np.nan_to_num(g, nan=0.0, posinf=1.0, neginf=0.0), 0, 1) if g is not None else None
            b = np.clip(np.nan_to_num(b, nan=0.0, posinf=1.0, neginf=0.0), 0, 1) if b is not None else None

            if r is None or g is None or b is None:
                print(f"One of the channels is None for palette '{palette}'. Skipping this palette.")
                self.thumbnail_buttons[i].setIcon(QIcon())
                self.thumbnail_buttons[i].setText(palette)
                continue

            combined = self.combine_channels_to_color([r, g, b], f"Preview_{palette}")
            if combined is not None:
                # Convert NumPy array to QImage
                q_image = self.numpy_to_qimage(combined)
                if q_image.isNull():
                    print(f"Failed to convert preview for palette '{palette}' to QImage.")
                    continue

                pixmap = QPixmap.fromImage(q_image)
                if pixmap.isNull():
                    print(f"Failed to create QPixmap for palette '{palette}'.")
                    continue

                # Scale pixmap
                scaled_pixmap = pixmap.scaled(
                    int(pixmap.width() * self.preview_scale),
                    int(pixmap.height() * self.preview_scale),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )

                # Add text overlay
                painter = QPainter(scaled_pixmap)
                painter.setRenderHint(QPainter.Antialiasing)
                painter.setPen(QPen(text_color))
                painter.setFont(QFont("Helvetica", 8))
                painter.drawText(scaled_pixmap.rect(), Qt.AlignCenter, palette)
                painter.end()

                # Set pixmap to the corresponding button
                self.thumbnail_buttons[i].setIcon(QIcon(scaled_pixmap))
                self.thumbnail_buttons[i].setIconSize(scaled_pixmap.size())
                self.thumbnail_buttons[i].setToolTip(f"Palette: {palette}")
                QApplication.processEvents()
            else:
                self.thumbnail_buttons[i].setIcon(QIcon())
                self.thumbnail_buttons[i].setText(palette)

        self.status_label.setText("Preview palettes generated successfully.")





    def generate_final_palette_image(self, palette_name):
        """
        Generates the final combined image for the selected palette.
        Handles substitution of SII for Ha or Ha for SII if one is missing.
        """
        try:
            print(f"Generating final palette image for: {palette_name}")
            
            # Determine availability
            ha_available = self.ha_image is not None
            sii_available = self.sii_image is not None
            
            # Define substitution
            if not ha_available and sii_available:
                # Substitute SII for Ha
                substituted_ha = self.sii_image
                substituted_sii = None
                print("Substituting SII for Ha.")
            elif not sii_available and ha_available:
                # Substitute Ha for SII
                substituted_sii = self.ha_image
                substituted_ha = None
                print("Substituting Ha for SII.")
            else:
                substituted_ha = self.ha_image
                substituted_sii = self.sii_image
            
            # Temporarily assign substituted channels
            original_ha = self.ha_image
            original_sii = self.sii_image
            
            self.ha_image = substituted_ha
            self.sii_image = substituted_sii
            
            # Combine channels
            combined_image = self.combine_channels(palette_name)
            
            # Restore original channels
            self.ha_image = original_ha
            self.sii_image = original_sii
            
            if combined_image is not None:
                # Ensure the combined image has the correct shape
                if combined_image.ndim == 4 and combined_image.shape[3] == 3:
                    combined_image = combined_image[:, :, :, 0]  # Remove the extra dimension

                # Convert to QImage
                q_image = self.numpy_to_qimage(combined_image)
                if q_image.isNull():
                    raise ValueError(f"Failed to convert combined image for palette '{palette_name}' to QImage.")

                pixmap = QPixmap.fromImage(q_image)
                if pixmap.isNull():
                    raise ValueError(f"Failed to create QPixmap for palette '{palette_name}'.")

                # Scale the pixmap based on zoom factor
                scaled_pixmap = pixmap.scaled(
                    int(pixmap.width() * self.zoom_factor),
                    int(pixmap.height() * self.zoom_factor),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )

                # Display the scaled pixmap in the main preview area
                self.image_label.setPixmap(scaled_pixmap)
                self.image_label.resize(scaled_pixmap.size())
                self.combined_image = combined_image
                self.status_label.setText(f"Final palette '{palette_name}' generated successfully.")

                self.selected_palette = palette_name
                self.update_preview_thumbnails(None)  # Trigger re-render with updated text colors

            else:
                raise ValueError(f"Failed to generate combined image for palette '{palette_name}'.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to generate final image: {e}")
            self.status_label.setText(f"Failed to generate palette '{palette_name}'.")
            print(f"[Error] {e}")

    def highlight_selected_button(self, palette_name):
        """
        Highlights the clicked button by changing its text color and resets others.
        """
        for button in self.thumbnail_buttons:
            if button.text() == palette_name:
                # Change text color to indicate selection
                button.setStyleSheet("color: green; font-weight: bold;")
                self.selected_palette_button = button
            else:
                # Reset text color for non-selected buttons
                button.setStyleSheet("")


    def combine_channels(self, palette_name):
        """
        Combines Ha, OIII, SII channels based on the palette name.
        Ensures that all combined channel values are within the [0, 1] range.
        """
        if palette_name in self.palette_names[:9]:  # Standard palettes
            r, g, b = self.map_channels(palette_name, self.ha_image, self.oiii_image, self.sii_image)
        elif palette_name in self.palette_names[9:]:  # Special palettes
            r, g, b = self.map_special_palettes(palette_name, self.ha_image, self.oiii_image, self.sii_image)
        else:
            # Fallback to SHO
            r, g, b = self.map_channels("SHO", self.ha_image, self.oiii_image, self.sii_image)

        if r is not None and g is not None and b is not None:
            # Replace NaN and Inf with 0
            r = np.nan_to_num(r, nan=0.0, posinf=1.0, neginf=0.0)
            g = np.nan_to_num(g, nan=0.0, posinf=1.0, neginf=0.0)
            b = np.nan_to_num(b, nan=0.0, posinf=1.0, neginf=0.0)

            # Normalize to [0,1]
            r = np.clip(r, 0, 1)
            g = np.clip(g, 0, 1)
            b = np.clip(b, 0, 1)

            # Ensure single-channel
            if r.ndim == 3:
                r = r[:, :, 0]
            if g.ndim == 3:
                g = g[:, :, 0]
            if b.ndim == 3:
                b = b[:, :, 0]

            combined = np.stack([r, g, b], axis=2)
            return combined
        else:
            return None


    def combine_channels_to_color(self, channels, output_id):
        """
        Combines three grayscale images into an RGB image.
        Ensures that all channels are consistent and have no extra dimensions.
        """
        try:
            # Validate input channels
            if len(channels) != 3:
                raise ValueError(f"Expected 3 channels, got {len(channels)}")
            
            # Ensure all channels have the same shape
            for i, channel in enumerate(channels):
                if channel is None:
                    raise ValueError(f"Channel {i} is None.")
                if channel.shape != channels[0].shape:
                    raise ValueError(f"Channel {i} has shape {channel.shape}, expected {channels[0].shape}")
            
            # Ensure all channels are 2D
            channels = [channel[:, :, 0] if channel.ndim == 3 else channel for channel in channels]
            
            # Debugging: Print channel shapes after extraction
            for idx, channel in enumerate(channels):
                print(f"Channel {idx} shape after extraction: {channel.shape}")
            
            # Stack channels along the third axis to create RGB
            rgb_image = np.stack(channels, axis=2)
            print(f"Combined RGB image shape: {rgb_image.shape}")
            return rgb_image
        except Exception as e:
            print(f"Error in combine_channels_to_color: {e}")
            return None

    def map_channels(self, palette_name, ha, oiii, sii):
        """
        Maps the Ha, OIII, SII channels based on the palette name.
        Substitutes SII for Ha or Ha for SII if one is missing.
        """
        # Substitute SII for Ha if Ha is missing
        if ha is None and sii is not None:
            ha = sii
            print("Ha is missing. Substituting SII for Ha.")
        
        # Substitute Ha for SII if SII is missing
        if sii is None and ha is not None:
            sii = ha
            print("SII is missing. Substituting Ha for SII.")
        
        # Define the channel mappings
        mapping = {
            "SHO": [sii, ha, oiii],
            "HOO": [ha, oiii, oiii],
            "HSO": [ha, sii, oiii],
            "HOS": [ha, oiii, sii],
            "OSS": [oiii, sii, sii],
            "OHH": [oiii, ha, ha],
            "OSH": [oiii, sii, ha],
            "OHS": [oiii, ha, sii],
            "HSS": [ha, sii, sii],
        }
        
        # Retrieve the mapped channels based on the palette name
        mapped_channels = mapping.get(palette_name, [ha, oiii, sii])
             
        return mapped_channels


    def map_special_palettes(self, palette_name, ha, oiii, sii):
        """
        Maps channels for special palettes like Realistic1, Realistic2, Foraxx.
        Ensures all expressions produce values within the [0, 1] range.
        Substitutes SII for Ha or Ha for SII if one is missing.
        """
        try:
            # Substitute SII for Ha if Ha is missing
            if ha is None and sii is not None:
                ha = sii
                print("Ha is missing in special palette. Substituting SII for Ha.")
        
            # Substitute Ha for SII if SII is missing
            if sii is None and ha is not None:
                sii = ha
                print("SII is missing in special palette. Substituting Ha for SII.")
        
            # Realistic1 mapping
            if palette_name == "Realistic1":
                expr_r = (ha + sii) / 2 if (ha is not None and sii is not None) else (ha if ha is not None else 0)
                expr_g = (0.3 * ha) + (0.7 * oiii) if (ha is not None and oiii is not None) else (ha if ha is not None else 0)
                expr_b = (0.9 * oiii) + (0.1 * ha) if (ha is not None and oiii is not None) else (oiii if oiii is not None else 0)
        
            # Realistic2 mapping
            elif palette_name == "Realistic2":
                expr_r = (0.7 * ha + 0.3 * sii) if (ha is not None and sii is not None) else (ha if ha is not None else 0)
                expr_g = (0.3 * sii + 0.7 * oiii) if (sii is not None and oiii is not None) else (oiii if oiii is not None else 0)
                expr_b = oiii if oiii is not None else 0
        
            # Foraxx mapping
            elif palette_name == "Foraxx":
                if ha is not None and oiii is not None and sii is None:
                    expr_r = ha
                    temp = ha * oiii
                    expr_g = (temp ** (1 - temp)) * ha + (1 - (temp ** (1 - temp))) * oiii
                    expr_b = oiii
                elif ha is not None and oiii is not None and sii is not None:
                    temp = oiii ** (1 - oiii)
                    expr_r = (temp * sii) + ((1 - temp) * ha)
                    temp_ha_oiii = ha * oiii
                    expr_g = (temp_ha_oiii ** (1 - temp_ha_oiii)) * ha + (1 - (temp_ha_oiii ** (1 - temp_ha_oiii))) * oiii
                    expr_b = oiii
                else:
                    # Fallback to SHO
                    return self.map_channels("SHO", ha, oiii, sii)
        
            else:
                # Fallback to SHO for any undefined palette
                return self.map_channels("SHO", ha, oiii, sii)
        
            # Replace invalid values and normalize
            expr_r = np.clip(np.nan_to_num(expr_r, nan=0.0, posinf=1.0, neginf=0.0), 0, 1)
            expr_g = np.clip(np.nan_to_num(expr_g, nan=0.0, posinf=1.0, neginf=0.0), 0, 1)
            expr_b = np.clip(np.nan_to_num(expr_b, nan=0.0, posinf=1.0, neginf=0.0), 0, 1)
        
            return expr_r, expr_g, expr_b
        except Exception as e:
            print(f"[Error] Failed to map palette {palette_name}: {e}")
            return None, None, None


    def extract_oscc_channels(self, osc_image, base_id):
        """
        Extracts R, G, B channels from the OSC image and assigns unique postfixes.
        
        Parameters:
            osc_image (numpy.ndarray): The OSC image array.
            base_id (str): The base identifier for naming.
        
        Returns:
            list: A list containing the extracted R, G, B channels as NumPy arrays.
        """
        if osc_image is None or osc_image.shape[2] < 3:
            print(f"[!] OSC image {base_id} has fewer than 3 channels—skipping extraction.")
            return []

        # Extract channels
        R = osc_image[:, :, 0]  # Red channel
        G = osc_image[:, :, 1]  # Green channel
        B = osc_image[:, :, 2]  # Blue channel

        # Assign unique postfixes
        R_name = f"{base_id}_pppR"
        G_name = f"{base_id}_pppG"
        B_name = f"{base_id}_pppB"

        # For Seti Astro Suite, we might need to create separate image objects or handle naming differently
        # Here, we'll assume that we can manage the names via dictionaries or similar structures

        # Store the extracted channels with their names
        extracted_channels = {
            R_name: R,
            G_name: G,
            B_name: B
        }

        # Optionally, hide these images in the GUI or manage them as needed
        # For example, you might add them to an internal list for cleanup

        # For demonstration, we'll return the list of channels
        return [R, G, B]




    def numpy_to_qimage(self, image_array):
        """
        Converts a NumPy array to QImage.
        Assumes image_array is in the range [0, 1] and in RGB format.
        """
        try:
            # Validate input shape
            if image_array.ndim == 2:
                # Grayscale image
                
                image_uint8 = (np.clip(image_array, 0, 1) * 255).astype(np.uint8)
                height, width = image_uint8.shape
                bytes_per_line = width
                q_image = QImage(image_uint8.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
                return q_image.copy()
            elif image_array.ndim == 3 and image_array.shape[2] == 3:
                # RGB image
                
                image_uint8 = (np.clip(image_array, 0, 1) * 255).astype(np.uint8)
                height, width, channels = image_uint8.shape
                if channels != 3:
                    raise ValueError(f"Expected 3 channels for RGB, got {channels}")
                bytes_per_line = 3 * width
                q_image = QImage(image_uint8.data, width, height, bytes_per_line, QImage.Format_RGB888)
                return q_image.copy()
            else:
                # Invalid shape
                raise ValueError(f"Invalid image shape for QImage conversion: {image_array.shape}")
        except Exception as e:
            print(f"Error converting NumPy array to QImage: {e}")
            return QImage()



    def save_image(self):
        """
        Save the current combined image to a selected path.
        """
        if self.combined_image is not None:
            save_file, _ = QFileDialog.getSaveFileName(
                self,
                "Save As",
                "",
                "Images (*.png *.tif *.tiff *.fits *.fit);;All Files (*)"
            )

            if save_file:
                # Prompt the user for bit depth
                bit_depth, ok = QInputDialog.getItem(
                    self,
                    "Select Bit Depth",
                    "Choose bit depth for saving:",
                    ["16-bit", "32-bit floating point"],
                    0,
                    False
                )
                if ok:
                    # Determine the user-selected format from the filename
                    _, ext = os.path.splitext(save_file)
                    selected_format = ext.lower().strip('.')

                    # Validate the selected format
                    valid_formats = ['png', 'tif', 'tiff', 'fits', 'fit']
                    if selected_format not in valid_formats:
                        QMessageBox.critical(
                            self,
                            "Error",
                            f"Unsupported file format: {selected_format}. Supported formats are: {', '.join(valid_formats)}"
                        )
                        return

                    # Ensure correct data ordering for FITS format
                    final_image = self.combined_image
                    if selected_format in ['fits', 'fit']:
                        if self.combined_image.ndim == 3:  # RGB image
                            # Transpose to (channels, height, width)
                            final_image = np.transpose(self.combined_image, (2, 0, 1))
                            print(f"Transposed for FITS: {final_image.shape}")
                        elif self.combined_image.ndim == 2:  # Mono image
                            print(f"Mono image, no transposition needed: {final_image.shape}")
                        else:
                            QMessageBox.critical(
                                self,
                                "Error",
                                "Unsupported image dimensions for FITS saving."
                            )
                            return

                    # Check if any loaded image file paths have the `.xisf` extension
                    loaded_file_paths = [
                        self.ha_filename, self.oiii_filename,
                        self.sii_filename, self.osc_filename
                    ]
                    contains_xisf = any(
                        file_path.lower().endswith('.xisf') for file_path in loaded_file_paths if file_path
                    )

                    # Create a minimal header if any loaded image is XISF
                    sanitized_header = self.original_header if not contains_xisf else self.create_minimal_fits_header(final_image)

                    # Pass the correctly ordered image to the global save_image function
                    try:
                        save_image(
                            img_array=final_image,
                            filename=save_file,
                            original_format=selected_format,
                            bit_depth=bit_depth,
                            original_header=sanitized_header,  # Pass minimal or original header
                            is_mono=self.is_mono
                        )
                        print(f"Image successfully saved to {save_file}.")
                        self.status_label.setText(f"Image saved to: {save_file}")
                    except Exception as e:
                        QMessageBox.critical(self, "Error", f"Failed to save image: {e}")
                        print(f"Error saving image: {e}")
            else:
                self.status_label.setText("Save canceled.")
        else:
            QMessageBox.warning(self, "Warning", "No combined image to save.")
            self.status_label.setText("No combined image to save.")


    def create_minimal_fits_header(self, img_array):
        """
        Creates a minimal FITS header when the original header is missing.
        """
        from astropy.io.fits import Header

        header = Header()
        header['SIMPLE'] = (True, 'Standard FITS file')
        header['BITPIX'] = -32  # 32-bit floating-point data
        header['NAXIS'] = 2 if self.is_mono else 3
        header['NAXIS1'] = self.combined_image.shape[1]  # Image width
        header['NAXIS2'] = self.combined_image.shape[0]  # Image height
        if not self.is_mono:
            header['NAXIS3'] = self.combined_image.shape[2]  # Number of color channels
        header['BZERO'] = 0.0  # No offset
        header['BSCALE'] = 1.0  # No scaling
        header['COMMENT'] = "Minimal FITS header generated by Perfect Palette Picker."

        return header








    def zoom_in(self):
        """
        Zooms into the main preview image.
        """
        if self.zoom_factor < 5.0:  # Maximum zoom factor
            self.zoom_factor *= 1.25
            self.update_main_preview()
        else:
            print("Maximum zoom level reached.")
            self.status_label.setText("Maximum zoom level reached.")

    def zoom_out(self):
        """
        Zooms out of the main preview image.
        """
        if self.zoom_factor > 0.2:  # Minimum zoom factor
            self.zoom_factor /= 1.25
            self.update_main_preview()
        else:
            print("Minimum zoom level reached.")
            self.status_label.setText("Minimum zoom level reached.")

    def fit_to_preview(self):
        """
        Fits the main preview image to the scroll area.
        """
        if self.combined_image is not None:
            q_image = self.numpy_to_qimage(self.combined_image)
            if q_image.isNull():
                QMessageBox.critical(self, "Error", "Cannot fit image to preview due to conversion error.")
                return
            pixmap = QPixmap.fromImage(q_image)
            scroll_area_width = self.scroll_area.viewport().width()
            self.zoom_factor = scroll_area_width / pixmap.width()
            self.update_main_preview()
            self.status_label.setText("Image fitted to preview area.")
        else:
            QMessageBox.warning(self, "Warning", "No image loaded to fit.")

    def update_main_preview(self):
        """
        Updates the main preview image based on the current zoom factor.
        """
        if self.combined_image is not None:
            q_image = self.numpy_to_qimage(self.combined_image)
            pixmap = QPixmap.fromImage(q_image)
            if pixmap.isNull():
                QMessageBox.critical(self, "Error", "Failed to update main preview. Invalid QPixmap.")
                return

            # Ensure dimensions are integers
            scaled_width = int(pixmap.width() * self.zoom_factor)
            scaled_height = int(pixmap.height() * self.zoom_factor)

            scaled_pixmap = pixmap.scaled(
                scaled_width,
                scaled_height,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.image_label.setPixmap(scaled_pixmap)
            self.image_label.resize(scaled_pixmap.size())
        else:
            self.image_label.clear()





    def create_palette_preview(self, palette_name):
        """
        Creates a mini-preview image for the given palette.
        Returns the combined RGB image as a NumPy array.
        """
        print(f"Creating mini-preview for palette: {palette_name}")
        combined = self.combine_channels(palette_name)
        return combined

    def push_final_palette_to_image_manager(self):
        """
        Pushes the final combined image to the ImageManager for further processing.
        """
        if self.combined_image is not None:
            # Check if any of the loaded file paths have an XISF extension
            loaded_files = [self.ha_filename, self.oiii_filename, self.sii_filename, self.osc1_filename, self.osc2_filename]
            was_xisf = any(file_path and file_path.lower().endswith('.xisf') for file_path in loaded_files)

            # Generate a minimal FITS header if the original header is missing or if the format was XISF
            sanitized_header = self.original_header
            if was_xisf or sanitized_header is None:
                sanitized_header = None

            # Ensure a valid file path exists
            file_path = self.ha_filename if self.ha_image is not None else "Combined Image"

            # Create metadata for the combined image
            metadata = {
                'file_path': file_path,
                'original_header': sanitized_header,  # Use the sanitized or minimal header
                'bit_depth': self.bit_depth if hasattr(self, 'bit_depth') else "Unknown",
                'is_mono': self.is_mono if hasattr(self, 'is_mono') else False,
                'processing_parameters': {
                    'zoom_factor': self.zoom_factor,
                    'preview_scale': self.preview_scale
                },
                'processing_timestamp': datetime.now().isoformat(),
                'source_images': {
                    'Ha': self.ha_filename if self.ha_image is not None else "Not Provided",
                    'OIII': self.oiii_filename if self.oiii_image is not None else "Not Provided",
                    'SII': self.sii_filename if self.sii_image is not None else "Not Provided",
                    'OSC1': self.osc1_filename if self.osc1_image is not None else "Not Provided",
                    'OSC2': self.osc2_filename if self.osc2_image is not None else "Not Provided"
                }
            }

            # Push the image and metadata into the ImageManager
            if self.image_manager:
                try:
                    self.image_manager.update_image(
                        updated_image=self.combined_image, metadata=metadata
                    )
                    print(f"Image pushed to ImageManager with metadata: {metadata}")
                    self.status_label.setText("Final palette image pushed for further processing.")
                except Exception as e:
                    print(f"Error updating ImageManager: {e}")
                    QMessageBox.critical(self, "Error", f"Failed to update ImageManager:\n{e}")
            else:
                print("ImageManager is not initialized.")
                QMessageBox.warning(self, "Warning", "ImageManager is not initialized. Cannot store the combined image.")
        else:
            QMessageBox.warning(self, "Warning", "No final palette image to push.")
            self.status_label.setText("No final palette image to push.")



    def mousePressEvent(self, event):
        """
        Starts dragging when the left mouse button is pressed.
        """
        if event.button() == Qt.LeftButton:
            self.dragging = True
            self.last_mouse_position = event.pos()
            self.image_label.setCursor(Qt.ClosedHandCursor)

    def mouseMoveEvent(self, event):
        """
        Handles dragging by adjusting the scroll area's position.
        """
        if self.dragging and self.last_mouse_position is not None:
            # Calculate the difference in mouse movement
            delta = event.pos() - self.last_mouse_position
            self.last_mouse_position = event.pos()

            # Adjust the scroll area's scroll position
            self.scroll_area.horizontalScrollBar().setValue(
                self.scroll_area.horizontalScrollBar().value() - delta.x()
            )
            self.scroll_area.verticalScrollBar().setValue(
                self.scroll_area.verticalScrollBar().value() - delta.y()
            )

    def mouseReleaseEvent(self, event):
        """
        Stops dragging when the left mouse button is released.
        """
        if event.button() == Qt.LeftButton:
            self.dragging = False
            self.last_mouse_position = None
            self.image_label.setCursor(Qt.OpenHandCursor)


    def cleanup_preview_windows(self):
        """
        Cleans up temporary preview images by resetting image variables and clearing GUI elements.
        """
        print("Cleaning up preview windows...")
        
        # 1. Reset Temporary Image Variables
        self.ha_image = None
        self.oiii_image = None
        self.sii_image = None
        print("Temporary preview images (Ha, OIII, SII) have been cleared.")
        
        # 2. Clear GUI Elements Displaying Previews
        # Update the list below with the actual names of your preview labels or buttons
        preview_labels = ['ha_preview_label', 'oiii_preview_label', 'sii_preview_label']
        for label_name in preview_labels:
            if hasattr(self, label_name):
                label = getattr(self, label_name)
                label.clear()  # Removes the pixmap or any displayed content
                print(f"{label_name} has been cleared.")
        
        # 3. Clear Final Image Display (if applicable)
        # Update 'final_image_label' with your actual final image display widget name
        if hasattr(self, 'image_label'):
            self.image_label.clear()
            print("Final image label has been cleared.")
        
        # 4. Reset Thumbnail Buttons (if used for previews)
        # Ensure 'self.thumbnail_buttons' is a list of your thumbnail QPushButtons
        for button in self.thumbnail_buttons:
            button.setIcon(QIcon())    # Remove existing icon



        print("Thumbnail buttons have been reset.")
        
        # 5. Update Status Label
        self.status_label.setText("Preview windows cleaned up.")
        print("Status label updated to indicate cleanup.")
        
        # 6. Process UI Events to Reflect Changes Immediately
        QApplication.processEvents()


    def closeEvent(self, event):
        """
        Handle the close event to perform cleanup.
        """
        self.cleanup_preview_windows()
        event.accept()


class NBtoRGBstarsTab(QWidget):
    def __init__(self, image_manager=None):
        super().__init__()
        self.image_manager = image_manager  # Reference to the ImageManager
        self.initUI()
        self.ha_image = None
        self.oiii_image = None
        self.sii_image = None
        self.osc_image = None
        self.combined_image = None
        self.is_mono = False
        # Filenames
        self.ha_filename = None
        self.oiii_filename = None
        self.sii_filename = None
        self.osc_filename = None        
        self.filename = None  # Store the selected file path
        self.zoom_factor = 1.0  # Initialize to 1.0 for normal size
        self.dragging = False
        self.last_pos = QPoint()
        self.processing_thread = None
        self.original_header = None
        self.original_pixmap = None  # To store the original QPixmap for zooming
        self.bit_depth = "Unknown"

        if self.image_manager:
            # Connect to ImageManager's image_changed signal if needed
            self.image_manager.image_changed.connect(self.on_image_changed)

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

        # **Remove Zoom Buttons from Left Panel (Not present)**
        # No existing zoom buttons to remove in the left panel

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

        # **Create Right Panel Layout**
        right_widget = QWidget(self)
        right_layout = QVBoxLayout(right_widget)

        # Zoom controls

        zoom_layout = QHBoxLayout()
        zoom_in_button = QPushButton("Zoom In")
        zoom_in_button.clicked.connect(self.zoom_in)
        zoom_layout.addWidget(zoom_in_button)

        zoom_out_button = QPushButton("Zoom Out")
        zoom_out_button.clicked.connect(self.zoom_out)
        zoom_layout.addWidget(zoom_out_button)

        # **Add "Fit to Preview" Button**
        self.fitToPreviewButton = QPushButton("Fit to Preview")
        self.fitToPreviewButton.clicked.connect(self.fit_to_preview)
        zoom_layout.addWidget(self.fitToPreviewButton)

        right_layout.addLayout(zoom_layout)

        # Right side for the preview inside a QScrollArea
        self.scrollArea = QScrollArea(self)
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scrollArea.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        self.imageLabel = QLabel(self)
        self.imageLabel.setAlignment(Qt.AlignCenter)
        self.scrollArea.setWidget(self.imageLabel)
        self.scrollArea.setMinimumSize(400, 400)

        right_layout.addWidget(self.scrollArea)

        # Add the right widget to the main layout
        main_layout.addWidget(right_widget)

        self.setLayout(main_layout)
        self.scrollArea.viewport().setMouseTracking(True)
        self.scrollArea.viewport().installEventFilter(self)

    def on_image_changed(self, slot, image, metadata):
        """
        Slot to handle image changes from ImageManager.
        Updates the display if the current slot is affected.
        """
        if slot == self.image_manager.current_slot:
            # Ensure the image is a numpy array
            if not isinstance(image, np.ndarray):
                image = np.array(image)  # Convert to numpy array if needed

            # Update internal state with the new image and metadata
            self.combined_image = image
            self.original_header = metadata.get('original_header', None)
            self.bit_depth = metadata.get('bit_depth', None)
            self.is_mono = metadata.get('is_mono', False)

            # Update image display (assuming updateImageDisplay handles the proper rendering)
            self.updateImageDisplay()

            print(f"NBtoRGBstarsTab: Image updated from ImageManager slot {slot}.")



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
        selected_file, _ = QFileDialog.getOpenFileName(
            self, f"Select {image_type} Image", "", "Images (*.png *.tif *.tiff *.fits *.fit *.xisf)"
        )
        if selected_file:
            try:
                image, original_header, bit_depth, is_mono = load_image(selected_file)
                if image is None:
                    raise ValueError("Failed to load image data.")

                if image_type == 'Ha':
                    self.ha_image = image
                    self.ha_filename = selected_file
                    self.original_header = original_header
                    self.bit_depth = bit_depth
                    self.is_mono = is_mono
                    self.haLabel.setText(f"{os.path.basename(selected_file)} selected")
                elif image_type == 'OIII':
                    self.oiii_image = image
                    self.oiii_filename = selected_file
                    self.original_header = original_header
                    self.bit_depth = bit_depth
                    self.is_mono = is_mono
                    self.oiiiLabel.setText(f"{os.path.basename(selected_file)} selected")
                elif image_type == 'SII':
                    self.sii_image = image
                    self.sii_filename = selected_file
                    self.original_header = original_header
                    self.bit_depth = bit_depth
                    self.is_mono = is_mono
                    self.siiLabel.setText(f"{os.path.basename(selected_file)} selected")
                elif image_type == 'OSC':
                    self.osc_image = image
                    self.osc_filename = selected_file
                    self.original_header = original_header
                    self.bit_depth = bit_depth
                    self.is_mono = is_mono
                    self.oscLabel.setText(f"{os.path.basename(selected_file)} selected")

            except Exception as e:
                print(f"Failed to load {image_type} image: {e}")
                QMessageBox.critical(self, "Error", f"Failed to load {image_type} image:\n{e}")

    def previewCombine(self):
        ha_to_oii_ratio = self.haToOiiRatioSlider.value() / 100.0
        enable_star_stretch = self.starStretchCheckBox.isChecked()
        stretch_factor = self.stretchSlider.value() / 100.0

        # Show spinner before starting processing
        self.showSpinner()

        # Reset zoom factor when a new preview is generated
        self.zoom_factor = 1.0

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
        try:
            preview_image = (combined_image * 255).astype(np.uint8)
            h, w = preview_image.shape[:2]
            q_image = QImage(preview_image.data, w, h, 3 * w, QImage.Format_RGB888)
        except Exception as e:
            print(f"Error converting combined image for display: {e}")
            QMessageBox.critical(self, "Error", f"Failed to prepare image for display:\n{e}")
            self.hideSpinner()
            return

        # Store original pixmap for zooming
        self.original_pixmap = QPixmap.fromImage(q_image)

        # Apply initial zoom
        scaled_pixmap = self.original_pixmap.scaled(
            self.original_pixmap.size() * self.zoom_factor,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.imageLabel.setPixmap(scaled_pixmap)
        self.imageLabel.resize(scaled_pixmap.size())

        # Hide the spinner after processing is done
        self.hideSpinner()

        # Prepare metadata with safeguards
        metadata = {
            'file_path': self.ha_filename if self.ha_image is not None else "Combined Image",
            'original_header': self.original_header if self.original_header else {},
            'bit_depth': self.bit_depth if self.bit_depth else "Unknown",
            'is_mono': self.is_mono,
            'processing_parameters': {
                'ha_to_oii_ratio': self.haToOiiRatioSlider.value() / 100.0,
                'enable_star_stretch': self.starStretchCheckBox.isChecked(),
                'stretch_factor': self.stretchSlider.value() / 100.0
            },
            'processing_timestamp': datetime.now().isoformat(),
            'source_images': {
                'Ha': self.ha_filename if self.ha_image is not None else "Not Provided",
                'OIII': self.oiii_filename if self.oiii_image is not None else "Not Provided",
                'SII': self.sii_filename if self.sii_image is not None else "Not Provided",
                'OSC': self.osc_filename if self.osc_image is not None else "Not Provided"
            }
        }

        # Update ImageManager with the new combined image
        if self.image_manager:
            try:
                self.image_manager.update_image(updated_image=self.combined_image, metadata=metadata)
                print("NBtoRGBstarsTab: Combined image stored in ImageManager.")
                
            except Exception as e:
                print(f"Error updating ImageManager: {e}")
                QMessageBox.critical(self, "Error", f"Failed to update ImageManager:\n{e}")
        else:
            print("ImageManager is not initialized.")
            QMessageBox.warning(self, "Warning", "ImageManager is not initialized. Cannot store the combined image.")

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

    def zoom_in(self):
        if self.zoom_factor < 20.0:  # Set a maximum zoom limit (e.g., 500%)
            self.zoom_factor *= 1.25  # Increase zoom by 25%
            self.updateImageDisplay()
        else:
            print("Maximum zoom level reached.")

    def zoom_out(self):
        if self.zoom_factor > 0.01:  # Set a minimum zoom limit (e.g., 20%)
            self.zoom_factor /= 1.25  # Decrease zoom by 20%
            self.updateImageDisplay()
        else:
            print("Minimum zoom level reached.")

    def fit_to_preview(self):
        """Adjust the zoom factor so that the image's width fits within the preview area's width."""
        if self.image is not None:
            # Get the width of the scroll area's viewport (preview area)
            preview_width = self.scrollArea.viewport().width()
            
            # Get the original image width from the numpy array
            # Assuming self.image has shape (height, width, channels) or (height, width) for grayscale
            if self.image.ndim == 3:
                image_width = self.image.shape[1]
            elif self.image.ndim == 2:
                image_width = self.image.shape[1]
            else:
                print("Unexpected image dimensions!")
                self.statusLabel.setText("Cannot fit image to preview due to unexpected dimensions.")
                return
            
            # Calculate the required zoom factor to fit the image's width into the preview area
            new_zoom_factor = preview_width / image_width
            
            # Update the zoom factor without enforcing any limits
            self.zoom_factor = new_zoom_factor
            
            # Apply the new zoom factor to update the display
            self.apply_zoom()
            
            # Update the status label to reflect the new zoom level
        else:
            print("No image loaded. Cannot fit to preview.")

    def apply_zoom(self):
        """Apply the current zoom level to the image."""
        self.updateImageDisplay()  # Call without extra arguments; it will calculate dimensions based on zoom factor

    def reset_zoom(self):
        self.zoom_factor = 1.0
        self.updateImageDisplay()

    def updateImageDisplay(self):
        if self.original_pixmap:
            scaled_pixmap = self.original_pixmap.scaled(
                self.original_pixmap.size() * self.zoom_factor,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.imageLabel.setPixmap(scaled_pixmap)
            self.imageLabel.resize(scaled_pixmap.size())

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

    # Placeholder methods for functionalities
    def handleImageMouseMove(self, x, y):
        # Implement handling mouse movement over the image
        pass


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
        # Normalize input images to [0, 1]
        if self.ha_image is not None:
            self.ha_image = np.clip(self.ha_image, 0, 1)
        if self.oiii_image is not None:
            self.oiii_image = np.clip(self.oiii_image, 0, 1)
        if self.sii_image is not None:
            self.sii_image = np.clip(self.sii_image, 0, 1)
        if self.osc_image is not None:
            self.osc_image = np.clip(self.osc_image, 0, 1)

        # Combined RGB logic
        if self.osc_image is not None:
            # Extract OSC channels
            r_channel = self.osc_image[..., 0]
            g_channel = self.osc_image[..., 1]
            b_channel = self.osc_image[..., 2]

            # Fallbacks for missing Ha, OIII, or SII images
            r_combined = 0.5 * r_channel + 0.5 * (self.sii_image if self.sii_image is not None else r_channel)
            g_combined = self.ha_to_oii_ratio * (self.ha_image if self.ha_image is not None else r_channel) + \
                         (1 - self.ha_to_oii_ratio) * g_channel
            b_combined = b_channel if self.oiii_image is None else self.oiii_image
        else:
            # Use narrowband images directly (default logic)
            r_combined = 0.5 * self.ha_image + 0.5 * (self.sii_image if self.sii_image is not None else self.ha_image)
            g_combined = self.ha_to_oii_ratio * self.ha_image + (1 - self.ha_to_oii_ratio) * self.oiii_image
            b_combined = self.oiii_image

        # Normalize combined channels to [0, 1]
        r_combined = np.clip(r_combined, 0, 1)
        g_combined = np.clip(g_combined, 0, 1)
        b_combined = np.clip(b_combined, 0, 1)

        # Stack the channels to create an RGB image
        combined_image = np.stack((r_combined, g_combined, b_combined), axis=-1)

        # Apply star stretch if enabled
        if self.enable_star_stretch:
            combined_image = self.apply_star_stretch(combined_image)

        # Apply SCNR (remove green cast)
        combined_image = self.apply_scnr(combined_image)

        # Emit the processed image for preview
        self.preview_generated.emit(combined_image)

    def apply_star_stretch(self, image):
        # Ensure input image is in the range [0, 1]
        assert np.all(image >= 0) and np.all(image <= 1), "Image must be normalized to [0, 1] before star stretch."
        stretched = ((3 ** self.stretch_factor) * image) / ((3 ** self.stretch_factor - 1) * image + 1)
        return np.clip(stretched, 0, 1)

    def apply_scnr(self, image):
        green_channel = image[..., 1]
        max_rg = np.maximum(image[..., 0], image[..., 2])
        green_channel[green_channel > max_rg] = max_rg[green_channel > max_rg]
        image[..., 1] = green_channel
        return image

class HaloBGonTab(QWidget):
    def __init__(self, image_manager=None):
        super().__init__()
        self.image_manager = image_manager  # Reference to the ImageManager

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
        self.initUI()

        if self.image_manager:
            # Connect to ImageManager's image_changed signal
            self.image_manager.image_changed.connect(self.on_image_changed)
        

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
            3. Click Refresh Preview to apply the halo reduction.
        """)
        instruction_box.setWordWrap(True)
        left_layout.addWidget(instruction_box)

        # File selection button
        self.fileButton = QPushButton("Load Image", self)
        self.fileButton.clicked.connect(self.selectImage)
        left_layout.addWidget(self.fileButton)

        self.fileLabel = QLabel('', self)
        left_layout.addWidget(self.fileLabel)

        # Reduction amount slider
        self.reductionLabel = QLabel("Reduction Amount: Extra Low", self)
        self.reductionSlider = QSlider(Qt.Horizontal, self)
        self.reductionSlider.setMinimum(0)
        self.reductionSlider.setMaximum(3)
        self.reductionSlider.setValue(0)  # 0: Extra Low, 1: Low, 2: Medium, 3: High
        self.reductionSlider.setToolTip("Adjust the amount of halo reduction (Extra Low, Low, Medium, High)")
        self.reductionSlider.valueChanged.connect(self.updateReductionLabel)
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

        # **Create a horizontal layout for Refresh Preview and Undo buttons**
        action_buttons_layout = QHBoxLayout()

        # Refresh Preview button
        self.executeButton = QPushButton("Refresh Preview", self)
        self.executeButton.clicked.connect(self.generatePreview)
        action_buttons_layout.addWidget(self.executeButton)

        # Undo button with left arrow icon
        self.undoButton = QPushButton("Undo", self)
        undo_icon = self.style().standardIcon(QStyle.SP_ArrowBack)  # Standard left arrow icon
        self.undoButton.setIcon(undo_icon)
        self.undoButton.clicked.connect(self.undoAction)
        self.undoButton.setEnabled(False)  # Disabled by default
        action_buttons_layout.addWidget(self.undoButton)

        # Add the horizontal layout to the left layout
        left_layout.addLayout(action_buttons_layout)

        self.saveButton = QPushButton("Save Image", self)
        self.saveButton.clicked.connect(self.saveImage)
        left_layout.addWidget(self.saveButton)

        # **Remove Zoom Buttons from Left Panel**
        # Comment out or remove the existing zoom buttons in the left panel
        # zoom_layout = QHBoxLayout()
        # self.zoomInButton = QPushButton("Zoom In", self)
        # self.zoomInButton.clicked.connect(self.zoomIn)
        # zoom_layout.addWidget(self.zoomInButton)
        #
        # self.zoomOutButton = QPushButton("Zoom Out", self)
        # self.zoomOutButton.clicked.connect(self.zoomOut)
        # zoom_layout.addWidget(self.zoomOutButton)
        # left_layout.addLayout(zoom_layout)

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

        # **Create Right Panel Layout**
        right_widget = QWidget(self)
        right_layout = QVBoxLayout(right_widget)

        # Zoom controls

        zoom_layout = QHBoxLayout()
        zoom_in_button = QPushButton("Zoom In")
        zoom_in_button.clicked.connect(self.zoomIn)
        zoom_layout.addWidget(zoom_in_button)

        zoom_out_button = QPushButton("Zoom Out")
        zoom_out_button.clicked.connect(self.zoomOut)
        zoom_layout.addWidget(zoom_out_button)

        # **Add "Fit to Preview" Button**
        self.fitToPreviewButton = QPushButton("Fit to Preview")
        self.fitToPreviewButton.clicked.connect(self.fit_to_preview)
        zoom_layout.addWidget(self.fitToPreviewButton)

        right_layout.addLayout(zoom_layout)

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

        right_layout.addWidget(self.scrollArea)

        # Add the right widget to the main layout
        main_layout.addWidget(right_widget)

        self.setLayout(main_layout)
        self.scrollArea.viewport().setMouseTracking(True)
        self.scrollArea.viewport().installEventFilter(self)

    def on_image_changed(self, slot, image, metadata):
        """
        Slot to handle image changes from ImageManager.
        Updates the display if the current slot is affected.
        """
        if slot == self.image_manager.current_slot:
            # Ensure the image is a numpy array before proceeding
            if not isinstance(image, np.ndarray):
                image = np.array(image)  # Convert to numpy array if necessary
            
            self.image = image  # Set the original image
            self.preview_image = None  # Reset the preview image
            self.original_header = metadata.get('original_header', None)
            self.is_mono = metadata.get('is_mono', False)
            self.filename = metadata.get('file_path', self.filename)

            # Update the image display
            self.updateImageDisplay()

            print(f"Halo-B-Gon Tab: Image updated from ImageManager slot {slot}.")

            # **Update Undo and Redo Button States**
            if self.image_manager:
                self.undoButton.setEnabled(self.image_manager.can_undo())



    def updateImageDisplay(self):
        if self.image is not None:
            # Prepare the image for display by normalizing and converting to uint8
            display_image = (self.image * 255).astype(np.uint8)
            h, w = display_image.shape[:2]

            if display_image.ndim == 3:  # RGB Image
                # Convert the image to QImage format
                q_image = QImage(display_image.tobytes(), w, h, 3 * w, QImage.Format_RGB888)
            else:  # Grayscale Image
                q_image = QImage(display_image.tobytes(), w, h, w, QImage.Format_Grayscale8)

            # Create a QPixmap from QImage
            pixmap = QPixmap.fromImage(q_image)
            self.current_pixmap = pixmap  # Store the original pixmap for future reference

            # Scale the pixmap based on the zoom factor
            scaled_pixmap = pixmap.scaled(pixmap.size() * self.zoom_factor, Qt.KeepAspectRatio, Qt.SmoothTransformation)

            # Set the pixmap on the image label
            self.imageLabel.setPixmap(scaled_pixmap)
            self.imageLabel.resize(scaled_pixmap.size())  # Resize the label to fit the image
        else:
            # If no image is available, clear the label and show a message
            self.imageLabel.clear()
            self.imageLabel.setText('No image loaded.')



    def undoAction(self):
        if self.image_manager and self.image_manager.can_undo():
            try:
                # Perform the undo operation
                self.image_manager.undo()
                print("HaloBGonTab: Undo performed.")
            except Exception as e:
                print(f"Error performing undo: {e}")
                QMessageBox.critical(self, "Error", f"Failed to perform undo:\n{e}")
        else:
            QMessageBox.information(self, "Info", "Nothing to undo.")
            print("HaloBGonTab: No actions to undo.")

        # Update the state of the Undo button
        self.undoButton.setEnabled(self.image_manager.can_undo())

    def updateReductionLabel(self, value):
        labels = ["Extra Low", "Low", "Medium", "High"]
        if 0 <= value < len(labels):
            self.reductionLabel.setText(f"Reduction Amount: {labels[value]}")
        else:
            self.reductionLabel.setText("Reduction Amount: Unknown")

    def zoomIn(self):
        self.zoom_factor *= 1.2  # Increase zoom by 20%
        self.updateImageDisplay()

    def zoomOut(self):
        self.zoom_factor /= 1.2  # Decrease zoom by 20%
        self.updateImageDisplay()
    
    def fit_to_preview(self):
        """Adjust the zoom factor so that the image's width fits within the preview area's width."""
        if self.image is not None:
            # Get the width of the scroll area's viewport (preview area)
            preview_width = self.scrollArea.viewport().width()
            
            # Get the original image width from the numpy array
            # Assuming self.image has shape (height, width, channels) or (height, width) for grayscale
            if self.image.ndim == 3:
                image_width = self.image.shape[1]
            elif self.image.ndim == 2:
                image_width = self.image.shape[1]
            else:
                print("Unexpected image dimensions!")
                self.statusLabel.setText("Cannot fit image to preview due to unexpected dimensions.")
                return
            
            # Calculate the required zoom factor to fit the image's width into the preview area
            new_zoom_factor = preview_width / image_width
            
            # Update the zoom factor without enforcing any limits
            self.zoom_factor = new_zoom_factor
            
            # Apply the new zoom factor to update the display
            self.apply_zoom()
            
            # Update the status label to reflect the new zoom level
        else:
            print("No image loaded. Cannot fit to preview.")

    def apply_zoom(self):
        """Apply the current zoom level to the image."""
        self.updateImageDisplay()

    def selectImage(self):
        selected_file, _ = QFileDialog.getOpenFileName(
            self, 
            "Select Stars Only Image", 
            "", 
            "Images (*.png *.tif *.tiff *.fits *.fit *.xisf)"
        )
        if selected_file:
            try:
                # Load the image with header information
                self.image, self.original_header, _, self.is_mono = load_image(selected_file)  # Ensure load_image returns (image, header, bit_depth, is_mono)
                self.filename = selected_file 
                self.fileLabel.setText(os.path.basename(selected_file))
                
                # Update ImageManager with the loaded image
                if self.image_manager:
                    metadata = {
                        'file_path': selected_file,
                        'original_header': self.original_header,
                        'bit_depth': 'Unknown',  # Update if available
                        'is_mono': self.is_mono
                    }
                    self.image_manager.update_image(updated_image=self.image, metadata=metadata)
                    print(f"HaloBGonTab: Loaded image stored in ImageManager.")
                
                self.generatePreview()  # Generate preview after loading
            except Exception as e:
                self.fileLabel.setText(f"Error: {str(e)}")
                QMessageBox.critical(self, "Error", f"Failed to load image:\n{e}")
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

    def updatePreview(self, stretched_image):
        # Store the stretched image for saving
        self.preview_image = stretched_image

        # Update the ImageManager with the new stretched image
        metadata = {
            'file_path': self.filename if self.filename else "Stretched Image",
            'original_header': self.original_header if self.original_header else {},
            'bit_depth': "Unknown",  # Update if bit_depth is available
            'is_mono': self.is_mono,
            'processing_timestamp': datetime.now().isoformat(),
            'source_images': {
                'Original': self.filename if self.filename else "Not Provided"
            }
        }

        # Update ImageManager with the new processed image
        if self.image_manager:
            try:
                self.image_manager.update_image(updated_image=self.preview_image, metadata=metadata)
                print("StarStretchTab: Processed image stored in ImageManager.")
            except Exception as e:
                print(f"Error updating ImageManager: {e}")
                QMessageBox.critical(self, "Error", f"Failed to update ImageManager:\n{e}")
        else:
            print("ImageManager is not initialized.")
            QMessageBox.warning(self, "Warning", "ImageManager is not initialized. Cannot store the processed image.")

        # Update the preview once the processing thread emits the result
        preview_image = (stretched_image * 255).astype(np.uint8)
        h, w = preview_image.shape[:2]
        if preview_image.ndim == 3:
            q_image = QImage(preview_image.data, w, h, 3 * w, QImage.Format_RGB888)
        else:
            q_image = QImage(preview_image.data, w, h, w, QImage.Format_Grayscale8)

        pixmap = QPixmap.fromImage(q_image)
        self.current_pixmap = pixmap  # **Store the original pixmap**
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
            self.processing_thread = HaloProcessingThread(
                self.image, 
                self.reductionSlider.value(), 
                self.linearDataCheckbox.isChecked()
            )
            self.processing_thread.preview_generated.connect(self.updatePreview)
            self.processing_thread.start()
        else:
            QMessageBox.warning(self, "Warning", "No image loaded. Please load an image first.")
            print("HaloBGonTab: No image loaded. Cannot generate preview.")

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


    def createLightnessMask(self, image):
        # Check if the image is already single-channel (grayscale)
        if image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):
            # Normalize the grayscale image
            lightness_mask = image.astype(np.float32) / 255.0
        else:
            # Convert to grayscale to create a lightness mask
            lightness_mask = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

        # Apply a Gaussian blur to smooth the mask
        blurred = cv2.GaussianBlur(lightness_mask, (0, 0), sigmaX=2)

        # Apply an unsharp mask for enhancement
        lightness_mask = cv2.addWeighted(lightness_mask, 1.66, blurred, -0.66, 0)

        return lightness_mask

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
        # Ensure the image values are in range [0, 1]
        image = np.clip(image, 0, 1)

        # Convert linear to non-linear if the image is linear
        if is_linear:
            image = image ** (1 / 5)  # Gamma correction for linear data

        # Apply halo reduction logic
        lightness_mask = self.createLightnessMask(image)  # Single-channel mask
        inverted_mask = 1.0 - lightness_mask
        duplicated_mask = cv2.GaussianBlur(lightness_mask, (0, 0), sigmaX=2)
        enhanced_mask = inverted_mask - duplicated_mask * reduction_amount * 0.33

        # Expand the mask to match the number of channels in the image
        if image.ndim == 3 and image.shape[2] == 3:  # Color image
            enhanced_mask = np.expand_dims(enhanced_mask, axis=-1)  # Add a channel dimension
            enhanced_mask = np.repeat(enhanced_mask, 3, axis=-1)  # Repeat for all 3 channels

        # Verify that the image and mask dimensions match
        if image.shape != enhanced_mask.shape:
            raise ValueError(
                f"Shape mismatch between image {image.shape} and enhanced_mask {enhanced_mask.shape}"
            )

        # Apply the mask to the image
        masked_image = cv2.multiply(image, enhanced_mask)

        # Apply curves to the resulting image
        final_image = self.applyCurvesToImage(masked_image, reduction_amount)

        # Ensure the final image values are within [0, 1]
        return np.clip(final_image, 0, 1)


    def createLightnessMask(self, image):
        # Check if the image is already grayscale
        if image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):
            # Image is already grayscale; normalize it
            lightness_mask = image.astype(np.float32) / 255.0
        else:
            # Convert RGB image to grayscale
            lightness_mask = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

        # Apply Gaussian blur to smooth the mask
        blurred = cv2.GaussianBlur(lightness_mask, (0, 0), sigmaX=2)

        # Apply an unsharp mask for enhancement
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
    def __init__(self, image_manager):
        super().__init__()
        self.image_manager = image_manager
        self.initUI()
        self.nb_image = None  # Selected NB image
        self.continuum_image = None  # Selected Continuum image
        self.filename = None  # Store the selected file path
        self.is_mono = True
        self.combined_image = None  # Store the result of the continuum subtraction
        self.zoom_factor = 1.0  # Initialize zoom factor for preview scaling
        self.dragging = False
        self.last_pos = None
        self.processing_thread = None  # For background processing
        self.original_header = None
        self.original_pixmap = None  # To store the original QPixmap for zooming

        if self.image_manager:
            # Connect to ImageManager's image_changed signal if needed
            self.image_manager.image_changed.connect(self.on_image_changed)

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
        self.nb_label = QLabel("No NB image selected")
        left_layout.addWidget(self.nb_button)
        left_layout.addWidget(self.nb_label)

        self.continuum_button = QPushButton("Load Continuum Image")
        self.continuum_button.clicked.connect(lambda: self.selectImage("continuum"))
        self.continuum_label = QLabel("No Continuum image selected")
        left_layout.addWidget(self.continuum_button)
        left_layout.addWidget(self.continuum_label)

        # Linear Output Checkbox
        self.linear_output_checkbox = QCheckBox("Output Linear Image Only")
        left_layout.addWidget(self.linear_output_checkbox)

        # Progress indicator (spinner) label
        self.spinnerLabel = QLabel(self)
        self.spinnerLabel.setAlignment(Qt.AlignCenter)
        self.spinnerMovie = QMovie(resource_path("spinner.gif"))  # Ensure spinner.gif is in the correct path
        self.spinnerLabel.setMovie(self.spinnerMovie)
        self.spinnerLabel.hide()  # Hide spinner by default
        left_layout.addWidget(self.spinnerLabel)

        # Status label to show processing status
        self.statusLabel = QLabel(self)
        self.statusLabel.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(self.statusLabel)

        # Execute Button
        self.execute_button = QPushButton("Execute")
        self.execute_button.clicked.connect(self.startContinuumSubtraction)
        left_layout.addWidget(self.execute_button)

        # **Remove Zoom Buttons from Left Panel**
        # The following code is removed to eliminate zoom buttons from the left panel
        # zoom_layout = QHBoxLayout()
        # self.zoomInButton = QPushButton("Zoom In")
        # self.zoomInButton.clicked.connect(self.zoom_in)
        # zoom_layout.addWidget(self.zoomInButton)
        #
        # self.zoomOutButton = QPushButton("Zoom Out")
        # self.zoomOutButton.clicked.connect(self.zoom_out)
        # zoom_layout.addWidget(self.zoomOutButton)
        # left_layout.addLayout(zoom_layout)

        # Save Button
        self.save_button = QPushButton("Save Continuum Subtracted Image")
        self.save_button.clicked.connect(self.save_continuum_subtracted)
        self.save_button.setEnabled(False)  # Disable until an image is processed
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

        # Add left widget to the main layout
        main_layout.addWidget(left_widget)

        # **Create Right Panel Layout**
        right_widget = QWidget(self)
        right_layout = QVBoxLayout(right_widget)

        # **Add Zoom Buttons to Right Panel**
        zoom_layout = QHBoxLayout()
        self.zoomInButton = QPushButton("Zoom In")
        self.zoomInButton.clicked.connect(self.zoom_in)
        zoom_layout.addWidget(self.zoomInButton)

        self.zoomOutButton = QPushButton("Zoom Out")
        self.zoomOutButton.clicked.connect(self.zoom_out)
        zoom_layout.addWidget(self.zoomOutButton)

        # **Add "Fit to Preview" Button**
        self.fitToPreviewButton = QPushButton("Fit to Preview")
        self.fitToPreviewButton.clicked.connect(self.fit_to_preview)
        zoom_layout.addWidget(self.fitToPreviewButton)        

        # Add the zoom buttons layout to the right panel
        right_layout.addLayout(zoom_layout)

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

        right_layout.addWidget(self.scrollArea)

        # Add the right widget to the main layout
        main_layout.addWidget(right_widget)

        self.setLayout(main_layout)
        self.scrollArea.viewport().setMouseTracking(True)
        self.scrollArea.viewport().installEventFilter(self)

        # Initially disable zoom buttons until an image is loaded and previewed
        self.zoomInButton.setEnabled(False)
        self.zoomOutButton.setEnabled(False)

    def on_image_changed(self, slot, image, metadata):
        """
        Slot to handle image changes from ImageManager.
        Updates the display if the current slot is affected.
        """
        if slot == 0:  # Assuming slot 0 is used for shared images
            # Ensure the image is a numpy array
            if not isinstance(image, np.ndarray):
                image = np.array(image)  # Convert to numpy array if needed

            # Update internal state with the new image and metadata
            self.combined_image = image
            self.original_header = metadata.get('original_header', None)
            self.is_mono = metadata.get('is_mono', False)
            self.filename = metadata.get('file_path', None)

            # Update the preview
            self.update_preview()

            print(f"ContinuumSubtractTab: Image updated from ImageManager slot {slot}.")



    def selectImage(self, image_type):
        selected_file, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.tif *.tiff *.fits *.fit *xisf)")
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
            self.processing_thread = ContinuumProcessingThread(
                self.nb_image,
                self.continuum_image,
                self.linear_output_checkbox.isChecked()
            )
            self.processing_thread.processing_complete.connect(self.display_image)
            self.processing_thread.finished.connect(self.hideSpinner)
            self.processing_thread.status_update.connect(self.update_status_label)
            self.processing_thread.start()
        else:
            self.statusLabel.setText("Please select both NB and Continuum images.")
            print("Please select both NB and Continuum images.")

    def update_status_label(self, message):
        self.statusLabel.setText(message)

    def zoom_in(self):
        if self.zoom_factor < 5.0:  # Maximum 500% zoom
            self.zoom_factor *= 1.2  # Increase zoom by 20%
            self.update_preview()
            self.statusLabel.setText(f"Zoom: {self.zoom_factor * 100:.0f}%")

        else:
            self.statusLabel.setText("Maximum zoom level reached.")

    def zoom_out(self):
        if self.zoom_factor > 0.01:  # Minimum 20% zoom
            self.zoom_factor /= 1.2  # Decrease zoom by ~17%
            self.update_preview()
            self.statusLabel.setText(f"Zoom: {self.zoom_factor * 100:.0f}%")

        else:
            self.statusLabel.setText("Minimum zoom level reached.")

    def fit_to_preview(self):
        """Adjust the zoom factor so that the image's width fits within the preview area's width."""
        # Check if the original pixmap exists
        if self.original_pixmap is not None:
            # Get the width of the scroll area's viewport (preview area)
            preview_width = self.scrollArea.viewport().width()
            
            # Get the width of the original image from the original_pixmap
            image_width = self.original_pixmap.width()
            
            # Calculate the required zoom factor to fit the image's width into the preview area
            new_zoom_factor = preview_width / image_width
            
            # Update the zoom factor without enforcing any limits
            self.zoom_factor = new_zoom_factor
            
            # Apply the new zoom factor to update the display
            self.apply_zoom()
            
            # Update the status label to reflect the new zoom level
            self.statusLabel.setText(f"Fit to Preview: {self.zoom_factor * 100:.0f}%")

        else:

            self.statusLabel.setText("No image to fit to preview.")


    def apply_zoom(self):
        """Apply the current zoom level to the image."""
        self.update_preview()  # Call without extra arguments; it will calculate dimensions based on zoom factor            

    def update_preview(self):
        if self.original_pixmap is not None:
            scaled_pixmap = self.original_pixmap.scaled(
                self.original_pixmap.size() * self.zoom_factor,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.imageLabel.setPixmap(scaled_pixmap)
            self.imageLabel.resize(scaled_pixmap.size())
            print(f"Preview updated with zoom factor: {self.zoom_factor}")
        else:
            print("Original pixmap is not set. Cannot update preview.")

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
                    bit_depth, ok = QInputDialog.getItem(
                        self, "Select Bit Depth", "Choose bit depth for saving:", bit_depth_options, 0, False
                    )
                    
                    if ok and bit_depth:
                        # Call save_image with the necessary parameters
                        save_image(
                            self.combined_image, 
                            save_filename, 
                            original_format, 
                            bit_depth, 
                            self.original_header, 
                            self.is_mono
                        )
                        self.statusLabel.setText(f'Image saved as: {save_filename}')
                        print(f"Image saved as: {save_filename}")
                    else:
                        self.statusLabel.setText('Save canceled.')
                        print("Save operation canceled.")
                else:
                    # For non-TIFF/FITS formats, save directly without bit depth selection
                    save_image(self.combined_image, save_filename, original_format)
                    self.statusLabel.setText(f'Image saved as: {save_filename}')
                    print(f"Image saved as: {save_filename}")
            else:
                self.statusLabel.setText('Save canceled.')
                print("Save operation canceled.")
        else:
            self.statusLabel.setText("No processed image to save.")
            print("No processed image to save.")

    def display_image(self, processed_image):
        if processed_image is not None:
            self.combined_image = processed_image

            # Convert the processed image to a displayable format
            preview_image = (processed_image * 255).astype(np.uint8)
            
            # Check if the image is mono or RGB
            if preview_image.ndim == 2:  # Mono image
                # Create a 3-channel RGB image by duplicating the single channel
                preview_image = np.stack([preview_image] * 3, axis=-1)  # Stack to create RGB

            h, w = preview_image.shape[:2]

            # Ensure the array is contiguous
            preview_image = np.ascontiguousarray(preview_image)

            # Change the format to RGB888 for displaying an RGB image
            q_image = QImage(preview_image.data, w, h, 3 * w, QImage.Format_RGB888)

            pixmap = QPixmap.fromImage(q_image)

            # Store the original pixmap only once
            if self.original_pixmap is None:
                self.original_pixmap = pixmap.copy()

            # Scale from original pixmap based on zoom_factor
            scaled_pixmap = self.original_pixmap.scaled(
                self.original_pixmap.size() * self.zoom_factor,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.imageLabel.setPixmap(scaled_pixmap)
            self.imageLabel.resize(scaled_pixmap.size())

            # Enable save and zoom buttons now that an image is processed
            self.save_button.setEnabled(True)
            self.zoomInButton.setEnabled(True)
            self.zoomOutButton.setEnabled(True)

            self.statusLabel.setText("Continuum subtraction completed.")
            # Push the processed image to ImageManager
            if self.image_manager:
                metadata = {
                    'file_path': self.filename,
                    'original_header': self.original_header,
                    'is_mono': self.is_mono,
                    'source': 'Continuum Subtraction'
                }
                self.image_manager.update_image(self.combined_image, metadata, slot=0)

                print("ContinuumSubtractTab: Image pushed to ImageManager.")
        else:
            self.statusLabel.setText("Continuum subtraction failed.")
            print("Continuum subtraction failed.")

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
                self.scrollArea.horizontalScrollBar().setValue(
                    self.scrollArea.horizontalScrollBar().value() - delta.x()
                )
                self.scrollArea.verticalScrollBar().setValue(
                    self.scrollArea.verticalScrollBar().value() - delta.y()
                )
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





def load_image(filename, max_retries=3, wait_seconds=3):
    """
    Loads an image from the specified filename with support for various formats.
    If a "buffer is too small for requested array" error occurs, it retries loading after waiting.

    Parameters:
        filename (str): Path to the image file.
        max_retries (int): Number of times to retry on specific buffer error.
        wait_seconds (int): Seconds to wait before retrying.

    Returns:
        tuple: (image, original_header, bit_depth, is_mono) or (None, None, None, None) on failure.
    """
    attempt = 0
    while attempt <= max_retries:
        try:
            image = None  # Ensure 'image' is explicitly declared
            bit_depth = None
            is_mono = False
            original_header = None

            if filename.lower().endswith(('.fits', '.fit')):
                print(f"Loading FITS file: {filename}")
                with fits.open(filename) as hdul:
                    image_data = hdul[0].data
                    original_header = hdul[0].header  # Capture the FITS header

                    # Ensure native byte order
                    if image_data.dtype.byteorder not in ('=', '|'):
                        image_data = image_data.astype(image_data.dtype.newbyteorder('='))

                    # Determine bit depth
                    if image_data.dtype == np.uint8:
                        bit_depth = "8-bit"
                        print("Identified 8-bit FITS image.")
                        image = image_data.astype(np.float32) / 255.0
                    elif image_data.dtype == np.uint16:
                        bit_depth = "16-bit"
                        print("Identified 16-bit FITS image.")
                        image = image_data.astype(np.float32) / 65535.0
                    elif image_data.dtype == np.float32:
                        bit_depth = "32-bit floating point"
                        print("Identified 32-bit floating point FITS image.")
                    elif image_data.dtype == np.uint32:
                        bit_depth = "32-bit unsigned"
                        print("Identified 32-bit unsigned FITS image.")
                    else:
                        raise ValueError("Unsupported FITS data type!")

                    # Handle 3D FITS data (e.g., RGB or multi-layered)
                    if image_data.ndim == 3 and image_data.shape[0] == 3:
                        image = np.transpose(image_data, (1, 2, 0))  # Reorder to (height, width, channels)

                        if bit_depth == "8-bit":
                            image = image.astype(np.float32) / 255.0
                        elif bit_depth == "16-bit":
                            image = image.astype(np.float32) / 65535.0
                        elif bit_depth == "32-bit unsigned":
                            bzero = original_header.get('BZERO', 0)
                            bscale = original_header.get('BSCALE', 1)
                            image = image.astype(np.float32) * bscale + bzero

                            # Normalize based on range
                            image_min = image.min()
                            image_max = image.max()
                            image = (image - image_min) / (image_max - image_min)
                        # No normalization needed for 32-bit float
                        is_mono = False

                    # Handle 2D FITS data (grayscale)
                    elif image_data.ndim == 2:
                        if bit_depth == "8-bit":
                            image = image_data.astype(np.float32) / 255.0
                        elif bit_depth == "16-bit":
                            image = image_data.astype(np.float32) / 65535.0
                        elif bit_depth == "32-bit unsigned":
                            bzero = original_header.get('BZERO', 0)
                            bscale = original_header.get('BSCALE', 1)
                            image = image_data.astype(np.float32) * bscale + bzero

                            # Normalize based on range
                            image_min = image.min()
                            image_max = image.max()
                            image = (image - image_min) / (image_max - image_min)
                        elif bit_depth == "32-bit floating point":
                            image = image_data
                        else:
                            raise ValueError("Unsupported FITS data type!")

                        # Mono or RGB handling
                        if image_data.ndim == 2:  # Mono
                            is_mono = True
                            return image, original_header, bit_depth, is_mono
                        elif image_data.ndim == 3 and image_data.shape[0] == 3:  # RGB
                            image = np.transpose(image_data, (1, 2, 0))  # Convert to (H, W, C)
                            is_mono = False
                            return image, original_header, bit_depth, is_mono

                    else:
                        raise ValueError("Unsupported FITS format or dimensions!")

            elif filename.lower().endswith(('.tiff', '.tif')):
                print(f"Loading TIFF file: {filename}")
                image_data = tiff.imread(filename)
                print(f"Loaded TIFF image with dtype: {image_data.dtype}")

                # Determine bit depth and normalize
                if image_data.dtype == np.uint8:
                    bit_depth = "8-bit"
                    image = image_data.astype(np.float32) / 255.0
                elif image_data.dtype == np.uint16:
                    bit_depth = "16-bit"
                    image = image_data.astype(np.float32) / 65535.0
                elif image_data.dtype == np.uint32:
                    bit_depth = "32-bit unsigned"
                    image = image_data.astype(np.float32) / 4294967295.0
                elif image_data.dtype == np.float32:
                    bit_depth = "32-bit floating point"
                    image = image_data
                else:
                    raise ValueError("Unsupported TIFF format!")

                # Handle mono or RGB TIFFs
                if image_data.ndim == 2:  # Mono
                    is_mono = True
                elif image_data.ndim == 3 and image_data.shape[2] == 3:  # RGB
                    is_mono = False
                else:
                    raise ValueError("Unsupported TIFF image dimensions!")

            elif filename.lower().endswith('.xisf'):
                print(f"Loading XISF file: {filename}")
                xisf = XISF(filename)

                # Read image data (assuming the first image in the XISF file)
                image_data = xisf.read_image(0)  # Adjust the index if multiple images are present

                # Retrieve metadata
                image_meta = xisf.get_images_metadata()[0]  # Assuming single image
                file_meta = xisf.get_file_metadata()


                # Determine bit depth and normalize
                if image_data.dtype == np.uint8:
                    bit_depth = "8-bit"
                    image = image_data.astype(np.float32) / 255.0
                elif image_data.dtype == np.uint16:
                    bit_depth = "16-bit"
                    image = image_data.astype(np.float32) / 65535.0
                elif image_data.dtype == np.uint32:
                    bit_depth = "32-bit unsigned"
                    image = image_data.astype(np.float32) / 4294967295.0
                elif image_data.dtype == np.float32:
                    bit_depth = "32-bit floating point"
                    image = image_data
                else:
                    raise ValueError("Unsupported XISF data type!")

                # Handle mono or RGB XISF
                if image_data.ndim == 2 or (image_data.ndim == 3 and image_data.shape[2] == 1):  # Mono
                    is_mono = True
                    if image_data.ndim == 3:
                        image = np.squeeze(image_data, axis=2)
                    image = np.stack([image] * 3, axis=-1)  # Convert to RGB by stacking
                elif image_data.ndim == 3 and image_data.shape[2] == 3:  # RGB
                    is_mono = False
                else:
                    raise ValueError("Unsupported XISF image dimensions!")

                # For XISF, you can choose what to set as original_header
                # It could be a combination of file_meta and image_meta or any other relevant information
                original_header = {
                    "file_meta": file_meta,
                    "image_meta": image_meta
                }

                print(f"Loaded XISF image: shape={image.shape}, bit depth={bit_depth}, mono={is_mono}")
                return image, original_header, bit_depth, is_mono

            elif filename.lower().endswith(('.cr2', '.nef', '.arw', '.dng', '.orf', '.rw2', '.pef')):
                print(f"Loading RAW file: {filename}")
                with rawpy.imread(filename) as raw:
                    # Get the raw Bayer data
                    bayer_image = raw.raw_image_visible.astype(np.float32)
                    print(f"Raw Bayer image dtype: {bayer_image.dtype}, min: {bayer_image.min()}, max: {bayer_image.max()}")

                    # Ensure Bayer image is normalized
                    bayer_image /= bayer_image.max()

                    if bayer_image.ndim == 2:
                        image = bayer_image  # Keep as 2D mono image
                        is_mono = True
                    elif bayer_image.ndim == 3 and bayer_image.shape[2] == 3:
                        image = bayer_image  # Already RGB
                        is_mono = False
                    else:
                        raise ValueError(f"Unexpected RAW Bayer image shape: {bayer_image.shape}")
                    bit_depth = "16-bit"  # Assuming 16-bit raw data
                    is_mono = True

                    # Populate `original_header` with RAW metadata
                    original_header_dict = {
                        'CAMERA': raw.camera_whitebalance[0] if raw.camera_whitebalance else 'Unknown',
                        'EXPTIME': raw.shutter if hasattr(raw, 'shutter') else 0.0,
                        'ISO': raw.iso_speed if hasattr(raw, 'iso_speed') else 0,
                        'FOCAL': raw.focal_len if hasattr(raw, 'focal_len') else 0.0,
                        'DATE': raw.timestamp if hasattr(raw, 'timestamp') else 'Unknown',
                    }

                    # Extract CFA pattern
                    cfa_pattern = raw.raw_colors_visible
                    cfa_mapping = {
                        0: 'R',  # Red
                        1: 'G',  # Green
                        2: 'B',  # Blue
                    }
                    cfa_description = ''.join([cfa_mapping.get(color, '?') for color in cfa_pattern.flatten()[:4]])

                    # Add CFA pattern to header
                    original_header_dict['CFA'] = (cfa_description, 'Color Filter Array pattern')

                    # Convert original_header_dict to fits.Header
                    original_header = fits.Header()
                    for key, value in original_header_dict.items():
                        original_header[key] = value

                    print(f"RAW file loaded with CFA pattern: {cfa_description}")

            elif filename.lower().endswith('.png'):
                print(f"Loading PNG file: {filename}")
                img = Image.open(filename)

                # Convert unsupported modes to RGB
                if img.mode not in ('L', 'RGB'):
                    print(f"Unsupported PNG mode: {img.mode}, converting to RGB")
                    img = img.convert("RGB")

                # Convert image to numpy array and normalize pixel values to [0, 1]
                image = np.array(img, dtype=np.float32) / 255.0
                bit_depth = "8-bit"

                # Determine if the image is grayscale or RGB
                if len(image.shape) == 2:  # Grayscale image
                    is_mono = True
                elif len(image.shape) == 3 and image.shape[2] == 3:  # RGB image
                    is_mono = False
                else:
                    raise ValueError(f"Unsupported PNG dimensions: {image.shape}")

                print(f"Loaded PNG image: shape={image.shape}, bit depth={bit_depth}, mono={is_mono}")

            elif filename.lower().endswith(('.jpg', '.jpeg')):
                print(f"Loading JPG file: {filename}")
                img = Image.open(filename)
                if img.mode == 'L':  # Grayscale
                    is_mono = True
                    image = np.array(img, dtype=np.float32) / 255.0
                    bit_depth = "8-bit"
                elif img.mode == 'RGB':  # RGB
                    is_mono = False
                    image = np.array(img, dtype=np.float32) / 255.0
                    bit_depth = "8-bit"
                else:
                    raise ValueError("Unsupported JPG format!")            

            else:
                raise ValueError("Unsupported file format!")

            print(f"Loaded image: shape={image.shape}, bit depth={bit_depth}, mono={is_mono}")
            return image, original_header, bit_depth, is_mono

        except Exception as e:
            error_message = str(e)
            if "buffer is too small for requested array" in error_message.lower():
                if attempt < max_retries:
                    attempt += 1
                    print(f"Error reading image {filename}: {e}")
                    print(f"Retrying in {wait_seconds} seconds... (Attempt {attempt}/{max_retries})")
                    time.sleep(wait_seconds)
                    continue  # Retry loading the image
                else:
                    print(f"Error reading image {filename} after {max_retries} retries: {e}")
            else:
                print(f"Error reading image {filename}: {e}")
            return None, None, None, None








def save_image(img_array, filename, original_format, bit_depth=None, original_header=None, is_mono=False, image_meta=None, file_meta=None):
    """
    Save an image array to a file in the specified format and bit depth.
    """
    img_array = ensure_native_byte_order(img_array)  # Ensure correct byte order
    xisf_metadata = original_header

    try:
        if original_format == 'png':
            img = Image.fromarray((img_array * 255).astype(np.uint8))  # Convert to 8-bit and save as PNG
            img.save(filename)
            print(f"Saved 8-bit PNG image to: {filename}")
        
        elif original_format in ['tiff', 'tif']:
            # Save TIFF files based on bit depth
            if bit_depth == "8-bit":
                tiff.imwrite(filename, (img_array * 255).astype(np.uint8))  # Save as 8-bit TIFF
            elif bit_depth == "16-bit":
                tiff.imwrite(filename, (img_array * 65535).astype(np.uint16))  # Save as 16-bit TIFF
            elif bit_depth == "32-bit unsigned":
                tiff.imwrite(filename, (img_array * 4294967295).astype(np.uint32))  # Save as 32-bit unsigned TIFF
            elif bit_depth == "32-bit floating point":
                tiff.imwrite(filename, img_array.astype(np.float32))  # Save as 32-bit floating point TIFF
            else:
                raise ValueError("Unsupported bit depth for TIFF!")
            print(f"Saved {bit_depth} TIFF image to: {filename}")

        elif original_format in ['fits', 'fit']:
            # Preserve the original extension
            if not filename.lower().endswith(f".{original_format}"):
                filename = filename.rsplit('.', 1)[0] + f".{original_format}"

            if original_header is not None:
                # Convert original_header (dictionary) to astropy Header object
                fits_header = fits.Header()
                for key, value in original_header.items():
                    fits_header[key] = value
                fits_header['BSCALE'] = 1.0  # Scaling factor
                fits_header['BZERO'] = 0.0   # Offset for brightness    

                # Handle mono (2D) images
                if is_mono or img_array.ndim == 2:
                    if bit_depth == "16-bit":
                        img_array_fits = (img_array * 65535).astype(np.uint16)
                    elif bit_depth == "32-bit unsigned":
                        bzero = fits_header.get('BZERO', 0)
                        bscale = fits_header.get('BSCALE', 1)
                        img_array_fits = (img_array.astype(np.float32) * bscale + bzero).astype(np.uint32)
                    else:  # 32-bit float
                        img_array_fits = img_array.astype(np.float32)

                    # Update header for a 2D (grayscale) image
                    fits_header['NAXIS'] = 2
                    fits_header['NAXIS1'] = img_array.shape[1]  # Width
                    fits_header['NAXIS2'] = img_array.shape[0]  # Height
                    fits_header.pop('NAXIS3', None)  # Remove if present

                    hdu = fits.PrimaryHDU(img_array_fits, header=fits_header)

                # Handle RGB (3D) images
                else:
                    img_array_transposed = np.transpose(img_array, (2, 0, 1))  # Channels, Height, Width
                    if bit_depth == "16-bit":
                        img_array_fits = (img_array_transposed * 65535).astype(np.uint16)
                    elif bit_depth == "32-bit unsigned":
                        bzero = fits_header.get('BZERO', 0)
                        bscale = fits_header.get('BSCALE', 1)
                        img_array_fits = img_array_transposed.astype(np.float32) * bscale + bzero
                        fits_header['BITPIX'] = -32
                    else:  # Default to 32-bit float
                        img_array_fits = img_array_transposed.astype(np.float32)

                    # Update header for a 3D (RGB) image
                    fits_header['NAXIS'] = 3
                    fits_header['NAXIS1'] = img_array_transposed.shape[2]  # Width
                    fits_header['NAXIS2'] = img_array_transposed.shape[1]  # Height
                    fits_header['NAXIS3'] = img_array_transposed.shape[0]  # Channels

                    hdu = fits.PrimaryHDU(img_array_fits, header=fits_header)

                # Write the FITS file
                try:
                    hdu.writeto(filename, overwrite=True)
                    print(f"Saved as {original_format.upper()} to: {filename}")
                except Exception as e:
                    print(f"Error saving FITS file: {e}")
            else:
                raise ValueError("Original header is required for FITS format!")

        elif original_format in ['.cr2', '.nef', '.arw', '.dng', '.orf', '.rw2', '.pef']:
            # Save as FITS file with metadata
            print("RAW formats are not writable. Saving as FITS instead.")
            filename = filename.rsplit('.', 1)[0] + ".fits"

            if original_header is not None:
                # Convert original_header (dictionary) to astropy Header object
                fits_header = fits.Header()
                for key, value in original_header.items():
                    fits_header[key] = value
                fits_header['BSCALE'] = 1.0  # Scaling factor
                fits_header['BZERO'] = 0.0   # Offset for brightness    

                if is_mono:  # Grayscale FITS
                    if bit_depth == "16-bit":
                        img_array_fits = (img_array[:, :, 0] * 65535).astype(np.uint16)
                    elif bit_depth == "32-bit unsigned":
                        bzero = fits_header.get('BZERO', 0)
                        bscale = fits_header.get('BSCALE', 1)
                        img_array_fits = (img_array[:, :, 0].astype(np.float32) * bscale + bzero).astype(np.uint32)
                    else:  # 32-bit float
                        img_array_fits = img_array[:, :, 0].astype(np.float32)

                    # Update header for a 2D (grayscale) image
                    fits_header['NAXIS'] = 2
                    fits_header['NAXIS1'] = img_array.shape[1]  # Width
                    fits_header['NAXIS2'] = img_array.shape[0]  # Height
                    fits_header.pop('NAXIS3', None)  # Remove if present

                    hdu = fits.PrimaryHDU(img_array_fits, header=fits_header)
                else:  # RGB FITS
                    img_array_transposed = np.transpose(img_array, (2, 0, 1))  # Channels, Height, Width
                    if bit_depth == "16-bit":
                        img_array_fits = (img_array_transposed * 65535).astype(np.uint16)
                    elif bit_depth == "32-bit unsigned":
                        bzero = fits_header.get('BZERO', 0)
                        bscale = fits_header.get('BSCALE', 1)
                        img_array_fits = img_array_transposed.astype(np.float32) * bscale + bzero
                        fits_header['BITPIX'] = -32
                    else:  # Default to 32-bit float
                        img_array_fits = img_array_transposed.astype(np.float32)

                    # Update header for a 3D (RGB) image
                    fits_header['NAXIS'] = 3
                    fits_header['NAXIS1'] = img_array_transposed.shape[2]  # Width
                    fits_header['NAXIS2'] = img_array_transposed.shape[1]  # Height
                    fits_header['NAXIS3'] = img_array_transposed.shape[0]  # Channels

                    hdu = fits.PrimaryHDU(img_array_fits, header=fits_header)

                # Write the FITS file
                try:
                    hdu.writeto(filename, overwrite=True)
                    print(f"RAW processed and saved as FITS to: {filename}")
                except Exception as e:
                    print(f"Error saving FITS file: {e}")
            else:
                raise ValueError("Original header is required for FITS format!")

        elif original_format == 'xisf':
            try:
                print(f"Original image shape: {img_array.shape}, dtype: {img_array.dtype}")
                print(f"Bit depth: {bit_depth}")

                # Adjust bit depth for saving
                if bit_depth == "16-bit":
                    processed_image = (img_array * 65535).astype(np.uint16)
                elif bit_depth == "32-bit unsigned":
                    processed_image = (img_array * 4294967295).astype(np.uint32)
                else:  # Default to 32-bit float
                    processed_image = img_array.astype(np.float32)

                # Handle mono images explicitly
                if is_mono:
                    print("Detected mono image. Preparing for XISF...")
                    if processed_image.ndim == 3 and processed_image.shape[2] > 1:
                        processed_image = processed_image[:, :, 0]  # Extract single channel
                    processed_image = processed_image[:, :, np.newaxis]  # Add back channel dimension

                    # Update metadata for mono images
                    if image_meta and isinstance(image_meta, list):
                        image_meta[0]['geometry'] = (processed_image.shape[1], processed_image.shape[0], 1)
                        image_meta[0]['colorSpace'] = 'Gray'
                    else:
                        # Create default metadata for mono images
                        image_meta = [{
                            'geometry': (processed_image.shape[1], processed_image.shape[0], 1),
                            'colorSpace': 'Gray'
                        }]

                # Handle RGB images
                else:
                    if image_meta and isinstance(image_meta, list):
                        image_meta[0]['geometry'] = (processed_image.shape[1], processed_image.shape[0], processed_image.shape[2])
                        image_meta[0]['colorSpace'] = 'RGB'
                    else:
                        # Create default metadata for RGB images
                        image_meta = [{
                            'geometry': (processed_image.shape[1], processed_image.shape[0], processed_image.shape[2]),
                            'colorSpace': 'RGB'
                        }]

                # Ensure fallback for `image_meta` and `file_meta`
                if image_meta is None or not isinstance(image_meta, list):
                    image_meta = [{
                        'geometry': (processed_image.shape[1], processed_image.shape[0], 1 if is_mono else 3),
                        'colorSpace': 'Gray' if is_mono else 'RGB'
                    }]
                if file_meta is None:
                    file_meta = {}

                # Debug: Print processed image details and metadata
                print(f"Processed image shape for XISF: {processed_image.shape}, dtype: {processed_image.dtype}")

                # Save the image using XISF.write
                XISF.write(
                    filename,                    # Output path
                    processed_image,             # Final processed image
                    creator_app="Seti Astro Cosmic Clarity",
                    image_metadata=image_meta[0],  # First block of image metadata
                    xisf_metadata=file_meta,       # File-level metadata
                    shuffle=True
                )

                print(f"Saved {bit_depth} XISF image to: {filename}")

            except Exception as e:
                print(f"Error saving XISF file: {e}")
                raise


        else:
            raise ValueError("Unsupported file format!")

    except Exception as e:
        print(f"Error saving image to {filename}: {e}")
        raise






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
        return array.byteswap().view(array.dtype.newbyteorder('='))
    return array


# Determine if running inside a PyInstaller bundle
if hasattr(sys, '_MEIPASS'):
    # Set path for PyInstaller bundle
    data_path = os.path.join(sys._MEIPASS, "astroquery", "simbad", "data")
else:
    # Set path for regular Python environment
    data_path = "/Users/franklinmarek/cosmicclarity/env/lib/python3.12/site-packages/astroquery/simbad/data"

# Ensure the final path doesn't contain 'data/data' duplication
if 'data/data' in data_path:
    data_path = data_path.replace('data/data', 'data')

conf.dataurl = f'file://{data_path}/'

# Access wrench_icon.png, adjusting for PyInstaller executable
if hasattr(sys, '_MEIPASS'):
    wrench_path = os.path.join(sys._MEIPASS, 'wrench_icon.png')
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
    wrench_path = 'wrench_icon.png'  # Path for running as a script
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

    def delete_selected_object(self):
        if self.selected_object is None:
            self.parent.status_label.setText("No object selected to delete.")
            return

        # Remove the selected object from the results list
        self.parent.results = [obj for obj in self.parent.results if obj != self.selected_object]

        # Remove the corresponding row from the TreeBox
        for i in range(self.parent.results_tree.topLevelItemCount()):
            item = self.parent.results_tree.topLevelItem(i)
            if item.text(2) == self.selected_object["name"]:  # Match the name in the third column
                self.parent.results_tree.takeTopLevelItem(i)
                break

        # Clear the selection
        self.selected_object = None
        self.parent.results_tree.clearSelection()

        # Redraw the main and mini previews without the deleted marker
        self.draw_query_results()
        self.update_mini_preview()

        # Update the status label
        self.parent.status_label.setText("Selected object and marker removed.")



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

    def zoom_to_coordinates(self, ra, dec):
        """Zoom to the specified RA/Dec coordinates and center the view on that position."""
        # Calculate the pixel position from RA and Dec
        pixel_x, pixel_y = self.parent.calculate_pixel_from_ra_dec(ra, dec)
        
        if pixel_x is not None and pixel_y is not None:
            # Center the view on the calculated pixel position
            self.centerOn(pixel_x, pixel_y)
            
            # Reset the zoom level to 1.0 by adjusting the transformation matrix
            self.resetTransform()
            self.scale(1.0, 1.0)

            # Optionally, update the mini preview to reflect the new zoom and center
            self.update_mini_preview()

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
            #if self.circle_center is not None and self.circle_radius > 0:
            #    self.update_circle()

            # Draw object markers (circle or crosshair)
            for obj in self.parent.results:
                ra, dec, name = obj["ra"], obj["dec"], obj["name"]
                x, y = self.parent.calculate_pixel_from_ra_dec(ra, dec)
                if x is not None and y is not None:
                    # Determine color: green if selected, red otherwise
                    pen_color = QColor(0, 255, 0) if obj == self.selected_object else QColor(255, 0, 0)
                    pen = QPen(pen_color, 2)

                    if self.parent.marker_style == "Circle":
                        # Draw a circle around the object
                        self.parent.main_scene.addEllipse(int(x - 5), int(y - 5), 10, 10, pen)
                    elif self.parent.marker_style == "Crosshair":
                        # Draw crosshair with a 5-pixel gap in the middle
                        crosshair_size = 10
                        gap = 5
                        line1 = QLineF(x - crosshair_size, y, x - gap, y)
                        line2 = QLineF(x + gap, y, x + crosshair_size, y)
                        line3 = QLineF(x, y - crosshair_size, x, y - gap)
                        line4 = QLineF(x, y + gap, x, y + crosshair_size)
                        for line in [line1, line2, line3, line4]:
                            crosshair_item = QGraphicsLineItem(line)
                            crosshair_item.setPen(pen)
                            self.parent.main_scene.addItem(crosshair_item)
                    if self.parent.show_names:
                        #print(f"Drawing name: {name} at ({x}, {y})")  # Debugging statement
                        text_color = obj.get("color", QColor(Qt.white))
                        text_item = QGraphicsTextItem(name)
                        text_item.setPos(x + 10, y + 10)  # Offset to avoid overlapping the marker
                        text_item.setDefaultTextColor(text_color)
                        text_item.setFont(self.parent.selected_font)
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
        self.metadata = {}
        self.circle_center = None
        self.circle_radius = 0    
        self.show_names = False  # Boolean to toggle showing names on the main image
        self.max_results = 100  # Default maximum number of query results     
        self.current_tool = None  # Track the active annotation tool
        self.marker_style = "Circle" 
            

        main_layout = QHBoxLayout()

        # Left Column Layout
        left_panel = QVBoxLayout()

        # Load the image using the resource_path function
        wimilogo_path = resource_path("wimilogo.png")

        # Create a QLabel to display the logo
        self.logo_label = QLabel()

        # Set the logo image to the label
        logo_pixmap = QPixmap(wimilogo_path)

        # Scale the pixmap to fit within a desired size, maintaining the aspect ratio
        scaled_pixmap = logo_pixmap.scaled(200, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        # Set the scaled pixmap to the label
        self.logo_label.setPixmap(scaled_pixmap)

        # Set alignment to center the logo horizontally
        self.logo_label.setAlignment(Qt.AlignCenter)

        # Optionally, you can set a fixed size for the label (this is for layout purposes)
        self.logo_label.setFixedSize(200, 100)  # Adjust the size as needed

        # Add the logo_label to your layout
        left_panel.addWidget(self.logo_label)
       
        button_layout = QHBoxLayout()
        
        # Load button
        self.load_button = QPushButton("Load Image")
        self.load_button.setIcon(QApplication.style().standardIcon(QStyle.SP_FileDialogStart))
        self.load_button.clicked.connect(self.open_image)

        # AutoStretch button
        self.auto_stretch_button = QPushButton("AutoStretch")
        self.auto_stretch_button.clicked.connect(self.toggle_autostretch)

        # Add both buttons to the horizontal layout
        button_layout.addWidget(self.load_button)
        button_layout.addWidget(self.auto_stretch_button)

        # Add the button layout to the left panel
        left_panel.addLayout(button_layout)

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
        self.settings_button.setIcon(QIcon(wrench_path))  # Adjust icon path as needed
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

        # Create a horizontal layout for the labels
        label_layout = QHBoxLayout()

        # Create the label to display the count of objects
        self.object_count_label = QLabel("Objects Found: 0")

        # Create the label with instructions
        self.instructions_label = QLabel("Right Click a Row for More Options")

        # Add both labels to the horizontal layout
        label_layout.addWidget(self.object_count_label)
        label_layout.addWidget(self.instructions_label)

        # Add the horizontal layout to the main panel layout
        right_panel.addLayout(label_layout)

        self.results_tree = QTreeWidget()
        self.results_tree.setHeaderLabels(["RA", "Dec", "Name", "Diameter", "Type", "Long Type", "Redshift", "Comoving Radial Distance (GLy)"])
        self.results_tree.setFixedHeight(150)
        self.results_tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.results_tree.customContextMenuRequested.connect(self.open_context_menu)
        self.results_tree.itemClicked.connect(self.on_tree_item_clicked)
        self.results_tree.itemDoubleClicked.connect(self.on_tree_item_double_clicked)
        self.results_tree.setSortingEnabled(True)
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

        # Delete Selected Object button
        self.delete_selected_object_button = QPushButton("Delete Selected Object")
        self.delete_selected_object_button.setIcon(QApplication.style().standardIcon(QStyle.SP_DialogCloseButton))  # Trash icon
        self.delete_selected_object_button.clicked.connect(self.main_preview.delete_selected_object)  # Connect to delete_selected_object in CustomGraphicsView

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
        annotation_tools_layout.addWidget(self.delete_selected_object_button, 3, 4)

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

    def update_object_count(self):
        count = self.results_tree.topLevelItemCount()
        self.object_count_label.setText(f"Objects Found: {count}")

    def open_context_menu(self, position):
        
        # Get the item at the mouse position
        item = self.results_tree.itemAt(position)
        if not item:
            return  # If no item is clicked, do nothing
        
        self.on_tree_item_clicked(item)

        # Create the context menu
        menu = QMenu(self)

        # Define actions
        open_website_action = QAction("Open Website", self)
        open_website_action.triggered.connect(lambda: self.results_tree.itemDoubleClicked.emit(item, 0))
        menu.addAction(open_website_action)

        zoom_to_object_action = QAction("Zoom to Object", self)
        zoom_to_object_action.triggered.connect(lambda: self.zoom_to_object(item))
        menu.addAction(zoom_to_object_action)

        copy_info_action = QAction("Copy Object Information", self)
        copy_info_action.triggered.connect(lambda: self.copy_object_information(item))
        menu.addAction(copy_info_action)

        # Display the context menu at the cursor position
        menu.exec_(self.results_tree.viewport().mapToGlobal(position))

    def toggle_autostretch(self):
        if not hasattr(self, 'original_image'):
            # Store the original image the first time AutoStretch is applied
            self.original_image = self.image_data.copy()
        
        # Determine if the image is mono or color based on the number of dimensions
        if self.image_data.ndim == 2:
            # Call stretch_mono_image if the image is mono

            stretched_image = stretch_mono_image(self.image_data, target_median=0.25, normalize=True)
        else:
            # Call stretch_color_image if the image is color

            stretched_image = stretch_color_image(self.image_data, target_median=0.25, linked=True, normalize=True)
        
        # If the AutoStretch is toggled off (using the same button), restore the original image
        if self.auto_stretch_button.text() == "AutoStretch":
            # Store the stretched image and update the button text to indicate it's on
            self.stretched_image = stretched_image
            self.auto_stretch_button.setText("Turn Off AutoStretch")
        else:
            # Revert to the original image and update the button text to indicate it's off
            stretched_image = self.original_image
            self.auto_stretch_button.setText("AutoStretch")
        

        stretched_image = (stretched_image * 255).astype(np.uint8)


        # Update the display with the stretched image (or original if toggled off)

        height, width = stretched_image.shape[:2]
        bytes_per_line = 3 * width

        # Ensure the image has 3 channels (RGB)
        if stretched_image.ndim == 2:
            stretched_image = np.stack((stretched_image,) * 3, axis=-1)
        elif stretched_image.shape[2] == 1:
            stretched_image = np.repeat(stretched_image, 3, axis=2)



        qimg = QImage(stretched_image.tobytes(), width, height, bytes_per_line, QImage.Format_RGB888)
        if qimg.isNull():
            print("Failed to create QImage")
            return

        pixmap = QPixmap.fromImage(qimg)
        if pixmap.isNull():
            print("Failed to create QPixmap")
            return

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

        # Optionally, you can also update any other parts of the UI after stretching the image
        print(f"AutoStretch {'applied to' if self.auto_stretch_button.text() == 'Turn Off AutoStretch' else 'removed from'} the image.")




    def zoom_to_object(self, item):
        """Zoom to the object in the main preview."""
        ra = float(item.text(0))  # Assuming RA is in the first column
        dec = float(item.text(1))  # Assuming Dec is in the second column
        self.main_preview.zoom_to_coordinates(ra, dec)
        

    def copy_object_information(self, item):
        """Copy object information to the clipboard."""
        info = f"RA: {item.text(0)}, Dec: {item.text(1)}, Name: {item.text(2)}, Diameter: {item.text(3)}, Type: {item.text(4)}"
        clipboard = QApplication.clipboard()
        clipboard.setText(info)

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
        self.query_simbad(radius_deg, max_results=100000)




    def toggle_advanced_search(self):
        """Toggle visibility of the advanced search panel."""
        self.advanced_search_panel.setVisible(not self.advanced_search_panel.isVisible())

    def toggle_all_items(self):
        """Toggle selection for all items in the object tree."""
        # Check if all items are currently selected
        all_checked = all(
            self.object_tree.topLevelItem(i).checkState(0) == Qt.Checked
            for i in range(self.object_tree.topLevelItemCount())
        )

        # Determine the new state: Uncheck if all are checked, otherwise check all
        new_state = Qt.Unchecked if all_checked else Qt.Checked

        # Apply the new state to all items
        for i in range(self.object_tree.topLevelItemCount()):
            item = self.object_tree.topLevelItem(i)
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
        self.show_names = bool(state)        
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
        self.image_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.jpeg *.tif *.tiff *.fit *.fits *.xisf)")
        if self.image_path:
            img_array, original_header, bit_depth, is_mono = load_image(self.image_path)
            if img_array is not None:

                self.image_data = img_array
                self.original_header = original_header
                self.bit_depth = bit_depth
                self.is_mono = is_mono

                # Prepare image for display
                if img_array.ndim == 2:  # Single-channel image
                    img_array = np.stack([img_array] * 3, axis=-1)  # Expand to 3 channels


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
                elif self.image_path.lower().endswith('.xisf'):
                    # Load WCS from XISF properties
                    xisf_meta = self.extract_xisf_metadata(self.image_path)
                    self.metadata = xisf_meta  # Ensure metadata is stored in self.metadata for later use

                    # Construct WCS header from XISF properties
                    header = self.construct_fits_header_from_xisf(xisf_meta)
                    if header:
                        try:
                            self.initialize_wcs_from_header(header)
                        except ValueError as e:
                            print("Error initializing WCS from XISF:", e)
                            QMessageBox.warning(self, "WCS Error", "Failed to load WCS data from XISF properties.")
                else:
                    # For non-FITS images (e.g., JPEG, PNG), prompt directly for a blind solve
                    self.prompt_blind_solve()

    def extract_xisf_metadata(self, xisf_path):
        """
        Extract metadata from a .xisf file, focusing on WCS and essential image properties.
        """
        try:
            # Load the XISF file
            xisf = XISF(xisf_path)
            
            # Extract file and image metadata
            self.file_meta = xisf.get_file_metadata()
            self.image_meta = xisf.get_images_metadata()[0]  # Get metadata for the first image
            return self.image_meta
        except Exception as e:
            print(f"Error reading XISF metadata: {e}")
            return None

    def initialize_wcs_from_header(self, header):
        """ Initialize WCS data from a FITS header or constructed XISF header """
        try:
            # Use only the first two dimensions for WCS
            self.wcs = WCS(header, naxis=2, relax=True)
            
            # Calculate and set pixel scale
            pixel_scale_matrix = self.wcs.pixel_scale_matrix
            self.pixscale = np.sqrt(pixel_scale_matrix[0, 0]**2 + pixel_scale_matrix[1, 0]**2) * 3600  # arcsec/pixel
            self.center_ra, self.center_dec = self.wcs.wcs.crval
            self.wcs_header = self.wcs.to_header(relax=True)  # Store the full WCS header, including non-standard keywords
            self.print_corner_coordinates()
            
            # Display WCS information
            if 'CROTA2' in header:
                self.orientation = header['CROTA2']
            else:
                self.orientation = calculate_orientation(header)
                if self.orientation is None:
                    print("Orientation: CD matrix elements not found in WCS header.")

            if self.orientation is not None:
                print(f"Orientation: {self.orientation:.2f}°")
                self.orientation_label.setText(f"Orientation: {self.orientation:.2f}°")
            else:
                self.orientation_label.setText("Orientation: N/A")

            print(f"WCS data loaded from header: RA={self.center_ra}, Dec={self.center_dec}, Pixel Scale={self.pixscale} arcsec/px")
        except ValueError as e:
            raise ValueError(f"WCS initialization error: {e}")

    def construct_fits_header_from_xisf(self, xisf_meta):
        """ Convert XISF metadata to a FITS header compatible with WCS """
        header = fits.Header()

        # Define WCS keywords to populate
        wcs_keywords = ["CTYPE1", "CTYPE2", "CRPIX1", "CRPIX2", "CRVAL1", "CRVAL2", "CDELT1", "CDELT2", 
                        "A_ORDER", "B_ORDER", "AP_ORDER", "BP_ORDER"]

        # Populate WCS and FITS keywords
        if 'FITSKeywords' in xisf_meta:
            for keyword, values in xisf_meta['FITSKeywords'].items():
                for entry in values:
                    if 'value' in entry:
                        value = entry['value']
                        if keyword in wcs_keywords:
                            try:
                                value = int(value)
                            except ValueError:
                                value = float(value)
                        header[keyword] = value

        # Manually add WCS information if missing
        header.setdefault('CTYPE1', 'RA---TAN')
        header.setdefault('CTYPE2', 'DEC--TAN')

        # Add SIP distortion suffix if SIP coefficients are present
        if any(key in header for key in ["A_ORDER", "B_ORDER", "AP_ORDER", "BP_ORDER"]):
            header['CTYPE1'] = 'RA---TAN-SIP'
            header['CTYPE2'] = 'DEC--TAN-SIP'

        # Set default reference pixel to the center of the image
        header.setdefault('CRPIX1', self.image_data.shape[1] / 2)
        header.setdefault('CRPIX2', self.image_data.shape[0] / 2)

        # Retrieve RA and DEC values if available
        if 'RA' in xisf_meta['FITSKeywords']:
            header['CRVAL1'] = float(xisf_meta['FITSKeywords']['RA'][0]['value'])  # Reference RA
        if 'DEC' in xisf_meta['FITSKeywords']:
            header['CRVAL2'] = float(xisf_meta['FITSKeywords']['DEC'][0]['value'])  # Reference DEC

        # Calculate pixel scale if focal length and pixel size are available
        if 'FOCALLEN' in xisf_meta['FITSKeywords'] and 'XPIXSZ' in xisf_meta['FITSKeywords']:
            focal_length = float(xisf_meta['FITSKeywords']['FOCALLEN'][0]['value'])  # in mm
            pixel_size = float(xisf_meta['FITSKeywords']['XPIXSZ'][0]['value'])  # in μm
            pixel_scale = (pixel_size * 206.265) / focal_length  # arcsec/pixel
            header['CDELT1'] = -pixel_scale / 3600.0
            header['CDELT2'] = pixel_scale / 3600.0
        else:
            header['CDELT1'] = -2.77778e-4  # ~1 arcsecond/pixel
            header['CDELT2'] = 2.77778e-4

        # Populate CD matrix using the XISF LinearTransformationMatrix if available
        if 'XISFProperties' in xisf_meta and 'PCL:AstrometricSolution:LinearTransformationMatrix' in xisf_meta['XISFProperties']:
            linear_transform = xisf_meta['XISFProperties']['PCL:AstrometricSolution:LinearTransformationMatrix']['value']
            header['CD1_1'] = linear_transform[0][0]
            header['CD1_2'] = linear_transform[0][1]
            header['CD2_1'] = linear_transform[1][0]
            header['CD2_2'] = linear_transform[1][1]
        else:
            # Use pixel scale for CD matrix if no linear transformation is defined
            header['CD1_1'] = header['CDELT1']
            header['CD1_2'] = 0.0
            header['CD2_1'] = 0.0
            header['CD2_2'] = header['CDELT2']

        # Ensure numeric types for SIP distortion keywords if present
        sip_keywords = ["A_ORDER", "B_ORDER", "AP_ORDER", "BP_ORDER"]
        for sip_key in sip_keywords:
            if sip_key in xisf_meta['XISFProperties']:
                try:
                    value = xisf_meta['XISFProperties'][sip_key]['value']
                    header[sip_key] = int(value) if isinstance(value, str) and value.isdigit() else float(value)
                except ValueError:
                    pass  # Ignore any invalid conversion

        return header

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
            # Check if the file is XISF format
            file_extension = os.path.splitext(image_path)[-1].lower()
            if file_extension == ".xisf":
                # Load the XISF image
                xisf = XISF(image_path)
                im_data = xisf.read_image(0)
                
                # Convert to a temporary TIFF file for upload
                temp_image_path = os.path.splitext(image_path)[0] + "_converted.tif"
                if im_data.dtype == np.float32 or im_data.dtype == np.float64:
                    im_data = np.clip(im_data, 0, 1) * 65535
                im_data = im_data.astype(np.uint16)

                # Save as TIFF
                if im_data.shape[-1] == 1:  # Grayscale
                    tiff.imwrite(temp_image_path, np.squeeze(im_data, axis=-1))
                else:  # RGB
                    tiff.imwrite(temp_image_path, im_data)

                print(f"Converted XISF file to TIFF at {temp_image_path} for upload.")
                image_path = temp_image_path  # Use the converted file for upload

            # Upload the image file
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

        finally:
            # Clean up temporary file if created
            if file_extension == ".xisf" and os.path.exists(temp_image_path):
                os.remove(temp_image_path)
                print(f"Temporary TIFF file {temp_image_path} deleted after upload.")



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
            self.update_object_count()

        except Exception as e:
            # Fallback to legacy region query if TAP fails
            try:
                QMessageBox.warning(self, "Query Failed", f"TAP service failed, falling back to legacy region query. Error: {str(e)}")
                
                # Legacy region query fallback
                coord = SkyCoord(ra_center, dec_center, unit="deg")
                legacy_result = Simbad.query_region(coord, radius=radius_deg * u.deg)

                if legacy_result is None or len(legacy_result) == 0:
                    QMessageBox.information(self, "No Results", "No objects found in the specified area (fallback query).")
                    return

                # Process legacy query results
                query_results = []
                self.results_tree.clear()

                for row in legacy_result:
                    try:
                        # Convert RA/Dec to degrees
                        coord = SkyCoord(row["RA"], row["DEC"], unit=(u.hourangle, u.deg))
                        ra = coord.ra.deg  # RA in degrees
                        dec = coord.dec.deg  # Dec in degrees
                    except Exception as coord_error:
                        print(f"Failed to convert RA/Dec for {row['MAIN_ID']}: {coord_error}")
                        continue  # Skip this object if conversion fails

                    # Retrieve other data fields
                    main_id = row["MAIN_ID"]
                    short_type = row["OTYPE"]
                    long_type = otype_long_name_lookup.get(short_type, short_type)

                    # Fallback does not provide some fields, so we use placeholders
                    diameter = "N/A"
                    redshift = "N/A"
                    comoving_distance = "N/A"

                    # Add to TreeWidget for display
                    item = QTreeWidgetItem([
                        f"{ra:.6f}", f"{dec:.6f}", main_id, diameter, short_type, long_type, redshift, comoving_distance
                    ])
                    self.results_tree.addTopLevelItem(item)

                    # Append full details to query_results
                    query_results.append({
                        'ra': ra,  # Ensure degrees format
                        'dec': dec,  # Ensure degrees format
                        'name': main_id,
                        'diameter': diameter,
                        'short_type': short_type,
                        'long_type': long_type,
                        'redshift': redshift,
                        'comoving_distance': comoving_distance,
                        'source': "Simbad (Legacy)"
                    })

                # Pass fallback results to graphics and updates
                self.main_preview.set_query_results(query_results)
                self.query_results = query_results  # Keep a reference to results in MainWindow
                self.update_object_count()

            except Exception as fallback_error:
                QMessageBox.critical(self, "Query Failed", f"Both TAP and fallback queries failed: {str(fallback_error)}")

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
            self.update_object_count()
            
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
            self.update_object_count()

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
        """Open settings dialog to adjust max results and marker type."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Settings")
        
        layout = QFormLayout(dialog)
        
        # Max Results setting
        max_results_spinbox = QSpinBox()
        max_results_spinbox.setRange(1, 100000)
        max_results_spinbox.setValue(self.max_results)
        layout.addRow("Max Results:", max_results_spinbox)
        
        # Marker Style selection
        marker_style_combo = QComboBox()
        marker_style_combo.addItems(["Circle", "Crosshair"])
        marker_style_combo.setCurrentText(self.marker_style)
        layout.addRow("Marker Style:", marker_style_combo)

        # Force Blind Solve button
        force_blind_solve_button = QPushButton("Force Blind Solve")
        force_blind_solve_button.clicked.connect(lambda: self.force_blind_solve(dialog))
        layout.addWidget(force_blind_solve_button)
        
        # OK and Cancel buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(lambda: self.update_settings(max_results_spinbox.value(), marker_style_combo.currentText(), dialog))
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        
        dialog.setLayout(layout)
        dialog.exec_()

    def update_settings(self, max_results, marker_style, dialog):
        """Update settings based on dialog input."""
        self.max_results = max_results
        self.marker_style = marker_style  # Store the selected marker style
        self.main_preview.draw_query_results()
        dialog.accept()

    def force_blind_solve(self, dialog):
        """Force a blind solve on the currently loaded image."""
        dialog.accept()  # Close the settings dialog
        self.prompt_blind_solve()  # Call the blind solve function


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



# Set the directory for the images in the /imgs folder
if getattr(sys, 'frozen', False):  # Check if running as a PyInstaller bundle
    phase_folder = os.path.join(sys._MEIPASS, "imgs")  # Use PyInstaller's temporary directory with /imgs
else:
    phase_folder = os.path.join(os.path.dirname(__file__), "imgs")  # Use the directory of the script file with /imgs


# Set precision for Decimal operations
getcontext().prec = 24

# Suppress warnings
warnings.filterwarnings("ignore")


class CalculationThread(QThread):
    calculation_complete = pyqtSignal(pd.DataFrame, str)
    lunar_phase_calculated = pyqtSignal(int, str)  # phase_percentage, phase_image_name
    lst_calculated = pyqtSignal(str)
    status_update = pyqtSignal(str)

    def __init__(self, latitude, longitude, date, time, timezone, min_altitude, catalog_filters, object_limit):
        super().__init__()
        self.latitude = latitude
        self.longitude = longitude
        self.date = date
        self.time = time
        self.timezone = timezone
        self.min_altitude = min_altitude
        self.catalog_filters = catalog_filters
        self.object_limit = object_limit

    def get_catalog_file_path(self):
        # Define a user-writable location for the catalog (e.g., in the user's home directory)
        user_catalog_path = os.path.join(os.path.expanduser("~"), "celestial_catalog.csv")

        # Check if we are running in a PyInstaller bundle
        if not os.path.exists(user_catalog_path):
            bundled_catalog = os.path.join(getattr(sys, '_MEIPASS', os.path.dirname(__file__)), "celestial_catalog.csv")
            if os.path.exists(bundled_catalog):
                # Copy the bundled catalog to a writable location
                shutil.copyfile(bundled_catalog, user_catalog_path)

        return user_catalog_path  # Return the path to the user-writable catalog

    def run(self):
        try:
            # Convert date and time to astropy Time
            datetime_str = f"{self.date} {self.time}"
            local = pytz.timezone(self.timezone)
            naive_datetime = datetime.strptime(datetime_str, "%Y-%m-%d %H:%M")
            local_datetime = local.localize(naive_datetime)
            astropy_time = Time(local_datetime)

            # Define observer's location
            location = EarthLocation(lat=self.latitude * u.deg, lon=self.longitude * u.deg, height=0 * u.m)

            # Calculate Local Sidereal Time
            lst = astropy_time.sidereal_time('apparent', self.longitude * u.deg)
            self.lst_calculated.emit(f"Local Sidereal Time: {lst.to_string(unit=u.hour, precision=3)}")

            # Calculate lunar phase
            phase_percentage, phase_image_name = self.calculate_lunar_phase(astropy_time, location)

            # Emit lunar phase data
            self.lunar_phase_calculated.emit(phase_percentage, phase_image_name)

            # Determine the path to celestial_catalog.csv
            catalog_file = os.path.join(
                getattr(sys, '_MEIPASS', os.path.dirname(__file__)), "celestial_catalog.csv"
            )

            # Load celestial catalog from CSV
            if not os.path.exists(catalog_file):
                self.calculation_complete.emit(pd.DataFrame(), "Catalog file not found.")
                return

            df = pd.read_csv(catalog_file, encoding='ISO-8859-1')

            # Apply catalog filters
            df = df[df['Catalog'].isin(self.catalog_filters)]
            df.dropna(subset=['RA', 'Dec'], inplace=True)

            # Check altitude and calculate additional metrics
            altaz_frame = AltAz(obstime=astropy_time, location=location)
            altitudes, azimuths, minutes_to_transit, degrees_from_moon = [], [], [], []
            before_or_after = []

            moon = get_body("moon", astropy_time, location).transform_to(altaz_frame)

            for _, row in df.iterrows():
                sky_coord = SkyCoord(ra=row['RA'] * u.deg, dec=row['Dec'] * u.deg, frame='icrs')
                altaz = sky_coord.transform_to(altaz_frame)
                altitudes.append(round(altaz.alt.deg, 1))
                azimuths.append(round(altaz.az.deg, 1))

                # Calculate time difference to transit
                ra = row['RA'] * u.deg.to(u.hourangle)  # Convert RA from degrees to hour angle
                time_diff = ((ra - lst.hour) * u.hour) % (24 * u.hour)
                minutes = round(time_diff.value * 60, 1)
                if minutes > 720:
                    minutes = 1440 - minutes
                    before_or_after.append("After")
                else:
                    before_or_after.append("Before")
                minutes_to_transit.append(minutes)

                # Calculate angular distance from the moon
                moon_sep = sky_coord.separation(moon).deg
                degrees_from_moon.append(round(moon_sep, 2))

            df['Altitude'] = altitudes
            df['Azimuth'] = azimuths
            df['Minutes to Transit'] = minutes_to_transit
            df['Before/After Transit'] = before_or_after
            df['Degrees from Moon'] = degrees_from_moon

            # Apply altitude filter
            df = df[df['Altitude'] >= self.min_altitude]

            # Sort by "Minutes to Transit"
            df = df.sort_values(by='Minutes to Transit')

            # Limit the results to the object_limit
            df = df.head(self.object_limit)

            self.calculation_complete.emit(df, "Calculation complete.")
        except Exception as e:
            self.calculation_complete.emit(pd.DataFrame(), f"Error: {str(e)}")

    def calculate_lunar_phase(self, astropy_time, location):
        moon = get_body("moon", astropy_time, location)
        sun = get_sun(astropy_time)
        elongation = moon.separation(sun).deg

        # Determine lunar phase percentage
        phase_percentage = (1 - np.cos(np.radians(elongation))) / 2 * 100
        phase_percentage = round(phase_percentage)

        # Determine if it is waxing or waning
        future_time = astropy_time + (6 * u.hour)
        future_moon = get_body("moon", future_time, location)
        future_sun = get_sun(future_time)
        future_elongation = future_moon.separation(future_sun).deg
        is_waxing = future_elongation > elongation

        phase_folder = os.path.join(sys._MEIPASS, "imgs") if getattr(sys, 'frozen', False) else os.path.join(os.path.dirname(__file__), "imgs")


        # Select appropriate lunar phase image based on phase angle
        phase_image_name = "new_moon.png"  # Default

        if 0 <= elongation < 9:
            phase_image_name = "new_moon.png"
        elif 9 <= elongation < 18:
            phase_image_name = "waxing_crescent_1.png" if is_waxing else "waning_crescent_5.png"
        elif 18 <= elongation < 27:
            phase_image_name = "waxing_crescent_2.png" if is_waxing else "waning_crescent_4.png"
        elif 27 <= elongation < 36:
            phase_image_name = "waxing_crescent_3.png" if is_waxing else "waning_crescent_3.png"
        elif 36 <= elongation < 45:
            phase_image_name = "waxing_crescent_4.png" if is_waxing else "waning_crescent_2.png"
        elif 45 <= elongation < 54:
            phase_image_name = "waxing_crescent_5.png" if is_waxing else "waning_crescent_1.png"
        elif 54 <= elongation < 90:
            phase_image_name = "first_quarter.png"
        elif 90 <= elongation < 108:
            phase_image_name = "waxing_gibbous_1.png" if is_waxing else "waning_gibbous_4.png"
        elif 108 <= elongation < 126:
            phase_image_name = "waxing_gibbous_2.png" if is_waxing else "waning_gibbous_3.png"
        elif 126 <= elongation < 144:
            phase_image_name = "waxing_gibbous_3.png" if is_waxing else "waning_gibbous_2.png"
        elif 144 <= elongation < 162:
            phase_image_name = "waxing_gibbous_4.png" if is_waxing else "waning_gibbous_1.png"
        elif 162 <= elongation <= 180:
            phase_image_name = "full_moon.png"


        self.lunar_phase_calculated.emit(phase_percentage, phase_image_name)
        return phase_percentage, phase_image_name



class WhatsInMySky(QWidget):
    def __init__(self):
        super().__init__()
        self.settings_file = os.path.join(os.path.expanduser("~"), "sky_settings.json")
        self.settings = {}  # Initialize empty settings dictionary
        self.initUI()  # Build the UI
        self.load_settings()  # Load settings after UI is built
        self.object_limit = self.settings.get("object_limit", 100)

    def initUI(self):
        layout = QGridLayout()
        fixed_width = 150

        # Latitude, Longitude, Date, Time, Time Zone
        self.latitude_entry, self.longitude_entry, self.date_entry, self.time_entry, self.timezone_combo = self.setup_basic_info_fields(layout, fixed_width)

        # Minimum Altitude, Catalog Filters, RA/Dec format
        self.min_altitude_entry, self.catalog_vars, self.ra_dec_format = self.setup_filters(layout, fixed_width)

        # Calculate Button, Status Label, Sidereal Time, Treeview for Results, Custom Object and Save Buttons
        self.setup_controls(layout, fixed_width)

        self.setLayout(layout)
        self.setMinimumWidth(1000)  # Ensures a wide enough starting window

    def setup_basic_info_fields(self, layout, fixed_width):
        self.latitude_entry = QLineEdit()
        self.latitude_entry.setFixedWidth(fixed_width)
        layout.addWidget(QLabel("Latitude:"), 0, 0)
        layout.addWidget(self.latitude_entry, 0, 1)

        self.longitude_entry = QLineEdit()
        self.longitude_entry.setFixedWidth(fixed_width)
        layout.addWidget(QLabel("Longitude:"), 1, 0)
        layout.addWidget(self.longitude_entry, 1, 1)

        self.date_entry = QLineEdit()
        self.date_entry.setFixedWidth(fixed_width)
        layout.addWidget(QLabel("Date (YYYY-MM-DD):"), 2, 0)
        layout.addWidget(self.date_entry, 2, 1)

        self.time_entry = QLineEdit()
        self.time_entry.setFixedWidth(fixed_width)
        layout.addWidget(QLabel("Time (HH:MM):"), 3, 0)
        layout.addWidget(self.time_entry, 3, 1)

        self.timezone_combo = QComboBox()
        self.timezone_combo.addItems(pytz.all_timezones)
        self.timezone_combo.setFixedWidth(fixed_width)
        layout.addWidget(QLabel("Time Zone:"), 4, 0)
        layout.addWidget(self.timezone_combo, 4, 1)

        return self.latitude_entry, self.longitude_entry, self.date_entry, self.time_entry, self.timezone_combo

    def setup_filters(self, layout, fixed_width):
        self.min_altitude_entry = QLineEdit()
        self.min_altitude_entry.setFixedWidth(fixed_width)
        layout.addWidget(QLabel("Min Altitude (0-90 degrees):"), 5, 0)
        layout.addWidget(self.min_altitude_entry, 5, 1)

        catalog_frame = QScrollArea()
        catalog_widget = QWidget()
        catalog_layout = QGridLayout()
        self.catalog_vars = {}
        for i, catalog in enumerate(["Messier", "NGC", "IC", "Caldwell", "Abell", "Sharpless", "LBN", "LDN", "PNG", "User"]):
            chk = QCheckBox(catalog)
            chk.setChecked(False)
            catalog_layout.addWidget(chk, i // 5, i % 5)
            self.catalog_vars[catalog] = chk
        catalog_widget.setLayout(catalog_layout)
        catalog_frame.setWidget(catalog_widget)
        catalog_frame.setFixedWidth(fixed_width + 250)
        layout.addWidget(QLabel("Catalog Filters:"), 6, 0)
        layout.addWidget(catalog_frame, 6, 1)

        # RA/Dec format setup
        self.ra_dec_degrees = QRadioButton("Degrees")
        self.ra_dec_hms = QRadioButton("H:M:S / D:M:S")
        ra_dec_group = QButtonGroup()
        ra_dec_group.addButton(self.ra_dec_degrees)
        ra_dec_group.addButton(self.ra_dec_hms)
        self.ra_dec_degrees.setChecked(True)  # Default to Degrees format
        ra_dec_layout = QHBoxLayout()
        ra_dec_layout.addWidget(self.ra_dec_degrees)
        ra_dec_layout.addWidget(self.ra_dec_hms)
        layout.addWidget(QLabel("RA/Dec Format:"), 7, 0)
        layout.addLayout(ra_dec_layout, 7, 1)

        # Connect the radio buttons to the update function
        self.ra_dec_degrees.toggled.connect(self.update_ra_dec_format)
        self.ra_dec_hms.toggled.connect(self.update_ra_dec_format)

        return self.min_altitude_entry, self.catalog_vars, self.ra_dec_degrees

    def setup_controls(self, layout, fixed_width):
        # Calculate button
        calculate_button = QPushButton("Calculate")
        calculate_button.setFixedWidth(fixed_width)
        layout.addWidget(calculate_button, 8, 0)
        calculate_button.clicked.connect(self.start_calculation)

        # Status label
        self.status_label = QLabel("Status: Idle")
        layout.addWidget(self.status_label, 9, 0, 1, 2)

        # Sidereal time label
        self.lst_label = QLabel("Local Sidereal Time: {:.3f}".format(0.0))
        layout.addWidget(self.lst_label, 10, 0, 1, 2)

        # Lunar phase image and label
        self.lunar_phase_image_label = QLabel()
        layout.addWidget(self.lunar_phase_image_label, 0, 2, 4, 1)  # Position it appropriately

        self.lunar_phase_label = QLabel("Lunar Phase: N/A")
        layout.addWidget(self.lunar_phase_label, 4, 2)

        # Treeview for results (expand dynamically)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels([
            "Name", "RA", "Dec", "Altitude", "Azimuth", "Minutes to Transit", "Before/After Transit",
            "Degrees from Moon", "Alt Name", "Type", "Magnitude", "Size (arcmin)"
        ])
        self.tree.setSortingEnabled(True)
        self.tree.header().setSectionResizeMode(QHeaderView.Stretch)
        self.tree.sortByColumn(5, Qt.AscendingOrder)
        layout.addWidget(self.tree, 11, 0, 1, 3)
        self.tree.itemDoubleClicked.connect(self.on_row_double_click)

        # Buttons at the bottom
        add_object_button = QPushButton("Add Custom Object")
        add_object_button.setFixedWidth(fixed_width)
        layout.addWidget(add_object_button, 12, 0)
        add_object_button.clicked.connect(self.add_custom_object)

        save_button = QPushButton("Save to CSV")
        save_button.setFixedWidth(fixed_width)
        layout.addWidget(save_button, 12, 1)
        save_button.clicked.connect(self.save_to_csv)

        # Settings button to change the number of objects displayed
        settings_button = QPushButton()
        settings_button.setIcon(QIcon(wrench_path))  # Use icon_path for the button's icon
        settings_button.setFixedWidth(fixed_width)
        layout.addWidget(settings_button, 12, 2)
        settings_button.clicked.connect(self.open_settings)        

        # Allow the main window to expand
        layout.setColumnStretch(2, 1)  # Makes the right column (with tree widget) expand as the window grows


    def start_calculation(self):
        # Gather the inputs
        latitude = float(self.latitude_entry.text())
        longitude = float(self.longitude_entry.text())
        date_str = self.date_entry.text()
        time_str = self.time_entry.text()
        timezone_str = self.timezone_combo.currentText()
        min_altitude = float(self.min_altitude_entry.text())
        catalog_filters = [catalog for catalog, var in self.catalog_vars.items() if var.isChecked()]
        object_limit = self.object_limit

        # Set up and start the calculation thread
        self.calc_thread = CalculationThread(
            latitude, longitude, date_str, time_str, timezone_str,
            min_altitude, catalog_filters, object_limit
        )
        self.calc_thread.calculation_complete.connect(self.on_calculation_complete)
        self.calc_thread.lunar_phase_calculated.connect(self.update_lunar_phase)
        self.calc_thread.lst_calculated.connect(self.update_lst) 
        self.calc_thread.status_update.connect(self.update_status)
        self.update_status("Calculating...")
        self.calc_thread.start()


    def update_lunar_phase(self, phase_percentage, phase_image_name):
        # Update the lunar phase label
        self.lunar_phase_label.setText(f"Lunar Phase: {phase_percentage}% illuminated")

        # Define the path to the image
        phase_folder = os.path.join(sys._MEIPASS, "imgs") if getattr(sys, 'frozen', False) else os.path.join(os.path.dirname(__file__), "imgs")
        phase_image_path = os.path.join(phase_folder, phase_image_name)

        # Load and display the lunar phase image if it exists
        if os.path.exists(phase_image_path):
            pixmap = QPixmap(phase_image_path).scaled(100, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.lunar_phase_image_label.setPixmap(pixmap)
        else:
            print(f"Image not found: {phase_image_path}")     

    def on_calculation_complete(self, df, message):
        # Handle the data received from the calculation thread
        self.update_status(message)
        if not df.empty:
            self.tree.clear()
            for _, row in df.iterrows():
                # Prepare RA and Dec display based on selected format
                ra_display = row['RA']
                dec_display = row['Dec']

                if self.ra_dec_hms.isChecked():
                    # Convert degrees to H:M:S format
                    sky_coord = SkyCoord(ra=row['RA'] * u.deg, dec=row['Dec'] * u.deg)
                    ra_display = sky_coord.ra.to_string(unit=u.hour, sep=':')
                    dec_display = sky_coord.dec.to_string(unit=u.deg, sep=':')

                # Calculate Before/After Transit string
                before_after = row['Before/After Transit']

                # Ensure Size (arcmin) displays correctly as a string
                size_arcmin = row.get('Info', '')
                if pd.notna(size_arcmin):
                    size_arcmin = str(size_arcmin)  # Ensure it's treated as a string

                # Populate each row with the calculated data
                values = [
                    str(row['Name']) if pd.notna(row['Name']) else '',  # Ensure Name is a string or empty
                    str(ra_display),  # RA in either H:M:S or degrees format
                    str(dec_display),  # Dec in either H:M:S or degrees format
                    str(row['Altitude']) if pd.notna(row['Altitude']) else '',  # Altitude as string or empty
                    str(row['Azimuth']) if pd.notna(row['Azimuth']) else '',  # Azimuth as string or empty
                    str(int(row['Minutes to Transit'])) if pd.notna(row['Minutes to Transit']) else '',  # Minutes to Transit as integer string
                    before_after,  # Before/After Transit (already a string)
                    str(round(row['Degrees from Moon'], 2)) if pd.notna(row['Degrees from Moon']) else '',  # Degrees from Moon as rounded string or empty
                    row.get('Alt Name', '') if pd.notna(row.get('Alt Name', '')) else '',  # Alt Name as string or empty
                    row.get('Type', '') if pd.notna(row.get('Type', '')) else '',  # Type as string or empty
                    str(row.get('Magnitude', '')) if pd.notna(row.get('Magnitude', '')) else '',  # Magnitude as string or empty
                    str(size_arcmin) if pd.notna(size_arcmin) else ''  # Size in arcmin as string or empty
                ]

                # Use SortableTreeWidgetItem instead of QTreeWidgetItem
                item = SortableTreeWidgetItem(values)
                self.tree.addTopLevelItem(item)


    def update_status(self, message):
        self.status_label.setText(f"Status: {message}")

    def update_lst(self, message):
        self.lst_label.setText(message)


    def save_settings(self, latitude, longitude, date, time, timezone, min_altitude):
        """Save the user-provided settings to a JSON file."""
        settings = {
            "latitude": latitude,
            "longitude": longitude,
            "date": date,
            "time": time,
            "timezone": timezone,
            "min_altitude": min_altitude,
            "object_limit": self.object_limit
        }
        with open(self.settings_file, 'w') as f:
            json.dump(settings, f)
        print("Settings saved:", settings)

    def load_settings(self):
        """Load settings from the JSON file if it exists."""
        if os.path.exists(self.settings_file):
            with open(self.settings_file, 'r') as f:
                self.settings = json.load(f)
            # Populate fields with loaded settings
            self.latitude_entry.setText(str(self.settings.get("latitude", "")))
            self.longitude_entry.setText(str(self.settings.get("longitude", "")))

            # Use today's date for the date entry instead of the saved date
            today_date = datetime.now().strftime("%Y-%m-%d")
            self.date_entry.setText(today_date)

            self.time_entry.setText(self.settings.get("time", ""))
            self.timezone_combo.setCurrentText(self.settings.get("timezone", "UTC"))
            self.min_altitude_entry.setText(str(self.settings.get("min_altitude", "0")))
            self.object_limit = self.settings.get("object_limit", 100)
        else:
            self.settings = {}
            # Default to today's date
            self.date_entry.setText(datetime.now().strftime("%Y-%m-%d"))


    def open_settings(self):
        object_limit, ok = QInputDialog.getInt(self, "Settings", "Enter number of objects to display:", value=self.object_limit, min=1, max=1000)
        if ok:
            self.object_limit = object_limit

    def treeview_sort_column(self, tv, col, reverse):
        l = [(tv.set(k, col), k) for k in tv.get_children('')]
        try:
            l.sort(key=lambda t: float(t[0]) if t[0] else float('inf'), reverse=reverse)
        except ValueError:
            l.sort(reverse=reverse)

        for index, (val, k) in enumerate(l):
            tv.move(k, '', index)

        tv.heading(col, command=lambda: self.treeview_sort_column(tv, col, not reverse))

    def on_row_double_click(self, item: QTreeWidgetItem, column: int):
        """Handle double-clicking an item in the tree view."""
        object_name = item.text(0).replace(" ", "")  # Assuming the name is in the first column
        search_url = f"https://www.astrobin.com/search/?q={object_name}"
        print(f"Opening URL: {search_url}")  # Debugging output
        webbrowser.open(search_url)

    def add_custom_object(self):
        # Gather information for the custom object
        name, ok_name = QInputDialog.getText(self, "Add Custom Object", "Enter object name:")
        if not ok_name or not name:
            return

        ra, ok_ra = QInputDialog.getDouble(self, "Add Custom Object", "Enter RA (in degrees):", decimals=3)
        if not ok_ra:
            return

        dec, ok_dec = QInputDialog.getDouble(self, "Add Custom Object", "Enter Dec (in degrees):", decimals=3)
        if not ok_dec:
            return

        # Create the custom object entry
        new_object = {
            "Name": name,
            "RA": ra,
            "Dec": dec,
            "Catalog": "User Defined",
            "Alt Name": "User Defined",
            "Type": "Custom",
            "Magnitude": "",
            "Info": ""
        }

        # Load the catalog, add the custom object, and save it back
        df = pd.read_csv(self.calc_thread.catalog_file, encoding='ISO-8859-1')
        df = pd.concat([df, pd.DataFrame([new_object])], ignore_index=True)
        df.to_csv(self.calc_thread.catalog_file, index=False, encoding='ISO-8859-1')
        self.update_status(f"Added custom object: {name}")

    def update_ra_dec_format(self):
        """Update the RA/Dec format in the tree based on the selected radio button."""
        is_degrees_format = self.ra_dec_degrees.isChecked()  # Check if degrees format is selected

        for i in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(i)
            ra_value = item.text(1)  # RA is in the second column
            dec_value = item.text(2)  # Dec is in the third column

            try:
                if is_degrees_format:
                    # Convert H:M:S to degrees only if in H:M:S format
                    if ":" in ra_value:
                        # Conversion from H:M:S format to degrees
                        sky_coord = SkyCoord(ra=ra_value, dec=dec_value, unit=(u.hourangle, u.deg))
                        ra_display = str(round(sky_coord.ra.deg, 3))
                        dec_display = str(round(sky_coord.dec.deg, 3))
                    else:
                        # Already in degrees format; no conversion needed
                        ra_display = ra_value
                        dec_display = dec_value
                else:
                    # Convert degrees to H:M:S only if in degrees format
                    if ":" not in ra_value:
                        # Conversion from degrees to H:M:S format
                        ra_deg = float(ra_value)
                        dec_deg = float(dec_value)
                        sky_coord = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg)
                        ra_display = sky_coord.ra.to_string(unit=u.hour, sep=':')
                        dec_display = sky_coord.dec.to_string(unit=u.deg, sep=':')
                    else:
                        # Already in H:M:S format; no conversion needed
                        ra_display = ra_value
                        dec_display = dec_value

            except ValueError as e:
                print(f"Conversion error: {e}")
                ra_display = ra_value
                dec_display = dec_value
            except Exception as e:
                print(f"Unexpected error: {e}")
                ra_display = ra_value
                dec_display = dec_value

            # Update item with the new RA/Dec display format
            item.setText(1, ra_display)
            item.setText(2, dec_display)



    def save_to_csv(self):
        # Ask user where to save the CSV file
        file_path, _ = QFileDialog.getSaveFileName(self, "Save CSV File", "", "CSV files (*.csv);;All Files (*)")
        if file_path:
            # Extract data from QTreeWidget
            columns = [self.tree.headerItem().text(i) for i in range(self.tree.columnCount())]
            data = [columns]
            for i in range(self.tree.topLevelItemCount()):
                item = self.tree.topLevelItem(i)
                row = [item.text(j) for j in range(self.tree.columnCount())]
                data.append(row)

            # Convert data to DataFrame and save as CSV
            df = pd.DataFrame(data[1:], columns=data[0])
            df.to_csv(file_path, index=False)
            self.update_status(f"Data saved to {file_path}")

class SortableTreeWidgetItem(QTreeWidgetItem):
    def __lt__(self, other):
        # Get the column index being sorted
        column = self.treeWidget().sortColumn()

        # Columns with numeric data for custom sorting (adjust column indices as needed)
        numeric_columns = [3, 4, 5, 7, 10]  # Altitude, Azimuth, Minutes to Transit, Degrees from Moon, Magnitude

        # Check if the column is in numeric_columns for numeric sorting
        if column in numeric_columns:
            try:
                # Attempt to compare as floats
                return float(self.text(column)) < float(other.text(column))
            except ValueError:
                # If conversion fails, fall back to string comparison
                return self.text(column) < other.text(column)
        else:
            # Default string comparison for other columns
            return self.text(column) < other.text(column)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = AstroEditingSuite()
    window.show()
    sys.exit(app.exec_())
