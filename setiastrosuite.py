# Standard library imports
import os
import tempfile
import sys
import time
import json
import logging
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
from scipy.interpolate import Rbf

import rawpy

# PyQt5 imports
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QFileDialog, QGraphicsView, QGraphicsScene, QMessageBox, QInputDialog, QTreeWidget, 
    QTreeWidgetItem, QCheckBox, QDialog, QFormLayout, QSpinBox, QDialogButtonBox, QGridLayout,
    QGraphicsEllipseItem, QGraphicsLineItem, QGraphicsRectItem, QGraphicsPathItem, QDoubleSpinBox,
    QColorDialog, QFontDialog, QStyle, QSlider, QTabWidget, QScrollArea, QSizePolicy, QSpacerItem, QAbstractItemView, QToolBar,QGraphicsPixmapItem,QRubberBand,QVBoxLayout,QGroupBox,
    QGraphicsTextItem, QComboBox, QLineEdit, QRadioButton, QButtonGroup, QHeaderView, QStackedWidget, QSplitter, QMenu, QAction, QMenuBar, QTextEdit, QProgressBar, QGraphicsItem, QToolButton, QStatusBar
)
from PyQt5.QtGui import (
    QPixmap, QImage, QPainter, QPen, QColor, QTransform, QIcon, QPainterPath, QFont, QMovie, QCursor, QBrush, QPolygon
)
from PyQt5.QtCore import Qt, QRectF, QLineF, QPointF, QThread, pyqtSignal, QCoreApplication, QPoint, QTimer, QRect, QFileSystemWatcher, QEvent, pyqtSlot, QProcess, QSize, QObject,QSettings

# Math functions
from math import sqrt



if hasattr(sys, '_MEIPASS'):
    # PyInstaller path
    icon_path = os.path.join(sys._MEIPASS, 'astrosuite.png')
    windowslogo_path = os.path.join(sys._MEIPASS, 'astrosuite.ico')
    green_path = os.path.join(sys._MEIPASS, 'green.png')
    neutral_path = os.path.join(sys._MEIPASS, 'neutral.png')
    whitebalance_path = os.path.join(sys._MEIPASS, 'whitebalance.png')
    morpho_path = os.path.join(sys._MEIPASS, 'morpho.png')
    clahe_path = os.path.join(sys._MEIPASS, 'clahe.png')
    starnet_path = os.path.join(sys._MEIPASS, 'starnet.png')
    staradd_path = os.path.join(sys._MEIPASS, 'staradd.png')
    LExtract_path = os.path.join(sys._MEIPASS, 'LExtract.png')
    LInsert_path = os.path.join(sys._MEIPASS, 'LInsert.png')
    slot0_path = os.path.join(sys._MEIPASS, 'slot0.png')
    slot1_path = os.path.join(sys._MEIPASS, 'slot1.png')
    slot2_path = os.path.join(sys._MEIPASS, 'slot2.png')
    slot3_path = os.path.join(sys._MEIPASS, 'slot3.png')
    slot4_path = os.path.join(sys._MEIPASS, 'slot4.png')
    rgbcombo_path = os.path.join(sys._MEIPASS, 'rgbcombo.png')
    rgbextract_path = os.path.join(sys._MEIPASS, 'rgbextract.png')
    copyslot_path = os.path.join(sys._MEIPASS, 'copyslot.png')
    graxperticon_path = os.path.join(sys._MEIPASS, 'graxpert.png')
    cropicon_path = os.path.join(sys._MEIPASS, 'cropicon.png')
    openfile_path = os.path.join(sys._MEIPASS, 'openfile.png')
    abeicon_path = os.path.join(sys._MEIPASS, 'abeicon.png')    
    undoicon_path = os.path.join(sys._MEIPASS, 'undoicon.png')  
    redoicon_path = os.path.join(sys._MEIPASS, 'redoicon.png')  
else:
    # Development path
    icon_path = 'astrosuite.png'
    windowslogo_path = 'astrosuite.ico'
    green_path = 'green.png'
    neutral_path = 'neutral.png'
    whitebalance_path = 'whitebalance.png'
    morpho_path = 'morpho.png'
    clahe_path = 'clahe.png'
    starnet_path = 'starnet.png'
    staradd_path = 'staradd.png'
    LExtract_path = 'LExtract.png'
    LInsert_path = 'LInsert.png'
    slot1_path = 'slot1.png'
    slot0_path = 'slot0.png'
    slot2_path = 'slot2.png'
    slot3_path  = 'slot3.png'
    slot4_path  = 'slot4.png'
    rgbcombo_path = 'rgbcombo.png'
    rgbextract_path = 'rgbextract.png'
    copyslot_path = 'copyslot.png'
    graxperticon_path = 'graxpert.png'
    cropicon_path = 'cropicon.png'
    openfile_path = 'openfile.png'
    abeicon_path = 'abeicon.png'
    undoicon_path = 'undoicon.png'
    redoicon_path = 'redoicon.png'


class AstroEditingSuite(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowIcon(QIcon(icon_path))
        self.current_theme = "dark"  # Default theme
        self.image_manager = ImageManager(max_slots=5)  # Initialize ImageManager
        self.image_manager.image_changed.connect(self.update_file_name)
        self.settings = QSettings("Seti Astro", "Seti Astro Suite")  # Replace "Seti Astro" with your actual organization name
        self.starnet_exe_path = self.settings.value("starnet/exe_path", type=str)  # Load saved path if available
        self.preview_windows = {}
        print("Initialized preview_windows dictionary.")
        self.initUI()

    def initUI(self):
        # Set the window icon
        self.setWindowIcon(QIcon(icon_path))

        # Enable drag and drop
        self.setAcceptDrops(True)

        # Create a menu bar
        menubar = self.menuBar()  # Use the menu bar directly from QMainWindow

        # --------------------
        # File Menu
        # --------------------
        file_menu = menubar.addMenu("File")
        
        # Create File Menu Actions
        open_action = QAction("Open Image", self)
        open_action.setShortcut('Ctrl+O')
        open_action.setStatusTip('Open an image file')
        open_action.triggered.connect(self.open_image)
        
        save_action = QAction("Save As", self)
        save_action.setShortcut('Ctrl+S')
        save_action.setStatusTip('Save the image to disk')
        save_action.triggered.connect(self.save_image)
        
        undo_action = QAction("Undo", self)
        undo_action.setShortcut('Ctrl+Z')
        undo_action.setStatusTip('Undo the last action')
        undo_action.triggered.connect(self.undo_image)
        
        redo_action = QAction("Redo", self)
        redo_action.setShortcut('Ctrl+Y')
        redo_action.setStatusTip('Redo the last undone action')
        redo_action.triggered.connect(self.redo_image)
        
        exit_action = QAction("Exit", self)
        exit_action.setShortcut('Ctrl+Q')  # Common shortcut for Exit
        exit_action.setStatusTip('Exit the application')
        exit_action.triggered.connect(self.close)  # Close the application

        # Add actions to the File menu
        file_menu.addAction(open_action)
        file_menu.addAction(save_action)
        file_menu.addSeparator()
        file_menu.addAction(undo_action)
        file_menu.addAction(redo_action)
        file_menu.addSeparator()
        file_menu.addAction(exit_action)

        # --------------------
        # Themes Menu
        # --------------------
        theme_menu = menubar.addMenu("Themes")
        light_theme_action = QAction("Light Theme", self)
        dark_theme_action = QAction("Dark Theme", self)

        light_theme_action.triggered.connect(lambda: self.apply_theme("light"))
        dark_theme_action.triggered.connect(lambda: self.apply_theme("dark"))

        theme_menu.addAction(light_theme_action)
        theme_menu.addAction(dark_theme_action)

        # --------------------
        # Functions Menu
        # --------------------
        functions_menu = menubar.addMenu("Functions")

        gradient_removal_icon = QIcon(abeicon_path)  # Replace with the actual path variable
        gradient_removal_action = QAction(gradient_removal_icon, "Remove Gradient with SetiAstro ABE", self)
        gradient_removal_action.setShortcut('Ctrl+Shift+G')  # Assign a keyboard shortcut
        gradient_removal_action.setStatusTip('Remove gradient from the current image')
        gradient_removal_action.triggered.connect(self.remove_gradient)

        # Add the new action to the Functions menu
        functions_menu.addAction(gradient_removal_action)

        remove_gradient_action = QAction(QIcon(graxperticon_path), "Remove Gradient with GraXpert", self)
        remove_gradient_action.triggered.connect(self.remove_gradient_with_graxpert)
        functions_menu.addAction(remove_gradient_action)        
        
        # Add Crop to Functions menu
        crop_action = QAction(QIcon(cropicon_path), "Crop Image", self)
        crop_action.setShortcut('Ctrl+K')
        crop_action.setStatusTip('Crop the current image')
        crop_action.triggered.connect(self.open_crop_tool)
        functions_menu.addAction(crop_action)

        # Create Remove Green QAction
        remove_green_action = QAction("Remove Green", self)
        remove_green_action.setShortcut('Ctrl+G')  # Assign a keyboard shortcut
        remove_green_action.setStatusTip('Remove green noise from the image')
        remove_green_action.triggered.connect(self.open_remove_green_dialog)
        
        # Add Remove Green to Functions menu
        functions_menu.addAction(remove_green_action)

        background_neutralization_action = QAction("Background Neutralization", self)
        background_neutralization_action.setShortcut('Ctrl+N')  # Assign a keyboard shortcut
        background_neutralization_action.setStatusTip('Neutralize background colors based on a sample region')
        background_neutralization_action.triggered.connect(self.open_background_neutralization_dialog)
        
        # Add to Functions menu
        functions_menu.addAction(background_neutralization_action)        

        # White Balance Action
        whitebalance_action = QAction("White Balance", self)
        whitebalance_action.setShortcut('Ctrl+Shift+W')  # Assign a keyboard shortcut
        whitebalance_action.setStatusTip('Adjust white balance of the image')
        whitebalance_action.triggered.connect(self.open_whitebalance_dialog)
        
        # Add White Balance to Functions menu
        functions_menu.addAction(whitebalance_action)   

        # Extract Luminance Action with Icon
        extract_luminance_icon = QIcon(LExtract_path)
        extract_luminance_action = QAction(extract_luminance_icon, "Extract Luminance", self)
        extract_luminance_action.setShortcut('Ctrl+Shift+E')  # Assign a keyboard shortcut
        extract_luminance_action.setStatusTip('Extract luminance from the current image')
        extract_luminance_action.triggered.connect(self.extract_luminance)

        # Add Extract Luminance to Functions menu
        functions_menu.addAction(extract_luminance_action)

        # Recombine Luminance Action with Icon
        recombine_luminance_icon = QIcon(LInsert_path)
        recombine_luminance_action = QAction(recombine_luminance_icon, "Recombine Luminance", self)
        recombine_luminance_action.setShortcut('Ctrl+Shift+R')  # Assign a keyboard shortcut
        recombine_luminance_action.setStatusTip('Recombine luminance into the RGB image in slot 1')
        recombine_luminance_action.triggered.connect(self.recombine_luminance)

        # Add Recombine Luminance to Functions menu
        functions_menu.addAction(recombine_luminance_action)

        # RGB Combination Action
        rgb_combination_icon = QIcon(rgbcombo_path)
        rgb_combination_action = QAction(rgb_combination_icon, "RGB Combination", self)
        rgb_combination_action.setShortcut('Ctrl+Shift+C')  # Assign a keyboard shortcut
        rgb_combination_action.setStatusTip('Combine separate R, G, B images into an RGB image')
        rgb_combination_action.triggered.connect(self.rgb_combination)
        # Add RGB Combination to Functions menu
        functions_menu.addAction(rgb_combination_action)
        
        # RGB Extract Action
        rgb_extract_icon = QIcon(rgbextract_path)
        rgb_extract_action = QAction(rgb_extract_icon, "RGB Extract", self)
        rgb_extract_action.setShortcut('Ctrl+Shift+X')  # Assign a keyboard shortcut
        rgb_extract_action.setStatusTip('Extract R, G, B channels from an RGB image')
        rgb_extract_action.triggered.connect(self.rgb_extract)
        # Add RGB Extract to Functions menu
        functions_menu.addAction(rgb_extract_action)

        clahe_action = QAction("CLAHE", self)
        clahe_action.setShortcut('Ctrl+Shift+C')  # Assign a keyboard shortcut
        clahe_action.setStatusTip('Apply Contrast Limited Adaptive Histogram Equalization')
        clahe_action.triggered.connect(self.open_clahe_dialog)
        
        # Add CLAHE to Functions menu
        functions_menu.addAction(clahe_action)


        # Morphological Operations Action
        morpho_action = QAction("Morphological Operations", self)
        morpho_action.setShortcut('Ctrl+Shift+M')  # Assign a keyboard shortcut
        morpho_action.setStatusTip('Apply morphological operations to the image')
        morpho_action.triggered.connect(self.open_morpho_dialog)
        
        # Add Morphological Operations to Functions menu
        functions_menu.addAction(morpho_action)        

        remove_stars_action = QAction("Remove Stars", self)
        remove_stars_action.setShortcut('Ctrl+R')  # Assign a keyboard shortcut
        remove_stars_action.setStatusTip('Remove stars from the image using StarNet')
        remove_stars_action.triggered.connect(self.remove_stars)

        # Add Remove Stars to Functions menu
        functions_menu.addAction(remove_stars_action)

        add_stars_action = QAction("Add Stars", self)
        add_stars_action.setShortcut('Ctrl+A')  # Assign a keyboard shortcut
        add_stars_action.setStatusTip('Add stars back to the current image')
        add_stars_action.triggered.connect(self.add_stars)

        # Add Add Stars to Functions menu
        functions_menu.addAction(add_stars_action)        

        # --------------------
        # Slot Menu
        # --------------------
        slot_menu = menubar.addMenu("Slots")

        # Define the number of slots based on ImageManager
        num_slots = self.image_manager.max_slots

        for slot in range(num_slots):
            # Dynamically get the slot icon path
            slot_icon_path = getattr(sys.modules[__name__], f'slot{slot}_path', 'slot0.png')  # Default to slot0.png if not found
            slot_icon = QIcon(slot_icon_path)

            # Create a QAction for each slot
            slot_action = QAction(slot_icon, f"Slot {slot}", self)
            slot_action.setStatusTip(f"Open preview for Slot {slot}")

            # Connect the action to a method with the slot number as an argument
            slot_action.triggered.connect(lambda checked, s=slot: self.open_preview_window(s))

            # Add the action to the Slot Menu
            slot_menu.addAction(slot_action)

        # --------------------
        # Toolbar
        # --------------------
        filebar = QToolBar("File Toolbar")
        self.addToolBar(filebar)

        # Add Open File icon and action
        open_icon = QIcon(openfile_path)  # Replace with the actual path to your "Open File" icon
        open_action = QAction(open_icon, "Open File", self)
        open_action.setStatusTip("Open an image file")
        open_action.triggered.connect(self.open_image)  # Connect to the existing open_image method
        filebar.addAction(open_action)

        # Add Save As disk icon and action
        save_as_icon = QIcon(disk_icon_path)  # Replace with the actual path to your "Save As" icon
        save_as_action = QAction(save_as_icon, "Save As", self)
        save_as_action.setStatusTip("Save the current image")
        save_as_action.triggered.connect(self.save_image)  # Connect to the existing save_image method
        filebar.addAction(save_as_action)

        # Add Undo icon and action
        undo_icon = QIcon(undoicon_path)  # Replace with the actual path to your Undo icon
        undo_action_toolbar = QAction(undo_icon, "Undo", self)
        undo_action_toolbar.setStatusTip("Undo the last action")
        undo_action_toolbar.triggered.connect(self.undo_image)
        filebar.addAction(undo_action_toolbar)

        # Add Redo icon and action
        redo_icon = QIcon(redoicon_path)  # Replace with the actual path to your Redo icon
        redo_action_toolbar = QAction(redo_icon, "Redo", self)
        redo_action_toolbar.setStatusTip("Redo the last undone action")
        redo_action_toolbar.triggered.connect(self.redo_image)
        filebar.addAction(redo_action_toolbar)


        toolbar = QToolBar("Main Toolbar")
        self.addToolBar(toolbar)

        # Add "Copy Slot" Button to Toolbar with Icon
        copy_slot_icon = QIcon(copyslot_path)  # Ensure 'copyslot.png' is the correct path
        copy_slot_action = QAction(copy_slot_icon, "Copy Slot", self)
        copy_slot_action.setStatusTip("Copy the current image in Slot 0 to another slot")
        copy_slot_action.triggered.connect(self.copy_slot0_to_target)
        toolbar.addAction(copy_slot_action)

        crop_icon = QIcon(cropicon_path)
        crop_action.setIcon(crop_icon)
        toolbar.addAction(crop_action)

        toolbar.addAction(gradient_removal_action)

        remove_gradient_icon = QIcon(graxperticon_path)
        remove_gradient_action.setIcon(remove_gradient_icon)
        remove_gradient_action.setStatusTip("Remove Gradient with GraXpert AI")
        toolbar.addAction(remove_gradient_action)

        # Add Remove Stars Button to Toolbar with Icon
        remove_stars_icon = QIcon(starnet_path)
        remove_stars_action.setIcon(remove_stars_icon)  # Set the icon to the QAction
        remove_stars_action.setToolTip("Remove Stars using StarNet")
        toolbar.addAction(remove_stars_action)  # Add the same QAction to the toolbar

        # Add Add Stars Button to Toolbar with Icon
        add_stars_icon = QIcon(staradd_path)
        add_stars_action.setIcon(add_stars_icon)  # Set the icon to the QAction
        add_stars_action.setToolTip("Add Stars back to the image")
        toolbar.addAction(add_stars_action)  # Add the same QAction to the toolbar

        # Add "Remove Green" Button to Toolbar with Icon
        remove_green_icon = QIcon(green_path)
        remove_green_action.setIcon(remove_green_icon)  # Set the icon to the QAction
        toolbar.addAction(remove_green_action)  # Add the same QAction to the toolbar

        # Add "Background Neutralization" Button to Toolbar with Icon
        background_neutralization_icon = QIcon(neutral_path)
        background_neutralization_action.setIcon(background_neutralization_icon)  # Set the icon
        background_neutralization_action.setToolTip("Neutralize background colors based on a sample region.")
        toolbar.addAction(background_neutralization_action)  # Add the QAction to the toolbar

        # Add White Balance Button to Toolbar with Icon
        whitebalance_icon = QIcon(whitebalance_path)
        whitebalance_action.setIcon(whitebalance_icon)
        whitebalance_action.setToolTip("Adjust white balance of the image.")
        toolbar.addAction(whitebalance_action)

        extract_luminance_icon = QIcon(LExtract_path)
        extract_luminance_action = QAction(extract_luminance_icon, "Extract Luminance", self)
        extract_luminance_action.triggered.connect(self.extract_luminance)
        toolbar.addAction(extract_luminance_action)

        recombine_luminance_icon = QIcon(LInsert_path)
        recombine_luminance_action = QAction(recombine_luminance_icon, "Recombine Luminance", self)
        recombine_luminance_action.triggered.connect(self.recombine_luminance)
        toolbar.addAction(recombine_luminance_action)

        # Add RGB Combination Button to Toolbar
        toolbar.addAction(rgb_combination_action)

        # Add RGB Extract Button to Toolbar
        toolbar.addAction(rgb_extract_action)

        # Add CLAHE Button to Toolbar with Icon
        clahe_icon = QIcon(clahe_path)
        clahe_action.setIcon(clahe_icon)
        clahe_action.setToolTip("Apply Contrast Limited Adaptive Histogram Equalization.")
        toolbar.addAction(clahe_action)      

        # Add Morphological Operations Button to Toolbar with Icon
        morpho_icon = QIcon(morpho_path)
        morpho_action.setIcon(morpho_icon)
        morpho_action.setToolTip("Apply morphological operations to the image.")
        toolbar.addAction(morpho_action)    



        # --------------------
        # Status Bar
        # --------------------
        self.statusBar = QStatusBar(self)
        self.setStatusBar(self.statusBar)

        # Create the file label to display the current file name in the status bar
        self.file_name_label = QLabel("No file selected")
        self.statusBar.addWidget(self.file_name_label, 1)  # Add label to status bar

        # Dimension label in the status bar
        self.dim_label = QLabel("0 x 0")
        self.statusBar.addWidget(self.dim_label)

        # --------------------
        # Tab Widget
        # --------------------
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

        # --------------------
        # Quick Navigation Menu
        # --------------------
        quicknav_menu = menubar.addMenu("Quick Navigation")
        for i in range(self.tabs.count()):
            tab_title = self.tabs.tabText(i)
            action = QAction(tab_title, self)
            # Use lambda with default argument to capture the current value of i
            action.triggered.connect(lambda checked, index=i: self.tabs.setCurrentIndex(index))
            quicknav_menu.addAction(action)

        # --------------------
        # Preferences Menu
        # --------------------
        preferences_menu = menubar.addMenu("Preferences")
        preferences_action = QAction("Open Preferences", self)
        preferences_action.setStatusTip('Modify application settings')
        preferences_action.triggered.connect(self.open_preferences_dialog)
        preferences_menu.addAction(preferences_action)


        # --------------------
        # Apply Default Theme
        # --------------------
        self.apply_theme(self.current_theme)

        # --------------------
        # Window Properties
        # --------------------
        self.setWindowTitle('Seti Astro\'s Suite V2.6.3')
        self.setGeometry(100, 100, 1000, 700)  # Set window size as needed

    def remove_gradient(self):
        """Handle the Remove Gradient action."""
        if self.image_manager.image is None:
            QMessageBox.warning(self, "No Image", "Please load an image before removing the gradient.")
            return

        # Initialize the GradientRemovalDialog with the current image
        gradient_dialog = GradientRemovalDialog(image=self.image_manager.image.copy(), parent=self)
        gradient_dialog.processing_completed.connect(self.handle_gradient_removal)
        gradient_dialog.exec_()


    def handle_gradient_removal(self, corrected_image, gradient_background):
        """
        Handle the processed image after gradient removal.

        Args:
            corrected_image (np.ndarray): The image after gradient removal.
            gradient_background (np.ndarray): The gradient background that was removed.
        """
        try:
            # Update the image in ImageManager for the current slot
            current_slot = self.image_manager.current_slot
            metadata = self.image_manager._metadata.get(current_slot, {}).copy()
            metadata['description'] = "Gradient removed"
            metadata['gradient_background'] = gradient_background  # Store gradient background

            # Call set_image with corrected_image and metadata only
            self.image_manager.set_image(new_image=corrected_image, metadata=metadata)

            # Assign gradient_background to Slot 1 directly
            slot_1 = 1  # Slot 1 is typically reserved
            metadata_slot1 = {
                'file_path': "Gradient Background",
                'description': "Gradient background extracted",
                'bit_depth': "32-bit floating point",
                'is_mono': len(gradient_background.shape) < 3,
                'gradient_background': gradient_background
            }

            # Directly assign to Slot 1
            self.image_manager._images[slot_1] = gradient_background
            self.image_manager._metadata[slot_1] = metadata_slot1

            # Notify the user
            QMessageBox.information(self, "Success", "Gradient removal completed successfully.")
            print(f"Gradient removal completed and image updated in Slot {current_slot}.")
            print(f"Gradient background stored in Slot {slot_1}.")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to apply gradient removal:\n{e}")
            print(f"Error in handle_gradient_removal: {e}")





    def open_preferences_dialog(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Preferences")
        layout = QVBoxLayout(dialog)
        
        # Display stored settings
        settings_form = QFormLayout()
        
        # Add settings fields dynamically
        settings_fields = {
            "GraXpert Path": ("graxpert/path", self.settings.value("graxpert/path", "")),
            "StarNet Executable Path": ("starnet/exe_path", self.settings.value("starnet/exe_path", "")),
            "Cosmic Clarity Folder": ("cosmic_clarity_folder", self.settings.value("cosmic_clarity_folder", ""))
        }
        
        # Create fields for each setting with folder selection icons
        input_fields = {}
        for label, (key, value) in settings_fields.items():
            field_widget = QWidget()
            field_layout = QHBoxLayout(field_widget)
            field_layout.setContentsMargins(0, 0, 0, 0)
            
            # Text field
            field = QLineEdit(value)
            input_fields[key] = field
            field_layout.addWidget(field)
            
            # Selection button
            select_button = QPushButton("...")
            select_button.setFixedWidth(30)
            if label == "StarNet Executable Path":
                select_button.setToolTip(f"Select file for {label}")
                select_button.clicked.connect(lambda _, f=field: self.select_file(f))  # File selection for StarNet
            else:
                select_button.setToolTip(f"Select folder for {label}")
                select_button.clicked.connect(lambda _, f=field: self.select_folder(f))  # Folder selection for others
            field_layout.addWidget(select_button)
            
            settings_form.addRow(label, field_widget)
        
        # Add the other fields without folder selection (e.g., Astrometry API Key, Latitude, etc.)
        additional_fields = {
            "Astrometry API Key": ("astrometry_api_key", self.settings.value("astrometry_api_key", "")),
            "Latitude": ("latitude", self.settings.value("latitude", "")),
            "Longitude": ("longitude", self.settings.value("longitude", "")),
            "Date": ("date", self.settings.value("date", "")),
            "Time": ("time", self.settings.value("time", "")),
            "Timezone": ("timezone", self.settings.value("timezone", "")),
            "Minimum Altitude": ("min_altitude", self.settings.value("min_altitude", ""))
        }
        for label, (key, value) in additional_fields.items():
            field = QLineEdit(value)
            settings_form.addRow(label, field)
            input_fields[key] = field
        
        layout.addLayout(settings_form)
        
        # Add Clear and Save buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Reset | QDialogButtonBox.Cancel)
        
        # Save button logic
        buttons.accepted.connect(lambda: self.save_preferences(input_fields, dialog))
        
        # Clear button logic
        buttons.button(QDialogButtonBox.Reset).clicked.connect(lambda: self.clear_preferences(input_fields))
        
        # Close dialog on cancel
        buttons.rejected.connect(dialog.reject)
        
        layout.addWidget(buttons)
        dialog.exec()

    def select_file(self, field):
        file_path = QFileDialog.getOpenFileName(self, "Select File", "", "Executables (*.exe);;All Files (*)")[0]
        if file_path:
            field.setText(file_path)

    def select_folder(self, field):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder_path:
            field.setText(folder_path)

    def save_preferences(self, input_fields, dialog):
        for key, field in input_fields.items():
            self.settings.setValue(key, field.text())
        dialog.accept()
        QMessageBox.information(self, "Preferences Saved", "Settings have been updated successfully.")

    def clear_preferences(self, input_fields):
        reply = QMessageBox.question(self, "Clear Preferences", "Are you sure you want to clear all preferences?", QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            for key in input_fields.keys():
                self.settings.remove(key)
                input_fields[key].clear()
            QMessageBox.information(self, "Preferences Cleared", "All settings have been reset.")



    def open_crop_tool(self):
        """Open the crop tool to crop the current image."""
        if self.image_manager.image is None:
            QMessageBox.warning(self, "No Image", "Please load an image before cropping.")
            return

        # Open the Crop Tool
        crop_tool = CropTool(self.image_manager.image, self)
        crop_tool.crop_applied.connect(self.apply_cropped_image)
        crop_tool.exec_()

    def apply_cropped_image(self, cropped_image):
        """Apply the cropped image to the current slot."""
        # Update the current slot with the cropped image
        current_slot = self.image_manager.current_slot
        metadata = self.image_manager._metadata.get(current_slot, {}).copy()
        metadata['file_path'] = "Cropped Image"

        # Save current state to undo stack
        self.image_manager._undo_stacks[current_slot].append(
            (self.image_manager._images[current_slot].copy(), metadata.copy())
        )
        print(f"ImageManager: Current state of Slot {current_slot} pushed to undo stack.")

        # Update with the cropped image
        self.image_manager._images[current_slot] = cropped_image
        self.image_manager._metadata[current_slot] = metadata

        # Emit signal to update UI
        self.image_manager.image_changed.emit(current_slot, cropped_image, metadata)
        QMessageBox.information(self, "Success", "Cropped image applied.")


    def rgb_combination(self):
        """Handle the RGB Combination action."""
        dialog = RGBCombinationDialog(self, image_manager=self.image_manager)
        if dialog.exec_() == QDialog.Accepted:
            combined_rgb = dialog.rgb_image  # Numpy array with shape (H, W, 3) normalized to [0,1]
            metadata = {
                'file_path': "RGB Combination",
                'is_mono': False,
                'bit_depth': "32-bit floating point",
                'original_header': None  # Add header information if available
            }
            # Store the combined RGB image in Slot 0
            self.image_manager._images[0] = combined_rgb
            self.image_manager._metadata[0] = metadata
            self.image_manager.image_changed.emit(0, combined_rgb, metadata)
            print("RGB image stored in Slot 0.")
            QMessageBox.information(self, "Success", "RGB image combined and stored in Slot 0.")
        else:
            print("RGB Combination cancelled by the user.")

    def rgb_extract(self):
        """Handle the RGB Extract action."""
        # Determine which slot to extract from
        # For this example, we'll extract from Slot 0
        slot_to_extract = 0
        image = self.image_manager._images.get(slot_to_extract, None)
        
        if image is None:
            QMessageBox.warning(self, "No Image", f"Slot {slot_to_extract} does not contain an image to extract from.")
            print(f"Slot {slot_to_extract} is empty. Cannot perform RGB Extract.")
            return
        
        if image.ndim != 3 or image.shape[2] != 3:
            QMessageBox.warning(self, "Invalid Image", "The selected image is not a valid RGB image.")
            print("Invalid image format for RGB Extract. Expected a 3-channel RGB image.")
            return
        
        try:
            # Split the RGB channels
            r_channel = image[..., 0].copy()
            g_channel = image[..., 1].copy()
            b_channel = image[..., 2].copy()
            
            # Define metadata for each channel
            metadata_r = {
                'file_path': f"RGB Extract - Red Channel from Slot {slot_to_extract}",
                'is_mono': True,
                'bit_depth': "32-bit floating point",
                'original_header': None
            }
            metadata_g = {
                'file_path': f"RGB Extract - Green Channel from Slot {slot_to_extract}",
                'is_mono': True,
                'bit_depth': "32-bit floating point",
                'original_header': None
            }
            metadata_b = {
                'file_path': f"RGB Extract - Blue Channel from Slot {slot_to_extract}",
                'is_mono': True,
                'bit_depth': "32-bit floating point",
                'original_header': None
            }
            
            # Store each channel in Slot 2, 3, and 4
            self.image_manager._images[2] = r_channel
            self.image_manager._images[3] = g_channel
            self.image_manager._images[4] = b_channel
            self.image_manager._metadata[2] = metadata_r
            self.image_manager._metadata[3] = metadata_g
            self.image_manager._metadata[4] = metadata_b
            
            
            print(f"Extracted R, G, B channels from Slot {slot_to_extract} and stored in Slots 2, 3, 4 respectively.")
            QMessageBox.information(self, "Success", "RGB channels extracted and stored in Slots 2, 3, and 4.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to extract RGB channels: {e}")
            print(f"Error during RGB Extract: {e}")


    def extract_luminance(self):
        """Extracts the luminance from the current image and updates slots."""
        if self.image_manager.image is None:
            QMessageBox.warning(self, "No Image", "Please load an image before extracting luminance.")
            return

        # Ensure the image is RGB
        current_image = self.image_manager.image
        if current_image.ndim != 3 or current_image.shape[2] != 3:
            QMessageBox.warning(self, "Invalid Image", "Luminance extraction requires an RGB image.")
            return

        # Clip the current image to [0, 1] to avoid any unexpected values outside the valid range
        current_image = np.clip(current_image, 0.0, 1.0)

        # Convert the RGB image to Lab to extract L* (luminance)
        lab_image = self.rgb_to_lab(current_image)
        luminance = lab_image[..., 0] / 100.0  # Normalize L* to [0, 1] for storage

        # Update slot 1 with the original RGB image (do not change current_slot)
        self.image_manager._images[1] = current_image
        self.image_manager._metadata[1] = self.image_manager._metadata[self.image_manager.current_slot].copy()
        print("Original RGB image moved to slot 1.")

        # Update slot 0 with the luminance image
        luminance_metadata = {
            'file_path': "Luminance Extracted",
            'is_mono': True,
            'bit_depth': "32-bit floating point",
        }
        self.image_manager._images[0] = luminance
        self.image_manager._metadata[0] = luminance_metadata
        print("Luminance image updated in slot 0.")

        # Emit signals for both slots to refresh views if necessary
        self.image_manager.image_changed.emit(0, luminance, luminance_metadata)
        self.image_manager.image_changed.emit(1, current_image, self.image_manager._metadata[1])

        # Open a preview for the original RGB image in slot 1
        self.open_preview_window(slot=1)

    def remove_gradient_with_graxpert(self):
        """Integrate GraXpert for gradient removal."""
        if self.image_manager.image is None:
            QMessageBox.warning(self, "No Image", "Please load an image before removing the gradient.")
            return

        # Prompt user for smoothing value
        smoothing, ok = QInputDialog.getDouble(
            self,
            "GraXpert Smoothing Amount",
            "Enter smoothing amount (0.0 to 1.0):",
            decimals=2,
            min=0.0,
            max=1.0,
            value=0.1
        )
        if not ok:
            return  # User cancelled

        # Save the current image as a TIFF file
        input_basename = "input_image"
        input_path = os.path.join(os.getcwd(), f"{input_basename}.tif")
        save_image(self.image_manager.image, input_path, "tiff", "16-bit", None, is_mono=False)

        # Output will have the same base name with `_GraXpert` suffix
        output_basename = f"{input_basename}_GraXpert"
        output_directory = os.getcwd()

        # Determine the platform-specific GraXpert command
        current_os = platform.system()
        if current_os == "Windows":
            graxpert_cmd = "GraXpert.exe"
        elif current_os == "Darwin":  # macOS
            graxpert_cmd = "/Applications/GraXpert.app/Contents/MacOS/GraXpert"
        elif current_os == "Linux":
            graxpert_cmd = self.get_graxpert_path()
            if not graxpert_cmd:
                return  # User cancelled
        else:
            QMessageBox.critical(self, "Unsupported OS", f"Unsupported operating system: {current_os}")
            return

        # Build the command
        command = [
            graxpert_cmd,
            "-cmd", "background-extraction",
            input_path,
            "-cli",
            "-smoothing", str(smoothing),
            "-gpu", "true"
        ]

        # Run the command
        self.run_graxpert_command(command, output_basename, output_directory)

    def get_graxpert_path(self):
        """Prompt user to select the GraXpert path on Linux and save it."""
        graxpert_path = self.settings.value("graxpert/path", type=str)

        if not graxpert_path or not os.path.exists(graxpert_path):
            QMessageBox.information(self, "GraXpert Path", "Please select the GraXpert executable.")
            options = QFileDialog.Options()
            options |= QFileDialog.ReadOnly
            graxpert_path, _ = QFileDialog.getOpenFileName(
                self,
                "Select GraXpert Executable",
                "",
                "Executable Files (*)",
                options=options
            )
            if not graxpert_path:
                QMessageBox.warning(self, "Cancelled", "GraXpert path selection was cancelled.")
                return None  # User cancelled
            if not os.access(graxpert_path, os.X_OK):
                try:
                    os.chmod(graxpert_path, 0o755)  # Add execute permissions
                except Exception as e:
                    QMessageBox.critical(self, "Permission Error", f"Failed to set execute permissions:\n{e}")
                    return None

            # Save the path for future use
            self.settings.setValue("graxpert/path", graxpert_path)

        return graxpert_path



    def run_graxpert_command(self, command, output_basename, output_directory):
        """Execute GraXpert asynchronously."""
        dialog = QDialog(self)
        dialog.setWindowTitle("GraXpert Progress")
        dialog.setMinimumSize(600, 400)
        layout = QVBoxLayout(dialog)
        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        layout.addWidget(text_edit)
        cancel_button = QPushButton("Cancel")
        layout.addWidget(cancel_button)

        thread = GraXpertThread(command)
        thread.stdout_signal.connect(text_edit.append)
        thread.stderr_signal.connect(text_edit.append)
        thread.finished_signal.connect(lambda code: self.on_graxpert_finished(code, output_basename, output_directory, dialog))
        cancel_button.clicked.connect(thread.terminate)

        thread.start()
        dialog.exec_()

    def on_graxpert_finished(self, return_code, output_basename, output_directory, dialog):
        """Handle GraXpert process completion."""
        dialog.close()
        if return_code != 0:
            QMessageBox.critical(self, "Error", "GraXpert process failed.")
            return

        # Locate the output file with any extension
        output_file = None
        for ext in ["fits", "tif", "tiff", "png"]:
            candidate = os.path.join(output_directory, f"{output_basename}.{ext}")
            if os.path.exists(candidate):
                output_file = candidate
                break

        if not output_file:
            QMessageBox.critical(self, "Error", "GraXpert output file not found.")
            return

        # Load the processed image back
        processed_image, _, _, _ = load_image(output_file)

        # Check the number of dimensions to determine if the image is mono
        if processed_image.ndim == 2:
            print("GraXpert output is a mono image. Converting to RGB...")
            processed_image = np.stack([processed_image] * 3, axis=-1)

        # Set the processed image in the image manager
        self.image_manager.set_image(
            processed_image,
            {'file_path': output_file, 'description': "GraXpert Gradient Removed"}
        )

        QMessageBox.information(self, "Success", "Gradient removed successfully.")




    def recombine_luminance(self):
        """Recombines luminance from slot 0 with the RGB image in slot 1."""
        # Ensure slot 1 has the original RGB image
        original_rgb = self.image_manager._images[1]
        if original_rgb is None:
            QMessageBox.warning(self, "No Image", "Slot 1 does not contain an RGB image for recombination.")
            return
        if original_rgb.ndim != 3 or original_rgb.shape[2] != 3:
            QMessageBox.warning(self, "Invalid Image", "Slot 1 must contain an RGB image for recombination.")
            return

        # Ensure slot 0 has the luminance image
        luminance = self.image_manager._images[0]
        if luminance is None or luminance.ndim != 2:
            QMessageBox.warning(self, "No Luminance", "Slot 0 must contain a luminance image for recombination.")
            return

        # Clip luminance to [0, 1] to ensure valid data
        luminance = np.clip(luminance, 0.0, 1.0)

        # Convert the RGB image to Lab color space
        lab_image = self.rgb_to_lab(original_rgb)

        # Replace the L* channel with the luminance
        lab_image[..., 0] = luminance * 100.0  # L* is scaled to [0, 100] in Lab

        # Convert the modified Lab image back to RGB color space
        updated_rgb = self.lab_to_rgb(lab_image)

        # Clip to [0, 1] to ensure valid RGB values
        updated_rgb = np.clip(updated_rgb, 0.0, 1.0)

        # Update slot 0 with the recombined image
        metadata = self.image_manager._metadata[1]
        metadata['file_path'] = "Luminance Recombined"
        self.image_manager.set_image(updated_rgb, metadata)
        print("Recombined image updated in slot 0.")


    def rgb_to_lab(self, rgb_image):
        """Convert a 32-bit floating-point RGB image to Lab color space."""
        # Transformation matrix for RGB to XYZ (D65 reference white)
        M = np.array([
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041]
        ], dtype=np.float32)

        # Convert RGB to linear RGB (no gamma correction needed for 32-bit normalized data)
        rgb_image = np.clip(rgb_image, 0.0, 1.0)

        # Convert RGB to XYZ
        xyz_image = np.dot(rgb_image.reshape(-1, 3), M.T).reshape(rgb_image.shape)
        xyz_image[..., 0] /= 0.95047  # Normalize by D65 reference white
        xyz_image[..., 2] /= 1.08883

        # Convert XYZ to Lab
        def f(t):
            delta = 6 / 29
            return np.where(t > delta**3, np.cbrt(t), (t / (3 * delta**2)) + (4 / 29))

        fx = f(xyz_image[..., 0])
        fy = f(xyz_image[..., 1])
        fz = f(xyz_image[..., 2])

        L = (116.0 * fy) - 16.0
        a = 500.0 * (fx - fy)
        b = 200.0 * (fy - fz)

        return np.stack([L, a, b], axis=-1)


    def lab_to_rgb(self, lab_image):
        """Convert a 32-bit floating-point Lab image to RGB color space."""
        # Transformation matrix for XYZ to RGB (D65 reference white)
        M_inv = np.array([
            [3.2404542, -1.5371385, -0.4985314],
            [-0.9692660,  1.8760108,  0.0415560],
            [0.0556434, -0.2040259,  1.0572252]
        ], dtype=np.float32)

        # Convert Lab to XYZ
        fy = (lab_image[..., 0] + 16.0) / 116.0
        fx = fy + lab_image[..., 1] / 500.0
        fz = fy - lab_image[..., 2] / 200.0

        def f_inv(t):
            delta = 6 / 29
            return np.where(t > delta, t**3, 3 * delta**2 * (t - 4 / 29))

        X = 0.95047 * f_inv(fx)
        Y = f_inv(fy)
        Z = 1.08883 * f_inv(fz)

        xyz_image = np.stack([X, Y, Z], axis=-1)

        # Convert XYZ to RGB
        rgb_image = np.dot(xyz_image.reshape(-1, 3), M_inv.T).reshape(xyz_image.shape)

        # Clip RGB to [0, 1] to maintain valid color ranges
        rgb_image = np.clip(rgb_image, 0.0, 1.0)

        return rgb_image

    def swap_slots(self, slot_a, slot_b):
        """
        Swap images and metadata between two slots.
        
        :param slot_a: The first slot number.
        :param slot_b: The second slot number.
        """
        try:
            # Retrieve images and metadata from both slots
            image_a = self.image_manager._images.get(slot_a, None)
            metadata_a = self.image_manager._metadata.get(slot_a, {}).copy()
            
            image_b = self.image_manager._images.get(slot_b, None)
            metadata_b = self.image_manager._metadata.get(slot_b, {}).copy()
            
            # Swap the images and metadata
            self.image_manager._images[slot_a] = image_b
            self.image_manager._metadata[slot_a] = metadata_b
            
            self.image_manager._images[slot_b] = image_a
            self.image_manager._metadata[slot_b] = metadata_a
            
            # Emit image_changed signals for both slots
            self.image_manager.image_changed.emit(slot_a, image_b, metadata_b)
            self.image_manager.image_changed.emit(slot_b, image_a, metadata_a)
            
            print(f"Swapped images between Slot {slot_a} and Slot {slot_b}.")
            QMessageBox.information(self, "Success", f"Swapped images between Slot {slot_a} and Slot {slot_b}.")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to swap images between Slot {slot_a} and Slot {slot_b}: {e}")
            print(f"Error during swapping slots {slot_a} and {slot_b}: {e}")

    def copy_slot0_to_target(self):
        """Copy the image from Slot 0 to a user-defined slot."""
        # Define available target slots (Slot 1 to Slot 4)
        available_slots = [f"Slot {i}" for i in range(1, self.image_manager.max_slots)]

        # Open the CopySlotDialog
        dialog = CopySlotDialog(self, available_slots)
        result = dialog.exec_()

        if result == QDialog.Accepted:
            target_slot_str = dialog.get_selected_slot()
            target_slot_num = int(target_slot_str.split()[-1])  # Extract slot number
            print(f"User selected to copy to {target_slot_str}.")

            # Check if the target slot already has an image
            if self.image_manager._images[target_slot_num] is not None:
                overwrite = QMessageBox.question(
                    self,
                    "Overwrite Confirmation",
                    f"{target_slot_str} already contains an image. Do you want to overwrite it?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                )
                if overwrite != QMessageBox.Yes:
                    QMessageBox.information(self, "Operation Cancelled", "Copy operation cancelled.")
                    print("User chose not to overwrite the target slot.")
                    return

            # Perform the copy operation
            source_image = self.image_manager._images.get(0, None)
            source_metadata = self.image_manager._metadata.get(0, {}).copy()

            if source_image is None:
                QMessageBox.warning(self, "No Image in Slot 0", "There is no image in Slot 0 to copy.")
                print("Slot 0 is empty. Cannot perform copy.")
                return

            try:
                # Save current state of the target slot to undo stack
                if self.image_manager._images[target_slot_num] is not None:
                    self.image_manager._undo_stacks[target_slot_num].append(
                        (self.image_manager._images[target_slot_num].copy(),
                        self.image_manager._metadata[target_slot_num].copy())
                    )
                    print(f"ImageManager: Current state of Slot {target_slot_num} pushed to undo stack.")

                # Clear redo stack since new action invalidates the redo history
                self.image_manager._redo_stacks[target_slot_num].clear()
                print(f"ImageManager: Redo stack for Slot {target_slot_num} cleared.")

                # Deep copy to prevent unintended modifications
                copied_image = source_image.copy()
                copied_metadata = source_metadata.copy()

                # Assign to target slot
                self.image_manager._images[target_slot_num] = copied_image
                self.image_manager._metadata[target_slot_num] = copied_metadata

                # Emit image_changed signal for the target slot
                self.image_manager.image_changed.emit(target_slot_num, copied_image, copied_metadata)

                QMessageBox.information(self, "Copy Successful", f"Image copied from Slot 0 to {target_slot_str}.")
                print(f"Image successfully copied from Slot 0 to {target_slot_str}.")

            except Exception as e:
                QMessageBox.critical(self, "Copy Failed", f"Failed to copy image to {target_slot_str}.\nError: {e}")
                print(f"Failed to copy image to {target_slot_str}. Error: {e}")
        else:
            print("Copy Slot operation cancelled by the user.")

    # --------------------
    # Slot Preview Methods
    # --------------------
    def open_preview_window(self, slot):
        """Opens a separate preview window for the specified slot."""
        print(f"Attempting to open preview window for Slot {slot}. Current preview_windows: {self.preview_windows}")
        # Check if the slot index is valid
        if slot < 0 or slot >= self.image_manager.max_slots:
            QMessageBox.warning(self, "Invalid Slot", f"Slot {slot} is out of range.")
            return

        # Check if the slot has an image
        image = self.image_manager._images[slot]
        if image is None:
            QMessageBox.warning(self, "No Image", f"Slot {slot} does not contain an image.")
            return

        # Check if a preview window for this slot already exists
        if slot in self.preview_windows:
            # If the window is already open, bring it to the front
            existing_window = self.preview_windows[slot]
            existing_window.raise_()
            existing_window.activateWindow()
            print(f"Preview window for Slot {slot} is already open.")
            return

        # Create a new ImagePreview window with a copy of the image data
        image_copy = image.copy()
        preview = ImagePreview(image_data=image_copy, slot=slot, parent=self)  # Pass parent=self
        preview.setWindowTitle(f"Preview - Slot {slot}")

        # Store the reference to prevent garbage collection
        self.preview_windows[slot] = preview
        print(f"Stored preview window for Slot {slot} in preview_windows.")

        # Connect the custom closed signal to the on_preview_closed method
        preview.closed.connect(self.on_preview_closed)

        # Show the preview window
        preview.show()
        print(f"Opened preview window for Slot {slot}.")

    def on_preview_closed(self, slot):
        """Handles the cleanup when a preview window is closed."""
        if slot in self.preview_windows:
            del self.preview_windows[slot]
            print(f"Preview window for Slot {slot} has been closed and removed from tracking.")
        else:
            print(f"No preview window found for Slot {slot} to remove.")


    def on_image_changed(self, slot, image, metadata):
        """Update the file name in the status bar and refresh preview if open."""
        file_path = metadata.get('file_path', None)
        if file_path:
            self.file_name_label.setText(os.path.basename(file_path))  # Update the label with file name
        else:
            self.file_name_label.setText("No file selected")

        # If a preview window for this slot is open, update its image
        if slot in self.preview_windows:
            preview_window = self.preview_windows[slot]
            preview_window.update_image_data(image.copy())
            print(f"Preview window for Slot {slot} updated with new image.")

     

    def add_stars(self):
        """
        Add stars back to the current image using the chosen stars-only image and method.
        Allows using a stars-only image from a slot or from a file.
        """
        try:
            print("Starting star addition process...")

            # Prompt the user to choose the source of stars-only image
            source_choice, ok = QInputDialog.getItem(
                self,
                "Select Stars-Only Image Source",
                "Choose the source of the stars-only image:",
                ["From File", "From Slot"],
                editable=False
            )

            if not ok or not source_choice:
                QMessageBox.warning(self, "Cancelled", "Star addition process cancelled.")
                print("Star addition process cancelled by the user.")
                return

            print(f"Stars-only image source selected: {source_choice}")

            if source_choice == "From File":
                # Prompt the user to select the stars-only image file
                stars_only_path, _ = QFileDialog.getOpenFileName(
                    self,
                    "Select Stars-Only Image",
                    "",
                    "Image Files (*.tif *.tiff *.png)"
                )

                if not stars_only_path:
                    QMessageBox.warning(self, "No Image Selected", "No stars-only image selected. Operation cancelled.")
                    print("No stars-only image selected. Exiting star addition process.")
                    return

                print(f"Stars-only image selected: {stars_only_path}")

                # Load the stars-only image
                stars_only_image = cv2.imread(stars_only_path, cv2.IMREAD_UNCHANGED)
                if stars_only_image is None:
                    QMessageBox.critical(self, "Error", "Failed to load stars-only image. Please try again.")
                    print(f"Failed to load stars-only image from {stars_only_path}.")
                    return

                # Determine bit depth based on image data type
                if stars_only_image.dtype == np.uint16:
                    stars_only_image = stars_only_image.astype('float32') / 65535.0
                elif stars_only_image.dtype == np.uint8:
                    stars_only_image = stars_only_image.astype('float32') / 255.0
                else:
                    stars_only_image = stars_only_image.astype('float32')  # Assuming it's already normalized

            elif source_choice == "From Slot":
                # Prompt the user to select a slot containing a stars-only image
                available_slots = [f"Slot {i}" for i in range(1, self.image_manager.max_slots) if self.image_manager._images.get(i, None) is not None]
                if not available_slots:
                    QMessageBox.warning(self, "No Available Slots", "No slots contain a stars-only image. Please add a stars-only image to a slot first.")
                    print("No slots contain a stars-only image.")
                    return

                slot_choice, ok = QInputDialog.getItem(
                    self,
                    "Select Slot",
                    "Choose a slot containing the stars-only image:",
                    available_slots,
                    editable=False
                )

                if not ok or not slot_choice:
                    QMessageBox.warning(self, "Cancelled", "Star addition process cancelled.")
                    print("Star addition process cancelled by the user.")
                    return

                target_slot_num = int(slot_choice.split()[-1])
                stars_only_image = self.image_manager._images.get(target_slot_num, None)

                if stars_only_image is None:
                    QMessageBox.warning(self, "Empty Slot", f"{slot_choice} does not contain a stars-only image.")
                    print(f"{slot_choice} is empty. Cannot perform star addition.")
                    return

                print(f"Stars-only image selected from {slot_choice}.")

                # If the stars-only image is integer type, normalize it
                if stars_only_image.dtype == np.uint16:
                    stars_only_image = stars_only_image.astype('float32') / 65535.0
                elif stars_only_image.dtype == np.uint8:
                    stars_only_image = stars_only_image.astype('float32') / 255.0
                else:
                    stars_only_image = stars_only_image.astype('float32')  # Assuming it's already normalized

            else:
                QMessageBox.warning(self, "Invalid Choice", "Invalid source choice. Operation cancelled.")
                print("Invalid source choice. Exiting star addition process.")
                return

            # Normalize stars-only image to [0, 1] range
            stars_only_image = np.clip(stars_only_image, 0.0, 1.0)

            # Check if current image exists
            current_slot = self.image_manager.current_slot
            current_image = self.image_manager._images.get(current_slot, None)

            if current_image is None:
                QMessageBox.warning(self, "No Image", f"Slot {current_slot} does not contain an image.")
                print(f"Slot {current_slot} is empty. Cannot perform star addition.")
                return

            # Prompt the user to choose the addition method
            addition_method, ok = QInputDialog.getItem(
                self,
                "Choose Addition Method",
                "Select how to add stars back to the image:",
                ["Screen", "Add"],
                editable=False
            )

            if not ok or not addition_method:
                QMessageBox.warning(self, "Cancelled", "Star addition process cancelled.")
                print("Star addition process cancelled by the user.")
                return

            print(f"Addition method selected: {addition_method}")

            # Perform the star addition
            print("Performing star addition...")
            if addition_method == "Screen":
                combined_image = current_image + stars_only_image - current_image * stars_only_image
            elif addition_method == "Add":
                combined_image = current_image + stars_only_image

            # Clip the combined image to [0, 1]
            combined_image = np.clip(combined_image, 0.0, 1.0)
            print("Star addition completed successfully.")

            # Save current state of the current slot to undo stack
            if self.image_manager._images[current_slot] is not None:
                self.image_manager._undo_stacks[current_slot].append(
                    (self.image_manager._images[current_slot].copy(),
                    self.image_manager._metadata[current_slot].copy())
                )
                print(f"ImageManager: Current state of Slot {current_slot} pushed to undo stack.")

            # Clear redo stack since new action invalidates the redo history
            self.image_manager._redo_stacks[current_slot].clear()
            print(f"ImageManager: Redo stack for Slot {current_slot} cleared.")

            # Get current metadata
            current_metadata = self.image_manager._metadata.get(current_slot, {}).copy()

            # Optionally, update metadata fields if necessary
            # For example, you might want to update the 'file_path' or add a note about star addition
            # Here, we'll add a simple note
            addition_note = f"Stars added using method: {addition_method}"
            if 'notes' in current_metadata and isinstance(current_metadata['notes'], list):
                current_metadata['notes'].append(addition_note)
            else:
                current_metadata['notes'] = [addition_note]

            # Assign the combined image and updated metadata to the current slot
            self.image_manager._images[current_slot] = combined_image
            self.image_manager._metadata[current_slot] = current_metadata

            # Emit the image_changed signal with all required arguments
            self.image_manager.image_changed.emit(current_slot, combined_image, current_metadata)

            QMessageBox.information(self, "Success", "Stars added successfully.")
            print("Stars added successfully.")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"An unexpected error occurred:\n{e}")
            print(f"An unexpected error occurred: {e}")



    def remove_stars(self):
        """
        Removes stars from the current image using StarNet and generates a stars-only image.
        Supports Windows, macOS, and Linux platforms.
        """

        print("Starting star removal process...")

        # Step 1: Verify StarNet Executable Path
        if not self.starnet_exe_path:
            print("StarNet executable path not set. Prompting user to select executable.")
            self.select_starnet_exe()
            if not self.starnet_exe_path:
                print("User cancelled StarNet executable selection.")
                return  # User cancelled the selection
            else:
                print(f"StarNet executable selected: {self.starnet_exe_path}")
        else:
            print(f"Using existing StarNet executable path: {self.starnet_exe_path}")

        # Step 2: Ensure current image is loaded
        if self.image_manager.image is None:
            QMessageBox.warning(self, "No Image", "Please load an image before removing stars.")
            print("No image loaded. Exiting star removal process.")
            return

        print("Image is loaded. Proceeding with star removal.")

        # Step 3: Determine the Operating System
        current_os = platform.system()
        print(f"Operating System detected: {current_os}")
        if current_os not in ["Windows", "Darwin", "Linux"]:
            QMessageBox.critical(self, "Unsupported OS", f"The current operating system '{current_os}' is not supported.")
            print(f"Unsupported operating system: {current_os}")
            return

        # Step 4: Ask if the image is linear
        reply = QMessageBox.question(
            self,
            "Image Linearity",
            "Is the current image linear?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if self.image_manager.image.ndim == 2 or (self.image_manager.image.ndim == 3 and self.image_manager.image.shape[2] == 1):
            print("Converting single-channel image to 3-channel RGB...")
            processing_image = np.stack([self.image_manager.image] * 3, axis=-1)
        else:
            processing_image = self.image_manager.image

        if reply == QMessageBox.Yes:
            print("Image is linear. Applying stretch.")
            dialog_msg = QMessageBox(self)
            dialog_msg.setWindowTitle("Stretching Image")
            dialog_msg.setText("Stretching the image for StarNet processing...")
            dialog_msg.setStandardButtons(QMessageBox.Ok)
            dialog_msg.exec_()

            # Apply stretch
            stretched_image = self.stretch_image(processing_image)
            # Use stretched_image for processing
            processing_image = stretched_image
            print("Image stretched successfully.")
            self.image_was_stretched = True
        else:
            print("Image is not linear. Proceeding without stretching.")
            processing_image = processing_image
            self.image_was_stretched = False

        # Step 4: Set Command Parameters Based on OS
        self.starnet_dir = os.path.dirname(self.starnet_exe_path)
        self.input_image_path = os.path.join(self.starnet_dir, "imagetoremovestars.tif")
        self.output_image_path = os.path.join(self.starnet_dir, "starless.tif")
        original_image = processing_image

        print(f"StarNet directory: {self.starnet_dir}")
        print(f"Input image path: {self.input_image_path}")
        print(f"Output image path: {self.output_image_path}")

        # Convert image from [0,1] to [0, 65535] for 16-bit TIFF
        image_16bit = (original_image * 65535).astype(np.uint16)
        # Convert RGB to BGR for OpenCV
        image_bgr_16bit = cv2.cvtColor(image_16bit, cv2.COLOR_RGB2BGR)
        cv2.imwrite(self.input_image_path, image_bgr_16bit)
        print(f"Input image saved at {self.input_image_path}")

        # Prepare the command based on the OS
        if current_os == "Windows":
            # Windows requires the stride parameter
            stride = 256
            command = [
                self.starnet_exe_path,
                self.input_image_path,
                self.output_image_path,
                str(stride)
            ]
            print("Preparing command for Windows.")
        elif current_os == "Linux":
            # Linux requires the stride parameter
            stride = 256
            command = [
                self.starnet_exe_path,
                self.input_image_path,
                self.output_image_path,
                str(stride)
            ]
            print("Preparing command for Linux.")
        elif current_os == "Darwin":
            # macOS does NOT require the stride parameter
            stride = None
            command = [
                self.starnet_exe_path,
                self.input_image_path,
                self.output_image_path
            ]
            print("Preparing command for macOS.")

        print(f"StarNet command: {' '.join(command)}")

        # Step 5: Ensure the StarNet Executable has Execute Permissions (for macOS and Linux)
        if current_os in ["Darwin", "Linux"]:
            if not os.access(self.starnet_exe_path, os.X_OK):
                print(f"StarNet executable not executable. Setting execute permissions for {self.starnet_exe_path}")
                os.chmod(self.starnet_exe_path, 0o755)  # Add execute permissions
                print("Execute permissions set.")
            else:
                print("StarNet executable already has execute permissions.")
        else:
            print("No need to set execute permissions for Windows.")

        # Step 6: Initialize and Show StarNetDialog
        starnet_dialog = StarNetDialog()
        starnet_dialog.show()

        # Step 7: Initialize StarNetThread
        self.starnet_thread = StarNetThread(command, self.starnet_dir)
        self.starnet_thread.stdout_signal.connect(starnet_dialog.append_text)
        self.starnet_thread.stderr_signal.connect(starnet_dialog.append_text)
        self.starnet_thread.finished_signal.connect(lambda return_code: self.on_starnet_finished(return_code, starnet_dialog, self.output_image_path))

        # Handle cancellation
        starnet_dialog.cancel_button.clicked.connect(self.starnet_thread.stop)

        # Start the thread
        self.starnet_thread.start()

    def on_starnet_finished(self, return_code, dialog, output_image_path):
        """
        Handles the completion of the StarNet process.
        """
        dialog.append_text(f"\nProcess finished with return code {return_code}.\n")
        if return_code != 0:
            QMessageBox.critical(self, "StarNet Error", f"StarNet failed with return code {return_code}.")
            print(f"StarNet failed with return code {return_code}.")
            dialog.close()
            return

        # Step 8: Load the starless image
        if not os.path.exists(output_image_path):
            QMessageBox.critical(self, "StarNet Error", "Starless image was not created.")
            print(f"Starless image was not created at {output_image_path}.")
            dialog.close()
            return

        print(f"Starless image found at {output_image_path}. Loading image...")
        dialog.append_text(f"Starless image found at {output_image_path}. Loading image...\n")

        starless_bgr = cv2.imread(output_image_path, cv2.IMREAD_UNCHANGED)
        if starless_bgr is None:
            QMessageBox.critical(self, "StarNet Error", "Failed to load starless image.")
            print(f"Failed to load starless image from {output_image_path}.")
            dialog.close()
            return

        print("Starless image loaded successfully.")
        dialog.append_text("Starless image loaded successfully.\n")

        # Convert back to RGB and normalize to [0,1]
        starless_rgb = cv2.cvtColor(starless_bgr, cv2.COLOR_BGR2RGB).astype('float32') / 65535.0



        # Check and apply unstretch if necessary
        if getattr(self, 'image_was_stretched', False):
            print("Unstretching the starless image...")
            starless_rgb = self.unstretch_image(starless_rgb)
            print("Starless image unstretched successfully.")
            dialog.append_text("Starless image unstretched successfully.\n")
        else:
            print("Image was not stretched. Proceeding without unstretching.")
            dialog.append_text("Image was not stretched. Proceeding without unstretching.\n")

        # Convert image_manager.image to 3-channel if needed
        if starless_rgb.ndim == 2 or (starless_rgb.ndim == 3 and starless_rgb.shape[2] == 1):
            print("Converting single-channel original image to 3-channel RGB...")
            starless_rgb = np.stack([starless_rgb] * 3, axis=-1)
        else:
            starless_rgb = starless_rgb

        # Convert image_manager.image to 3-channel if needed
        if self.image_manager.image.ndim == 2 or (self.image_manager.image.ndim == 3 and self.image_manager.image.shape[2] == 1):
            print("Converting single-channel original image to 3-channel RGB...")
            original_image_rgb = np.stack([self.image_manager.image] * 3, axis=-1)
        else:
            original_image_rgb = self.image_manager.image            

        # Step 9: Generate Stars Only Image
        print("Generating stars-only image...")
        dialog.append_text("Generating stars-only image...\n")
        with np.errstate(divide='ignore', invalid='ignore'):
            stars_only = (original_image_rgb - starless_rgb) / (1.0 - starless_rgb)
            stars_only = np.nan_to_num(stars_only, nan=0.0, posinf=0.0, neginf=0.0)
        stars_only = np.clip(stars_only, 0.0, 1.0)
        print("Stars-only image generated.")
        dialog.append_text("Stars-only image generated.\n")

        # Step 10: Prompt user to save Stars Only Image
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Stars Only Image",
            self.starnet_dir,  # Updated to use instance attribute
            "TIFF Files (*.tif *.tiff);;PNG Files (*.png)"
        )

        if save_path:
            print(f"Saving stars-only image to {save_path}...")
            dialog.append_text(f"Saving stars-only image to {save_path}...\n")
            try:
                # Determine the format and bit depth based on the file extension
                _, ext = os.path.splitext(save_path)
                ext = ext.lower()
                if ext in ['.tif', '.tiff']:
                    original_format = 'tiff'
                    bit_depth = '16-bit'
                elif ext == '.png':
                    original_format = 'png'
                    bit_depth = '8-bit'
                else:
                    QMessageBox.warning(self, "Unsupported Format", f"The selected format '{ext}' is not supported.")
                    print(f"Unsupported file extension: {ext}")
                    dialog.append_text(f"Unsupported file extension: {ext}\n")
                    dialog.close()
                    return

                # Call the global save_image function
                save_image(
                    img_array=stars_only,
                    filename=save_path,
                    original_format=original_format,
                    bit_depth=bit_depth,
                    original_header=None,  # Pass actual header if available
                    is_mono=False,        # Set to True if image is monochrome
                    image_meta=None,      # Pass image metadata if needed
                    file_meta=None        # Pass file metadata if needed
                )
                QMessageBox.information(self, "Success", "Stars only image saved successfully.")
                print("Stars-only image saved successfully.")
                dialog.append_text("Stars-only image saved successfully.\n")
            except Exception as e:
                QMessageBox.critical(self, "Save Error", f"Failed to save stars only image:\n{e}")
                print(f"Failed to save stars only image: {e}")
                dialog.append_text(f"Failed to save stars only image: {e}\n")
        else:
            QMessageBox.warning(self, "Save Cancelled", "Stars only image was not saved.")
            print("User cancelled saving the stars-only image.")
            dialog.append_text("User cancelled saving the stars-only image.\n")

        # Step 11: Update ImageManager with Starless Image
        print("Updating ImageManager with the starless image.")
        dialog.append_text("Updating ImageManager with the starless image.\n")
        self.image_manager.update_image(
            updated_image=starless_rgb,
            metadata=self.image_manager._metadata.get(self.image_manager.current_slot, {})
        )
        QMessageBox.information(self, "Success", "Starless image updated successfully.")
        print("ImageManager updated with starless image.")
        dialog.append_text("ImageManager updated with starless image.\n")

        # Optional: Clean up temporary files
        try:
            print("Cleaning up temporary files...")
            dialog.append_text("Cleaning up temporary files...\n")
            if os.path.exists(self.input_image_path):
                os.remove(self.input_image_path)
                print(f"Deleted temporary input image at {self.input_image_path}.")
                dialog.append_text(f"Deleted temporary input image at {self.input_image_path}.\n")
            if os.path.exists(self.output_image_path):
                os.remove(self.output_image_path)
                print(f"Deleted temporary output image at {self.output_image_path}.")
                dialog.append_text(f"Deleted temporary output image at {self.output_image_path}.\n")
        except Exception as e:
            QMessageBox.warning(self, "Cleanup Warning", f"Failed to delete temporary files:\n{e}")
            print(f"Failed to delete temporary files: {e}")
            dialog.append_text(f"Failed to delete temporary files: {e}\n")

        dialog.close()
 
    def stretch_image(self, image):
        """
        Perform an unlinked linear stretch on the image.
        Each channel is stretched independently by subtracting its own minimum,
        recording its own median, and applying the stretch formula.
        Returns the stretched image.
        """
        was_single_channel = False  # Flag to check if image was single-channel

        # Check if the image is single-channel
        if image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):
            was_single_channel = True
            image = np.stack([image] * 3, axis=-1)  # Convert to 3-channel by duplicating

        # Ensure the image is a float32 array for precise calculations and writable
        image = image.astype(np.float32).copy()

        # Initialize lists to store per-channel minima and medians
        self.stretch_original_mins = []
        self.stretch_original_medians = []

        # Initialize stretched_image as a copy of the input image
        stretched_image = image.copy()

        # Define the target median for stretching
        target_median = 0.25

        # Apply the stretch for each channel independently
        for c in range(3):
            # Record the minimum of the current channel
            channel_min = np.min(stretched_image[..., c])
            self.stretch_original_mins.append(channel_min)

            # Subtract the channel's minimum to shift the image
            stretched_image[..., c] -= channel_min

            # Record the median of the shifted channel
            channel_median = np.median(stretched_image[..., c])
            self.stretch_original_medians.append(channel_median)

            if channel_median != 0:
                numerator = (channel_median - 1) * target_median * stretched_image[..., c]
                denominator = (
                    channel_median * (target_median + stretched_image[..., c] - 1)
                    - target_median * stretched_image[..., c]
                )
                # To avoid division by zero
                denominator = np.where(denominator == 0, 1e-6, denominator)
                stretched_image[..., c] = numerator / denominator
            else:
                print(f"Channel {c} - Median is zero. Skipping stretch.")

        # Clip stretched image to [0, 1] range
        stretched_image = np.clip(stretched_image, 0.0, 1.0)

        # Store stretch parameters
        self.was_single_channel = was_single_channel

        return stretched_image


    def unstretch_image(self, image):
        """
        Undo the unlinked linear stretch to return the image to its original state.
        Each channel is unstretched independently by reverting the stretch formula
        using the stored medians and adding back the individual channel minima.
        Returns the unstretched image.
        """
        original_mins = self.stretch_original_mins
        original_medians = self.stretch_original_medians
        was_single_channel = self.was_single_channel

        # Ensure the image is a float32 array for precise calculations and writable
        image = image.astype(np.float32).copy()

        # Apply the unstretch for each channel independently
        for c in range(3):
            channel_median = np.median(image[..., c])
            original_median = original_medians[c]
            original_min = original_mins[c]

            if channel_median != 0 and original_median != 0:
                numerator = (channel_median - 1) * original_median * image[..., c]
                denominator = (
                    channel_median * (original_median + image[..., c] - 1)
                    - original_median * image[..., c]
                )
                # To avoid division by zero
                denominator = np.where(denominator == 0, 1e-6, denominator)
                image[..., c] = numerator / denominator
            else:
                print(f"Channel {c} - Median or original median is zero. Skipping unstretch.")

            # Add back the channel's original minimum
            image[..., c] += original_min

        # Clip to [0, 1] range
        image = np.clip(image, 0, 1)

        # If the image was originally single-channel, convert back to single-channel
        if was_single_channel:
            image = np.mean(image, axis=2, keepdims=True)  # Convert back to single-channel

        return image



    def select_starnet_exe(self):
        """
        Prompts the user to select the StarNet executable based on the operating system.
        Saves the path using QSettings for future use.
        """
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        current_os = platform.system()

        if current_os == "Windows":
            filter_str = "Executable Files (*.exe)"
        elif current_os in ["Linux", "Darwin"]:
            # For Unix-based systems, executables may not have extensions
            filter_str = "All Executable Files (*)"
        else:
            QMessageBox.critical(self, "Unsupported OS", f"The current operating system '{current_os}' is not supported.")
            return

        exe_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select StarNet Executable",
            "",
            filter_str,
            options=options
        )
        if exe_path:
            # For Windows, ensure the file has .exe extension
            if current_os == "Windows" and not exe_path.lower().endswith('.exe'):
                QMessageBox.warning(self, "Invalid File", "Please select a valid .exe file for StarNet.")
                return
            # For Unix-based systems, optionally check if it's executable
            elif current_os in ["Linux", "Darwin"]:
                if not os.access(exe_path, os.X_OK):
                    reply = QMessageBox.question(
                        self,
                        "Set Execute Permissions",
                        "The selected file does not have execute permissions. Would you like to add them?",
                        QMessageBox.Yes | QMessageBox.No,
                        QMessageBox.Yes
                    )
                    if reply == QMessageBox.Yes:
                        try:
                            os.chmod(exe_path, 0o755)
                        except Exception as e:
                            QMessageBox.critical(self, "Permission Error", f"Failed to set execute permissions:\n{e}")
                            return
                    else:
                        QMessageBox.information(self, "Cancelled", "Execute permissions not set. Cannot proceed.")
                        return
            self.starnet_exe_path = exe_path
            # Save the path using QSettings
            self.settings.setValue("starnet/exe_path", self.starnet_exe_path)
            QMessageBox.information(self, "StarNet Executable Set", f"StarNet executable set to:\n{exe_path}")
        else:
            QMessageBox.information(self, "Cancelled", "StarNet executable selection was cancelled.")



    def open_clahe_dialog(self):
        """Opens the CLAHE dialog window."""
        dialog = CLAHEDialog(self.image_manager, self)
        dialog.exec_()

    def open_morpho_dialog(self):
        """Opens the Morphological Operations dialog window."""
        dialog = MorphologyDialog(self.image_manager, self)
        dialog.exec_()

    def open_whitebalance_dialog(self):
        """Opens the White Balance dialog window."""
        dialog = WhiteBalanceDialog(self.image_manager, self)
        dialog.exec_()

    def open_background_neutralization_dialog(self):
        """Opens the Background Neutralization dialog window."""
        dialog = BackgroundNeutralizationDialog(self.image_manager, self)
        dialog.exec_()

    def open_remove_green_dialog(self):
        """Opens the Remove Green dialog window."""
        dialog = RemoveGreenDialog(self.image_manager, self)
        dialog.exec_()


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

        # If slot == 0 and we have a valid image, update dimensions
        if slot == 0 and image is not None:
            # image should be a numpy array with shape (height, width[, channels])
            h, w = image.shape[:2]
            self.dim_label.setText(f"{w} x {h}")
        else:
            # If another slot changed or no image, you might want to blank it
            # or just leave it as-is. Example: set to "—"
            self.dim_label.setText("—")            

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
                                            "Images (*.png *.tif *.tiff *.fits *.fit *.xisf *.cr2 *.nef *.arw *.dng *.orf *.rw2 *.pef);;All Files (*)")

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

class CopySlotDialog(QDialog):
    def __init__(self, parent=None, available_slots=None):
        super().__init__(parent)
        self.setWindowTitle("Copy Image to Slot")
        self.setModal(True)
        self.selected_slot = None  # To store the user's selection

        # Create layout components
        label = QLabel("Copy current image to:")
        self.slot_combo = QComboBox()
        if available_slots:
            self.slot_combo.addItems(available_slots)

        apply_button = QPushButton("Apply")
        apply_button.clicked.connect(self.apply_clicked)

        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)

        # Layout setup
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        button_layout.addWidget(apply_button)
        button_layout.addWidget(cancel_button)

        layout = QVBoxLayout()
        layout.addWidget(label)
        layout.addWidget(self.slot_combo)
        layout.addLayout(button_layout)

        self.setLayout(layout)

    def apply_clicked(self):
        """Handle the Apply button click."""
        self.selected_slot = self.slot_combo.currentText()
        self.accept()

    def get_selected_slot(self):
        """Return the selected slot."""
        return self.selected_slot

class CropTool(QDialog):
    """A cropping tool using QGraphicsView for better rectangle handling."""
    crop_applied = pyqtSignal(object)

    def __init__(self, image_data, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Crop Tool")
        self.setGeometry(150, 150, 800, 600)  # Initial size

        self.original_image_data = image_data.copy()  # Keep a copy of the original image
        self.image_data = image_data  # Displayed image (can be autostretched)
        self.scene = QGraphicsScene()
        self.graphics_view = QGraphicsView(self.scene)
        self.pixmap_item = None

        self.origin = QPointF()
        self.current_rect = QRectF()
        self.selection_rect_item = None
        self.drawing = False

        # Set up the layout
        layout = QVBoxLayout()
        layout.addWidget(self.graphics_view)

        # Buttons
        self.autostretch_button = QPushButton("Toggle Autostretch")
        self.autostretch_button.clicked.connect(self.toggle_autostretch)
        layout.addWidget(self.autostretch_button)

        self.crop_button = QPushButton("Apply Crop")
        self.crop_button.clicked.connect(self.apply_crop)
        layout.addWidget(self.crop_button)

        self.setLayout(layout)

        # Load and display the image
        self.load_image()
        self.graphics_view.viewport().installEventFilter(self)

    def load_image(self):
        """Load and display the image in the QGraphicsView."""
        height, width, channels = self.image_data.shape
        if channels == 3:
            q_image = QImage(
                (self.image_data * 255).astype(np.uint8).tobytes(),
                width,
                height,
                3 * width,
                QImage.Format_RGB888
            )
        else:
            q_image = QImage(
                (self.image_data * 255).astype(np.uint8).tobytes(),
                width,
                height,
                width,
                QImage.Format_Grayscale8
            )
        pixmap = QPixmap.fromImage(q_image)
        self.pixmap_item = QGraphicsPixmapItem(pixmap)
        self.scene.addItem(self.pixmap_item)
        self.graphics_view.fitInView(self.pixmap_item, Qt.KeepAspectRatio)

    def eventFilter(self, source, event):
        """Handle mouse events for drawing the cropping rectangle."""
        if source is self.graphics_view.viewport():
            if event.type() == event.MouseButtonPress:
                if event.button() == Qt.LeftButton:
                    self.drawing = True
                    self.origin = self.graphics_view.mapToScene(event.pos())
                    if self.selection_rect_item:
                        self.scene.removeItem(self.selection_rect_item)
                        self.selection_rect_item = None
            elif event.type() == event.MouseMove:
                if self.drawing:
                    current_pos = self.graphics_view.mapToScene(event.pos())
                    self.current_rect = QRectF(self.origin, current_pos).normalized()
                    if self.selection_rect_item:
                        self.scene.removeItem(self.selection_rect_item)
                    pen = QPen(QColor(255, 0, 0), 5, Qt.DashLine)
                    self.selection_rect_item = self.scene.addRect(self.current_rect, pen)
            elif event.type() == event.MouseButtonRelease:
                if event.button() == Qt.LeftButton and self.drawing:
                    self.drawing = False
                    current_pos = self.graphics_view.mapToScene(event.pos())
                    self.current_rect = QRectF(self.origin, current_pos).normalized()
                    if self.selection_rect_item:
                        self.scene.removeItem(self.selection_rect_item)
                    pen = QPen(QColor(0, 255, 0), 5, Qt.SolidLine)
                    self.selection_rect_item = self.scene.addRect(self.current_rect, pen)
        return super().eventFilter(source, event)

    def toggle_autostretch(self):
        """Apply autostretch for visualization purposes only."""
        stretched_image = None
        if len(self.original_image_data.shape) == 2:  # Mono image
            stretched_image = stretch_mono_image(self.original_image_data, target_median=0.5)
        elif len(self.original_image_data.shape) == 3:  # Color image
            stretched_image = stretch_color_image(self.original_image_data, target_median=0.5, linked=False)
        
        if stretched_image is not None:
            self.image_data = stretched_image
            # Save rectangle data before clearing the scene
            saved_rect = self.current_rect if not self.current_rect.isNull() else None

            self.scene.clear()
            self.load_image()

            # Redraw the rectangle if it exists
            if saved_rect:
                pen = QPen(QColor(0, 255, 0), 5, Qt.SolidLine)
                self.selection_rect_item = self.scene.addRect(saved_rect, pen)

    def apply_crop(self):
        """Crop the original image based on the selected rectangle."""
        if not self.current_rect.isNull():
            # Get the scene dimensions and scale accordingly
            scene_rect = self.scene.sceneRect()
            scale_x = self.original_image_data.shape[1] / scene_rect.width()
            scale_y = self.original_image_data.shape[0] / scene_rect.height()

            # Convert scene rectangle to image coordinates
            x = int(self.current_rect.left() * scale_x)
            y = int(self.current_rect.top() * scale_y)
            w = int(self.current_rect.width() * scale_x)
            h = int(self.current_rect.height() * scale_y)

            # Ensure bounds are valid
            x = max(0, min(x, self.original_image_data.shape[1] - 1))
            y = max(0, min(y, self.original_image_data.shape[0] - 1))
            w = max(1, min(w, self.original_image_data.shape[1] - x))
            h = max(1, min(h, self.original_image_data.shape[0] - y))

            # Crop the original image
            cropped_image = self.original_image_data[y:y + h, x:x + w]

            # Emit the cropped image to the parent
            self.crop_applied.emit(cropped_image)

            # Close the dialog
            self.accept()
        else:
            QMessageBox.warning(self, "No Selection", "Please draw a crop rectangle before applying.")

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
        self.active_previews = {}  # Track active preview windows by slot

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
        if slot is None:
            slot = self.current_slot

        if slot == 1:
            print("Warning: Attempting to update reserved slot 1.")
            return  # Prevent overwriting slot 1 unless explicitly allowed

        self._images[slot] = updated_image
        if metadata:
            self._metadata[slot] = metadata
        self.image_changed.emit(slot, updated_image, metadata)

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



class GradientRemovalDialog(QDialog):
    # Define signals to communicate with AstroEditingSuite
    processing_completed = pyqtSignal(np.ndarray, np.ndarray)  # Corrected Image, Gradient Background

    def __init__(self, image, parent=None):
        """
        Initializes the GradientRemoval dialog.

        Args:
            image: Original image as a NumPy array (float32, normalized 0-1).
            parent: Parent widget.
        """
        super().__init__(parent)
        self.setWindowTitle("Gradient Removal")
        self.image = image.copy()  # Original image (float32, 0-1)
        self.exclusion_polygons = []  # List of polygons (each polygon is a list of QPoint)
        self.drawing = False
        self.current_polygon = []

        # Initialize parameters with default values
        self.num_sample_points = 100
        self.poly_degree = 2
        self.rbf_smooth = 0.1
        self.show_gradient = False

        # Downsample scale factor (can be made user-definable if needed)
        self.downsample_scale = 4

        # Calculate scale factor to fit image within max_display_size
        original_height, original_width = self.image.shape[:2]
        max_display_size = (800, 600)
        max_width, max_height = max_display_size

        scale_w = max_width / original_width
        scale_h = max_height / original_height
        scale = min(scale_w, scale_h, 1.0)  # Prevent upscaling if image is smaller

        self.scale_factor = scale

        scaled_width = int(original_width * scale)
        scaled_height = int(original_height * scale)

        # Scale the image for display
        if len(self.image.shape) == 2:
            # Grayscale
            display_image = (self.image * 255).astype(np.uint8)
        else:
            # Color
            display_image = (self.image * 255).astype(np.uint8)


        # Resize to fit the max display size
        display_image = cv2.resize(
            display_image,
            (scaled_width, scaled_height),
            interpolation=cv2.INTER_AREA,
        )

        # Convert to QImage
        if len(display_image.shape) == 2:
            # Grayscale
            q_img = QImage(
                display_image.data,
                scaled_width,
                scaled_height,
                display_image.strides[0],
                QImage.Format_Grayscale8,
            )
        else:
            # Color
            q_img = QImage(
                display_image.data,
                scaled_width,
                scaled_height,
                display_image.strides[0],
                QImage.Format_RGB888,
            )

        self.base_pixmap = QPixmap.fromImage(q_img)
        self.pixmap = self.base_pixmap.copy()

        # Set up QLabel to display the image
        self.label = QLabel(self)
        self.label.setPixmap(self.pixmap)
        self.label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.label.mousePressEvent = self.mouse_press_event
        self.label.mouseMoveEvent = self.mouse_move_event
        self.label.mouseReleaseEvent = self.mouse_release_event

        # Set up controls
        self.setup_controls()

        # Set up layout
        main_layout = QHBoxLayout()
        main_layout.addWidget(self.label)

        # Controls layout
        controls_layout = QVBoxLayout()
        controls_layout.addWidget(self.controls_groupbox)
        controls_layout.addStretch(1)

        # Add a QLabel to display the current step
        self.status_label = QLabel("Ready")
        self.status_label.setAlignment(Qt.AlignCenter)
        controls_layout.addWidget(self.status_label)


        main_layout.addLayout(controls_layout)

        self.setLayout(main_layout)
        self.setMinimumSize(1000, 700)

        # Initialize thread as None (if using threading)
        self.thread = None

    def setup_controls(self):
        """
        Sets up the user controls for parameters.
        """
        self.controls_groupbox = QGroupBox("Parameters")
        form_layout = QFormLayout()

        # Number of sample points
        self.sample_points_spinbox = QSpinBox()
        self.sample_points_spinbox.setRange(10, 1000)
        self.sample_points_spinbox.setValue(self.num_sample_points)
        self.sample_points_spinbox.setSingleStep(10)
        self.sample_points_spinbox.valueChanged.connect(self.update_num_sample_points)
        form_layout.addRow("Number of Sample Points:", self.sample_points_spinbox)

        # Polynomial degree
        self.poly_degree_spinbox = QSpinBox()
        self.poly_degree_spinbox.setRange(1, 10)
        self.poly_degree_spinbox.setValue(self.poly_degree)
        self.poly_degree_spinbox.setSingleStep(1)
        self.poly_degree_spinbox.valueChanged.connect(self.update_poly_degree)
        form_layout.addRow("Polynomial Degree:", self.poly_degree_spinbox)

        # RBF smoothing
        self.rbf_smooth_spinbox = QDoubleSpinBox()
        self.rbf_smooth_spinbox.setRange(0.0, 10.0)
        self.rbf_smooth_spinbox.setValue(self.rbf_smooth)
        self.rbf_smooth_spinbox.setSingleStep(0.1)
        self.rbf_smooth_spinbox.valueChanged.connect(self.update_rbf_smooth)
        form_layout.addRow("RBF Smoothness:", self.rbf_smooth_spinbox)

        # Show gradient removal
        self.show_gradient_checkbox = QCheckBox("Show Gradient Removed")
        self.show_gradient_checkbox.stateChanged.connect(self.update_show_gradient)
        form_layout.addRow(self.show_gradient_checkbox)

        # Add AutoStretch button
        self.autostretch_button = QPushButton("AutoStretch")
        self.autostretch_button.setStatusTip("Apply auto-stretch to the displayed image")
        self.autostretch_button.clicked.connect(self.autostretch_image)
        form_layout.addRow(self.autostretch_button)

        # Clear Drawn Exclusion Areas button
        self.clear_exclusion_button = QPushButton("Clear Exclusion Areas")
        self.clear_exclusion_button.setStatusTip("Clear all drawn exclusion areas")
        self.clear_exclusion_button.clicked.connect(self.clear_exclusion_areas)
        form_layout.addRow(self.clear_exclusion_button)

        # Process button
        self.process_button = QPushButton("Process")
        self.process_button.clicked.connect(self.process_image)
        form_layout.addRow(self.process_button)

        # Instruction for Exclusion Zones
        instructions = QLabel("Draw exclusion zones by clicking and dragging on the image.\n"
                            "Press 'Enter' to finalize each polygon.")
        form_layout.addRow(instructions)

        self.controls_groupbox.setLayout(form_layout)

    def autostretch_image(self):
        """
        Applies auto-stretch to the displayed image without affecting the original image.
        """
        stretched_image = self.stretch_image(self.image)  # Stretch the original image for display

        # Scale the stretched image for display
        scaled_height, scaled_width = self.pixmap.height(), self.pixmap.width()
        display_image = (stretched_image * 255).astype(np.uint8)

        # Resize for display
        display_image = cv2.resize(
            display_image,
            (scaled_width, scaled_height),
            interpolation=cv2.INTER_AREA,
        )

        # Convert to QImage
        if len(display_image.shape) == 2:
            # Grayscale
            q_img = QImage(
                display_image.data,
                scaled_width,
                scaled_height,
                display_image.strides[0],
                QImage.Format_Grayscale8,
            )
        else:
            # Color
            q_img = QImage(
                display_image.data,
                scaled_width,
                scaled_height,
                display_image.strides[0],
                QImage.Format_RGB888,
            )

        # Update the pixmap with the stretched image
        self.stretched_pixmap = QPixmap.fromImage(q_img)  # Save the stretched pixmap
        self.label.setPixmap(self.stretched_pixmap)

    def update_selection(self):
        """
        Updates the pixmap with all finalized polygons and the current polygon being drawn.
        """
        # Use the stretched pixmap if available; otherwise, use the base pixmap
        if hasattr(self, "stretched_pixmap") and self.stretched_pixmap:
            self.pixmap = self.stretched_pixmap.copy()
        else:
            self.pixmap = self.base_pixmap.copy()

        painter = QPainter(self.pixmap)

        # Draw all finalized exclusion polygons in semi-transparent green
        pen = QPen(QColor(0, 255, 0), 2, Qt.SolidLine)
        brush = QColor(0, 255, 0, 50)  # Semi-transparent
        painter.setPen(pen)
        painter.setBrush(brush)
        for polygon in self.exclusion_polygons:
            painter.drawPolygon(polygon)

        # If currently drawing, draw the current polygon outline in red
        if self.drawing and len(self.current_polygon) > 1:
            pen = QPen(QColor(255, 0, 0), 2, Qt.DashLine)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            temp_polygon = QPolygon(self.current_polygon)
            painter.drawPolyline(temp_polygon)

        painter.end()
        self.label.setPixmap(self.pixmap)

    def clear_exclusion_areas(self):
        """
        Clears all drawn exclusion polygons and updates the preview.
        """
        self.exclusion_polygons = []  # Clear the list of polygons
        self.current_polygon = []  # Clear any currently drawn polygon
        self.update_selection()  # Redraw the pixmap



    def update_num_sample_points(self, value):
        self.num_sample_points = value

    def update_poly_degree(self, value):
        self.poly_degree = value

    def update_rbf_smooth(self, value):
        self.rbf_smooth = value

    def update_show_gradient(self, state):
        self.show_gradient = state == Qt.Checked

    def mouse_press_event(self, event):
        """
        Handles the mouse press event to initiate drawing.
        """
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.current_polygon = [event.pos()]
            self.update_selection()

    def mouse_move_event(self, event):
        """
        Handles the mouse move event to update the current polygon being drawn.
        """
        if self.drawing:
            self.current_polygon.append(event.pos())
            self.update_selection()

    def mouse_release_event(self, event):
        """
        Handles the mouse release event to finalize the polygon.
        """
        if event.button() == Qt.LeftButton and self.drawing:
            self.drawing = False
            self.exclusion_polygons.append(QPolygon(self.current_polygon))
            self.current_polygon = []
            self.update_selection()

    def process_image(self):
        """
        Processes the image to subtract the background in two stages:
        1. Polynomial gradient removal.
        2. RBF gradient removal.
        """
        # Disable the process button to prevent multiple clicks
        self.process_button.setEnabled(False)

        # Stretch the image before processing
        self.status_label.setText("Normalizing image for processing...")
        QApplication.processEvents()
        stretched_image = self.stretch_image(self.image)

        # Check if the image is color
        is_color = len(stretched_image.shape) == 3

        # Store original median
        original_median = np.median(stretched_image)

        # Create exclusion mask
        exclusion_mask = self.create_exclusion_mask(stretched_image.shape, self.exclusion_polygons) if self.exclusion_polygons else None

        # ------------------ First Stage: Polynomial Gradient Removal ------------------
        self.status_label.setText("Step 1: Polynomial Gradient Removal")
        QApplication.processEvents()
        # Downsample for polynomial background fitting
        small_image_poly = self.downsample_image(stretched_image, self.downsample_scale)

        # Create a downsampled exclusion mask for polynomial fitting
        if exclusion_mask is not None:
            small_exclusion_mask_poly = self.downsample_image(exclusion_mask.astype(np.float32), self.downsample_scale) >= 0.5
        else:
            small_exclusion_mask_poly = None

        # Generate sample points for polynomial fitting with exclusions
        poly_sample_points = self.generate_sample_points(
            small_image_poly, num_points=self.num_sample_points, exclusion_mask=small_exclusion_mask_poly
        )

        # Fit the polynomial gradient
        if is_color:
            poly_background = np.zeros_like(stretched_image)
            for channel in range(3):  # Process each channel separately
                poly_bg_channel = self.fit_polynomial_gradient(
                    small_image_poly[:, :, channel], poly_sample_points, degree=self.poly_degree
                )
                poly_background[:, :, channel] = self.upscale_background(poly_bg_channel, stretched_image.shape[:2])
        else:
            poly_background_small = self.fit_polynomial_gradient(small_image_poly, poly_sample_points, degree=self.poly_degree)
            poly_background = self.upscale_background(poly_background_small, stretched_image.shape[:2])

        # Subtract the polynomial background
        image_after_poly = stretched_image - poly_background

        # Normalize to restore original median
        image_after_poly = self.normalize_image(image_after_poly, original_median)

        # Clip the values to valid range
        image_after_poly = np.clip(image_after_poly, 0, 1)

        # ------------------ Second Stage: RBF Gradient Removal ------------------
        self.status_label.setText("Step 2: RBF Gradient Removal")
        QApplication.processEvents()
        # Downsample the image after polynomial removal for RBF fitting
        small_image_rbf = self.downsample_image(image_after_poly, self.downsample_scale)

        # Create a downsampled exclusion mask for RBF fitting
        if exclusion_mask is not None:
            small_exclusion_mask_rbf = self.downsample_image(exclusion_mask.astype(np.float32), self.downsample_scale) >= 0.5
        else:
            small_exclusion_mask_rbf = None

        # Generate sample points for RBF fitting with exclusions
        rbf_sample_points = self.generate_sample_points(
            small_image_rbf, num_points=self.num_sample_points, exclusion_mask=small_exclusion_mask_rbf
        )

        # Fit the RBF gradient
        if is_color:
            rbf_background = np.zeros_like(stretched_image)
            for channel in range(3):  # Process each channel separately
                rbf_bg_channel = self.fit_background(
                    small_image_rbf[:, :, channel], rbf_sample_points, smooth=self.rbf_smooth, patch_size=15
                )
                rbf_background[:, :, channel] = self.upscale_background(rbf_bg_channel, stretched_image.shape[:2])
        else:
            rbf_background_small = self.fit_background(small_image_rbf, rbf_sample_points, smooth=self.rbf_smooth, patch_size=15)
            rbf_background = self.upscale_background(rbf_background_small, stretched_image.shape[:2])

        # Subtract the RBF background
        corrected_image = image_after_poly - rbf_background

        # Normalize to restore original median
        corrected_image = self.normalize_image(corrected_image, original_median)

        # Clip the values to valid range
        corrected_image = np.clip(corrected_image, 0, 1)

        # Unstretch both the corrected image and the gradient background
        self.status_label.setText("De-Normalizing the processed images...")
        QApplication.processEvents()
        corrected_image = self.unstretch_image(corrected_image)
        total_background = poly_background + rbf_background
        gradient_background = self.unstretch_image(total_background)

                # Ensure both images are 3-channel RGB
        # Ensure both images are 3-channel RGB
        corrected_image = self.ensure_rgb(corrected_image)
        gradient_background = self.ensure_rgb(gradient_background)


        print("[DEBUG] Step 2 Completed.")

        # ------------------ Emit Results ------------------
        print("[DEBUG] Emitting results...")
        self.status_label.setText("Processing Complete")
        self.process_button.setEnabled(True)
        QApplication.processEvents()

        # Emit the processed images back to AstroEditingSuite
        self.processing_completed.emit(corrected_image, gradient_background)

        # Close the dialog
        self.accept()

    # ------------------ Helper Functions ------------------
    # Ensure corrected_image and gradient_background are strictly 3-channel RGB
    def ensure_rgb(self,image):
        """
        Ensures the given image is 3-channel RGB.
        Args:
            image: The input NumPy array (can be 2D or 3D with a single channel).
        Returns:
            A 3D NumPy array with shape (height, width, 3).
        """
        if image.ndim == 2:  # Grayscale image
            return np.repeat(image[:, :, np.newaxis], 3, axis=2)
        elif image.ndim == 3 and image.shape[2] == 1:  # Single-channel image with an extra dimension
            return np.repeat(image, 3, axis=2)
        elif image.ndim == 3 and image.shape[2] == 3:  # Already RGB
            return image
        else:
            raise ValueError(f"Unexpected image shape: {image.shape}")




    def stretch_image(self, image):
        """
        Perform an unlinked linear stretch on the image.
        Each channel is stretched independently by subtracting its own minimum,
        recording its own median, and applying the stretch formula.
        Returns the stretched image.
        """
        was_single_channel = False  # Flag to check if image was single-channel

        # Check if the image is single-channel
        if image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):
            was_single_channel = True
            image = np.stack([image] * 3, axis=-1)  # Convert to 3-channel by duplicating

        # Ensure the image is a float32 array for precise calculations and writable
        image = image.astype(np.float32).copy()

        # Initialize lists to store per-channel minima and medians
        self.stretch_original_mins = []
        self.stretch_original_medians = []

        # Initialize stretched_image as a copy of the input image
        stretched_image = image.copy()

        # Define the target median for stretching
        target_median = 0.25

        # Apply the stretch for each channel independently
        for c in range(3):
            # Record the minimum of the current channel
            channel_min = np.min(stretched_image[..., c])
            self.stretch_original_mins.append(channel_min)

            # Subtract the channel's minimum to shift the image
            stretched_image[..., c] -= channel_min

            # Record the median of the shifted channel
            channel_median = np.median(stretched_image[..., c])
            self.stretch_original_medians.append(channel_median)

            if channel_median != 0:
                numerator = (channel_median - 1) * target_median * stretched_image[..., c]
                denominator = (
                    channel_median * (target_median + stretched_image[..., c] - 1)
                    - target_median * stretched_image[..., c]
                )
                # To avoid division by zero
                denominator = np.where(denominator == 0, 1e-6, denominator)
                stretched_image[..., c] = numerator / denominator
            else:
                print(f"Channel {c} - Median is zero. Skipping stretch.")

        # Clip stretched image to [0, 1] range
        stretched_image = np.clip(stretched_image, 0.0, 1.0)

        # Store stretch parameters
        self.was_single_channel = was_single_channel

        return stretched_image


    def unstretch_image(self, image):
        """
        Undo the unlinked linear stretch to return the image to its original state.
        Each channel is unstretched independently by reverting the stretch formula
        using the stored medians and adding back the individual channel minima.
        Returns the unstretched image.
        """
        original_mins = self.stretch_original_mins
        original_medians = self.stretch_original_medians
        was_single_channel = self.was_single_channel

        # Ensure the image is a float32 array for precise calculations and writable
        image = image.astype(np.float32).copy()

        # Apply the unstretch for each channel independently
        for c in range(3):
            channel_median = np.median(image[..., c])
            original_median = original_medians[c]
            original_min = original_mins[c]

            if channel_median != 0 and original_median != 0:
                numerator = (channel_median - 1) * original_median * image[..., c]
                denominator = (
                    channel_median * (original_median + image[..., c] - 1)
                    - original_median * image[..., c]
                )
                # To avoid division by zero
                denominator = np.where(denominator == 0, 1e-6, denominator)
                image[..., c] = numerator / denominator
            else:
                print(f"Channel {c} - Median or original median is zero. Skipping unstretch.")

            # Add back the channel's original minimum
            image[..., c] += original_min

        # Clip to [0, 1] range
        image = np.clip(image, 0, 1)

        # If the image was originally single-channel, convert back to single-channel
        if was_single_channel:
            image = np.mean(image, axis=2, keepdims=True)  # Convert back to single-channel

        return image



    def downsample_image(self, image, scale=4):
        """
        Downsamples the image by the specified scale factor using area interpolation.

        Args:
            image: 2D/3D NumPy array of the image.
            scale: Downsampling scale factor.

        Returns:
            downsampled_image: Downsampled image.
        """
        new_size = (max(1, image.shape[1] // scale), max(1, image.shape[0] // scale))
        return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

    def upscale_background(self, background, original_shape):
        """
        Upscales the background model to the original image size.

        Args:
            background: 2D NumPy array (single-channel background model).
            original_shape: Tuple of (height, width) for the target size.

        Returns:
            upscaled_background: Upscaled 2D background model.
        """
        if background.ndim == 2:
            # Single-channel (grayscale) input
            return cv2.resize(background, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_LINEAR)
        elif background.ndim == 3 and background.shape[2] == 1:
            # Ensure input shape is reduced to 2D for single-channel data
            background = background.squeeze()  # Remove singleton dimension

        return cv2.resize(background, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_LINEAR)



    def divide_into_quartiles(self, image):
        """
        Divides the image into four quartiles.

        Args:
            image: 2D/3D NumPy array of the image.

        Returns:
            quartiles: Dictionary containing quartile images.
        """
        h, w = image.shape[:2]
        half_h, half_w = h // 2, w // 2
        return {
            'top_left': image[:half_h, :half_w],
            'top_right': image[:half_h, half_w:],
            'bottom_left': image[half_h:, :half_w],
            'bottom_right': image[half_h:, half_w:],
        }

    def exclude_bright_regions(self, quartile, exclusion_fraction=0.5):
        """
        Excludes the brightest regions in a quartile based on the exclusion fraction.

        Args:
            quartile: 2D/3D NumPy array of the quartile image.
            exclusion_fraction: Fraction of the brightest pixels to exclude.

        Returns:
            mask: Boolean mask where True indicates eligible pixels.
        """
        flattened = quartile.flatten()
        threshold = np.percentile(flattened, 100 * (1 - exclusion_fraction))
        mask = quartile < threshold
        return mask

    def gradient_descent_to_dim_spot(self, image, x, y, max_iterations=100, patch_size=15):
        """
        Moves a point to a dimmer spot using gradient descent, considering the median of a patch.

        Args:
            image: 2D/3D NumPy array of the image.
            x, y: Initial coordinates of the point.
            max_iterations: Maximum number of descent steps.
            patch_size: Size of the square patch (e.g., 15 for a 15x15 patch).

        Returns:
            (x, y): Coordinates of the dimmest local spot found.
        """
        half_patch = patch_size // 2

        # Get image dimensions and convert to luminance if color
        if len(image.shape) == 3:
            h, w, _ = image.shape
            luminance = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
        else:
            h, w = image.shape
            luminance = image

        for _ in range(max_iterations):
            # Define patch around the current point
            xmin, xmax = max(0, x - half_patch), min(w, x + half_patch + 1)
            ymin, ymax = max(0, y - half_patch), min(h, y + half_patch + 1)
            patch = luminance[ymin:ymax, xmin:xmax]
            current_value = np.median(patch)

            # Define a 3x3 window around the point
            neighbors = [
                (nx, ny) for nx in range(max(0, x - 1), min(w, x + 2))
                          for ny in range(max(0, y - 1), min(h, y + 2))
                          if (nx, ny) != (x, y)
            ]

            # Find the dimmest neighbor using patch medians
            def patch_median(coord):
                nx, ny = coord
                xmin_n, xmax_n = max(0, nx - half_patch), min(w, nx + half_patch + 1)
                ymin_n, ymax_n = max(0, ny - half_patch), min(h, ny + half_patch + 1)
                neighbor_patch = luminance[ymin_n:ymax_n, xmin_n:xmax_n]
                return np.median(neighbor_patch)

            dimmest_neighbor = min(neighbors, key=patch_median)
            dimmest_value = patch_median(dimmest_neighbor)

            # If the current point is already the dimmest, stop
            if dimmest_value >= current_value:
                break

            # Move to the dimmest neighbor
            x, y = dimmest_neighbor

        return x, y

    def fit_polynomial_gradient(self, image, sample_points, degree=2, patch_size=15):
        """
        Fits a polynomial gradient (up to the specified degree) to the image using sample points.

        Args:
            image: 2D/3D NumPy array of the image.
            sample_points: Array of (x, y) sample point coordinates.
            degree: Degree of the polynomial (e.g., 1 for linear, 2 for quadratic).
            patch_size: Size of the square patch for median calculation.

        Returns:
            background: The polynomial gradient model across the image.
        """
        h, w = image.shape[:2]
        half_patch = patch_size // 2

        x, y = sample_points[:, 0].astype(np.int32), sample_points[:, 1].astype(np.int32)
        valid_indices = (x >= 0) & (x < w) & (y >= 0) & (y < h)
        x, y = x[valid_indices], y[valid_indices]

        if len(image.shape) == 3:  # Color image
            background = np.zeros_like(image)
            for channel in range(image.shape[2]):  # Process each channel separately
                z = []
                for xi, yi in zip(x, y):
                    xmin, xmax = max(0, xi - half_patch), min(w, xi + half_patch + 1)
                    ymin, ymax = max(0, yi - half_patch), min(h, yi + half_patch + 1)
                    patch = image[ymin:ymax, xmin:xmax, channel]
                    z.append(np.median(patch))
                z = np.array(z, dtype=np.float64)

                # Fit polynomial model for this channel
                terms = []
                for i in range(degree + 1):
                    for j in range(degree + 1 - i):
                        terms.append((x**i) * (y**j))
                A = np.column_stack(terms)
                coeffs, _, _, _ = np.linalg.lstsq(A, z, rcond=None)

                # Generate polynomial model
                xx, yy = np.meshgrid(np.arange(w), np.arange(h))
                terms = []
                for i in range(degree + 1):
                    for j in range(degree + 1 - i):
                        terms.append((xx**i) * (yy**j))
                terms = np.array(terms)
                background[:, :, channel] = np.sum(coeffs[:, None, None] * terms, axis=0)
            return background
        else:  # Grayscale image
            return self.fit_polynomial_gradient(image[:, :, np.newaxis], sample_points, degree, patch_size)

    def generate_sample_points(self, image, num_points=100, exclusion_mask=None):
        """
        Generates sample points for gradient fitting, avoiding exclusion zones.

        Args:
            image: 2D/3D NumPy array of the image.
            num_points: Total number of sample points to generate.
            exclusion_mask: 2D boolean NumPy array where False indicates exclusion.

        Returns:
            points: NumPy array of shape (N, 2) with (x, y) coordinates.
        """
        h, w = image.shape[:2]
        points = []

        # Add border points: 1 in each corner and 5 along each border
        border_margin = 10

        # Corner points
        corners = [
            (border_margin, border_margin),                # Top-left
            (w - border_margin - 1, border_margin),        # Top-right
            (border_margin, h - border_margin - 1),        # Bottom-left
            (w - border_margin - 1, h - border_margin - 1) # Bottom-right
        ]
        for x, y in corners:
            if exclusion_mask is not None and not exclusion_mask[y, x]:
                continue
            x_new, y_new = self.gradient_descent_to_dim_spot(image, x, y)
            if exclusion_mask is not None and not exclusion_mask[y_new, x_new]:
                continue
            points.append((x_new, y_new))

        # Top and bottom borders
        for x in np.linspace(border_margin, w - border_margin, 5, dtype=int):
            # Top border
            if exclusion_mask is not None and not exclusion_mask[border_margin, x]:
                continue
            x_top, y_top = self.gradient_descent_to_dim_spot(image, x, border_margin)
            if exclusion_mask is not None and not exclusion_mask[y_top, x_top]:
                continue
            points.append((x_top, y_top))
            # Bottom border
            if exclusion_mask is not None and not exclusion_mask[h - border_margin - 1, x]:
                continue
            x_bottom, y_bottom = self.gradient_descent_to_dim_spot(image, x, h - border_margin - 1)
            if exclusion_mask is not None and not exclusion_mask[y_bottom, x_bottom]:
                continue
            points.append((x_bottom, y_bottom))

        # Left and right borders
        for y in np.linspace(border_margin, h - border_margin, 5, dtype=int):
            # Left border
            if exclusion_mask is not None and not exclusion_mask[y, border_margin]:
                continue
            x_left, y_left = self.gradient_descent_to_dim_spot(image, border_margin, y)
            if exclusion_mask is not None and not exclusion_mask[y_left, x_left]:
                continue
            points.append((x_left, y_left))
            # Right border
            if exclusion_mask is not None and not exclusion_mask[y, w - border_margin - 1]:
                continue
            x_right, y_right = self.gradient_descent_to_dim_spot(image, w - border_margin - 1, y)
            if exclusion_mask is not None and not exclusion_mask[y_right, x_right]:
                continue
            points.append((x_right, y_right))

        # Add random points in eligible areas (using quartiles)
        quartiles = self.divide_into_quartiles(image)
        for key, quartile in quartiles.items():
            # Determine the coordinates of the quartile in the full image
            h_quart, w_quart = quartile.shape[:2]
            if "top" in key:
                y_start = 0
            else:
                y_start = h // 2
            if "left" in key:
                x_start = 0
            else:
                x_start = w // 2

            # Create local exclusion mask for the quartile
            if exclusion_mask is not None:
                quart_exclusion_mask = exclusion_mask[y_start:y_start + h_quart, x_start:x_start + w_quart]
            else:
                quart_exclusion_mask = None

            # Convert quartile to grayscale if it has multiple channels
            if quartile.ndim == 3:
                # Assuming the color channels are last, convert to luminance
                quartile_gray = np.dot(quartile[..., :3], [0.2989, 0.5870, 0.1140])
            else:
                quartile_gray = quartile

            # Exclude bright regions
            mask = self.exclude_bright_regions(quartile_gray, exclusion_fraction=0.5)
            if quart_exclusion_mask is not None:
                mask &= quart_exclusion_mask

            eligible_indices = np.argwhere(mask)

            if len(eligible_indices) == 0:
                continue  # Skip if no eligible points in this quartile

            # Ensure we don't request more points than available
            num_points_in_quartile = min(len(eligible_indices), self.num_sample_points // 4)
            selected_indices = eligible_indices[np.random.choice(len(eligible_indices), num_points_in_quartile, replace=False)]

            for idx in selected_indices:
                y_idx, x_idx = idx  # Unpack row to y, x
                y_coord = y_start + y_idx
                x_coord = x_start + x_idx

                # Apply gradient descent to move to a dimmer spot
                x_new, y_new = self.gradient_descent_to_dim_spot(image, x_coord, y_coord)

                # Check if the new point is in exclusion
                if exclusion_mask is not None and not exclusion_mask[y_new, x_new]:
                    continue  # Skip points in exclusion areas

                points.append((x_new, y_new))

        return np.array(points)

    def fit_background(self, image, sample_points, smooth=0.1, patch_size=15):
        """
        Fits a background model using RBF interpolation.

        Args:
            image: 2D/3D NumPy array of the image.
            sample_points: Array of (x, y) sample point coordinates.
            smooth: Smoothness parameter for the RBF fitting.
            patch_size: Size of the square patch for median calculation.

        Returns:
            background: The RBF-based background model.
        """
        h, w = image.shape[:2]
        half_patch = patch_size // 2

        x, y = sample_points[:, 0].astype(np.int32), sample_points[:, 1].astype(np.int32)
        valid_indices = (x >= 0) & (x < w) & (y >= 0) & (y < h)
        x, y = x[valid_indices], y[valid_indices]

        if len(image.shape) == 3:  # Color image
            background = np.zeros_like(image)
            for channel in range(image.shape[2]):  # Process each channel separately
                z = []
                for xi, yi in zip(x, y):
                    xmin, xmax = max(0, xi - half_patch), min(w, xi + half_patch + 1)
                    ymin, ymax = max(0, yi - half_patch), min(h, yi + half_patch + 1)
                    patch = image[ymin:ymax, xmin:xmax, channel]
                    z.append(np.median(patch))
                z = np.array(z, dtype=np.float64)

                # Fit RBF for this channel
                rbf = Rbf(x, y, z, function='multiquadric', smooth=smooth, epsilon=1.0)
                grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
                background[:, :, channel] = rbf(grid_x, grid_y)
            return background
        else:  # Grayscale image
            return self.fit_background(image[:, :, np.newaxis], sample_points, smooth, patch_size)

    def calculate_median(self, values):
        """
        Calculates the median of the given values.

        Args:
            values: NumPy array of values.

        Returns:
            median: Median value.
        """
        return np.median(values)

    def calculate_mad(self, values, median):
        """
        Calculates the Median Absolute Deviation (MAD).

        Args:
            values: NumPy array of values.
            median: Median of the values.

        Returns:
            mad: Median Absolute Deviation.
        """
        deviations = np.abs(values - median)
        return np.median(deviations)

    def calculate_noise_weight(self, median, mad):
        """
        Calculates the noise weight based on median and MAD.

        Args:
            median: Median value.
            mad: Median Absolute Deviation.

        Returns:
            noise_weight: Noise weight (0.0 to 1.0).
        """
        if median == 0:
            median = 1e-6  # Avoid division by zero
        noise_factor = 1.0 - (mad / median)
        return max(0.0, min(1.0, noise_factor))

    def calculate_brightness_weight(self, avg_brightness, median_brightness):
        """
        Calculates the brightness weight based on average and median brightness.

        Args:
            avg_brightness: Average brightness of the patch.
            median_brightness: Median brightness of the patch.

        Returns:
            brightness_weight: Brightness weight (0.8 to 1.0).
        """
        if median_brightness == 0:
            median_brightness = 1e-6  # Avoid division by zero
        weight = 1.0 - abs(avg_brightness - median_brightness) / median_brightness
        return max(0.8, min(1.0, weight))  # Limit range for stability

    def calculate_spatial_weight(self, x, y, width, height):
        """
        Calculates the spatial weight based on the position of the point.

        Args:
            x: X-coordinate.
            y: Y-coordinate.
            width: Image width.
            height: Image height.

        Returns:
            spatial_weight: Spatial weight (0.95 to 1.0).
        """
        center_x = width / 2
        center_y = height / 2
        distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        max_distance = np.sqrt(center_x ** 2 + center_y ** 2)
        normalized_distance = distance / max_distance
        return 0.95 + 0.05 * normalized_distance

    def create_exclusion_mask(self, image_shape, exclusion_polygons):
        """
        Creates a boolean mask with False in exclusion areas and True elsewhere.

        Args:
            image_shape: Shape of the image (height, width, channels).
            exclusion_polygons: List of QPolygon objects.

        Returns:
            mask: 2D boolean NumPy array.
        """
        mask = np.ones(image_shape[:2], dtype=bool)  # Initialize all True

        if not exclusion_polygons:
            return mask  # No exclusions

        # Prepare polygons for OpenCV
        polygons = []
        for polygon in exclusion_polygons:
            points = []
            for point in polygon:
                # Scale back to original image coordinates
                x_original = point.x() / self.scale_factor
                y_original = point.y() / self.scale_factor
                points.append([int(x_original), int(y_original)])
            polygons.append(np.array(points, dtype=np.int32))

        # Create a single-channel mask
        exclusion_mask = np.zeros(image_shape[:2], dtype=np.uint8)

        # Fill the polygons on the exclusion mask
        cv2.fillPoly(exclusion_mask, polygons, 1)  # 1 inside polygons

        # Update the main mask: False inside exclusion polygons
        mask[exclusion_mask == 1] = False

        return mask

    def normalize_image(self, image, target_median):
        """
        Normalizes the image so that its median matches the target median.

        Args:
            image: 2D/3D NumPy array of the image.
            target_median: The desired median value.

        Returns:
            normalized_image: The median-normalized image.
        """
        current_median = np.median(image)
        median_diff = target_median - current_median
        normalized_image = image + median_diff
        return normalized_image


class ImagePreview(QWidget):
    # Define a custom signal that emits the slot number
    closed = pyqtSignal(int)
    
    def __init__(self, image_data, slot, parent=None):
        super().__init__(parent, Qt.Window)
        self.setWindowTitle(f"Preview - Slot {slot}")
        self.image_data = image_data  # Numpy array containing the image
        self.zoom_factor = 1.0
        self.slot = slot
        self.is_autostretched = False  # Track if AutoStretch is applied
        self.stretched_image_data = None  # Store stretched image data for visual purposes

        # Create UI components
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidget(self.image_label)
        self.scroll_area.setAlignment(Qt.AlignCenter)
        self.scroll_area.setWidgetResizable(True)
        
        # Install event filter on the scroll area’s viewport
        self.scroll_area.viewport().installEventFilter(self)

        # Convert numpy image data to QImage and display it
        self.update_image_display()

        # Create Zoom controls
        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setRange(1, 400)  # Zoom range from 1% to 400%
        self.zoom_slider.setValue(100)  # Default zoom (100%)
        self.zoom_slider.valueChanged.connect(self.on_zoom_changed)

        self.zoom_in_button = QPushButton("Zoom In")
        self.zoom_in_button.clicked.connect(lambda: self.adjust_zoom(10))

        self.zoom_out_button = QPushButton("Zoom Out")
        self.zoom_out_button.clicked.connect(lambda: self.adjust_zoom(-10))

        self.fit_to_preview_button = QPushButton("Fit to Preview")
        self.fit_to_preview_button.clicked.connect(self.fit_to_preview)

        # AutoStretch button
        self.autostretch_button = QPushButton("AutoStretch")
        self.autostretch_button.clicked.connect(self.apply_autostretch)

        # Create Swap Button (visible for all slots except Slot 0)
        if self.slot != 0:
            self.swap_button = QPushButton(f"Swap with Slot 0")
            self.swap_button.clicked.connect(self.swap_with_slot_zero)
            swap_layout = QHBoxLayout()
            swap_layout.addStretch()
            swap_layout.addWidget(self.swap_button)
        else:
            swap_layout = QHBoxLayout()  # Empty layout

        # Layout for zoom controls
        zoom_layout = QHBoxLayout()
        zoom_layout.addWidget(self.zoom_out_button)
        zoom_layout.addWidget(self.zoom_slider)
        zoom_layout.addWidget(self.zoom_in_button)
        zoom_layout.addWidget(self.fit_to_preview_button)
        zoom_layout.addWidget(self.autostretch_button)

        # Main layout
        layout = QVBoxLayout(self)
        layout.addWidget(self.scroll_area)
        layout.addLayout(zoom_layout)
        layout.addLayout(swap_layout)  # Add swap button layout
        self.setLayout(layout)

        # Variables to handle panning
        self._panning = False
        self._pan_start_x = 0
        self._pan_start_y = 0

    def eventFilter(self, source, event):
        """
        Intercept mouse events on the scroll area's viewport to implement panning.
        """
        if source == self.scroll_area.viewport():
            if event.type() == QEvent.MouseButtonPress:
                if event.button() == Qt.LeftButton:
                    self._panning = True
                    self._pan_start_x = event.x()
                    self._pan_start_y = event.y()
                    self.scroll_area.viewport().setCursor(Qt.ClosedHandCursor)
                    return True  # Event handled
            elif event.type() == QEvent.MouseMove:
                if self._panning and (event.buttons() & Qt.LeftButton):
                    delta_x = event.x() - self._pan_start_x
                    delta_y = event.y() - self._pan_start_y
                    # Adjust scroll bars
                    new_h = self.scroll_area.horizontalScrollBar().value() - delta_x
                    new_v = self.scroll_area.verticalScrollBar().value() - delta_y
                    self.scroll_area.horizontalScrollBar().setValue(new_h)
                    self.scroll_area.verticalScrollBar().setValue(new_v)
                    # Update the start position
                    self._pan_start_x = event.x()
                    self._pan_start_y = event.y()
                    return True  # Event handled
            elif event.type() == QEvent.MouseButtonRelease:
                if event.button() == Qt.LeftButton:
                    self._panning = False
                    self.scroll_area.viewport().setCursor(Qt.ArrowCursor)
                    return True  # Event handled
        return super().eventFilter(source, event)

    def apply_autostretch(self):
        """Applies AutoStretch to the displayed image for visualization."""
        if self.is_autostretched:
            # If already stretched, reset to the original image
            self.is_autostretched = False
            self.update_image_display()
        else:
            # Perform AutoStretch and display it
            self.is_autostretched = True
            self.stretched_image_data = self.stretch_image(self.image_data)
            self.update_image_display()

    def stretch_image(self, image):
        """
        Perform an unlinked linear stretch on the image.
        Each channel is stretched independently to enhance visualization.
        """
        if image.ndim == 2:  # Grayscale image
            channels = [image]
        else:  # RGB image
            channels = [image[..., i] for i in range(image.shape[2])]

        stretched_channels = []
        for channel in channels:
            channel_min = np.min(channel)
            channel_max = np.max(channel)
            stretched_channel = (channel - channel_min) / (channel_max - channel_min + 1e-6)
            stretched_channels.append(stretched_channel)

        if len(stretched_channels) == 1:
            return stretched_channels[0]  # Return grayscale
        else:
            return np.stack(stretched_channels, axis=-1)  # Return RGB

    def update_image_display(self):
        """Update the QLabel with the current image."""
        # Use the stretched image data if AutoStretch is applied
        display_image = self.stretched_image_data if self.is_autostretched else self.image_data

        # Normalize image data to [0, 255] and convert to uint8
        if display_image.dtype != np.uint8:
            image_data_normalized = np.clip(display_image * 255, 0, 255).astype('uint8')
        else:
            image_data_normalized = display_image

        if len(image_data_normalized.shape) == 2:  # Grayscale image
            height, width = image_data_normalized.shape
            bytes_per_line = width
            qimage = QImage(image_data_normalized.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        elif len(image_data_normalized.shape) == 3 and image_data_normalized.shape[2] == 3:  # RGB image
            height, width, channels = image_data_normalized.shape
            bytes_per_line = 3 * width
            qimage = QImage(image_data_normalized.data, width, height, bytes_per_line, QImage.Format_RGB888)
        else:
            QMessageBox.warning(self, "Invalid Image", "Unsupported image format for display.")
            return

        pixmap = QPixmap.fromImage(qimage)
        scaled_pixmap = pixmap.scaled(
            self.image_label.size() * self.zoom_factor,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.image_label.setPixmap(scaled_pixmap)

    def resizeEvent(self, event):
        """Ensure the image scales appropriately when the window is resized."""
        self.update_image_display()
        super().resizeEvent(event)

    def on_zoom_changed(self, value):
        """Handle changes in zoom slider."""
        self.zoom_factor = value / 100.0  # Convert slider value to zoom factor
        self.update_image_display()

    def adjust_zoom(self, delta):
        """Adjust zoom by a specified delta."""
        new_value = self.zoom_slider.value() + delta
        self.zoom_slider.setValue(max(1, min(400, new_value)))

    def fit_to_preview(self):
        """Fit the image to the preview window."""
        self.zoom_factor = 1.0
        self.zoom_slider.setValue(100)
        self.update_image_display()

    def swap_with_slot_zero(self):
        """Swap images between the current slot and Slot 0."""
        confirmation = QMessageBox.question(
            self,
            "Confirm Swap",
            f"Are you sure you want to swap Slot {self.slot} with Slot 0?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if confirmation == QMessageBox.Yes:
            # Debug: Print parent details
            print(f"Attempting to swap Slot {self.slot} with Slot 0.")
            print(f"Parent: {self.parent()}, Type: {type(self.parent())}")
            print(f"Does parent have 'swap_slots'? {'Yes' if hasattr(self.parent(), 'swap_slots') else 'No'}")
            
            # Call the swap_slots method in the parent (AstroEditingSuite)
            if self.parent() and hasattr(self.parent(), 'swap_slots'):
                self.parent().swap_slots(self.slot, 0)

                # Optionally, close the preview window after swapping
                self.close()
            else:
                QMessageBox.critical(self, "Error", "Parent does not have a swap_slots method.")
                print("Error: Parent does not have a swap_slots method.")

    def closeEvent(self, event):
        """Override the close event to emit the custom closed signal."""
        self.closed.emit(self.slot)  # Emit the slot number
        event.accept()  # Proceed with the standard close event

class GraXpertThread(QThread):
    """Thread to execute GraXpert commands."""
    stdout_signal = pyqtSignal(str)
    stderr_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(int)

    def __init__(self, command):
        super().__init__()
        self.command = command

    def run(self):
        """Run the GraXpert command and capture output."""
        process = subprocess.Popen(
            self.command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        for line in process.stdout:
            self.stdout_signal.emit(line.strip())
        for line in process.stderr:
            self.stderr_signal.emit(line.strip())
        self.finished_signal.emit(process.wait())

class RGBCombinationDialog(QDialog):
    def __init__(self, parent=None, image_manager=None):
        super().__init__(parent)
        self.setWindowTitle("RGB Combination")
        self.setModal(True)
        self.image_manager = image_manager  # Reference to ImageManager
        
        self.r_image_path = None
        self.g_image_path = None
        self.b_image_path = None
        self.use_existing_slots = False
        
        # Create UI components
        self.mode_label = QLabel("Select RGB Combination Mode:")
        
        # Radio buttons for mode selection
        self.load_files_radio = QRadioButton("Load Individual Files")
        self.use_slots_radio = QRadioButton("Use Existing Slots (2, 3, 4)")
        self.load_files_radio.setChecked(True)  # Default mode
        
        # Button group to ensure only one radio button is selected
        self.mode_group = QButtonGroup()
        self.mode_group.addButton(self.load_files_radio)
        self.mode_group.addButton(self.use_slots_radio)
        self.mode_group.buttonClicked.connect(self.update_mode)
        
        # GroupBox for mode selection
        self.mode_groupbox = QGroupBox()
        mode_layout = QVBoxLayout()
        mode_layout.addWidget(self.load_files_radio)
        mode_layout.addWidget(self.use_slots_radio)
        self.mode_groupbox.setLayout(mode_layout)
        
        # Load File Mode Widgets
        self.load_r_button = QPushButton("Load Red Image")
        self.load_r_button.clicked.connect(self.load_r_image)
        
        self.load_g_button = QPushButton("Load Green Image")
        self.load_g_button.clicked.connect(self.load_g_image)
        
        self.load_b_button = QPushButton("Load Blue Image")
        self.load_b_button.clicked.connect(self.load_b_image)
        
        self.r_label = QLabel("Red Image: Not Selected")
        self.g_label = QLabel("Green Image: Not Selected")
        self.b_label = QLabel("Blue Image: Not Selected")
        
        # Layout for Load Files Mode
        self.load_files_layout = QVBoxLayout()
        self.load_files_layout.addWidget(self.r_label)
        self.load_files_layout.addWidget(self.load_r_button)
        self.load_files_layout.addWidget(self.g_label)
        self.load_files_layout.addWidget(self.load_g_button)
        self.load_files_layout.addWidget(self.b_label)
        self.load_files_layout.addWidget(self.load_b_button)
        
        # Use Existing Slots Mode Widgets
        self.use_slots_label = QLabel("Ensure Slots 2, 3, and 4 contain R, G, B channels respectively.")
        self.use_slots_button = QPushButton("Use Slots 2, 3, 4")
        self.use_slots_button.clicked.connect(self.use_existing_slots_method)
        
        # Layout for Use Slots Mode
        self.use_slots_layout = QVBoxLayout()
        self.use_slots_layout.addWidget(self.use_slots_label)
        self.use_slots_layout.addWidget(self.use_slots_button)
        
        # Combine and Cancel buttons
        self.combine_button = QPushButton("Combine")
        self.combine_button.clicked.connect(self.combine_images)
        self.combine_button.setEnabled(False)  # Disabled until required inputs are available
        
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        
        # Layout for buttons
        buttons_layout = QHBoxLayout()
        buttons_layout.addStretch()
        buttons_layout.addWidget(self.combine_button)
        buttons_layout.addWidget(self.cancel_button)
        
        # Main Layout
        self.main_layout = QVBoxLayout()
        self.main_layout.addWidget(self.mode_label)
        self.main_layout.addWidget(self.mode_groupbox)
        self.main_layout.addLayout(self.load_files_layout)
        self.main_layout.addLayout(self.use_slots_layout)
        self.main_layout.addLayout(buttons_layout)
        
        self.setLayout(self.main_layout)
    
    def update_mode(self):
        """Update the UI based on the selected mode."""
        if self.load_files_radio.isChecked():
            self.use_existing_slots = False
            self.load_r_button.setEnabled(True)
            self.load_g_button.setEnabled(True)
            self.load_b_button.setEnabled(True)
            self.r_label.setEnabled(True)
            self.g_label.setEnabled(True)
            self.b_label.setEnabled(True)
            self.use_slots_button.setEnabled(False)
        else:
            self.use_existing_slots = True
            self.load_r_button.setEnabled(False)
            self.load_g_button.setEnabled(False)
            self.load_b_button.setEnabled(False)
            self.r_label.setEnabled(False)
            self.g_label.setEnabled(False)
            self.b_label.setEnabled(False)
            self.use_slots_button.setEnabled(True)
        
        self.check_inputs()
    
    def load_r_image(self):
        """Load the Red channel image."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Red Image", "", 
            "Image Files (*.png *.tif *.tiff *.fits *.fit *.xisf *.jpg *.jpeg);;All Files (*)"
        )
        if file_path:
            self.r_image_path = file_path
            self.r_label.setText(f"Red Image: {os.path.basename(file_path)}")
            self.check_inputs()
    
    def load_g_image(self):
        """Load the Green channel image."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Green Image", "", 
            "Image Files (*.png *.tif *.tiff *.fits *.fit *.xisf *.jpg *.jpeg);;All Files (*)"
        )
        if file_path:
            self.g_image_path = file_path
            self.g_label.setText(f"Green Image: {os.path.basename(file_path)}")
            self.check_inputs()
    
    def load_b_image(self):
        """Load the Blue channel image."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Blue Image", "", 
            "Image Files (*.png *.tif *.tiff *.fits *.fit *.xisf *.jpg *.jpeg);;All Files (*)"
        )
        if file_path:
            self.b_image_path = file_path
            self.b_label.setText(f"Blue Image: {os.path.basename(file_path)}")
            self.check_inputs()
    
    def use_existing_slots_method(self):
        """Use existing images from slots 2, 3, and 4."""
        # Check if slots 2, 3, and 4 have images
        slots = [2, 3, 4]
        images = []
        for slot in slots:
            img = self.image_manager._images.get(slot, None)
            if img is None:
                QMessageBox.warning(
                    self, 
                    "Missing Image", 
                    f"Slot {slot} does not contain an image. Please extract RGB channels first."
                )
                print(f"Slot {slot} is empty. Cannot use existing slots for RGB Combination.")
                return
            images.append(img)
        
        self.r_image_path = None  # Indicate that we're using existing slots
        self.g_image_path = None
        self.b_image_path = None
        self.combine_button.setEnabled(True)  # Enable Combine button as inputs are ready
    
    def check_inputs(self):
        """Enable the Combine button if all required inputs are available."""
        if self.use_existing_slots:
            # Check if slots 2,3,4 have images
            slots = [2, 3, 4]
            for slot in slots:
                if self.image_manager._images.get(slot, None) is None:
                    self.combine_button.setEnabled(False)
                    return
            self.combine_button.setEnabled(True)
        else:
            # Check if all three images are loaded
            if self.r_image_path and self.g_image_path and self.b_image_path:
                self.combine_button.setEnabled(True)
            else:
                self.combine_button.setEnabled(False)
    
    def combine_images(self):
        """Combine the loaded R, G, B images into a single RGB image."""
        try:
            if self.use_existing_slots:
                # Use images from slots 2, 3, 4
                r = self.image_manager._images.get(2).copy()
                g = self.image_manager._images.get(3).copy()
                b = self.image_manager._images.get(4).copy()
                
                # Ensure all images have the same dimensions
                if not (r.shape == g.shape == b.shape):
                    raise ValueError("All images must have the same dimensions.")
            else:
                # Load images from file paths
                r = cv2.imread(self.r_image_path, cv2.IMREAD_GRAYSCALE)
                g = cv2.imread(self.g_image_path, cv2.IMREAD_GRAYSCALE)
                b = cv2.imread(self.b_image_path, cv2.IMREAD_GRAYSCALE)
                
                if r is None or g is None or b is None:
                    raise ValueError("One or more images failed to load.")
                
                # Ensure all images have the same dimensions
                if not (r.shape == g.shape == b.shape):
                    raise ValueError("All images must have the same dimensions.")
                
                # Normalize images to [0,1]
                r = r.astype('float32') / 255.0
                g = g.astype('float32') / 255.0
                b = b.astype('float32') / 255.0
            
            # Stack channels to form RGB image
            rgb_image = np.stack([r, g, b], axis=2)
            
            # Check for grayscale images stored as RGB (all channels same)
            if np.array_equal(r, g) and np.array_equal(r, b):
                print("Detected grayscale image stored as RGB. Using only the red channel.")
                rgb_image = np.stack([r, np.zeros_like(r), np.zeros_like(r)], axis=2)
            
            self.rgb_image = rgb_image  # Store the combined image
            self.accept()  # Close the dialog with success
            print("RGB Combination successful.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to combine images: {e}")
            print(f"Error in RGB Combination: {e}")

class StarNetThread(QThread):
    # Define signals to communicate with the main thread
    stdout_signal = pyqtSignal(str)
    stderr_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(int)  # Emit return code

    def __init__(self, command, cwd):
        super().__init__()
        self.command = command
        self.cwd = cwd
        self._process = None  # To handle process termination

    def run(self):
        try:
            # Start the StarNet process
            self._process = subprocess.Popen(
                self.command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=self.cwd,
                bufsize=1,
                universal_newlines=True
            )

            # Read stdout and stderr in real-time
            while True:
                output = self._process.stdout.readline()
                if output:
                    self.stdout_signal.emit(output.strip())
                elif self._process.poll() is not None:
                    break

            # Capture remaining stdout
            remaining_stdout, remaining_stderr = self._process.communicate()
            if remaining_stdout:
                self.stdout_signal.emit(remaining_stdout.strip())
            if remaining_stderr:
                self.stderr_signal.emit(remaining_stderr.strip())

            # Emit the return code
            self.finished_signal.emit(self._process.returncode)

        except Exception as e:
            self.stderr_signal.emit(str(e))
            self.finished_signal.emit(-1)

    def stop(self):
        if self._process and self._process.poll() is None:
            self._process.terminate()
            self.wait()

class StarNetDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("StarNet Progress")
        self.setMinimumSize(600, 400)
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        layout.addWidget(self.text_edit)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.cancel_process)
        layout.addWidget(self.cancel_button)

        self.setLayout(layout)

    def append_text(self, text):
        self.text_edit.append(text)
        # Auto-scroll to the bottom
        self.text_edit.verticalScrollBar().setValue(self.text_edit.verticalScrollBar().maximum())

    def cancel_process(self):
        self.reject()  # Close the dialog

class BackgroundNeutralizationDialog(QDialog):
    def __init__(self, image_manager, parent=None):
        super().__init__(parent)
        self.image_manager = image_manager
        self.setWindowTitle("Background Neutralization")
        self.setGeometry(150, 150, 800, 600)  # Set appropriate size
        self.initUI()
        self.selection_rect_item = None  # To store the QGraphicsRectItem

    def initUI(self):
        layout = QVBoxLayout()

        # Instruction Label
        instruction_label = QLabel("Draw a sample box on the image to define the neutralization region.")
        layout.addWidget(instruction_label)

        # Graphics View for Image Display
        self.graphics_view = QGraphicsView()
        self.scene = QGraphicsScene()
        self.graphics_view.setScene(self.scene)
        layout.addWidget(self.graphics_view)

        # Load and Display Image
        self.load_image()

        # Initialize Variables for Drawing
        self.origin = QPointF()
        self.current_rect = QRectF()
        self.drawing = False

        # Connect Mouse Events
        self.graphics_view.viewport().installEventFilter(self)

        # Apply and Cancel Buttons
        button_layout = QVBoxLayout()
        apply_button = QPushButton("Apply Neutralization")
        apply_button.clicked.connect(self.apply_neutralization)
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(apply_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)

        self.setLayout(layout)

    def load_image(self):
        """Loads the current image from the ImageManager and displays it."""
        image = self.image_manager.image
        if image is not None:
            # Assuming image is a NumPy array normalized to [0,1]
            height, width, channels = image.shape
            if channels == 3:
                q_image = QImage(
                    (image * 255).astype(np.uint8).tobytes(),
                    width,
                    height,
                    3 * width,
                    QImage.Format_RGB888
                )
            else:
                # Handle other channel numbers if necessary
                q_image = QImage(
                    (image * 255).astype(np.uint8).tobytes(),
                    width,
                    height,
                    QImage.Format_Grayscale8
                )
            pixmap = QPixmap.fromImage(q_image)
            self.pixmap_item = QGraphicsPixmapItem(pixmap)
            self.scene.addItem(self.pixmap_item)
            self.graphics_view.fitInView(self.pixmap_item, Qt.KeepAspectRatio)
        else:
            QMessageBox.warning(self, "No Image", "No image loaded to neutralize.")
            self.reject()

    def eventFilter(self, source, event):
        """Handles mouse events for drawing the sample box."""
        if source is self.graphics_view.viewport():
            if event.type() == event.MouseButtonPress:
                if event.button() == Qt.LeftButton:
                    self.drawing = True
                    self.origin = self.graphics_view.mapToScene(event.pos())
                    # Remove existing selection rectangle if any
                    if self.selection_rect_item:
                        self.scene.removeItem(self.selection_rect_item)
                        self.selection_rect_item = None
            elif event.type() == event.MouseMove:
                if self.drawing:
                    current_pos = self.graphics_view.mapToScene(event.pos())
                    self.current_rect = QRectF(self.origin, current_pos).normalized()
                    # Remove existing rectangle item if any
                    if self.selection_rect_item:
                        self.scene.removeItem(self.selection_rect_item)
                        self.selection_rect_item = None
                    # Draw the new rectangle
                    pen = QPen(QColor(0, 255, 0), 2, Qt.DashLine)
                    self.selection_rect_item = QGraphicsRectItem(self.current_rect)
                    self.selection_rect_item.setPen(pen)
                    self.scene.addItem(self.selection_rect_item)
            elif event.type() == event.MouseButtonRelease:
                if event.button() == Qt.LeftButton and self.drawing:
                    self.drawing = False
                    # Finalize the rectangle
                    current_pos = self.graphics_view.mapToScene(event.pos())
                    self.current_rect = QRectF(self.origin, current_pos).normalized()
                    # Ensure minimum size to avoid accidental small selections
                    min_size = 10  # pixels
                    if self.current_rect.width() < min_size or self.current_rect.height() < min_size:
                        QMessageBox.warning(self, "Selection Too Small", "Please draw a larger selection box.")
                        if self.selection_rect_item:
                            self.scene.removeItem(self.selection_rect_item)
                            self.selection_rect_item = None
                        self.current_rect = QRectF()
                    else:
                        # Redraw the rectangle to ensure it's persistent
                        if self.selection_rect_item:
                            self.scene.removeItem(self.selection_rect_item)
                        pen = QPen(QColor(255, 0, 0), 2, Qt.SolidLine)
                        self.selection_rect_item = QGraphicsRectItem(self.current_rect)
                        self.selection_rect_item.setPen(pen)
                        self.scene.addItem(self.selection_rect_item)
        return super().eventFilter(source, event)

    def apply_neutralization(self):
        """Applies background neutralization based on the selected sample region."""
        if self.current_rect.isNull():
            QMessageBox.warning(self, "No Selection", "Please draw a sample box on the image.")
            return

        # Map the selection rectangle to image coordinates
        image = self.image_manager.image
        if image is None:
            QMessageBox.warning(self, "No Image", "No image loaded to neutralize.")
            return

        # Get the image dimensions
        height, width, channels = image.shape

        # Calculate scaling factors
        scene_rect = self.scene.sceneRect()
        scale_x = width / scene_rect.width()
        scale_y = height / scene_rect.height()

        # Convert scene coordinates to image coordinates
        x = int(self.current_rect.left() * scale_x)
        y = int(self.current_rect.top() * scale_y)
        w = int(self.current_rect.width() * scale_x)
        h = int(self.current_rect.height() * scale_y)

        # Ensure the rectangle is within image bounds
        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))
        w = max(1, min(w, width - x))
        h = max(1, min(h, height - y))

        sample_region = image[y:y + h, x:x + w, :]  # Extract the sample region

        # Calculate medians for each channel
        medians = np.median(sample_region, axis=(0, 1))  # Shape: (3,)
        average_median = np.mean(medians)

        # Calculate adjustments: average_median - channel_median
        adjustments = average_median - medians  # Shape: (3,)

        # Apply adjustments to the entire image
        adjusted_image = image.copy()
        for channel in range(3):
            delta = adjustments[channel]
            if delta > 0:
                # Need to subtract delta from the channel
                adjusted_image[:, :, channel] -= delta
                # Prevent division by zero or negative values
                adjusted_image[:, :, channel] = np.clip(adjusted_image[:, :, channel], 0.0, 1.0)
                adjusted_image[:, :, channel] /= (1.0 - delta) if (1.0 - delta) != 0 else 1.0
            elif delta < 0:
                # Need to add delta (since delta is negative, this is subtraction)
                adjusted_image[:, :, channel] += delta
                adjusted_image[:, :, channel] = np.clip(adjusted_image[:, :, channel], 0.0, 1.0)
                adjusted_image[:, :, channel] /= (1.0 - delta) if (1.0 - delta) != 0 else 1.0
            # If delta == 0, no change

        # Update the image in ImageManager
        self.image_manager.update_image(
            updated_image=adjusted_image,
            metadata=self.image_manager._metadata[self.image_manager.current_slot]
        )

        # Inform the user
        QMessageBox.information(self, "Success", "Background neutralization applied successfully.")

        # Close the dialog
        self.accept()
        
class RemoveGreenDialog(QDialog):
    def __init__(self, image_manager, parent=None):
        super().__init__(parent)
        self.image_manager = image_manager
        self.setWindowTitle("Remove Green")
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        # Instruction Label
        instruction_label = QLabel("Select the amount to remove green noise:")
        layout.addWidget(instruction_label)

        # Slider Configuration
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(100)  # Represents 0.0 to 1.0
        self.slider.setValue(100)     # Default to 1.0
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.setTickInterval(10)
        self.slider.valueChanged.connect(self.update_label)
        layout.addWidget(self.slider)

        # Current Value Display
        self.value_label = QLabel("Amount: 1.00")
        layout.addWidget(self.value_label)

        # Buttons Layout
        button_layout = QHBoxLayout()
        apply_button = QPushButton("Apply")
        cancel_button = QPushButton("Cancel")
        apply_button.clicked.connect(self.apply)
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(apply_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)

        self.setLayout(layout)

    def update_label(self, value):
        amount = value / 100.0
        self.value_label.setText(f"Amount: {amount:.2f}")

    def apply(self):
        amount = self.slider.value() / 100.0

        if self.image_manager.image is not None:
            try:
                # Apply the global SCNR function
                new_image = apply_average_neutral_scnr(self.image_manager.image, amount=amount)

                # Update the ImageManager's current image with the processed image
                self.image_manager.update_image(updated_image=new_image, metadata=self.image_manager._metadata[self.image_manager.current_slot])

                # Inform the user of the successful operation
                QMessageBox.information(self, "Success", f"Remove Green applied with amount {amount:.2f}")

                # Close the dialog
                self.accept()
            except Exception as e:
                # Handle any errors during processing
                QMessageBox.critical(self, "Error", f"Failed to apply Remove Green:\n{e}")
        else:
            # Inform the user if no image is loaded
            QMessageBox.warning(self, "No Image", "No image loaded to apply Remove Green.")
            self.reject()

class CLAHEDialog(QDialog):
    def __init__(self, image_manager, parent=None):
        super().__init__(parent)
        self.image_manager = image_manager
        self.setWindowTitle("CLAHE")
        self.setGeometry(200, 200, 800, 500)  # Increased size for better layout
        self.initUI()
        self.current_zoom = 1.0  # Initial zoom level

    def initUI(self):
        main_layout = QVBoxLayout()

        # CLAHE Parameters Group
        parameters_group = QGroupBox("CLAHE Parameters")
        parameters_layout = QGridLayout()

        # Clip Limit Slider and Label
        clip_label = QLabel("Clip Limit:")
        self.clip_slider = QSlider(Qt.Horizontal)
        self.clip_slider.setMinimum(1)
        self.clip_slider.setMaximum(40)  # Represents 0.1 to 4.0
        self.clip_slider.setValue(20)     # Default 2.0
        self.clip_slider.setTickInterval(1)
        self.clip_slider.setTickPosition(QSlider.TicksBelow)
        self.clip_value_label = QLabel("2.0")  # Initial value
        self.clip_slider.setToolTip("Adjust the clip limit for contrast enhancement. Higher values increase contrast.")

        self.clip_slider.valueChanged.connect(self.update_clip_value)
        self.clip_slider.valueChanged.connect(self.debounce_preview)

        parameters_layout.addWidget(clip_label, 0, 0)
        parameters_layout.addWidget(self.clip_slider, 0, 1)
        parameters_layout.addWidget(self.clip_value_label, 0, 2)

        # Tile Grid Size Slider and Label
        tile_label = QLabel("Tile Grid Size:")
        self.tile_slider = QSlider(Qt.Horizontal)
        self.tile_slider.setMinimum(1)
        self.tile_slider.setMaximum(32)
        self.tile_slider.setValue(8)        # Default (8,8)
        self.tile_slider.setTickInterval(1)
        self.tile_slider.setTickPosition(QSlider.TicksBelow)
        self.tile_value_label = QLabel("8")  # Initial value
        self.tile_slider.setToolTip("Adjust the size of grid for histogram equalization. Larger values affect broader areas.")

        self.tile_slider.valueChanged.connect(self.update_tile_value)
        self.tile_slider.valueChanged.connect(self.debounce_preview)

        parameters_layout.addWidget(tile_label, 1, 0)
        parameters_layout.addWidget(self.tile_slider, 1, 1)
        parameters_layout.addWidget(self.tile_value_label, 1, 2)

        parameters_group.setLayout(parameters_layout)
        main_layout.addWidget(parameters_group)

        # Preview Area
        preview_group = QGroupBox("Preview")
        preview_layout = QVBoxLayout()

        # QGraphicsView and QGraphicsScene
        self.preview_view = QGraphicsView()
        self.preview_scene = QGraphicsScene()
        self.preview_view.setScene(self.preview_scene)

        self.preview_view.setDragMode(QGraphicsView.ScrollHandDrag)  # Enable panning
        self.preview_view.setFixedSize(800, 500)  # Fixed size to prevent resizing

        # Initialize QGraphicsPixmapItem
        self.pixmap_item = QGraphicsPixmapItem()
        self.preview_scene.addItem(self.pixmap_item)

        preview_layout.addWidget(self.preview_view)
        preview_group.setLayout(preview_layout)
        main_layout.addWidget(preview_group)

        # Zoom and Fit Buttons
        zoom_layout = QHBoxLayout()
        zoom_in_button = QPushButton("Zoom In (+)")
        zoom_in_button.setToolTip("Zoom in the preview image.")
        zoom_in_button.clicked.connect(self.zoom_in)

        zoom_out_button = QPushButton("Zoom Out (-)")
        zoom_out_button.setToolTip("Zoom out the preview image.")
        zoom_out_button.clicked.connect(self.zoom_out)

        fit_button = QPushButton("Fit to Preview")
        fit_button.setToolTip("Fit the image to the preview area.")
        fit_button.clicked.connect(self.fit_to_preview)

        zoom_layout.addStretch()
        zoom_layout.addWidget(zoom_in_button)
        zoom_layout.addWidget(zoom_out_button)
        zoom_layout.addWidget(fit_button)

        main_layout.addLayout(zoom_layout)

        # Apply, Reset, and Cancel Buttons
        button_layout = QHBoxLayout()
        apply_button = QPushButton("Apply")
        apply_button.clicked.connect(self.apply_clahe)
        reset_button = QPushButton("Reset")
        reset_button.clicked.connect(self.reset_parameters)
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_layout.addStretch()
        button_layout.addWidget(apply_button)
        button_layout.addWidget(reset_button)
        button_layout.addWidget(cancel_button)
        main_layout.addLayout(button_layout)

        self.setLayout(main_layout)

        # Initialize Debounce Timer
        self.debounce_timer = QTimer()
        self.debounce_timer.setSingleShot(True)
        self.debounce_timer.timeout.connect(self.update_preview)

        # Store Original Image
        if self.image_manager.image is not None:
            self.original_image = self.image_manager.image.copy()
        else:
            self.original_image = None

        # Initialize Preview
        self.update_preview()

    def update_clip_value(self, value):
        clip_limit = value / 10.0  # 0.1 to 4.0
        self.clip_value_label.setText(f"{clip_limit:.1f}")

    def update_tile_value(self, value):
        self.tile_value_label.setText(str(value))

    def debounce_preview(self):
        """
        Starts or restarts the debounce timer.
        """
        self.debounce_timer.start(300)  # 300 milliseconds delay

    def update_preview(self):
        """
        Updates the preview image based on current slider values.
        """
        if self.original_image is None:
            self.pixmap_item.setPixmap(QPixmap())
            self.preview_scene.addText("No image loaded.")
            return

        clip_limit = self.clip_slider.value() / 10.0  # 0.1 to 4.0
        tile_grid_size = self.tile_slider.value()

        try:
            preview_image = apply_clahe(self.original_image, clip_limit=clip_limit, tile_grid_size=(tile_grid_size, tile_grid_size))
            self.display_image(preview_image)
        except Exception as e:
            self.preview_scene.clear()
            self.preview_scene.addText("Failed to generate preview.")
            QMessageBox.critical(self, "Error", f"Failed to generate CLAHE preview:\n{e}")

    def display_image(self, image):
        """
        Converts a NumPy image array to QPixmap and displays it in the QGraphicsView.
        Maintains the current zoom and pan settings.
        """
        # Convert image from [0,1] to [0,255] and to uint8
        image_uint8 = (image * 255).astype('uint8')
        # Convert RGB to BGR for OpenCV
        image_bgr = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2BGR)
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        height, width, channel = image_rgb.shape
        bytes_per_line = 3 * width
        q_image = QImage(image_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)

        # Update the existing pixmap item
        self.pixmap_item.setPixmap(pixmap)



    def zoom_in(self):
        """
        Zooms in the preview image by 25%.
        """
        self.current_zoom *= 1.25
        self.preview_view.scale(1.25, 1.25)

    def zoom_out(self):
        """
        Zooms out the preview image by 20%.
        """
        self.current_zoom *= 0.8
        self.preview_view.scale(0.8, 0.8)

    def fit_to_preview(self):
        """
        Fits the image to the preview area.
        """
        self.preview_view.fitInView(self.preview_scene.sceneRect(), Qt.KeepAspectRatio)
        self.current_zoom = 1.0  # Reset zoom level


    def apply_clahe(self):
        """
        Applies CLAHE with current parameters to the main image.
        """
        if self.original_image is None:
            QMessageBox.warning(self, "No Image", "No image loaded to apply CLAHE.")
            self.reject()
            return

        clip_limit = self.clip_slider.value() / 10.0  # 0.1 to 4.0
        tile_grid_size = self.tile_slider.value()

        try:
            enhanced_image = apply_clahe(self.original_image, clip_limit=clip_limit, tile_grid_size=(tile_grid_size, tile_grid_size))
            self.image_manager.update_image(
                updated_image=enhanced_image,
                metadata=self.image_manager._metadata.get(self.image_manager.current_slot, {})
            )
            QMessageBox.information(self, "Success", "CLAHE applied successfully.")
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to apply CLAHE:\n{e}")
            self.reject()

    def reset_parameters(self):
        """
        Resets sliders to their default values and updates the preview.
        """
        self.clip_slider.setValue(20)  # Default 2.0
        self.tile_slider.setValue(8)   # Default (8,8)

class MorphologyDialog(QDialog):
    def __init__(self, image_manager, parent=None):
        super().__init__(parent)
        self.image_manager = image_manager
        self.setWindowTitle("Morphological Operations")
        self.setGeometry(200, 200, 800, 500)  # Increased size for better layout
        self.initUI()
        self.current_zoom = 1.0  # Initial zoom level

    def initUI(self):
        main_layout = QVBoxLayout()

        # Morphological Parameters Group
        parameters_group = QGroupBox("Morphological Parameters")
        parameters_layout = QGridLayout()

        # Operation Type Selection
        operation_label = QLabel("Operation Type:")
        self.operation_combo = QComboBox()
        self.operation_combo.addItems(["Erosion", "Dilation", "Opening", "Closing"])
        self.operation_combo.setToolTip("Select the type of morphological operation to apply.")
        self.operation_combo.currentTextChanged.connect(self.debounce_preview)
        self.operation_combo.currentTextChanged.connect(self.update_preview)

        parameters_layout.addWidget(operation_label, 0, 0)
        parameters_layout.addWidget(self.operation_combo, 0, 1, 1, 2)

        # Kernel Size Slider and Label
        kernel_label = QLabel("Kernel Size:")
        self.kernel_slider = QSlider(Qt.Horizontal)
        self.kernel_slider.setMinimum(1)
        self.kernel_slider.setMaximum(31)
        self.kernel_slider.setValue(3)        # Default kernel size
        self.kernel_slider.setTickInterval(2)
        self.kernel_slider.setTickPosition(QSlider.TicksBelow)
        self.kernel_value_label = QLabel("3")  # Initial value
        self.kernel_slider.setToolTip("Adjust the size of the structuring element. Must be an odd number.")

        self.kernel_slider.valueChanged.connect(self.update_kernel_value)
        self.kernel_slider.valueChanged.connect(self.debounce_preview)

        parameters_layout.addWidget(kernel_label, 1, 0)
        parameters_layout.addWidget(self.kernel_slider, 1, 1)
        parameters_layout.addWidget(self.kernel_value_label, 1, 2)

        # Iterations Slider and Label
        iterations_label = QLabel("Iterations:")
        self.iterations_slider = QSlider(Qt.Horizontal)
        self.iterations_slider.setMinimum(1)
        self.iterations_slider.setMaximum(10)
        self.iterations_slider.setValue(1)        # Default iterations
        self.iterations_slider.setTickInterval(1)
        self.iterations_slider.setTickPosition(QSlider.TicksBelow)
        self.iterations_value_label = QLabel("1")  # Initial value
        self.iterations_slider.setToolTip("Adjust the number of times the operation is applied.")

        self.iterations_slider.valueChanged.connect(self.update_iterations_value)
        self.iterations_slider.valueChanged.connect(self.debounce_preview)

        parameters_layout.addWidget(iterations_label, 2, 0)
        parameters_layout.addWidget(self.iterations_slider, 2, 1)
        parameters_layout.addWidget(self.iterations_value_label, 2, 2)

        parameters_group.setLayout(parameters_layout)
        main_layout.addWidget(parameters_group)

        # Preview Area
        preview_group = QGroupBox("Preview")
        preview_layout = QVBoxLayout()

        # QGraphicsView and QGraphicsScene
        self.preview_view = QGraphicsView()
        self.preview_scene = QGraphicsScene()
        self.preview_view.setScene(self.preview_scene)

        self.preview_view.setDragMode(QGraphicsView.ScrollHandDrag)  # Enable panning
        self.preview_view.setFixedSize(800, 500)  # Fixed size to prevent resizing

        # Initialize QGraphicsPixmapItem
        self.pixmap_item = QGraphicsPixmapItem()
        self.preview_scene.addItem(self.pixmap_item)

        preview_layout.addWidget(self.preview_view)
        preview_group.setLayout(preview_layout)
        main_layout.addWidget(preview_group)

        # Zoom and Fit Buttons
        zoom_layout = QHBoxLayout()
        zoom_in_button = QPushButton("Zoom In (+)")
        zoom_in_button.setToolTip("Zoom in the preview image.")
        zoom_in_button.clicked.connect(self.zoom_in)

        zoom_out_button = QPushButton("Zoom Out (-)")
        zoom_out_button.setToolTip("Zoom out the preview image.")
        zoom_out_button.clicked.connect(self.zoom_out)

        fit_button = QPushButton("Fit to Preview")
        fit_button.setToolTip("Fit the image to the preview area.")
        fit_button.clicked.connect(self.fit_to_preview)

        zoom_layout.addStretch()
        zoom_layout.addWidget(zoom_in_button)
        zoom_layout.addWidget(zoom_out_button)
        zoom_layout.addWidget(fit_button)

        main_layout.addLayout(zoom_layout)

        # Apply, Reset, and Cancel Buttons
        button_layout = QHBoxLayout()
        apply_button = QPushButton("Apply")
        apply_button.clicked.connect(self.apply_morphology)
        reset_button = QPushButton("Reset")
        reset_button.clicked.connect(self.reset_parameters)
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_layout.addStretch()
        button_layout.addWidget(apply_button)
        button_layout.addWidget(reset_button)
        button_layout.addWidget(cancel_button)
        main_layout.addLayout(button_layout)

        self.setLayout(main_layout)

        # Initialize Debounce Timer
        self.debounce_timer = QTimer()
        self.debounce_timer.setSingleShot(True)
        self.debounce_timer.timeout.connect(self.update_preview)

        # Store Original Image
        if self.image_manager.image is not None:
            self.original_image = self.image_manager.image.copy()
        else:
            self.original_image = None

        # Initialize Preview
        self.update_preview()

    def update_kernel_value(self, value):
        if value % 2 == 0:
            value += 1  # Ensure kernel size is odd
            if value > self.kernel_slider.maximum():
                value = self.kernel_slider.maximum() - 1 if self.kernel_slider.maximum() % 2 == 0 else self.kernel_slider.maximum()
            self.kernel_slider.setValue(value)
        self.kernel_value_label.setText(str(value))

    def update_iterations_value(self, value):
        self.iterations_value_label.setText(str(value))

    def debounce_preview(self):
        """
        Starts or restarts the debounce timer.
        """
        self.debounce_timer.start(300)  # 300 milliseconds delay

    def update_preview(self):
        """
        Updates the preview image based on current parameters.
        """
        if self.original_image is None:
            self.pixmap_item.setPixmap(QPixmap())
            self.preview_scene.addText("No image loaded.")
            return

        operation = self.operation_combo.currentText().lower()  # e.g., 'erosion'
        kernel_size = self.kernel_slider.value()
        iterations = self.iterations_slider.value()

        # Ensure kernel size is odd
        if kernel_size % 2 == 0:
            kernel_size += 1
            self.kernel_slider.setValue(kernel_size)

        try:
            preview_image = apply_morphology(
                self.original_image,
                operation=operation,
                kernel_size=kernel_size,
                iterations=iterations
            )
            self.display_image(preview_image)
        except Exception as e:
            self.preview_scene.clear()
            self.preview_scene.addText("Failed to generate preview.")
            QMessageBox.critical(self, "Error", f"Failed to generate Morphological preview:\n{e}")

    def display_image(self, image):
        """
        Converts a NumPy image array to QPixmap and displays it in the QGraphicsView.
        Maintains the current zoom and pan settings.
        """
        # Convert image from [0,1] to [0,255] and to uint8
        image_uint8 = (image * 255).astype('uint8')
        # Convert RGB to BGR for OpenCV
        image_bgr = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2BGR)
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        height, width, channel = image_rgb.shape
        bytes_per_line = 3 * width
        q_image = QImage(image_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)

        # Update the existing pixmap item
        self.pixmap_item.setPixmap(pixmap)


    def zoom_in(self):
        """
        Zooms in the preview image by 25%.
        """
        self.current_zoom *= 1.25
        self.preview_view.scale(1.25, 1.25)

    def zoom_out(self):
        """
        Zooms out the preview image by 20%.
        """
        self.current_zoom *= 0.8
        self.preview_view.scale(0.8, 0.8)

    def fit_to_preview(self):
        """
        Fits the image to the preview area.
        """
        self.preview_view.fitInView(self.preview_scene.sceneRect(), Qt.KeepAspectRatio)
        self.current_zoom = 1.0  # Reset zoom level


    def apply_morphology(self):
        """
        Applies the selected morphological operation with current parameters to the main image.
        """
        if self.original_image is None:
            QMessageBox.warning(self, "No Image", "No image loaded to apply morphological operations.")
            self.reject()
            return

        operation = self.operation_combo.currentText().lower()  # e.g., 'erosion'
        kernel_size = self.kernel_slider.value()
        iterations = self.iterations_slider.value()

        # Ensure kernel size is odd
        if kernel_size % 2 == 0:
            kernel_size += 1

        try:
            processed_image = apply_morphology(
                self.original_image,
                operation=operation,
                kernel_size=kernel_size,
                iterations=iterations
            )
            self.image_manager.update_image(
                updated_image=processed_image,
                metadata=self.image_manager._metadata.get(self.image_manager.current_slot, {})
            )
            QMessageBox.information(self, "Success", f"{self.operation_combo.currentText()} applied successfully.")
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to apply morphological operations:\n{e}")
            self.reject()

    def reset_parameters(self):
        """
        Resets sliders and operation type to default values and updates the preview.
        """
        self.operation_combo.setCurrentIndex(0)  # 'Erosion'
        self.kernel_slider.setValue(3)           # Default kernel size
        self.iterations_slider.setValue(1)       # Default iterations

class WhiteBalanceDialog(QDialog):
    def __init__(self, image_manager, parent=None):
        super().__init__(parent)
        self.image_manager = image_manager
        self.setWindowTitle("White Balance")
        self.setGeometry(200, 200, 800, 500)  # Adjusted size to remove preview area
        self.initUI()

    def initUI(self):
        main_layout = QVBoxLayout()

        # White Balance Type Selection
        type_label = QLabel("White Balance Type:")
        self.type_combo = QComboBox()
        self.type_combo.addItems(["Star-Based", "Manual", "Auto"])
        self.type_combo.currentTextChanged.connect(self.update_options)

        type_layout = QHBoxLayout()
        type_layout.addWidget(type_label)
        type_layout.addWidget(self.type_combo)
        type_layout.addStretch()

        main_layout.addLayout(type_layout)

        # Standard White Balance Options
        self.standard_widget = QWidget()
        self.standard_layout = QVBoxLayout()

        # Gain Sliders for R, G, B
        gain_group = QGroupBox("Adjust Gain for Each Channel")
        gain_layout = QGridLayout()

        self.r_slider = QSlider(Qt.Horizontal)
        self.r_slider.setMinimum(50)
        self.r_slider.setMaximum(150)
        self.r_slider.setValue(100)  # Represents 0.5 to 1.5
        self.r_slider.setTickInterval(10)
        self.r_slider.setTickPosition(QSlider.TicksBelow)
        self.r_label = QLabel("100")

        self.g_slider = QSlider(Qt.Horizontal)
        self.g_slider.setMinimum(50)
        self.g_slider.setMaximum(150)
        self.g_slider.setValue(100)
        self.g_slider.setTickInterval(10)
        self.g_slider.setTickPosition(QSlider.TicksBelow)
        self.g_label = QLabel("100")

        self.b_slider = QSlider(Qt.Horizontal)
        self.b_slider.setMinimum(50)
        self.b_slider.setMaximum(150)
        self.b_slider.setValue(100)
        self.b_slider.setTickInterval(10)
        self.b_slider.setTickPosition(QSlider.TicksBelow)
        self.b_label = QLabel("100")

        # Connect sliders to update labels
        self.r_slider.valueChanged.connect(lambda val: self.r_label.setText(str(val)))
        self.g_slider.valueChanged.connect(lambda val: self.g_label.setText(str(val)))
        self.b_slider.valueChanged.connect(lambda val: self.b_label.setText(str(val)))

        # Arrange sliders and labels in grid
        gain_layout.addWidget(QLabel("Red Gain:"), 0, 0)
        gain_layout.addWidget(self.r_slider, 0, 1)
        gain_layout.addWidget(self.r_label, 0, 2)

        gain_layout.addWidget(QLabel("Green Gain:"), 1, 0)
        gain_layout.addWidget(self.g_slider, 1, 1)
        gain_layout.addWidget(self.g_label, 1, 2)

        gain_layout.addWidget(QLabel("Blue Gain:"), 2, 0)
        gain_layout.addWidget(self.b_slider, 2, 1)
        gain_layout.addWidget(self.b_label, 2, 2)

        gain_group.setLayout(gain_layout)
        self.standard_layout.addWidget(gain_group)
        self.standard_widget.setLayout(self.standard_layout)
        main_layout.addWidget(self.standard_widget)

        # Star-Based White Balance Options
        self.star_widget = QWidget()
        self.star_layout = QVBoxLayout()
        self.star_widget.setLayout(self.star_layout)
        self.star_widget.hide()  # Hidden initially

        star_info = QLabel("Star-Based White Balance automatically detects stars to adjust colors.")
        self.star_layout.addWidget(star_info)

        # Sensitivity Slider for Star Detection Threshold
        sensitivity_group = QGroupBox("Detection Sensitivity")
        sensitivity_layout = QHBoxLayout()

        self.sensitivity_slider = QSlider(Qt.Horizontal)
        self.sensitivity_slider.setMinimum(100)
        self.sensitivity_slider.setMaximum(255)
        self.sensitivity_slider.setValue(180)  # Default threshold
        self.sensitivity_slider.setTickInterval(5)
        self.sensitivity_slider.setTickPosition(QSlider.TicksBelow)
        self.sensitivity_label = QLabel("Threshold: 180")

        # Connect slider to update label and re-run detection
        self.sensitivity_slider.valueChanged.connect(self.update_sensitivity_label)
        self.sensitivity_slider.valueChanged.connect(self.detect_and_display_stars)

        sensitivity_layout.addWidget(QLabel("Threshold:"))
        sensitivity_layout.addWidget(self.sensitivity_slider)
        sensitivity_layout.addWidget(self.sensitivity_label)
        sensitivity_group.setLayout(sensitivity_layout)
        self.star_layout.addWidget(sensitivity_group)

        # Label to show number of detected stars
        self.star_count_label = QLabel("Detecting stars...")
        self.star_layout.addWidget(self.star_count_label)

        # Image display for detected stars
        self.star_image_label = QLabel()
        self.star_image_label.setFixedSize(800, 500)  # Reduced size as no preview is needed
        self.star_image_label.setStyleSheet("border: 1px solid black;")
        self.star_layout.addWidget(self.star_image_label)

        main_layout.addWidget(self.star_widget)

        # Apply and Cancel Buttons
        button_layout = QHBoxLayout()
        apply_button = QPushButton("Apply")
        apply_button.clicked.connect(self.apply_white_balance)
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_layout.addStretch()
        button_layout.addWidget(apply_button)
        button_layout.addWidget(cancel_button)

        main_layout.addLayout(button_layout)

        self.setLayout(main_layout)

        # **Set initial selection to "Star-Based" and display relevant widgets**
        self.type_combo.setCurrentText("Star-Based")
        self.update_options("Star-Based")

    def update_sensitivity_label(self, value):
        self.sensitivity_label.setText(f"Threshold: {value}")

    def update_options(self, text):
        if text == "Manual":
            self.star_widget.hide()
            self.standard_widget.show()
        elif text == "Auto":
            self.standard_widget.hide()
            self.star_widget.hide()
        elif text == "Star-Based":
            self.standard_widget.hide()
            self.star_widget.show()
            self.star_count_label.setText("Detecting stars...")
            # Trigger star detection and display
            self.detect_and_display_stars()

    def detect_and_display_stars(self):
        try:
            image = self.image_manager.image
            if image is not None:
                threshold = self.sensitivity_slider.value()
                balanced_image, star_count, image_with_stars = apply_star_based_white_balance(image, threshold)
                
                # Convert the image with stars to QImage and then to QPixmap
                height, width, channel = image_with_stars.shape
                bytes_per_line = 3 * width
                q_image = QImage(image_with_stars.data, width, height, bytes_per_line, QImage.Format_BGR888)
                pixmap = QPixmap.fromImage(q_image).scaled(
                    self.star_image_label.width(),
                    self.star_image_label.height(),
                    Qt.KeepAspectRatio
                )
                self.star_image_label.setPixmap(pixmap)
                self.star_count_label.setText(f"Detected {star_count} stars.")
            else:
                self.star_count_label.setText("No image loaded.")
        except Exception as e:
            self.star_count_label.setText("Detection failed.")
            self.star_image_label.clear()
            QMessageBox.critical(self, "Error", f"Failed to detect stars:\n{e}")

    def apply_white_balance(self):
        wb_type = self.type_combo.currentText()

        try:
            image = self.image_manager.image
            if image is not None:
                if wb_type == "Manual":
                    r_gain = self.r_slider.value() / 100.0  # 0.5 to 1.5
                    g_gain = self.g_slider.value() / 100.0
                    b_gain = self.b_slider.value() / 100.0
                    balanced_image = apply_standard_white_balance(image, r_gain, g_gain, b_gain)
                    self.image_manager.update_image(
                        updated_image=balanced_image,
                        metadata=self.image_manager._metadata.get(self.image_manager.current_slot, {})
                    )
                    QMessageBox.information(self, "Success", "Manual White Balance applied successfully.")
                    self.accept()
                elif wb_type == "Auto":
                    balanced_image = apply_auto_white_balance(image)
                    self.image_manager.update_image(
                        updated_image=balanced_image,
                        metadata=self.image_manager._metadata.get(self.image_manager.current_slot, {})
                    )
                    QMessageBox.information(self, "Success", "Auto White Balance applied successfully.")
                    self.accept()
                elif wb_type == "Star-Based":
                    threshold = self.sensitivity_slider.value()
                    balanced_image, star_count, _ = apply_star_based_white_balance(image, threshold)
                    self.image_manager.update_image(
                        updated_image=balanced_image,
                        metadata=self.image_manager._metadata.get(self.image_manager.current_slot, {})
                    )
                    QMessageBox.information(self, "Success", f"Star-Based White Balance applied successfully.\nDetected {star_count} stars.")
                    self.accept()
                else:
                    raise ValueError("Invalid White Balance Type.")
            else:
                QMessageBox.warning(self, "No Image", "No image loaded to apply White Balance.")
                self.reject()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to apply White Balance:\n{e}")
            self.reject()

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

        # Handle mono images
        if im_data.ndim == 2:
            print(f"Mono image detected with 2D shape: {im_data.shape}. Converting to 3-channel RGB for display.")
            im_data = np.stack([im_data] * 3, axis=-1)  # Convert to 3-channel RGB
        elif im_data.ndim == 3 and im_data.shape[2] == 1:
            print(f"Mono image with a single channel detected: {im_data.shape}. Converting to 3-channel RGB for display.")
            im_data = np.repeat(im_data, 3, axis=-1)  # Expand single channel to 3 channels

        if im_data.ndim == 3 and im_data.shape[2] == 3:
            # For color images (or converted mono images)
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
        else:
            print(f"Unexpected image shape: {im_data.shape}")
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
        self.current_pixmap = pixmap  # Store the current pixmap
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
                print(f"Failed to load XISF metadata: {e}")

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
                print(f"Failed to load FITS metadata: {e}")

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
                print(f"Failed to load Camera RAW metadata: {e}")

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
        self.settings = QSettings("Seti Astro", "Seti Astro Suite")

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

        # Save button to save the processed image
        self.save_button = QPushButton("Save Image")
        self.save_button.clicked.connect(self.save_processed_image_to_disk)
        #left_layout.addWidget(self.save_button)  

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
        """Undo to the previous image using ImageManager."""
        if self.image_manager.can_undo():
            # Perform undo operation
            self.image_manager.undo()
            
            # Update the undo/redo button states
            self.undo_button.setEnabled(self.image_manager.can_undo())
            self.redo_button.setEnabled(self.image_manager.can_redo())
            
            # Optionally refresh the UI (e.g., display the updated image)
            self.refresh_preview()
            print("CosmicClarityTab: Undo operation performed.")
        else:
            print("CosmicClarityTab: No actions to undo.")

    def redo(self):
        """Redo to the next image using ImageManager."""
        if self.image_manager.can_redo():
            # Perform redo operation
            self.image_manager.redo()
            
            # Update the undo/redo button states
            self.undo_button.setEnabled(self.image_manager.can_undo())
            self.redo_button.setEnabled(self.image_manager.can_redo())
            
            # Optionally refresh the UI (e.g., display the updated image)
            self.refresh_preview()
            print("CosmicClarityTab: Redo operation performed.")
        else:
            print("CosmicClarityTab: No actions to redo.")



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
        """Save the Cosmic Clarity folder path using QSettings."""
        self.settings.setValue("cosmic_clarity_folder", folder)  # Save to QSettings
        print(f"Saved Cosmic Clarity folder to QSettings: {folder}")

    def load_cosmic_clarity_folder(self):
        """Load the saved Cosmic Clarity folder path from QSettings."""
        folder = self.settings.value("cosmic_clarity_folder", "")  # Load from QSettings
        if folder:
            self.cosmic_clarity_folder = folder
            self.cosmic_clarity_folder_label.setText(f"Folder: {folder}")
            print(f"Loaded Cosmic Clarity folder from QSettings: {folder}")
        else:
            print("No saved Cosmic Clarity folder found in QSettings.")

    def on_image_changed(self, slot, image, metadata):
        """
        Slot to handle image changes from ImageManager.
        Updates the display if the current slot is affected.
        """
        if slot == self.image_manager.current_slot:
            self.loaded_image_path = metadata.get('file_path', None)
            self.original_header = metadata.get('original_header', None)
            self.bit_depth = metadata.get('bit_depth', None)
            self.is_mono = metadata.get('is_mono', False)

            # Ensure image is in numpy array format
            if not isinstance(image, np.ndarray):
                image = np.array(image)

            # Handle mono and color images
            if self.is_mono:
                # Squeeze the singleton dimension for grayscale images if it exists
                if len(image.shape) == 3 and image.shape[2] == 1:
                    print(f"Mono image detected with shape: {image.shape}. Squeezing singleton dimension.")
                    image = np.squeeze(image, axis=2)  # Convert (H, W, 1) to (H, W)

                # Convert 2D grayscale to RGB by stacking it
                if len(image.shape) == 2:
                    print(f"Converting mono image with shape: {image.shape} to 3-channel RGB.")
                    image = np.stack([image] * 3, axis=-1)

            elif len(image.shape) == 3 and image.shape[2] not in [1, 3]:
                # Catch unexpected formats like (H, W, C) where C is not 1 or 3
                raise ValueError(f"Unexpected image format with shape {image.shape}. Must be RGB or Grayscale.")

            self.image = image

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
            image = self.image

        if image is None:
            print("[ERROR] No image to display.")
            QMessageBox.warning(self, "No Image", "No image data available to display.")
            return False

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
            try:
                if is_mono:
                    print("Processing mono image for display...")
                    # Stretch mono image and convert to RGB
                    stretched_mono = stretch_mono_image(display_image if display_image.ndim == 2 else display_image[:, :, 0], target_median)
                    display_image = np.stack([stretched_mono] * 3, axis=-1)  # Convert mono to RGB for display
                else:
                    print("Processing color image for display...")
                    display_image = stretch_color_image(display_image, target_median, linked=False)
            except Exception as e:
                print(f"[ERROR] Error during image stretching: {e}")
                QMessageBox.critical(self, "Error", f"Error stretching image for display:\n{e}")
                return False
        else:
            print("AutoStretch is disabled.")

        # Convert to QImage for display
        try:
            display_image_uint8 = (display_image * 255).astype(np.uint8)
        except Exception as e:
            print(f"[ERROR] Error converting image to uint8: {e}")
            QMessageBox.critical(self, "Error", f"Error processing image for display:\n{e}")
            return False

        print(f"Image shape after conversion to uint8: {display_image_uint8.shape}")

        try:
            if display_image_uint8.ndim == 3 and display_image_uint8.shape[2] == 3:  # RGB image
                height, width, _ = display_image_uint8.shape
                bytes_per_line = 3 * width
                qimage = QImage(display_image_uint8.data, width, height, bytes_per_line, QImage.Format_RGB888)
            elif display_image_uint8.ndim == 2:  # Grayscale image
                height, width = display_image_uint8.shape
                bytes_per_line = width
                qimage = QImage(display_image_uint8.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
            else:
                print("[ERROR] Unexpected image format!")
                return False
        except Exception as e:
            print(f"[ERROR] Error creating QImage: {e}")
            return False

        # Create QPixmap from QImage
        pixmap = QPixmap.fromImage(qimage)
        if pixmap.isNull():
            print("[ERROR] Failed to convert QImage to QPixmap.")
            return False

        self.image_label.setPixmap(pixmap)
        self.image_label.repaint()
        self.image_label.update()

        # Restore the previous scroll position
        self.scroll_area.horizontalScrollBar().setValue(current_scroll_position[0])
        self.scroll_area.verticalScrollBar().setValue(current_scroll_position[1])

        return True



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
            # Prepare metadata for the ImageManager
            metadata = {
                'file_path': self.loaded_image_path,      # Ensure this is correctly set elsewhere
                'original_header': self.original_header,  # Ensure this is correctly set elsewhere
                'bit_depth': self.bit_depth,              # Ensure this is correctly set elsewhere
                'is_mono': self.is_mono                   # Ensure this is correctly set elsewhere
            }

            # Use ImageManager's set_image method to manage undo/redo stack
            if self.image_manager:
                try:
                    self.image_manager.set_image(processed_image, metadata)
                    print("CosmicClarityTab: Processed image stored in ImageManager with undo/redo support.")
                except Exception as e:
                    # Handle potential errors during the update
                    QMessageBox.critical(self, "Error", f"Failed to store processed image in ImageManager:\n{e}")
                    print(f"Error storing processed image in ImageManager: {e}")
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
        self.settings = QSettings("Seti Astro", "Seti Astro Suite")
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
            self.settings.setValue("cosmic_clarity_folder", folder)  # Save to QSettings
            self.folder_label.setText(f"Folder: {folder}")
            print(f"Selected Cosmic Clarity folder: {folder}")

    def load_cosmic_clarity_folder(self):
        folder = self.settings.value("cosmic_clarity_folder", "")  # Load from QSettings
        if folder:
            self.cosmic_clarity_folder = folder
            self.folder_label.setText(f"Folder: {folder}")
            print(f"Loaded Cosmic Clarity folder: {folder}")
        else:
            print("No saved Cosmic Clarity folder found.")

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
                                            "Images (*.png *.tif *.tiff *.fit *.fits *.xisf *.cr2 *.nef *.arw *.dng *.orf *.rw2 *.pef);;All Files (*)")
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
                self.image_manager.set_image( self.image, metadata)
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
                self.image_manager.set_image(self.preview_image, metadata=metadata)
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
        """Apply the 8-bit LUT to the preview image for real-time updates on slot 0."""

        # Access slot0 (recombined image) from ImageManager
        if self.image is None:
            print("No preview image loaded.")
            QMessageBox.warning(self, "No Image", "Preview image is not loaded.")
            return

        try:
            current_scroll_x = self.scrollArea.horizontalScrollBar().value()
            current_scroll_y = self.scrollArea.verticalScrollBar().value()

            # 1) Copy the entire preview in float [0..1]
            base_image = self.image.copy()  # shape: (H, W, 3 or 2)

            # 2) Convert the entire base_image to 8-bit
            image_8bit = (base_image * 255).astype(np.uint8)

            # 3) Make a working copy for transformation
            adjusted_8bit = image_8bit.copy()

            if adjusted_8bit.ndim == 3:  # RGB image
                adjusted_image = adjusted_8bit.copy()

                if curve_mode == "K (Brightness)":
                    # Apply LUT to all channels equally (Brightness)
                    for channel in range(3):
                        adjusted_image[:, :, channel] = lut[adjusted_8bit[:, :, channel]]

                elif curve_mode in ["R", "G", "B"]:
                    # Apply LUT to a single channel
                    channel_index = {"R": 0, "G": 1, "B": 2}[curve_mode]
                    adjusted_image[:, :, channel_index] = lut[adjusted_8bit[:, :, channel_index]]

                elif curve_mode in ["L*", "a*", "b*"]:
                    # Manual RGB to Lab Conversion
                    M = self.M
                    M_inv = self.M_inv

                    # Normalize RGB to [0,1]
                    rgb = adjusted_8bit.astype(np.float32) / 255.0

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
                    M = self.M
                    M_inv = self.M_inv

                    # Normalize RGB to [0,1]
                    rgb = adjusted_8bit.astype(np.float32) / 255.0

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
                    rgb = adjusted_8bit.astype(np.float32) / 255.0

                    # Split channels
                    R, G, B = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

                    # Compute Cmax, Cmin, Delta
                    Cmax = np.maximum.reduce([R, G, B])
                    Cmin = np.minimum.reduce([R, G, B])
                    Delta = Cmax - Cmin

                    # Initialize Hue (H), Saturation (S), and Value (V)
                    H = np.zeros_like(Cmax)
                    S = np.zeros_like(Cmax)
                    V = Cmax.copy()

                    # Compute Hue (H)
                    mask = Delta != 0
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
                    QMessageBox.warning(self, "Unsupported Mode", f"Unsupported curve mode: {curve_mode}")
                    return

            else:  # Grayscale image
                # For grayscale images, apply LUT directly
                adjusted_image = lut[adjusted_8bit]

            # Convert adjusted_image back to float [0..1]
            preview_image = adjusted_image.astype(np.float32) / 255.0

            # Finally, show it
            self.show_image(preview_image)
            self.scrollArea.horizontalScrollBar().setValue(current_scroll_x)
            self.scrollArea.verticalScrollBar().setValue(current_scroll_y)            

        except Exception as e:
            print(f"Error in updatePreviewLUT: {e}")
            QMessageBox.critical(self, "Error", f"Failed to update preview: {e}")



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


        source_image = self.original_image.copy()

        # Push the current image to the undo stack before modifying
        self.pushUndo(self.original_image.copy())

        # Show the spinner before starting processing
        self.showSpinner()

        # Initialize and start the processing thread
        self.processing_thread = FullCurvesProcessingThread(source_image, curve_mode, curve_func)
        self.processing_thread.result_ready.connect(self.finishProcessing)
        self.processing_thread.start()
        print("Started FullCurvesProcessingThread.")

    def finishProcessing(self, adjusted_image):
        self.hideSpinner()

        # This is the new full-res float image
        self.original_image = adjusted_image.copy()

        # Also create a new preview_image
        self.preview_image = self.downsample_for_preview(adjusted_image, max_width=1080)

        # For display in the tab
        self.image = self.preview_image.copy()

        # Show it
        self.show_image(self.image)

        # Clear the draggable points on the curve editor
        self.curveEditor.initCurve()

        # Optionally update the ImageManager
        if self.image_manager:
            metadata = {
                'file_path': self.loaded_image_path,
                'original_header': self.original_header,
                'bit_depth': self.bit_depth,
                'is_mono': self.is_mono
            }
            self.image_manager.update_image(updated_image=self.original_image, metadata=metadata)
            print("FullCurvesTab: Full-resolution image updated in ImageManager.")


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
            self.image_manager.update_image(updated_image=last_state, metadata=metadata)
            print("Undo: Image reverted in ImageManager.")

        # Update the Undo button state
        self.updateUndoButtonState()


    def resetCurve(self):
        """
        Resets the draggable points in the curve editor without affecting other settings.
        """
        try:
            # Reset the draggable points in the curve editor
            self.curveEditor.initCurve()

            # Clear the preview LUT to match the reset state of draggable points
            self.current_lut = np.linspace(0, 255, 256, dtype=np.uint8)
            self.updatePreviewLUT(self.current_lut, self.curve_mode)

        except Exception as e:
            print(f"Error during curve reset: {e}")
            QMessageBox.critical(self, "Error", f"Failed to reset draggable points: {e}")

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

            self.preview_image = self.downsample_for_preview(image, max_width=1080)
            self.image = self.preview_image.copy()
            
            # Save the previous scroll position
            self.previous_scroll_pos = (
                self.scrollArea.horizontalScrollBar().value(),
                self.scrollArea.verticalScrollBar().value()
            )
            
            self.fileLabel.setText(self.loaded_image_path if self.loaded_image_path else "")
            
            # Update the UI elements (buttons, etc.)
            self.show_image(image)
            self.update_image_display()

            # Enable or disable buttons based on image processing state
            self.applyButton.setEnabled(True)
            self.saveButton.setEnabled(True)
            self.undoButton.setEnabled(len(self.undo_stack) > 0)

            print(f"FullCurvesTab: Image updated from ImageManager slot {slot}.")

    def downsample_for_preview(self, image_float32, max_width=1080):
        """
        If image width > max_width, scale it down proportionally.
        Returns a new float32 image in [0..1].
        """


        h, w = image_float32.shape[:2]

        if w <= max_width:
            # No need to downsample
            return image_float32.copy()

        scale_factor = max_width / float(w)
        new_w = max_width
        new_h = int(h * scale_factor)

        # Convert [0..1] float to [0..255] uint8 for OpenCV resizing
        temp_8u = (image_float32 * 255).clip(0,255).astype(np.uint8)

        # Resize with INTER_AREA for best downsampling
        resized_8u = cv2.resize(temp_8u, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Convert back to float32 in [0..1]
        return resized_8u.astype(np.float32) / 255.0



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

    def zoom_in(self):
        """
        Zoom into the image by increasing the zoom factor.
        """
        if self.image is not None:
            self.zoom_factor *= 1.2
            self.show_image(self.image)
            print(f"Zoomed in. New zoom factor: {self.zoom_factor:.2f}")
        else:
            print("No stretched image to zoom in.")
            QMessageBox.warning(self, "Warning", "No stretched image to zoom in.")

    def zoom_out(self):
        """
        Zoom out of the image by decreasing the zoom factor.
        """
        if self.image is not None:
            self.zoom_factor /= 1.2
            self.show_image(self.image)
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
                                            "Images (*.png *.tif *.tiff *.fit *.fits *.xisf *.cr2 *.nef *.arw *.dng *.orf *.rw2 *.pef);;All Files (*)")
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

        # New "Apply Changes" button
        self.apply_btn = QPushButton("Apply Changes/Push for Processing", self)
        self.apply_btn.clicked.connect(self.apply_changes)
        top_btn_layout.addWidget(self.apply_btn)

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

    def apply_changes(self):
        """
        Push the combined image to ImageManager's slot 0 for further processing.
        """
        if self.combined_image is None:
            QMessageBox.warning(self, "No Image", "There is no combined image to apply.")
            return

        # Metadata for the combined image
        metadata = {
            'file_path': "Combined HF+LF Applied",
            'original_header': self.original_header,
            'is_mono': self.is_mono,
            'bit_depth': "32-bit floating point"
        }

        # Push the combined image to slot 0
        self.image_manager.set_image(self.combined_image, metadata)
        QMessageBox.information(self, "Changes Applied", "The combined image has been pushed to slot 0 for processing.")

        # Close the preview window (optional)
        self.close()       

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
        title_label = QLabel("Perfect Palette Picker", self)
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
        Opens a dialog to load an image either from a file or from a slot based on user choice.
        
        Parameters:
            image_type (str): The type of image to load ('Ha', 'OIII', 'SII', 'OSC1', 'OSC2').
        """
        try:
            print(f"Initiating load process for {image_type} image.")
            
            # Step 1: Prompt user to choose the source
            source_choice, ok = QInputDialog.getItem(
                self,
                f"Select {image_type} Image Source",
                "Choose the source of the image:",
                ["From File", "From Slot"],
                editable=False
            )
            
            if not ok or not source_choice:
                QMessageBox.warning(self, "Cancelled", f"{image_type} image loading cancelled.")
                print(f"{image_type} image loading cancelled by the user.")
                return
            
            print(f"{image_type} image source selected: {source_choice}")
            
            if source_choice == "From File":
                result = self.load_image_from_file(image_type)
            elif source_choice == "From Slot":
                result = self.load_image_from_slot(image_type)
            else:
                QMessageBox.warning(self, "Invalid Choice", "Invalid source choice. Operation cancelled.")
                print("Invalid source choice. Exiting load process.")
                return
            
            if result is None:
                # Loading was unsuccessful or cancelled
                return
            
            image, original_header, bit_depth, is_mono, file_path = result
            
            # Assign the loaded image to the appropriate attribute and update the label
            if image_type == 'Ha':
                self.ha_image = image
                self.ha_filename = file_path
                self.original_header = original_header
                self.bit_depth = bit_depth
                self.is_mono = is_mono
                self.ha_label.setText(f"Loaded: {os.path.basename(file_path) if file_path else 'From Slot'}")
            elif image_type == 'OIII':
                self.oiii_image = image
                self.oiii_filename = file_path
                self.original_header = original_header
                self.bit_depth = bit_depth
                self.is_mono = is_mono
                self.oiii_label.setText(f"Loaded: {os.path.basename(file_path) if file_path else 'From Slot'}")
            elif image_type == 'SII':
                self.sii_image = image
                self.sii_filename = file_path
                self.original_header = original_header
                self.bit_depth = bit_depth
                self.is_mono = is_mono
                self.sii_label.setText(f"Loaded: {os.path.basename(file_path) if file_path else 'From Slot'}")
            elif image_type == 'OSC1':
                self.osc1_image = image
                self.osc1_filename = file_path
                self.original_header = original_header
                self.bit_depth = bit_depth
                self.is_mono = is_mono
                self.osc1_label.setText(f"Loaded: {os.path.basename(file_path) if file_path else 'From Slot'}")
            elif image_type == 'OSC2':
                self.osc2_image = image
                self.osc2_filename = file_path
                self.original_header = original_header
                self.bit_depth = bit_depth
                self.is_mono = is_mono
                self.osc2_label.setText(f"Loaded: {os.path.basename(file_path) if file_path else 'From Slot'}")
            else:
                QMessageBox.warning(self, "Unknown Image Type", f"Image type '{image_type}' is not recognized.")
                print(f"Unknown image type: {image_type}")
                return
            
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
            print(f"{image_type} image loaded successfully.")
        
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An unexpected error occurred while loading {image_type} image:\n{e}")
            print(f"An unexpected error occurred while loading {image_type} image: {e}")

    def load_image_from_slot(self, image_type):
        """
        Handles loading an image from a slot.
        
        Parameters:
            image_type (str): The type of image to load.
        
        Returns:
            tuple: (image, original_header, bit_depth, is_mono, file_path) or None on failure.
        """
        if not self.image_manager:
            QMessageBox.critical(self, "Error", "ImageManager is not initialized. Cannot load image from slot.")
            print("ImageManager is not initialized. Cannot load image from slot.")
            return None
        
        # Retrieve available slots
        available_slots = [
            f"Slot {i}" for i in range(1, self.image_manager.max_slots + 1)
            if self.image_manager._images.get(i, None) is not None
        ]
        
        if not available_slots:
            QMessageBox.warning(self, "No Available Slots", "No slots contain images. Please add images to slots first.")
            print("No available slots contain images.")
            return None
        
        slot_choice, ok = QInputDialog.getItem(
            self,
            f"Select Slot for {image_type} Image",
            "Choose a slot containing the image:",
            available_slots,
            editable=False
        )
        
        if not ok or not slot_choice:
            QMessageBox.warning(self, "Cancelled", f"{image_type} image loading cancelled.")
            print(f"{image_type} image loading cancelled by the user.")
            return None
        
        # Extract slot number
        target_slot_num = int(slot_choice.split()[-1])
        image = self.image_manager._images.get(target_slot_num, None)
        
        if image is None:
            QMessageBox.warning(self, "Empty Slot", f"{slot_choice} does not contain an image.")
            print(f"{slot_choice} is empty. Cannot load {image_type} image.")
            return None
        
        print(f"{image_type} image selected from {slot_choice}.")
        
        # Retrieve metadata from ImageManager._metadata
        metadata = self.image_manager._metadata.get(target_slot_num, {})
        original_header = metadata.get('header', None)
        bit_depth = metadata.get('bit_depth', "Unknown")
        is_mono = metadata.get('is_mono', False)
        file_path = metadata.get('file_path', None)
        
        if image is None:
            QMessageBox.critical(self, "Error", f"Failed to load {image_type} image from {slot_choice}.")
            print(f"Failed to load {image_type} image from slot {slot_choice}.")
            return None
        
        return image, original_header, bit_depth, is_mono, file_path

    def load_image_from_file(self, image_type):
        """
        Handles loading an image from a file.
        
        Parameters:
            image_type (str): The type of image to load.
        
        Returns:
            tuple: (image, original_header, bit_depth, is_mono, file_path) or None on failure.
        """
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_filter = "Images (*.png *.tif *.tiff *.fits *.fit *.xisf)"
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            f"Select {image_type} Image File",
            "",
            file_filter,
            options=options
        )
        
        if not file_path:
            QMessageBox.warning(self, "No File Selected", f"No {image_type} image file selected. Operation cancelled.")
            print(f"No {image_type} image file selected.")
            return None
        
        print(f"{image_type} image file selected: {file_path}")
        
        # Load the image using your existing load_image function
        image, original_header, bit_depth, is_mono = load_image(file_path)
        if image is None:
            QMessageBox.critical(self, "Error", f"Failed to load {image_type} image from file.")
            print(f"Failed to load {image_type} image from file: {file_path}")
            return None
        
        return image, original_header, bit_depth, is_mono, file_path


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
                        self.sii_filename, self.osc1_filename, self.osc2_filename
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

            # Determine the valid file path:
            # Prioritize Ha, then OSC1, then OSC2
            file_path = None
            if self.ha_image is not None and self.ha_filename:
                file_path = self.ha_filename
                print("Using Ha filename as file_path.")
            elif self.osc1_image is not None and self.osc1_filename:
                file_path = self.osc1_filename
                print("Using OSC1 filename as file_path.")
            elif self.osc2_image is not None and self.osc2_filename:
                file_path = self.osc2_filename
                print("Using OSC2 filename as file_path.")
            else:
                # No valid source file, save combined_image to a temporary file
                try:
                    temp_dir = tempfile.gettempdir()
                    timestamp = int(time.time())
                    temp_file_path = os.path.join(temp_dir, f"combined_image_{timestamp}.tif")
                    
                    # Save the combined image using your existing save_image function
                    save_image(
                        img_array=self.combined_image,
                        filename=temp_file_path,
                        original_format='tif',
                        bit_depth=self.bit_depth,
                        original_header=self.original_header,
                        is_mono=self.is_mono
                    )
                    
                    file_path = temp_file_path
                    print(f"Combined image saved to temporary file: {file_path}")
                except Exception as e:
                    print(f"Failed to save combined image to temporary file: {e}")
                    QMessageBox.critical(
                        self, 
                        "Error", 
                        f"Failed to save combined image to temporary file:\n{e}"
                    )
                    return

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
        if self.combined_image is not None:
            # Get the width of the scroll area's viewport (preview area)
            preview_width = self.scrollArea.viewport().width()
            
            # Get the original image width from the numpy array
            # Assuming self.image has shape (height, width, channels) or (height, width) for grayscale
            if self.combined_image.ndim == 3:
                image_width = self.combined_image.shape[1]
            elif self.combined_image.ndim == 2:
                image_width = self.combined_image.shape[1]
            else:
                print("Unexpected image dimensions!")

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
        # Preprocess input images to ensure mono images are single-channel
        self.ha_image = preprocess_narrowband_image(self.ha_image)
        self.oiii_image = preprocess_narrowband_image(self.oiii_image)
        self.sii_image = preprocess_narrowband_image(self.sii_image)

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
            r_channel = self.osc_image[..., 0]
            g_channel = self.osc_image[..., 1]
            b_channel = self.osc_image[..., 2]

            r_combined = 0.5 * r_channel + 0.5 * (self.sii_image if self.sii_image is not None else r_channel)
            g_combined = self.ha_to_oii_ratio * (self.ha_image if self.ha_image is not None else r_channel) + \
                        (1 - self.ha_to_oii_ratio) * g_channel
            b_combined = b_channel if self.oiii_image is None else self.oiii_image
        else:
            r_combined = 0.5 * self.ha_image + 0.5 * (self.sii_image if self.sii_image is not None else self.ha_image)
            g_combined = self.ha_to_oii_ratio * self.ha_image + (1 - self.ha_to_oii_ratio) * self.oiii_image
            b_combined = self.oiii_image

        # Debugging: Check shapes
        print(f"R combined shape: {r_combined.shape}")
        print(f"G combined shape: {g_combined.shape}")
        print(f"B combined shape: {b_combined.shape}")

        # Normalize combined channels to [0, 1]
        r_combined = np.clip(r_combined, 0, 1)
        g_combined = np.clip(g_combined, 0, 1)
        b_combined = np.clip(b_combined, 0, 1)

        # Stack the channels to create an RGB image
        try:
            combined_image = np.stack((r_combined, g_combined, b_combined), axis=-1)
        except ValueError as e:
            print(f"Error while stacking channels: {e}")
            print(f"R: {r_combined.shape}, G: {g_combined.shape}, B: {b_combined.shape}")
            return

        print(f"Combined image shape: {combined_image.shape}")

        # Apply star stretch if enabled
        if self.enable_star_stretch:
            combined_image = self.apply_star_stretch(combined_image)

        # Ensure combined_image is 3-channel
        if combined_image.ndim != 3 or combined_image.shape[2] != 3:
            raise ValueError("Combined image must have three channels (RGB).")

        # Apply SCNR (remove green cast)
        apply_average_neutral_scnr(combined_image)

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
        """
        Updates the preview with the stretched image and ensures it is undoable by using set_image.
        """
        # Create metadata for the new image
        metadata = {
            'file_path': self.filename if self.filename else "Stretched Image",
            'original_header': self.original_header if self.original_header else {},
            'bit_depth': "Unknown",  # Update dynamically if available
            'is_mono': self.is_mono,
            'processing_timestamp': datetime.now().isoformat(),
            'source_images': {
                'Original': self.filename if self.filename else "Not Provided"
            }
        }

        # Ensure ImageManager is initialized
        if self.image_manager:
            try:
                # Set the new image and metadata using the ImageManager
                self.image_manager.set_image(stretched_image, metadata)
                print("StarStretchTab: Processed image stored in ImageManager (undoable).")
            except Exception as e:
                print(f"Error updating ImageManager: {e}")
                QMessageBox.critical(self, "Error", f"Failed to update ImageManager:\n{e}")
                return
        else:
            print("ImageManager is not initialized.")
            QMessageBox.warning(self, "Warning", "ImageManager is not initialized. Cannot store the processed image.")
            return

        # Convert the stretched image to 8-bit for display in the preview
        preview_image = (stretched_image * 255).astype(np.uint8)
        h, w = preview_image.shape[:2]
        if preview_image.ndim == 3:
            q_image = QImage(preview_image.data, w, h, 3 * w, QImage.Format_RGB888)
        else:
            q_image = QImage(preview_image.data, w, h, w, QImage.Format_Grayscale8)

        # Update the pixmap and scale it for the preview label
        pixmap = QPixmap.fromImage(q_image)
        self.current_pixmap = pixmap  # Store the original pixmap
        scaled_pixmap = pixmap.scaled(pixmap.size() * self.zoom_factor, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.imageLabel.setPixmap(scaled_pixmap)
        self.imageLabel.resize(scaled_pixmap.size())

        # Hide the spinner after processing is complete
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

        # Ensure the mask matches the data type of the image
        enhanced_mask = enhanced_mask.astype(image.dtype)

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
        # Ensure the image is in a supported format (float32)
        image = image.astype(np.float32)

        # Check if the image is already grayscale
        if image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):
            # Image is already grayscale; normalize it
            lightness_mask = image / 255.0
        else:
            # Convert RGB image to grayscale
            lightness_mask = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) / 255.0

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

def preprocess_narrowband_image(image):
    """
    Preprocess narrowband images to ensure they are single-channel.
    If the image is detected as a mono image stored in 3-channel format, the red channel is used.
    """
    if image is not None:
        if image.ndim == 3:
            if image.shape[2] == 3:
                # Use the red channel if the image is multi-channel
                print("Detected multi-channel RGB data. Using the red channel as mono.")
                image = image[..., 0]
            elif image.shape[2] == 1:
                # Squeeze single redundant channel
                print("Detected 1-channel image with extra dimension. Squeezing to single channel.")
                image = np.squeeze(image, axis=-1)
        elif image.ndim != 2:
            raise ValueError(f"Unexpected image shape: {image.shape}")
    return image



def apply_standard_white_balance(image: np.ndarray, r_gain: float = 1.0, g_gain: float = 1.0, b_gain: float = 1.0) -> np.ndarray:
    """
    Applies standard white balance by adjusting the gain of each color channel.

    Parameters:
        image (np.ndarray): Input RGB image as a NumPy array normalized to [0,1].
        r_gain (float): Gain for the Red channel.
        g_gain (float): Gain for the Green channel.
        b_gain (float): Gain for the Blue channel.

    Returns:
        np.ndarray: White-balanced RGB image.
    """
    balanced = image.copy()
    balanced[:, :, 0] *= r_gain
    balanced[:, :, 1] *= g_gain
    balanced[:, :, 2] *= b_gain
    balanced = np.clip(balanced, 0.0, 1.0)
    return balanced

def apply_auto_white_balance(image: np.ndarray) -> np.ndarray:
    """
    Applies automatic white balance using the Gray World Assumption.

    Parameters:
        image (np.ndarray): Input RGB image as a NumPy array normalized to [0,1].

    Returns:
        np.ndarray: White-balanced RGB image.
    """
    # Calculate the mean of each channel
    mean_r = np.mean(image[:, :, 0])
    mean_g = np.mean(image[:, :, 1])
    mean_b = np.mean(image[:, :, 2])
    
    # Calculate the overall mean
    mean_all = (mean_r + mean_g + mean_b) / 3
    
    # Calculate gains
    gain_r = mean_all / mean_r if mean_r != 0 else 1.0
    gain_g = mean_all / mean_g if mean_g != 0 else 1.0
    gain_b = mean_all / mean_b if mean_b != 0 else 1.0
    
    # Apply gains
    balanced = image.copy()
    balanced[:, :, 0] *= gain_r
    balanced[:, :, 1] *= gain_g
    balanced[:, :, 2] *= gain_b
    balanced = np.clip(balanced, 0.0, 1.0)
    return balanced

def apply_star_based_white_balance(image: np.ndarray, threshold: int = 180) -> tuple:
    """
    Applies white balance based on detected stars in the image using thresholding and contour detection.

    Parameters:
        image (np.ndarray): Input RGB image as a NumPy array normalized to [0,1].
        threshold (int): Threshold value for binary segmentation to detect stars.

    Returns:
        tuple: (White-balanced RGB image, Number of detected stars, Image with detected stars marked)
    """
    # Convert to grayscale
    gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

    # Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Apply Contrast Limited Adaptive Histogram Equalization (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)

    # Apply binary thresholding to isolate bright regions (stars)
    # Lower the threshold to detect fainter stars
    _, thresh = cv2.threshold(enhanced, threshold, 255, cv2.THRESH_BINARY)

    # Perform morphological operations to enhance star features
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel, iterations=1)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    star_pixels = []
    image_with_stars = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 2 < area < 300:  # Adjusted area thresholds for more sensitivity
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            if circularity > 0.5:  # Lowered circularity to detect less perfect circles
                # Compute the centroid
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    star_pixels.append(image[cY, cX, :])
                    # Draw a circle around the detected star
                    cv2.circle(image_with_stars, (cX, cY), 10, (0, 255, 0), 3)

    star_count = len(star_pixels)

    if star_count == 0:
        raise ValueError("No stars detected for Star-Based White Balance.")

    # Calculate average color of stars
    star_pixels = np.array(star_pixels)
    avg_color = np.mean(star_pixels, axis=0)  # [R, G, B]

    # Calculate scaling factors to normalize average color to neutral gray (average of R, G, B)
    avg = np.mean(avg_color)
    if avg == 0:
        raise ValueError("Average star color is zero, cannot apply White Balance.")
    scaling_factors = avg / avg_color

    # Apply scaling factors
    balanced = image.copy()
    balanced[:, :, 0] *= scaling_factors[0]  # Red channel
    balanced[:, :, 1] *= scaling_factors[1]  # Green channel
    balanced[:, :, 2] *= scaling_factors[2]  # Blue channel
    balanced = np.clip(balanced, 0.0, 1.0)

    return balanced, star_count, image_with_stars

def apply_morphology(image: np.ndarray, operation: str = 'erosion', kernel_size: int = 3, iterations: int = 1) -> np.ndarray:
    """
    Applies a morphological operation to the image.

    Parameters:
        image (np.ndarray): Input RGB image as a NumPy array normalized to [0,1].
        operation (str): Morphological operation ('erosion', 'dilation', 'opening', 'closing').
        kernel_size (int): Size of the structuring element.
        iterations (int): Number of times the operation is applied.

    Returns:
        np.ndarray: Morphologically processed RGB image.
    """
    # Define the structuring element
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    # Convert image to uint8
    image_uint8 = (image * 255).astype(np.uint8)

    # Apply the selected operation
    if operation == 'erosion':
        processed = cv2.erode(image_uint8, kernel, iterations=iterations)
    elif operation == 'dilation':
        processed = cv2.dilate(image_uint8, kernel, iterations=iterations)
    elif operation == 'opening':
        processed = cv2.morphologyEx(image_uint8, cv2.MORPH_OPEN, kernel, iterations=iterations)
    elif operation == 'closing':
        processed = cv2.morphologyEx(image_uint8, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    else:
        raise ValueError("Unsupported morphological operation.")

    # Convert back to float [0,1]
    processed_image = processed.astype(np.float32) / 255.0
    return processed_image

def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)) -> np.ndarray:
    """
    Applies CLAHE to the image for adaptive contrast enhancement.

    Parameters:
        image (np.ndarray): Input RGB image as a NumPy array normalized to [0,1].
        clip_limit (float): Threshold for contrast limiting.
        tile_grid_size (tuple): Size of grid for histogram equalization.

    Returns:
        np.ndarray: Contrast-enhanced RGB image.
    """
    if image.ndim == 2:
        # Grayscale image
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        enhanced = clahe.apply((image * 255).astype(np.uint8))
        return enhanced / 255.0
    elif image.ndim == 3 and image.shape[2] == 3:
        # Convert to LAB color space
        lab = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
        # Split the channels
        l, a, b = cv2.split(lab)
        # Apply CLAHE to the L-channel
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        cl = clahe.apply(l)
        # Merge the channels back
        limg = cv2.merge((cl, a, b))
        # Convert back to RGB
        enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB) / 255.0
        return enhanced
    else:
        raise ValueError("Input image must be either grayscale or RGB.")

def apply_average_neutral_scnr(image: np.ndarray, amount: float = 1.0) -> np.ndarray:
    """
    Applies the Average Neutral SCNR method to remove green noise from an RGB image.

    Parameters:
        image (np.ndarray): Input RGB image as a NumPy array with shape (H, W, 3).
                            The image should be normalized to the [0, 1] range.
        amount (float): Blending factor between the original and SCNR-processed image.
                        0.0 returns the original image, 1.0 returns the fully SCNR-processed image.

    Returns:
        np.ndarray: The SCNR-processed RGB image.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("Input image must be a NumPy array.")

    if image.ndim != 3 or image.shape[2] != 3:
        print(f"apply_average_neutral_scnr received invalid image shape: {image.shape}")
        raise ValueError("Input image must have three channels (RGB).")

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Input image must have three channels (RGB).")

    if not (0.0 <= amount <= 1.0):
        raise ValueError("Amount parameter must be between 0.0 and 1.0.")

    # Ensure the image is in float format
    image = image.astype(np.float32)

    # Separate the channels
    R, G, B = image[..., 0], image[..., 1], image[..., 2]

    # Apply the Average Neutral SCNR formula: G' = min(G, 0.5*(R + B))
    G_scnr = np.minimum(G, 0.5 * (R + B))

    # Create the SCNR image
    scnr_image = image.copy()
    scnr_image[..., 1] = G_scnr  # Replace the green channel

    # Blend the original and SCNR images based on the amount parameter
    final_image = (1.0 - amount) * image + amount * scnr_image

    # Ensure the final image is still within [0, 1]
    final_image = np.clip(final_image, 0.0, 1.0)

    return final_image


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
    data_path = "C:/Users/Gaming/Desktop/Python Code/venv/Lib/site-packages/astroquery/simbad/data"

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

settings = QSettings("Seti Astro", "Seti Astro Suite")

def save_api_key(api_key):
    settings.setValue("astrometry_api_key", api_key)  # Save to QSettings
    print("API key saved.")

def load_api_key():
    api_key = settings.value("astrometry_api_key", "")  # Load from QSettings
    if api_key:
        print("API key loaded.")
    return api_key




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
        self.settings = QSettings("Seti Astro", "Seti Astro Suite")
            

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
        self.settings = QSettings("Seti Astro", "Seti Astro Suite")
        self.initUI()  # Build the UI
        self.load_settings()  # Load settings after UI is built
        self.object_limit = self.settings.value("object_limit", 100, type=int)

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
        header = self.tree.header()
        header.setSectionResizeMode(QHeaderView.Interactive)  # Allow users to resize columns
        header.setStretchLastSection(False)  # Ensure last column is not stretched automatically

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

        # Validate inputs
        try:
            latitude = float(latitude)
            longitude = float(longitude)
            min_altitude = float(min_altitude)
        except ValueError:
            self.update_status("Invalid input: Latitude, Longitude, and Min Altitude must be numeric.")
            return

        # Save the settings
        self.save_settings(latitude, longitude, date_str, time_str, timezone_str, min_altitude)


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
        self.settings.setValue("latitude", latitude)
        self.settings.setValue("longitude", longitude)
        self.settings.setValue("date", date)
        self.settings.setValue("time", time)
        self.settings.setValue("timezone", timezone)
        self.settings.setValue("min_altitude", min_altitude)
        print("Settings saved.")

    def load_settings(self):
        """Load settings from QSettings and populate UI fields."""
        def safe_cast(value, default, cast_type):
            """Safely cast a value to a specific type."""
            try:
                return cast_type(value)
            except (ValueError, TypeError):
                return default

        # Load and cast settings with fallbacks
        self.latitude = safe_cast(self.settings.value("latitude", 0.0), 0.0, float)
        self.longitude = safe_cast(self.settings.value("longitude", 0.0), 0.0, float)
        self.date = self.settings.value("date", datetime.now().strftime("%Y-%m-%d"))
        self.time = self.settings.value("time", "00:00:00")
        self.timezone = self.settings.value("timezone", "UTC")
        self.min_altitude = safe_cast(self.settings.value("min_altitude", 0.0), 0.0, float)
        self.object_limit = safe_cast(self.settings.value("object_limit", 100), 100, int)

        # Populate fields in the UI
        self.latitude_entry.setText(str(self.latitude))
        self.longitude_entry.setText(str(self.longitude))
        self.date_entry.setText(self.date)
        self.time_entry.setText(self.time)
        self.timezone_combo.setCurrentText(self.timezone)
        self.min_altitude_entry.setText(str(self.min_altitude))

        print("Settings loaded:", {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "date": self.date,
            "time": self.time,
            "timezone": self.timezone,
            "min_altitude": self.min_altitude,
            "object_limit": self.object_limit,
        })




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
    # Configure logging to capture errors for debugging
    logging.basicConfig(
        filename="astro_editing_suite.log",
        level=logging.ERROR,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon(icon_path))
    
    try:
        # Create and show the main window
        window = AstroEditingSuite()
        window.show()
        sys.exit(app.exec_())
    except Exception as e:
        # Log the error
        logging.error("Unhandled exception occurred", exc_info=True)
        
        # Display a critical error message to the user
        QMessageBox.critical(
            None,
            "Application Error",
            f"An unexpected error occurred:\n{str(e)}\n\n"
            "Please check the log file for more details."
        )
        sys.exit(1)
