import sys
import os
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog, QTreeWidget, QTreeWidgetItem, QSplitter, QScrollArea, QCheckBox
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QPixmap, QImage, QIcon
from xisf import XISF
import numpy as np
from astropy.io import fits
import tifffile as tiff
from PIL import Image
import csv
from PyQt5.QtWidgets import QFileDialog, QMessageBox
import platform
import xml.etree.ElementTree as ET
import numpy as np
import lz4.block  # https://python-lz4.readthedocs.io/en/stable/lz4.block.html
import zlib  # https://docs.python.org/3/library/zlib.html
import zstandard  # https://python-zstandard.readthedocs.io/en/stable/
import base64
import sys
from datetime import datetime
import ast

# Determine the path to the icon
if hasattr(sys, '_MEIPASS'):
    # PyInstaller's temporary folder in a bundled app
    icon_path = os.path.join(sys._MEIPASS, 'xisfliberator.png')
else:
    # Normal development environment
    icon_path = 'xisfliberator.png'

class XISFViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.image_data = None
        self.file_meta = None
        self.image_meta = None
        self.is_mono = False
        self.bit_depth = None
        self.scale_factor = 0.25
        self.dragging = False
        self.drag_start_pos = QPoint()
        self.autostretch_enabled = False
    
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
        
        self.load_button = QPushButton("Load XISF File")
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

        footer_label = QLabel("""
            Written by Franklin Marek<br>
            <a href='http://www.setiastro.com'>www.setiastro.com</a>
        """)
        footer_label.setAlignment(Qt.AlignCenter)
        footer_label.setOpenExternalLinks(True)
        footer_label.setStyleSheet("font-size: 10px;")
        left_layout.addWidget(footer_label)


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
        self.setWindowTitle("XISF Liberator V1.0")

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
            self.stretched_image = stretch_color_image(self.image_data, target_median=0.25, linked=True, normalize=True)



    def load_xisf(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open XISF File", "", "XISF Files (*.xisf)")
        
        if file_name:
            try:
                xisf = XISF(file_name)
                im_data = xisf.read_image(0)
                
                # Load metadata for saving later
                self.file_meta = xisf.get_file_metadata()
                self.image_meta = xisf.get_images_metadata()[0]
                
                # Display metadata
                self.display_metadata()

                # Determine if the image is mono or RGB, and set bit depth
                self.is_mono = im_data.shape[2] == 1
                self.bit_depth = str(im_data.dtype)
                self.image_data = im_data

                # Display image with scaling and normalization as before
                self.display_image()
                
                # Enable save button
                self.save_button.setEnabled(True)
            except Exception as e:
                self.image_label.setText(f"Failed to load XISF file: {e}")

    def display_image(self):
        if self.image_data is None:
            return

        im_data = self.stretched_image if self.autostretch_enabled else self.image_data
        if self.is_mono:
            im_data = np.squeeze(im_data, axis=2)
            height, width = im_data.shape
            bytes_per_line = width
            
            if im_data.dtype == np.uint8:
                q_image = QImage(im_data.tobytes(), width, height, bytes_per_line, QImage.Format_Grayscale8)
            elif im_data.dtype == np.uint16:
                im_data = (im_data / 256).astype(np.uint8)
                q_image = QImage(im_data.tobytes(), width, height, bytes_per_line, QImage.Format_Grayscale8)
            elif im_data.dtype == np.float32 or im_data.dtype == np.float64:
                im_data = np.clip((im_data - im_data.min()) / (im_data.max() - im_data.min()) * 255, 0, 255).astype(np.uint8)
                q_image = QImage(im_data.tobytes(), width, height, bytes_per_line, QImage.Format_Grayscale8)
        else:
            height, width, channels = im_data.shape
            bytes_per_line = channels * width
            
            if im_data.dtype == np.uint8:
                q_image = QImage(im_data.tobytes(), width, height, bytes_per_line, QImage.Format_RGB888)
            elif im_data.dtype == np.uint16:
                im_data = (im_data / 256).astype(np.uint8)
                q_image = QImage(im_data.tobytes(), width, height, bytes_per_line, QImage.Format_RGB888)
            elif im_data.dtype == np.float32 or im_data.dtype == np.float64:
                im_data = np.clip((im_data - im_data.min()) / (im_data.max() - im_data.min()) * 255, 0, 255).astype(np.uint8)
                q_image = QImage(im_data.tobytes(), width, height, bytes_per_line, QImage.Format_RGB888)

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

    def display_metadata(self):
        self.metadata_tree.clear()
        
        file_meta_item = QTreeWidgetItem(["File Metadata"])
        self.metadata_tree.addTopLevelItem(file_meta_item)
        for key, value in self.file_meta.items():
            item = QTreeWidgetItem([key, str(value['value'])])
            file_meta_item.addChild(item)

        image_meta_item = QTreeWidgetItem(["Image Metadata"])
        self.metadata_tree.addTopLevelItem(image_meta_item)
        for key, value in self.image_meta.items():
            if key == 'FITSKeywords':
                fits_item = QTreeWidgetItem(["FITS Keywords"])
                image_meta_item.addChild(fits_item)
                for kw, kw_values in value.items():
                    for kw_value in kw_values:
                        item = QTreeWidgetItem([kw, kw_value["value"]])
                        fits_item.addChild(item)
            elif key == 'XISFProperties':
                props_item = QTreeWidgetItem(["XISF Properties"])
                image_meta_item.addChild(props_item)
                for prop_name, prop in value.items():
                    item = QTreeWidgetItem([prop_name, str(prop["value"])])
                    props_item.addChild(item)
            else:
                item = QTreeWidgetItem([key, str(value)])
                image_meta_item.addChild(item)
        
        self.metadata_tree.expandAll()

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
                        self.save_tiff(output_path, image_data=image_to_save, bit_depth=16)
                    elif is_32bit_float:
                        self.save_tiff(output_path, image_data=image_to_save, bit_depth=32)
                    else:
                        self.save_tiff(output_path, image_data=image_to_save, bit_depth=8)
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

def stretch_mono_image(image, target_median, normalize=False):
    black_point = max(np.min(image), np.median(image) - 2.7 * np.std(image))
    rescaled_image = (image - black_point) / (1 - black_point)
    median_image = np.median(rescaled_image)
    stretched_image = ((median_image - 1) * target_median * rescaled_image) / (median_image * (target_median + rescaled_image - 1) - target_median * rescaled_image)

    if normalize:
        stretched_image = stretched_image / np.max(stretched_image)
    
    return np.clip(stretched_image, 0, 1)

def stretch_color_image(image, target_median, linked=True, normalize=False):
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
    
    if normalize:
        stretched_image = stretched_image / np.max(stretched_image)
    
    return np.clip(stretched_image, 0, 1)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    viewer = XISFViewer()
    viewer.show()
    sys.exit(app.exec_())
