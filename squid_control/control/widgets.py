# set QT_API environment variable
import os

import qtpy

import locale

# qt libraries
from qtpy.QtCore import *
from qtpy.QtWidgets import *
from qtpy.QtGui import *
import numpy as np
import pyqtgraph as pg

import pandas as pd

from datetime import datetime

from squid_control.control.config import CONFIG
from squid_control.control.camera import TriggerModeSetting


class WrapperWindow(QMainWindow):
    def __init__(self, content_widget, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setCentralWidget(content_widget)
        self.hide()

    def closeEvent(self, event):
        self.hide()
        event.ignore()

    def closeForReal(self, event):
        super().closeEvent(event)


class CollapsibleGroupBox(QGroupBox):
    def __init__(self, title):
        super(CollapsibleGroupBox, self).__init__(title)
        self.setCheckable(True)
        self.setChecked(True)
        self.higher_layout = QVBoxLayout()
        self.content = QVBoxLayout()
        # self.content.setAlignment(Qt.AlignTop)
        self.content_widget = QWidget()
        self.content_widget.setLayout(self.content)
        self.higher_layout.addWidget(self.content_widget)
        self.setLayout(self.higher_layout)
        self.toggled.connect(self.toggle_content)

    def toggle_content(self, state):
        self.content_widget.setVisible(state)


class ConfigEditorForAcquisitions(QDialog):
    def __init__(self, configManager, only_z_offset=True):
        super().__init__()

        self.config = configManager

        self.only_z_offset = only_z_offset

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area_widget = QWidget()
        self.scroll_area_layout = QVBoxLayout()
        self.scroll_area_widget.setLayout(self.scroll_area_layout)
        self.scroll_area.setWidget(self.scroll_area_widget)

        self.save_config_button = QPushButton("Save Config")
        self.save_config_button.clicked.connect(self.save_config)
        self.save_to_file_button = QPushButton("Save to File")
        self.save_to_file_button.clicked.connect(self.save_to_file)
        self.load_config_button = QPushButton("Load Config from File")
        self.load_config_button.clicked.connect(
            lambda: self.load_config_from_file(None)
        )

        layout = QVBoxLayout()
        layout.addWidget(self.scroll_area)
        layout.addWidget(self.save_config_button)
        layout.addWidget(self.save_to_file_button)
        layout.addWidget(self.load_config_button)

        self.config_value_widgets = {}

        self.setLayout(layout)
        self.setWindowTitle("Configuration Editor")
        self.init_ui(only_z_offset)

    def init_ui(self, only_z_offset=None):
        if only_z_offset is None:
            only_z_offset = self.only_z_offset
        self.groups = {}
        for section in self.config.configurations:
            if not only_z_offset:
                group_box = CollapsibleGroupBox(section.name)
            else:
                group_box = QGroupBox(section.name)

            group_layout = QVBoxLayout()

            section_value_widgets = {}

            self.groups[str(section.id)] = group_box

            for option in section.__dict__.keys():
                if option.startswith("_") and option.endswith("_options"):
                    continue
                if option == "id":
                    continue
                if only_z_offset and option != "z_offset":
                    continue
                option_value = str(getattr(section, option))
                option_name = QLabel(option)
                option_layout = QHBoxLayout()
                option_layout.addWidget(option_name)
                if f"_{option}_options" in list(section.__dict__.keys()):
                    option_value_list = getattr(section, f"_{option}_options")
                    values = option_value_list.strip("[]").split(",")
                    for i in range(len(values)):
                        values[i] = values[i].strip()
                    if option_value not in values:
                        values.append(option_value)
                    combo_box = QComboBox()
                    combo_box.addItems(values)
                    combo_box.setCurrentText(option_value)
                    option_layout.addWidget(combo_box)
                    section_value_widgets[option] = combo_box
                else:
                    option_input = QLineEdit(option_value)
                    option_layout.addWidget(option_input)
                    section_value_widgets[option] = option_input
                group_layout.addLayout(option_layout)

            self.config_value_widgets[str(section.id)] = section_value_widgets
            if not only_z_offset:
                group_box.content.addLayout(group_layout)
            else:
                group_box.setLayout(group_layout)

            self.scroll_area_layout.addWidget(group_box)

    def save_config(self):
        for section in self.config.configurations:
            for option in section.__dict__.keys():
                if option.startswith("_") and option.endswith("_options"):
                    continue
                old_val = getattr(section, option)
                if option == "id":
                    continue
                elif option == "camera_sn":
                    option_name_in_xml = "CameraSN"
                else:
                    option_name_in_xml = (
                        option.replace("_", " ").title().replace(" ", "")
                    )
                try:
                    widget = self.config_value_widgets[str(section.id)][option]
                except KeyError:
                    continue
                if type(widget) is QLineEdit:
                    self.config.update_configuration(
                        section.id, option_name_in_xml, widget.text()
                    )
                else:
                    self.config.update_configuration(
                        section.id, option_name_in_xml, widget.currentText()
                    )
        self.config.configurations = []
        self.config.read_configurations()

    def save_to_file(self):
        self.save_config()
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Acquisition Config File", "", "XML Files (*.xml);;All Files (*)"
        )
        if file_path:
            self.config.write_configuration(file_path)

    def load_config_from_file(self, only_z_offset=None):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Acquisition Config File", "", "XML Files (*.xml);;All Files (*)"
        )
        if file_path:
            self.config.config_filename = file_path
            self.config.configurations = []
            self.config.read_configurations()
            # Clear and re-initialize the UI
            self.scroll_area_widget.deleteLater()
            self.scroll_area_widget = QWidget()
            self.scroll_area_layout = QVBoxLayout()
            self.scroll_area_widget.setLayout(self.scroll_area_layout)
            self.scroll_area.setWidget(self.scroll_area_widget)
            self.init_ui(only_z_offset)


class ConfigEditor(QDialog):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area_widget = QWidget()
        self.scroll_area_layout = QVBoxLayout()
        self.scroll_area_widget.setLayout(self.scroll_area_layout)
        self.scroll_area.setWidget(self.scroll_area_widget)

        self.save_config_button = QPushButton("Save Config")
        self.save_config_button.clicked.connect(self.save_config)
        self.save_to_file_button = QPushButton("Save to File")
        self.save_to_file_button.clicked.connect(self.save_to_file)
        self.load_config_button = QPushButton("Load Config from File")
        self.load_config_button.clicked.connect(self.load_config_from_file)

        layout = QVBoxLayout()
        layout.addWidget(self.scroll_area)
        layout.addWidget(self.save_config_button)
        layout.addWidget(self.save_to_file_button)
        layout.addWidget(self.load_config_button)

        self.config_value_widgets = {}

        self.setLayout(layout)
        self.setWindowTitle("Configuration Editor")
        self.init_ui()

    def init_ui(self):
        self.groups = {}
        for section in self.config.sections():
            group_box = CollapsibleGroupBox(section)
            group_layout = QVBoxLayout()

            section_value_widgets = {}

            self.groups[section] = group_box

            for option in self.config.options(section):
                if option.startswith("_") and option.endswith("_options"):
                    continue
                option_value = self.config.get(section, option)
                option_name = QLabel(option)
                option_layout = QHBoxLayout()
                option_layout.addWidget(option_name)
                if f"_{option}_options" in self.config.options(section):
                    option_value_list = self.config.get(section, f"_{option}_options")
                    values = option_value_list.strip("[]").split(",")
                    for i in range(len(values)):
                        values[i] = values[i].strip()
                    if option_value not in values:
                        values.append(option_value)
                    combo_box = QComboBox()
                    combo_box.addItems(values)
                    combo_box.setCurrentText(option_value)
                    option_layout.addWidget(combo_box)
                    section_value_widgets[option] = combo_box
                else:
                    option_input = QLineEdit(option_value)
                    option_layout.addWidget(option_input)
                    section_value_widgets[option] = option_input
                group_layout.addLayout(option_layout)

            self.config_value_widgets[section] = section_value_widgets
            group_box.content.addLayout(group_layout)
            self.scroll_area_layout.addWidget(group_box)

    def save_config(self):
        for section in self.config.sections():
            for option in self.config.options(section):
                if option.startswith("_") and option.endswith("_options"):
                    continue
                old_val = self.config.get(section, option)
                widget = self.config_value_widgets[section][option]
                if type(widget) is QLineEdit:
                    self.config.set(section, option, widget.text())
                else:
                    self.config.set(section, option, widget.currentText())
                if old_val != self.config.get(section, option):
                    print(self.config.get(section, option))

    def save_to_file(self):
        self.save_config()
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Config File", "", "INI Files (*.ini);;All Files (*)"
        )
        if file_path:
            with open(file_path, "w") as configfile:
                self.config.write(configfile)

    def load_config_from_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Config File", "", "INI Files (*.ini);;All Files (*)"
        )
        if file_path:
            self.config.read(file_path)
            # Clear and re-initialize the UI
            self.scroll_area_widget.deleteLater()
            self.scroll_area_widget = QWidget()
            self.scroll_area_layout = QVBoxLayout()
            self.scroll_area_widget.setLayout(self.scroll_area_layout)
            self.scroll_area.setWidget(self.scroll_area_widget)
            self.init_ui()


class ConfigEditorBackwardsCompatible(ConfigEditor):
    def __init__(self, config, original_filepath, main_window):
        super().__init__(config)
        self.original_filepath = original_filepath
        self.main_window = main_window

        self.apply_exit_button = QPushButton("Apply and Exit")
        self.apply_exit_button.clicked.connect(self.apply_and_exit)

        self.layout().addWidget(self.apply_exit_button)

    def apply_and_exit(self):
        self.save_config()
        with open(self.original_filepath, "w") as configfile:
            self.config.write(configfile)
        try:
            self.main_window.close()
        except:
            pass
        self.close()


class SpinningDiskConfocalWidget(QWidget):
    def __init__(self, xlight, config_manager=None):
        super(SpinningDiskConfocalWidget, self).__init__()

        self.config_manager = config_manager

        self.xlight = xlight

        self.init_ui()

        self.dropdown_emission_filter.setCurrentText(
            str(self.xlight.get_emission_filter())
        )
        self.dropdown_dichroic.setCurrentText(str(self.xlight.get_dichroic()))

        self.dropdown_emission_filter.currentIndexChanged.connect(
            self.set_emission_filter
        )
        self.dropdown_dichroic.currentIndexChanged.connect(self.set_dichroic)

        self.disk_position_state = self.xlight.get_disk_position()

        if self.disk_position_state == 1:
            self.btn_toggle_widefield.setText("Switch to Widefield")

        if self.config_manager is not None:
            if self.disk_position_state == 1:
                self.config_manager.config_filename = "confocal_configurations.xml"
            else:
                self.config_manager.config_filename = "widefield_configurations.xml"
            self.config_manager.configurations = []
            self.config_manager.read_configurations()

        self.btn_toggle_widefield.clicked.connect(self.toggle_disk_position)

        self.btn_toggle_motor.clicked.connect(self.toggle_motor)

    def init_ui(self):

        emissionFilterLayout = QHBoxLayout()
        emissionFilterLayout.addWidget(QLabel("Emission Filter Position"))

        self.dropdown_emission_filter = QComboBox(self)
        self.dropdown_emission_filter.addItems([str(i + 1) for i in range(8)])

        emissionFilterLayout.addWidget(self.dropdown_emission_filter)

        dichroicLayout = QHBoxLayout()
        dichroicLayout.addWidget(QLabel("Dichroic Position"))

        self.dropdown_dichroic = QComboBox(self)
        self.dropdown_dichroic.addItems([str(i + 1) for i in range(5)])

        dichroicLayout.addWidget(self.dropdown_dichroic)

        dropdownLayout = QVBoxLayout()

        dropdownLayout.addLayout(dichroicLayout)
        dropdownLayout.addLayout(emissionFilterLayout)
        dropdownLayout.addStretch()

        self.btn_toggle_widefield = QPushButton("Switch to Confocal")

        self.btn_toggle_motor = QPushButton("Disk Motor On")
        self.btn_toggle_motor.setCheckable(True)

        layout = QVBoxLayout(self)
        layout.addWidget(self.btn_toggle_motor)

        layout.addWidget(self.btn_toggle_widefield)
        layout.addLayout(dropdownLayout)
        self.setLayout(layout)

    def disable_all_buttons(self):
        self.dropdown_emission_filter.setEnabled(False)
        self.dropdown_dichroic.setEnabled(False)
        self.btn_toggle_widefield.setEnabled(False)
        self.btn_toggle_motor.setEnabled(False)

    def enable_all_buttons(self):
        self.dropdown_emission_filter.setEnabled(True)
        self.dropdown_dichroic.setEnabled(True)
        self.btn_toggle_widefield.setEnabled(True)
        self.btn_toggle_motor.setEnabled(True)

    def toggle_disk_position(self):
        self.disable_all_buttons()
        if self.disk_position_state == 1:
            self.disk_position_state = self.xlight.set_disk_position(0)
            self.btn_toggle_widefield.setText("Switch to Confocal")
        else:
            self.disk_position_state = self.xlight.set_disk_position(1)
            self.btn_toggle_widefield.setText("Switch to Widefield")
        if self.config_manager is not None:
            if self.disk_position_state == 1:
                self.config_manager.config_filename = "confocal_configurations.xml"
            else:
                self.config_manager.config_filename = "widefield_configurations.xml"
            self.config_manager.configurations = []
            self.config_manager.read_configurations()
        self.enable_all_buttons()

    def toggle_motor(self):
        self.disable_all_buttons()
        if self.btn_toggle_motor.isChecked():
            self.xlight.set_disk_motor_state(True)
        else:
            self.xlight.set_disk_motor_state(False)
        self.enable_all_buttons()

    def set_emission_filter(self, index):
        self.disable_all_buttons()
        selected_pos = self.dropdown_emission_filter.currentText()
        self.xlight.set_emission_filter(selected_pos)
        self.enable_all_buttons()

    def set_dichroic(self, index):
        self.disable_all_buttons()
        selected_pos = self.dropdown_dichroic.currentText()
        self.xlight.set_dichroic(selected_pos)
        self.enable_all_buttons()


class ObjectivesWidget(QWidget):
    def __init__(self, objective_store):
        super(ObjectivesWidget, self).__init__()

        self.objectiveStore = objective_store

        self.init_ui()

        self.dropdown.setCurrentText(self.objectiveStore.current_objective)

    def init_ui(self):
        # Dropdown for selecting keys
        self.dropdown = QComboBox(self)
        self.dropdown.addItems(self.objectiveStore.objectives_dict.keys())
        self.dropdown.currentIndexChanged.connect(self.display_objective)

        # TextBrowser to display key-value pairs
        # self.text_browser = QTextBrowser(self)
        # Layout
        dropdownLayout = QHBoxLayout()
        dropdownLabel = QLabel("Objectives:")
        dropdownLayout.addWidget(dropdownLabel)
        dropdownLayout.addWidget(self.dropdown)
        # textLayout = QHBoxLayout()
        # textLayout.addWidget(self.text_browser)
        layout = QVBoxLayout(self)
        layout.addLayout(dropdownLayout)
        # layout.addLayout(textLayout)

    def display_objective(self, index):
        selected_key = self.dropdown.currentText()
        objective_data = self.objectiveStore.objectives_dict.get(selected_key, {})
        # text = "\n".join([f"{key}: {value}" for key, value in objective_data.items()])
        self.objectiveStore.current_objective = selected_key
        # self.text_browser.setPlainText(text)


class FocusMapWidget(QWidget):

    def __init__(self, autofocusController, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.autofocusController = autofocusController
        self.init_ui()

    def init_ui(self):
        self.btn_add_to_focusmap = QPushButton("Add to focus map")
        self.btn_enable_focusmap = QPushButton("Enable focus map")
        self.btn_clear_focusmap = QPushButton("Clear focus map")
        self.fmap_coord_1 = QLabel("Focus Map Point 1: (xxx,yyy,zzz)")
        self.fmap_coord_2 = QLabel("Focus Map Point 2: (xxx,yyy,zzz)")
        self.fmap_coord_3 = QLabel("Focus Map Point 3: (xxx,yyy,zzz)")
        layout = QVBoxLayout()
        layout.addWidget(self.fmap_coord_1)
        layout.addWidget(self.fmap_coord_2)
        layout.addWidget(self.fmap_coord_3)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.btn_add_to_focusmap)
        button_layout.addWidget(self.btn_clear_focusmap)

        layout.addLayout(button_layout)

        layout.addWidget(self.btn_enable_focusmap)

        self.setLayout(layout)

        self.btn_add_to_focusmap.clicked.connect(self.add_to_focusmap)
        self.btn_enable_focusmap.clicked.connect(self.enable_focusmap)
        self.btn_clear_focusmap.clicked.connect(self.clear_focusmap)

    def disable_all_buttons(self):
        self.btn_add_to_focusmap.setEnabled(False)
        self.btn_enable_focusmap.setEnabled(False)
        self.btn_clear_focusmap.setEnabled(False)

    def enable_all_buttons(self):
        self.btn_add_to_focusmap.setEnabled(True)
        self.btn_enable_focusmap.setEnabled(True)
        self.btn_clear_focusmap.setEnabled(True)

    def clear_focusmap(self):
        self.disable_all_buttons()
        self.autofocusController.clear_focus_map()
        self.update_focusmap_display()
        self.btn_enable_focusmap.setText("Enable focus map")
        self.enable_all_buttons()

    def update_focusmap_display(self):
        self.fmap_coord_1.setText("Focus Map Point 1: (xxx,yyy,zzz)")
        self.fmap_coord_2.setText("Focus Map Point 2: (xxx,yyy,zzz)")
        self.fmap_coord_3.setText("Focus Map Point 3: (xxx,yyy,zzz)")
        try:
            x, y, z = self.autofocusController.focus_map_coords[0]
            self.fmap_coord_1.setText(f"Focus Map Point 1: ({x:.3f},{y:.3f},{z:.3f})")
        except IndexError:
            pass
        try:
            x, y, z = self.autofocusController.focus_map_coords[1]
            self.fmap_coord_2.setText(f"Focus Map Point 2: ({x:.3f},{y:.3f},{z:.3f})")
        except IndexError:
            pass
        try:
            x, y, z = self.autofocusController.focus_map_coords[2]
            self.fmap_coord_3.setText(f"Focus Map Point 3: ({x:.3f},{y:.3f},{z:.3f})")
        except IndexError:
            pass

    def enable_focusmap(self):
        self.disable_all_buttons()
        if self.autofocusController.use_focus_map == False:
            self.autofocusController.set_focus_map_use(True)
        else:
            self.autofocusController.set_focus_map_use(False)
        if self.autofocusController.use_focus_map:
            self.btn_enable_focusmap.setText("Disable focus map")
        else:
            self.btn_enable_focusmap.setText("Enable focus map")
        self.enable_all_buttons()

    def add_to_focusmap(self):
        self.disable_all_buttons()
        try:
            self.autofocusController.add_current_coords_to_focus_map()
        except ValueError:
            pass
        self.update_focusmap_display()
        self.enable_all_buttons()


class CameraSettingsWidget(QFrame):

    def __init__(
        self,
        camera,
        include_gain_exposure_time=True,
        include_camera_temperature_setting=False,
        main=None,
        *args,
        **kwargs,
    ):

        super().__init__(*args, **kwargs)
        self.camera = camera
        self.add_components(
            include_gain_exposure_time, include_camera_temperature_setting
        )
        # set frame style
        self.setFrameStyle(QFrame.Panel | QFrame.Raised)

    def add_components(
        self, include_gain_exposure_time, include_camera_temperature_setting
    ):

        # add buttons and input fields
        self.entry_exposureTime = QDoubleSpinBox()
        self.entry_exposureTime.setMinimum(self.camera.EXPOSURE_TIME_MS_MIN)
        self.entry_exposureTime.setMaximum(self.camera.EXPOSURE_TIME_MS_MAX)
        self.entry_exposureTime.setSingleStep(1)
        self.entry_exposureTime.setValue(20)
        self.camera.set_exposure_time(20)

        self.entry_analogGain = QDoubleSpinBox()
        self.entry_analogGain.setMinimum(self.camera.GAIN_MIN)
        self.entry_analogGain.setMaximum(self.camera.GAIN_MAX)
        self.entry_analogGain.setSingleStep(self.camera.GAIN_STEP)
        self.entry_analogGain.setValue(0)
        self.camera.set_analog_gain(0)

        self.dropdown_pixelFormat = QComboBox()
        self.dropdown_pixelFormat.addItems(
            ["MONO8", "MONO12", "MONO14", "MONO16", "BAYER_RG8", "BAYER_RG12"]
        )
        if self.camera.pixel_format is not None:
            self.dropdown_pixelFormat.setCurrentText(self.camera.pixel_format)
        else:
            print("setting camera's default pixel format")
            self.camera.set_pixel_format(CONFIG.DEFAULT_PIXEL_FORMAT)
            self.dropdown_pixelFormat.setCurrentText(CONFIG.DEFAULT_PIXEL_FORMAT)
        # to do: load and save pixel format in configurations

        self.entry_ROI_offset_x = QSpinBox()
        self.entry_ROI_offset_x.setValue(self.camera.OffsetX)
        self.entry_ROI_offset_x.setSingleStep(8)
        self.entry_ROI_offset_x.setFixedWidth(60)
        self.entry_ROI_offset_x.setMinimum(0)
        self.entry_ROI_offset_x.setMaximum(self.camera.WidthMax)
        self.entry_ROI_offset_x.setKeyboardTracking(False)
        self.entry_ROI_offset_y = QSpinBox()
        self.entry_ROI_offset_y.setValue(self.camera.OffsetY)
        self.entry_ROI_offset_y.setSingleStep(8)
        self.entry_ROI_offset_y.setFixedWidth(60)
        self.entry_ROI_offset_y.setMinimum(0)
        self.entry_ROI_offset_y.setMaximum(self.camera.HeightMax)
        self.entry_ROI_offset_y.setKeyboardTracking(False)
        self.entry_ROI_width = QSpinBox()
        self.entry_ROI_width.setMinimum(16)
        self.entry_ROI_width.setMaximum(self.camera.WidthMax)
        self.entry_ROI_width.setValue(self.camera.Width)
        self.entry_ROI_width.setSingleStep(8)
        self.entry_ROI_width.setFixedWidth(60)
        self.entry_ROI_width.setKeyboardTracking(False)
        self.entry_ROI_height = QSpinBox()
        self.entry_ROI_height.setSingleStep(8)
        self.entry_ROI_height.setMinimum(16)
        self.entry_ROI_height.setMaximum(self.camera.HeightMax)
        self.entry_ROI_height.setValue(self.camera.Height)
        self.entry_ROI_height.setFixedWidth(60)
        self.entry_ROI_height.setKeyboardTracking(False)
        self.entry_temperature = QDoubleSpinBox()
        self.entry_temperature.setMaximum(25)
        self.entry_temperature.setMinimum(-50)
        self.entry_temperature.setDecimals(1)
        self.label_temperature_measured = QLabel()
        # self.label_temperature_measured.setNum(0)
        self.label_temperature_measured.setFrameStyle(QFrame.Panel | QFrame.Sunken)

        # connection
        self.entry_exposureTime.valueChanged.connect(self.camera.set_exposure_time)
        self.entry_analogGain.valueChanged.connect(self.camera.set_analog_gain)
        self.dropdown_pixelFormat.currentTextChanged.connect(
            self.camera.set_pixel_format
        )
        self.entry_ROI_offset_x.valueChanged.connect(self.set_ROI_offset)
        self.entry_ROI_offset_y.valueChanged.connect(self.set_ROI_offset)
        self.entry_ROI_height.valueChanged.connect(self.set_Height)
        self.entry_ROI_width.valueChanged.connect(self.set_Width)

        # layout
        grid_ctrl = QGridLayout()
        if include_gain_exposure_time:
            grid_ctrl.addWidget(QLabel("Exposure Time (ms)"), 0, 0)
            grid_ctrl.addWidget(self.entry_exposureTime, 0, 1)
            grid_ctrl.addWidget(QLabel("Analog Gain"), 1, 0)
            grid_ctrl.addWidget(self.entry_analogGain, 1, 1)
        grid_ctrl.addWidget(QLabel("Pixel Format"), 2, 0)
        grid_ctrl.addWidget(self.dropdown_pixelFormat, 2, 1)
        try:
            current_res = self.camera.resolution
            current_res_string = "x".join([str(current_res[0]), str(current_res[1])])
            res_options = [f"{res[0]}x{res[1]}" for res in self.camera.res_list]
            self.dropdown_res = QComboBox()
            self.dropdown_res.addItems(res_options)
            self.dropdown_res.setCurrentText(current_res_string)

            self.dropdown_res.currentTextChanged.connect(self.change_full_res)
            grid_ctrl.addWidget(QLabel("Full Resolution"), 2, 2)
            grid_ctrl.addWidget(self.dropdown_res, 2, 3)
        except AttributeError:
            pass
        if include_camera_temperature_setting:
            grid_ctrl.addWidget(QLabel("Set Temperature (C)"), 3, 0)
            grid_ctrl.addWidget(self.entry_temperature, 3, 1)
            grid_ctrl.addWidget(QLabel("Actual Temperature (C)"), 3, 2)
            grid_ctrl.addWidget(self.label_temperature_measured, 3, 3)
            try:
                self.entry_temperature.valueChanged.connect(self.set_temperature)
                self.camera.set_temperature_reading_callback(
                    self.update_measured_temperature
                )
            except AttributeError:
                pass

        hbox1 = QHBoxLayout()
        hbox1.addWidget(QLabel("ROI"))
        hbox1.addStretch()
        hbox1.addWidget(QLabel("height"))
        hbox1.addWidget(self.entry_ROI_height)
        hbox1.addWidget(QLabel("width"))
        hbox1.addWidget(self.entry_ROI_width)

        hbox1.addWidget(QLabel("offset y"))
        hbox1.addWidget(self.entry_ROI_offset_y)
        hbox1.addWidget(QLabel("offset x"))
        hbox1.addWidget(self.entry_ROI_offset_x)

        self.grid = QGridLayout()
        self.grid.addLayout(grid_ctrl, 0, 0)
        self.grid.addLayout(hbox1, 1, 0)
        self.setLayout(self.grid)

    def set_exposure_time(self, exposure_time):
        self.entry_exposureTime.setValue(exposure_time)

    def set_analog_gain(self, analog_gain):
        self.entry_analogGain.setValue(analog_gain)

    def set_Width(self):
        width = int(self.entry_ROI_width.value() // 8) * 8
        self.entry_ROI_width.blockSignals(True)
        self.entry_ROI_width.setValue(width)
        self.entry_ROI_width.blockSignals(False)
        offset_x = (self.camera.WidthMax - self.entry_ROI_width.value()) / 2
        offset_x = int(offset_x // 8) * 8
        self.entry_ROI_offset_x.blockSignals(True)
        self.entry_ROI_offset_x.setValue(offset_x)
        self.entry_ROI_offset_x.blockSignals(False)
        self.camera.set_ROI(
            self.entry_ROI_offset_x.value(),
            self.entry_ROI_offset_y.value(),
            self.entry_ROI_width.value(),
            self.entry_ROI_height.value(),
        )

    def set_Height(self):
        height = int(self.entry_ROI_height.value() // 8) * 8
        self.entry_ROI_height.blockSignals(True)
        self.entry_ROI_height.setValue(height)
        self.entry_ROI_height.blockSignals(False)
        offset_y = (self.camera.HeightMax - self.entry_ROI_height.value()) / 2
        offset_y = int(offset_y // 8) * 8
        self.entry_ROI_offset_y.blockSignals(True)
        self.entry_ROI_offset_y.setValue(offset_y)
        self.entry_ROI_offset_y.blockSignals(False)
        self.camera.set_ROI(
            self.entry_ROI_offset_x.value(),
            self.entry_ROI_offset_y.value(),
            self.entry_ROI_width.value(),
            self.entry_ROI_height.value(),
        )

    def set_ROI_offset(self):
        self.camera.set_ROI(
            self.entry_ROI_offset_x.value(),
            self.entry_ROI_offset_y.value(),
            self.entry_ROI_width.value(),
            self.entry_ROI_height.value(),
        )

    def set_temperature(self):
        try:
            self.camera.set_temperature(float(self.entry_temperature.value()))
        except AttributeError:
            pass

    def update_measured_temperature(self, temperature):
        self.label_temperature_measured.setNum(temperature)

    def change_full_res(self, index):
        res_strings = self.dropdown_res.currentText().split("x")
        res_x = int(res_strings[0])
        res_y = int(res_strings[1])
        self.camera.set_resolution(res_x, res_y)
        self.entry_ROI_offset_x.blockSignals(True)
        self.entry_ROI_offset_y.blockSignals(True)
        self.entry_ROI_height.blockSignals(True)
        self.entry_ROI_width.blockSignals(True)

        self.entry_ROI_height.setMaximum(self.camera.HeightMax)
        self.entry_ROI_width.setMaximum(self.camera.WidthMax)

        self.entry_ROI_offset_x.setMaximum(self.camera.WidthMax)
        self.entry_ROI_offset_y.setMaximum(self.camera.HeightMax)

        self.entry_ROI_offset_x.setValue(int(8 * self.camera.OffsetX // 8))
        self.entry_ROI_offset_y.setValue(int(8 * self.camera.OffsetY // 8))
        self.entry_ROI_height.setValue(int(8 * self.camera.Height // 8))
        self.entry_ROI_width.setValue(int(8 * self.camera.Width // 8))

        self.entry_ROI_offset_x.blockSignals(False)
        self.entry_ROI_offset_y.blockSignals(False)
        self.entry_ROI_height.blockSignals(False)
        self.entry_ROI_width.blockSignals(False)


class LiveControlWidget(QFrame):
    signal_newExposureTime = Signal(float)
    signal_newAnalogGain = Signal(float)
    signal_autoLevelSetting = Signal(bool)

    def __init__(
        self,
        streamHandler,
        liveController,
        configurationManager=None,
        show_trigger_options=True,
        show_display_options=True,
        show_autolevel=False,
        autolevel=False,
        main=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.liveController = liveController
        self.streamHandler = streamHandler
        self.configurationManager = configurationManager
        self.fps_trigger = 10
        self.fps_display = 10
        self.liveController.set_trigger_fps(self.fps_trigger)
        self.streamHandler.set_display_fps(self.fps_display)

        self.triggerMode = TriggerModeSetting.SOFTWARE
        # note that this references the object in self.configurationManager.configurations
        self.currentConfiguration = self.configurationManager.configurations[0]

        self.add_components(
            show_trigger_options, show_display_options, show_autolevel, autolevel
        )
        self.setFrameStyle(QFrame.Panel | QFrame.Raised)
        self.update_microscope_mode_by_name(self.currentConfiguration.name)

        self.is_switching_mode = False  # flag used to prevent from settings being set by twice - from both mode change slot and value change slot; another way is to use blockSignals(True)

    def add_components(
        self, show_trigger_options, show_display_options, show_autolevel, autolevel
    ):
        # line 0: trigger mode
        self.triggerMode = None
        self.dropdown_triggerManu = QComboBox()
        self.dropdown_triggerManu.addItems(
            [
                TriggerModeSetting.SOFTWARE.value,
                TriggerModeSetting.HARDWARE.value,
                TriggerModeSetting.CONTINUOUS.value,
            ]
        )

        # line 1: fps
        self.entry_triggerFPS = QDoubleSpinBox()
        self.entry_triggerFPS.setMinimum(0.02)
        self.entry_triggerFPS.setMaximum(1000)
        self.entry_triggerFPS.setSingleStep(1)
        self.entry_triggerFPS.setValue(self.fps_trigger)

        # line 2: choose microscope mode / toggle live mode
        self.dropdown_modeSelection = QComboBox()
        for microscope_configuration in self.configurationManager.configurations:
            self.dropdown_modeSelection.addItems([microscope_configuration.name])
        self.dropdown_modeSelection.setCurrentText(self.currentConfiguration.name)

        self.btn_live = QPushButton("Live")
        self.btn_live.setCheckable(True)
        self.btn_live.setChecked(False)
        self.btn_live.setDefault(False)

        # line 3: exposure time and analog gain associated with the current mode
        self.entry_exposureTime = QDoubleSpinBox()
        self.entry_exposureTime.setMinimum(
            self.liveController.camera.EXPOSURE_TIME_MS_MIN
        )
        self.entry_exposureTime.setMaximum(
            self.liveController.camera.EXPOSURE_TIME_MS_MAX
        )
        self.entry_exposureTime.setSingleStep(1)
        self.entry_exposureTime.setValue(0)

        self.entry_analogGain = QDoubleSpinBox()
        self.entry_analogGain = QDoubleSpinBox()
        self.entry_analogGain.setMinimum(0)
        self.entry_analogGain.setMaximum(24)
        self.entry_analogGain.setSingleStep(0.1)
        self.entry_analogGain.setValue(0)

        self.slider_illuminationIntensity = QSlider(Qt.Horizontal)
        self.slider_illuminationIntensity.setTickPosition(QSlider.TicksBelow)
        self.slider_illuminationIntensity.setMinimum(0)
        self.slider_illuminationIntensity.setMaximum(100)
        self.slider_illuminationIntensity.setValue(100)
        self.slider_illuminationIntensity.setSingleStep(1)

        self.entry_illuminationIntensity = QDoubleSpinBox()
        self.entry_illuminationIntensity.setMinimum(0)
        self.entry_illuminationIntensity.setMaximum(100)
        self.entry_illuminationIntensity.setSingleStep(1)
        self.entry_illuminationIntensity.setValue(100)

        # line 4: display fps and resolution scaling
        self.entry_displayFPS = QDoubleSpinBox()
        self.entry_displayFPS.setMinimum(1)
        self.entry_displayFPS.setMaximum(240)
        self.entry_displayFPS.setSingleStep(1)
        self.entry_displayFPS.setValue(self.fps_display)

        self.slider_resolutionScaling = QSlider(Qt.Horizontal)
        self.slider_resolutionScaling.setTickPosition(QSlider.TicksBelow)
        self.slider_resolutionScaling.setMinimum(10)
        self.slider_resolutionScaling.setMaximum(100)
        self.slider_resolutionScaling.setValue(CONFIG.DEFAULT_DISPLAY_CROP)
        self.slider_resolutionScaling.setSingleStep(10)

        # autolevel
        self.btn_autolevel = QPushButton("Autolevel")
        self.btn_autolevel.setCheckable(True)
        self.btn_autolevel.setChecked(autolevel)

        # connections
        self.entry_triggerFPS.valueChanged.connect(self.liveController.set_trigger_fps)
        self.entry_displayFPS.valueChanged.connect(self.streamHandler.set_display_fps)
        self.slider_resolutionScaling.valueChanged.connect(
            self.streamHandler.set_display_resolution_scaling
        )
        self.slider_resolutionScaling.valueChanged.connect(
            self.liveController.set_display_resolution_scaling
        )
        self.dropdown_modeSelection.currentTextChanged.connect(
            self.update_microscope_mode_by_name
        )
        self.dropdown_triggerManu.currentIndexChanged.connect(self.update_trigger_mode)
        self.btn_live.clicked.connect(self.toggle_live)
        self.entry_exposureTime.valueChanged.connect(self.update_config_exposure_time)
        self.entry_analogGain.valueChanged.connect(self.update_config_analog_gain)
        self.entry_illuminationIntensity.valueChanged.connect(
            self.update_config_illumination_intensity
        )
        self.entry_illuminationIntensity.valueChanged.connect(
            lambda x: self.slider_illuminationIntensity.setValue(int(x))
        )
        self.slider_illuminationIntensity.valueChanged.connect(
            self.entry_illuminationIntensity.setValue
        )
        self.btn_autolevel.clicked.connect(self.signal_autoLevelSetting.emit)

        # layout
        grid_line0 = QGridLayout()
        grid_line0.addWidget(QLabel("Trigger Mode"), 0, 0)
        grid_line0.addWidget(self.dropdown_triggerManu, 0, 1)
        grid_line0.addWidget(QLabel("Trigger FPS"), 0, 2)
        grid_line0.addWidget(self.entry_triggerFPS, 0, 3)

        grid_line1 = QGridLayout()
        grid_line1.addWidget(QLabel("Microscope Configuration"), 0, 0)
        grid_line1.addWidget(self.dropdown_modeSelection, 0, 1)
        grid_line1.addWidget(self.btn_live, 0, 2)

        grid_line2 = QGridLayout()
        grid_line2.addWidget(QLabel("Exposure Time (ms)"), 0, 0)
        grid_line2.addWidget(self.entry_exposureTime, 0, 1)
        grid_line2.addWidget(QLabel("Analog Gain"), 0, 2)
        grid_line2.addWidget(self.entry_analogGain, 0, 3)

        grid_line4 = QGridLayout()
        grid_line4.addWidget(QLabel("Illumination"), 0, 0)
        grid_line4.addWidget(self.slider_illuminationIntensity, 0, 1)
        grid_line4.addWidget(self.entry_illuminationIntensity, 0, 2)

        grid_line3 = QGridLayout()
        grid_line3.addWidget(QLabel("Display FPS"), 0, 0)
        grid_line3.addWidget(self.entry_displayFPS, 0, 1)
        grid_line3.addWidget(QLabel("Display Resolution"), 0, 2)
        grid_line3.addWidget(self.slider_resolutionScaling, 0, 3)
        if show_autolevel:
            grid_line3.addWidget(self.btn_autolevel, 0, 4)

        self.grid = QVBoxLayout()
        if show_trigger_options:
            self.grid.addLayout(grid_line0)
        self.grid.addLayout(grid_line1)
        self.grid.addLayout(grid_line2)
        self.grid.addLayout(grid_line4)
        if show_display_options:
            self.grid.addLayout(grid_line3)
        self.grid.addStretch()
        self.setLayout(self.grid)

    def toggle_live(self, pressed):
        if pressed:
            self.liveController.start_live()
        else:
            self.liveController.stop_live()

    def update_camera_settings(self):
        self.signal_newAnalogGain.emit(self.entry_analogGain.value())
        self.signal_newExposureTime.emit(self.entry_exposureTime.value())

    def update_microscope_mode_by_name(self, current_microscope_mode_name):
        self.is_switching_mode = True
        # identify the mode selected (note that this references the object in self.configurationManager.configurations)
        self.currentConfiguration = next(
            (
                config
                for config in self.configurationManager.configurations
                if config.name == current_microscope_mode_name
            ),
            None,
        )
        # update the microscope to the current configuration
        self.liveController.set_microscope_mode(self.currentConfiguration)
        # update the exposure time and analog gain settings according to the selected configuration
        self.entry_exposureTime.setValue(self.currentConfiguration.exposure_time)
        self.entry_analogGain.setValue(self.currentConfiguration.analog_gain)
        self.entry_illuminationIntensity.setValue(
            self.currentConfiguration.illumination_intensity
        )
        self.is_switching_mode = False

    def update_trigger_mode(self):
        self.liveController.set_trigger_mode(self.dropdown_triggerManu.currentText())

    def update_config_exposure_time(self, new_value):
        if self.is_switching_mode == False:
            self.currentConfiguration.exposure_time = new_value
            self.configurationManager.update_configuration(
                self.currentConfiguration.id, "ExposureTime", new_value
            )
            self.signal_newExposureTime.emit(new_value)

    def update_config_analog_gain(self, new_value):
        if self.is_switching_mode == False:
            self.currentConfiguration.analog_gain = new_value
            self.configurationManager.update_configuration(
                self.currentConfiguration.id, "AnalogGain", new_value
            )
            self.signal_newAnalogGain.emit(new_value)

    def update_config_illumination_intensity(self, new_value):
        if self.is_switching_mode == False:
            self.currentConfiguration.illumination_intensity = new_value
            self.configurationManager.update_configuration(
                self.currentConfiguration.id, "IlluminationIntensity", new_value
            )
            self.liveController.set_illumination(
                self.currentConfiguration.illumination_source,
                self.currentConfiguration.illumination_intensity,
            )

    def set_microscope_mode(self, config):
        # self.liveController.set_microscope_mode(config)
        self.dropdown_modeSelection.setCurrentText(config.name)

    def set_trigger_mode(self, trigger_mode):
        self.dropdown_triggerManu.setCurrentText(trigger_mode)
        self.liveController.set_trigger_mode(self.dropdown_triggerManu.currentText())


class RecordingWidget(QFrame):
    def __init__(self, streamHandler, imageSaver, main=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.imageSaver = imageSaver  # for saving path control
        self.streamHandler = streamHandler
        self.base_path_is_set = False
        self.add_components()
        self.setFrameStyle(QFrame.Panel | QFrame.Raised)

    def add_components(self):
        self.btn_setSavingDir = QPushButton("Browse")
        self.btn_setSavingDir.setDefault(False)
        self.btn_setSavingDir.setIcon(QIcon("icon/folder.png"))

        self.lineEdit_savingDir = QLineEdit()
        self.lineEdit_savingDir.setReadOnly(True)
        self.lineEdit_savingDir.setText("Choose a base saving directory")

        self.lineEdit_savingDir.setText(CONFIG.DEFAULT_SAVING_PATH)
        self.imageSaver.set_base_path(CONFIG.DEFAULT_SAVING_PATH)

        self.lineEdit_experimentID = QLineEdit()

        self.entry_saveFPS = QDoubleSpinBox()
        self.entry_saveFPS.setMinimum(0.02)
        self.entry_saveFPS.setMaximum(1000)
        self.entry_saveFPS.setSingleStep(1)
        self.entry_saveFPS.setValue(1)
        self.streamHandler.set_save_fps(1)

        self.entry_timeLimit = QSpinBox()
        self.entry_timeLimit.setMinimum(-1)
        self.entry_timeLimit.setMaximum(60 * 60 * 24 * 30)
        self.entry_timeLimit.setSingleStep(1)
        self.entry_timeLimit.setValue(-1)

        self.btn_record = QPushButton("Record")
        self.btn_record.setCheckable(True)
        self.btn_record.setChecked(False)
        self.btn_record.setDefault(False)

        grid_line1 = QGridLayout()
        grid_line1.addWidget(QLabel("Saving Path"))
        grid_line1.addWidget(self.lineEdit_savingDir, 0, 1)
        grid_line1.addWidget(self.btn_setSavingDir, 0, 2)

        grid_line2 = QGridLayout()
        grid_line2.addWidget(QLabel("Experiment ID"), 0, 0)
        grid_line2.addWidget(self.lineEdit_experimentID, 0, 1)

        grid_line3 = QGridLayout()
        grid_line3.addWidget(QLabel("Saving FPS"), 0, 0)
        grid_line3.addWidget(self.entry_saveFPS, 0, 1)
        grid_line3.addWidget(QLabel("Time Limit (s)"), 0, 2)
        grid_line3.addWidget(self.entry_timeLimit, 0, 3)
        grid_line3.addWidget(self.btn_record, 0, 4)

        self.grid = QGridLayout()
        self.grid.addLayout(grid_line1, 0, 0)
        self.grid.addLayout(grid_line2, 1, 0)
        self.grid.addLayout(grid_line3, 2, 0)
        self.setLayout(self.grid)

        # add and display a timer - to be implemented
        # self.timer = QTimer()

        # connections
        self.btn_setSavingDir.clicked.connect(self.set_saving_dir)
        self.btn_record.clicked.connect(self.toggle_recording)
        self.entry_saveFPS.valueChanged.connect(self.streamHandler.set_save_fps)
        self.entry_timeLimit.valueChanged.connect(
            self.imageSaver.set_recording_time_limit
        )
        self.imageSaver.stop_recording.connect(self.stop_recording)

    def set_saving_dir(self):
        dialog = QFileDialog()
        save_dir_base = dialog.getExistingDirectory(None, "Select Folder")
        self.imageSaver.set_base_path(save_dir_base)
        self.lineEdit_savingDir.setText(save_dir_base)
        self.base_path_is_set = True

    def toggle_recording(self, pressed):
        if self.base_path_is_set == False:
            self.btn_record.setChecked(False)
            msg = QMessageBox()
            msg.setText("Please choose base saving directory first")
            msg.exec_()
            return
        if pressed:
            self.lineEdit_experimentID.setEnabled(False)
            self.btn_setSavingDir.setEnabled(False)
            self.imageSaver.start_new_experiment(self.lineEdit_experimentID.text())
            self.streamHandler.start_recording()
        else:
            self.streamHandler.stop_recording()
            self.lineEdit_experimentID.setEnabled(True)
            self.btn_setSavingDir.setEnabled(True)

    # stop_recording can be called by imageSaver
    def stop_recording(self):
        self.lineEdit_experimentID.setEnabled(True)
        self.btn_record.setChecked(False)
        self.streamHandler.stop_recording()
        self.btn_setSavingDir.setEnabled(True)


class NavigationWidget(QFrame):
    def __init__(
        self,
        navigationController,
        slidePositionController=None,
        main=None,
        widget_configuration="full",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.navigationController = navigationController
        self.slidePositionController = slidePositionController
        self.widget_configuration = widget_configuration
        self.slide_position = None
        self.add_components()
        self.setFrameStyle(QFrame.Panel | QFrame.Raised)

    def add_components(self):
        self.label_Xpos = QLabel()
        self.label_Xpos.setNum(0)
        self.label_Xpos.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        self.entry_dX = QDoubleSpinBox()
        self.entry_dX.setMinimum(0)
        self.entry_dX.setMaximum(25)
        self.entry_dX.setSingleStep(0.2)
        self.entry_dX.setValue(0)
        self.entry_dX.setDecimals(3)
        self.entry_dX.setKeyboardTracking(False)
        self.btn_moveX_forward = QPushButton("Forward")
        self.btn_moveX_forward.setDefault(False)
        self.btn_moveX_backward = QPushButton("Backward")
        self.btn_moveX_backward.setDefault(False)

        self.btn_home_X = QPushButton("Home X")
        self.btn_home_X.setDefault(False)
        self.btn_home_X.setEnabled(CONFIG.HOMING_ENABLED_X)
        self.btn_zero_X = QPushButton("Zero X")
        self.btn_zero_X.setDefault(False)

        self.checkbox_clickToMove = QCheckBox("Click to move")
        self.checkbox_clickToMove.setChecked(False)

        self.label_Ypos = QLabel()
        self.label_Ypos.setNum(0)
        self.label_Ypos.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        self.entry_dY = QDoubleSpinBox()
        self.entry_dY.setMinimum(0)
        self.entry_dY.setMaximum(25)
        self.entry_dY.setSingleStep(0.2)
        self.entry_dY.setValue(0)
        self.entry_dY.setDecimals(3)
        self.entry_dY.setKeyboardTracking(False)
        self.btn_moveY_forward = QPushButton("Forward")
        self.btn_moveY_forward.setDefault(False)
        self.btn_moveY_backward = QPushButton("Backward")
        self.btn_moveY_backward.setDefault(False)

        self.btn_home_Y = QPushButton("Home Y")
        self.btn_home_Y.setDefault(False)
        self.btn_home_Y.setEnabled(CONFIG.HOMING_ENABLED_Y)
        self.btn_zero_Y = QPushButton("Zero Y")
        self.btn_zero_Y.setDefault(False)

        self.label_Zpos = QLabel()
        self.label_Zpos.setNum(0)
        self.label_Zpos.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        self.entry_dZ = QDoubleSpinBox()
        self.entry_dZ.setMinimum(0)
        self.entry_dZ.setMaximum(1000)
        self.entry_dZ.setSingleStep(0.2)
        self.entry_dZ.setValue(0)
        self.entry_dZ.setDecimals(3)
        self.entry_dZ.setKeyboardTracking(False)
        self.btn_moveZ_forward = QPushButton("Forward")
        self.btn_moveZ_forward.setDefault(False)
        self.btn_moveZ_backward = QPushButton("Backward")
        self.btn_moveZ_backward.setDefault(False)

        self.btn_home_Z = QPushButton("Home Z")
        self.btn_home_Z.setDefault(False)
        self.btn_home_Z.setEnabled(CONFIG.HOMING_ENABLED_Z)
        self.btn_zero_Z = QPushButton("Zero Z")
        self.btn_zero_Z.setDefault(False)

        self.btn_load_slide = QPushButton("To Slide Loading Position")

        grid_line0 = QGridLayout()
        grid_line0.addWidget(QLabel("X (mm)"), 0, 0)
        grid_line0.addWidget(self.label_Xpos, 0, 1)
        grid_line0.addWidget(self.entry_dX, 0, 2)
        grid_line0.addWidget(self.btn_moveX_forward, 0, 3)
        grid_line0.addWidget(self.btn_moveX_backward, 0, 4)

        grid_line1 = QGridLayout()
        grid_line1.addWidget(QLabel("Y (mm)"), 0, 0)
        grid_line1.addWidget(self.label_Ypos, 0, 1)
        grid_line1.addWidget(self.entry_dY, 0, 2)
        grid_line1.addWidget(self.btn_moveY_forward, 0, 3)
        grid_line1.addWidget(self.btn_moveY_backward, 0, 4)

        grid_line2 = QGridLayout()
        grid_line2.addWidget(QLabel("Z (um)"), 0, 0)
        grid_line2.addWidget(self.label_Zpos, 0, 1)
        grid_line2.addWidget(self.entry_dZ, 0, 2)
        grid_line2.addWidget(self.btn_moveZ_forward, 0, 3)
        grid_line2.addWidget(self.btn_moveZ_backward, 0, 4)

        grid_line3 = QHBoxLayout()

        grid_line3_buttons = QGridLayout()
        if self.widget_configuration == "full":
            grid_line3_buttons.addWidget(self.btn_zero_X, 0, 3)
            grid_line3_buttons.addWidget(self.btn_zero_Y, 0, 4)
            grid_line3_buttons.addWidget(self.btn_zero_Z, 0, 5)
            grid_line3_buttons.addWidget(self.btn_home_X, 0, 0)
            grid_line3_buttons.addWidget(self.btn_home_Y, 0, 1)
            grid_line3_buttons.addWidget(self.btn_home_Z, 0, 2)
        elif self.widget_configuration == "malaria":
            grid_line3_buttons.addWidget(self.btn_load_slide, 0, 0, 1, 2)
            grid_line3_buttons.addWidget(self.btn_home_Z, 0, 2, 1, 1)
            grid_line3_buttons.addWidget(self.btn_zero_Z, 0, 3, 1, 1)
        elif self.widget_configuration == "384 well plate":
            grid_line3_buttons.addWidget(self.btn_load_slide, 0, 0, 1, 2)
            grid_line3_buttons.addWidget(self.btn_home_Z, 0, 2, 1, 1)
            grid_line3_buttons.addWidget(self.btn_zero_Z, 0, 3, 1, 1)
        elif self.widget_configuration == "96 well plate":
            grid_line3_buttons.addWidget(self.btn_load_slide, 0, 0, 1, 2)
            grid_line3_buttons.addWidget(self.btn_home_Z, 0, 2, 1, 1)
            grid_line3_buttons.addWidget(self.btn_zero_Z, 0, 3, 1, 1)

        grid_line3.addLayout(grid_line3_buttons)

        grid_line3.addWidget(self.checkbox_clickToMove)

        self.grid = QGridLayout()
        self.grid.addLayout(grid_line0, 0, 0)
        self.grid.addLayout(grid_line1, 1, 0)
        self.grid.addLayout(grid_line2, 2, 0)
        self.grid.addLayout(grid_line3, 3, 0)
        self.setLayout(self.grid)

        self.entry_dX.valueChanged.connect(self.set_deltaX)
        self.entry_dY.valueChanged.connect(self.set_deltaY)
        self.entry_dZ.valueChanged.connect(self.set_deltaZ)

        self.btn_moveX_forward.clicked.connect(self.move_x_forward)
        self.btn_moveX_backward.clicked.connect(self.move_x_backward)
        self.btn_moveY_forward.clicked.connect(self.move_y_forward)
        self.btn_moveY_backward.clicked.connect(self.move_y_backward)
        self.btn_moveZ_forward.clicked.connect(self.move_z_forward)
        self.btn_moveZ_backward.clicked.connect(self.move_z_backward)

        self.btn_home_X.clicked.connect(self.home_x)
        self.btn_home_Y.clicked.connect(self.home_y)
        self.btn_home_Z.clicked.connect(self.home_z)
        self.btn_zero_X.clicked.connect(self.zero_x)
        self.btn_zero_Y.clicked.connect(self.zero_y)
        self.btn_zero_Z.clicked.connect(self.zero_z)

        self.checkbox_clickToMove.stateChanged.connect(
            self.navigationController.set_flag_click_to_move
        )

        self.btn_load_slide.clicked.connect(self.switch_position)
        self.btn_load_slide.setStyleSheet("background-color: #C2C2FF")

    def move_x_forward(self):
        self.navigationController.move_x(self.entry_dX.value())

    def move_x_backward(self):
        self.navigationController.move_x(-self.entry_dX.value())

    def move_y_forward(self):
        self.navigationController.move_y(self.entry_dY.value())

    def move_y_backward(self):
        self.navigationController.move_y(-self.entry_dY.value())

    def move_z_forward(self):
        self.navigationController.move_z(self.entry_dZ.value() / 1000)

    def move_z_backward(self):
        self.navigationController.move_z(-self.entry_dZ.value() / 1000)

    def set_deltaX(self, value):
        mm_per_ustep = CONFIG.SCREW_PITCH_X_MM / (
            self.navigationController.x_microstepping * CONFIG.FULLSTEPS_PER_REV_X
        )  # to implement a get_x_microstepping() in multipointController
        deltaX = round(value / mm_per_ustep) * mm_per_ustep
        self.entry_dX.setValue(deltaX)

    def set_deltaY(self, value):
        mm_per_ustep = CONFIG.SCREW_PITCH_Y_MM / (
            self.navigationController.y_microstepping * CONFIG.FULLSTEPS_PER_REV_Y
        )
        deltaY = round(value / mm_per_ustep) * mm_per_ustep
        self.entry_dY.setValue(deltaY)

    def set_deltaZ(self, value):
        mm_per_ustep = CONFIG.SCREW_PITCH_Z_MM / (
            self.navigationController.z_microstepping * CONFIG.FULLSTEPS_PER_REV_Z
        )
        deltaZ = round(value / 1000 / mm_per_ustep) * mm_per_ustep * 1000
        self.entry_dZ.setValue(deltaZ)

    def home_x(self):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setText("Confirm your action")
        msg.setInformativeText("Click OK to run homing")
        msg.setWindowTitle("Confirmation")
        msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        msg.setDefaultButton(QMessageBox.Cancel)
        retval = msg.exec_()
        if QMessageBox.Ok == retval:
            self.navigationController.home_x()

    def home_y(self):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setText("Confirm your action")
        msg.setInformativeText("Click OK to run homing")
        msg.setWindowTitle("Confirmation")
        msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        msg.setDefaultButton(QMessageBox.Cancel)
        retval = msg.exec_()
        if QMessageBox.Ok == retval:
            self.navigationController.home_y()

    def home_z(self):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setText("Confirm your action")
        msg.setInformativeText("Click OK to run homing")
        msg.setWindowTitle("Confirmation")
        msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        msg.setDefaultButton(QMessageBox.Cancel)
        retval = msg.exec_()
        if QMessageBox.Ok == retval:
            self.navigationController.home_z()

    def zero_x(self):
        self.navigationController.zero_x()

    def zero_y(self):
        self.navigationController.zero_y()

    def zero_z(self):
        self.navigationController.zero_z()

    def slot_slide_loading_position_reached(self):
        self.slide_position = "loading"
        self.btn_load_slide.setStyleSheet("background-color: #C2FFC2")
        self.btn_load_slide.setText("To Scanning Position")
        self.btn_moveX_forward.setEnabled(False)
        self.btn_moveX_backward.setEnabled(False)
        self.btn_moveY_forward.setEnabled(False)
        self.btn_moveY_backward.setEnabled(False)
        self.btn_moveZ_forward.setEnabled(False)
        self.btn_moveZ_backward.setEnabled(False)
        self.btn_load_slide.setEnabled(True)

    def slot_slide_scanning_position_reached(self):
        self.slide_position = "scanning"
        self.btn_load_slide.setStyleSheet("background-color: #C2C2FF")
        self.btn_load_slide.setText("To Loading Position")
        self.btn_moveX_forward.setEnabled(True)
        self.btn_moveX_backward.setEnabled(True)
        self.btn_moveY_forward.setEnabled(True)
        self.btn_moveY_backward.setEnabled(True)
        self.btn_moveZ_forward.setEnabled(True)
        self.btn_moveZ_backward.setEnabled(True)
        self.btn_load_slide.setEnabled(True)

    def switch_position(self):
        if self.slide_position != "loading":
            self.slidePositionController.move_to_slide_loading_position()
        else:
            self.slidePositionController.move_to_slide_scanning_position()
        self.btn_load_slide.setEnabled(False)


class DACControWidget(QFrame):
    def __init__(self, microcontroller, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.microcontroller = microcontroller
        self.add_components()
        self.setFrameStyle(QFrame.Panel | QFrame.Raised)

    def add_components(self):
        self.slider_DAC0 = QSlider(Qt.Horizontal)
        self.slider_DAC0.setTickPosition(QSlider.TicksBelow)
        self.slider_DAC0.setMinimum(0)
        self.slider_DAC0.setMaximum(100)
        self.slider_DAC0.setSingleStep(1)
        self.slider_DAC0.setValue(0)

        self.entry_DAC0 = QDoubleSpinBox()
        self.entry_DAC0.setMinimum(0)
        self.entry_DAC0.setMaximum(100)
        self.entry_DAC0.setSingleStep(0.1)
        self.entry_DAC0.setValue(0)
        self.entry_DAC0.setKeyboardTracking(False)

        self.slider_DAC1 = QSlider(Qt.Horizontal)
        self.slider_DAC1.setTickPosition(QSlider.TicksBelow)
        self.slider_DAC1.setMinimum(0)
        self.slider_DAC1.setMaximum(100)
        self.slider_DAC1.setValue(0)
        self.slider_DAC1.setSingleStep(1)

        self.entry_DAC1 = QDoubleSpinBox()
        self.entry_DAC1.setMinimum(0)
        self.entry_DAC1.setMaximum(100)
        self.entry_DAC1.setSingleStep(0.1)
        self.entry_DAC1.setValue(0)
        self.entry_DAC1.setKeyboardTracking(False)

        # connections
        self.entry_DAC0.valueChanged.connect(self.set_DAC0)
        self.entry_DAC0.valueChanged.connect(self.slider_DAC0.setValue)
        self.slider_DAC0.valueChanged.connect(self.entry_DAC0.setValue)
        self.entry_DAC1.valueChanged.connect(self.set_DAC1)
        self.entry_DAC1.valueChanged.connect(self.slider_DAC1.setValue)
        self.slider_DAC1.valueChanged.connect(self.entry_DAC1.setValue)

        # layout
        grid_line1 = QGridLayout()
        grid_line1.addWidget(QLabel("DAC0"), 0, 0)
        grid_line1.addWidget(self.slider_DAC0, 0, 1)
        grid_line1.addWidget(self.entry_DAC0, 0, 2)
        grid_line1.addWidget(QLabel("DAC1"), 1, 0)
        grid_line1.addWidget(self.slider_DAC1, 1, 1)
        grid_line1.addWidget(self.entry_DAC1, 1, 2)

        self.grid = QGridLayout()
        self.grid.addLayout(grid_line1, 1, 0)
        self.setLayout(self.grid)

    def set_DAC0(self, value):
        self.microcontroller.analog_write_onboard_DAC(0, int(value * 65535 / 100))

    def set_DAC1(self, value):
        self.microcontroller.analog_write_onboard_DAC(1, int(value * 65535 / 100))


class AutoFocusWidget(QFrame):
    def __init__(self, autofocusController, main=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.autofocusController = autofocusController
        self.add_components()
        self.setFrameStyle(QFrame.Panel | QFrame.Raised)

    def add_components(self):
        self.entry_delta = QDoubleSpinBox()
        self.entry_delta.setMinimum(0)
        self.entry_delta.setMaximum(20)
        self.entry_delta.setSingleStep(0.2)
        self.entry_delta.setDecimals(3)
        self.entry_delta.setValue(1.524)
        self.entry_delta.setKeyboardTracking(False)
        self.autofocusController.set_deltaZ(1.524)

        self.entry_N = QSpinBox()
        self.entry_N.setMinimum(3)
        self.entry_N.setMaximum(20)
        self.entry_N.setSingleStep(1)
        self.entry_N.setValue(10)
        self.entry_N.setKeyboardTracking(False)
        self.autofocusController.set_N(10)

        self.btn_autofocus = QPushButton("Autofocus")
        self.btn_autofocus.setDefault(False)
        self.btn_autofocus.setCheckable(True)
        self.btn_autofocus.setChecked(False)

        # layout
        grid_line0 = QGridLayout()
        grid_line0.addWidget(QLabel("delta Z (um)"), 0, 0)
        grid_line0.addWidget(self.entry_delta, 0, 1)
        grid_line0.addWidget(QLabel("N Z planes"), 0, 2)
        grid_line0.addWidget(self.entry_N, 0, 3)
        grid_line0.addWidget(self.btn_autofocus, 0, 4)

        self.grid = QGridLayout()
        self.grid.addLayout(grid_line0, 0, 0)
        self.setLayout(self.grid)

        # connections
        self.btn_autofocus.clicked.connect(
            lambda: self.autofocusController.autofocus(False)
        )
        self.entry_delta.valueChanged.connect(self.set_deltaZ)
        self.entry_N.valueChanged.connect(self.autofocusController.set_N)
        self.autofocusController.autofocusFinished.connect(self.autofocus_is_finished)

    def set_deltaZ(self, value):
        mm_per_ustep = CONFIG.SCREW_PITCH_Z_MM / (
            self.autofocusController.navigationController.z_microstepping
            * CONFIG.FULLSTEPS_PER_REV_Z
        )
        deltaZ = round(value / 1000 / mm_per_ustep) * mm_per_ustep * 1000
        self.entry_delta.setValue(deltaZ)
        self.autofocusController.set_deltaZ(deltaZ)

    def autofocus_is_finished(self):
        self.btn_autofocus.setChecked(False)


class StatsDisplayWidget(QFrame):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initUI()
        self.setFrameStyle(QFrame.Panel | QFrame.Raised)

    def initUI(self):
        self.layout = QVBoxLayout()
        self.table_widget = QTableWidget()
        self.table_widget.setColumnCount(2)
        self.table_widget.verticalHeader().hide()
        self.table_widget.horizontalHeader().hide()
        self.table_widget.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents
        )
        self.layout.addWidget(self.table_widget)
        self.setLayout(self.layout)

    def display_stats(self, stats):
        locale.setlocale(locale.LC_ALL, "")
        self.table_widget.setRowCount(len(stats))
        row = 0
        for key, value in stats.items():
            key_item = QTableWidgetItem(str(key))
            value_item = None
            try:
                value_item = QTableWidgetItem(f"{value:n}")
            except:
                value_item = QTableWidgetItem(str(value))
            self.table_widget.setItem(row, 0, key_item)
            self.table_widget.setItem(row, 1, value_item)
            row += 1


class MultiPointWidget(QFrame):
    def __init__(
        self,
        multipointController,
        configurationManager=None,
        main=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.multipointController = multipointController
        self.configurationManager = configurationManager
        self.base_path_is_set = False
        self.add_components()
        self.setFrameStyle(QFrame.Panel | QFrame.Raised)

    def add_components(self):

        self.btn_setSavingDir = QPushButton("Browse")
        self.btn_setSavingDir.setDefault(False)
        self.btn_setSavingDir.setIcon(QIcon("icon/folder.png"))

        self.lineEdit_savingDir = QLineEdit()
        self.lineEdit_savingDir.setReadOnly(True)
        self.lineEdit_savingDir.setText("Choose a base saving directory")

        self.lineEdit_savingDir.setText(CONFIG.DEFAULT_SAVING_PATH)
        self.multipointController.set_base_path(CONFIG.DEFAULT_SAVING_PATH)
        self.base_path_is_set = True

        self.lineEdit_experimentID = QLineEdit()

        self.entry_deltaX = QDoubleSpinBox()
        self.entry_deltaX.setMinimum(0)
        self.entry_deltaX.setMaximum(5)
        self.entry_deltaX.setSingleStep(0.1)
        self.entry_deltaX.setValue(CONFIG.Acquisition.DX)
        self.entry_deltaX.setDecimals(3)
        self.entry_deltaX.setKeyboardTracking(False)

        self.entry_NX = QSpinBox()
        self.entry_NX.setMinimum(1)
        self.entry_NX.setMaximum(50)
        self.entry_NX.setSingleStep(1)
        self.entry_NX.setValue(CONFIG.Acquisition.NX)
        self.entry_NX.setKeyboardTracking(False)

        self.entry_deltaY = QDoubleSpinBox()
        self.entry_deltaY.setMinimum(0)
        self.entry_deltaY.setMaximum(5)
        self.entry_deltaY.setSingleStep(0.1)
        self.entry_deltaY.setValue(CONFIG.Acquisition.DX)
        self.entry_deltaY.setDecimals(3)
        self.entry_deltaY.setKeyboardTracking(False)

        self.entry_NY = QSpinBox()
        self.entry_NY.setMinimum(1)
        self.entry_NY.setMaximum(50)
        self.entry_NY.setSingleStep(1)
        self.entry_NY.setValue(CONFIG.Acquisition.NY)
        self.entry_NY.setKeyboardTracking(False)

        self.entry_deltaZ = QDoubleSpinBox()
        self.entry_deltaZ.setMinimum(0)
        self.entry_deltaZ.setMaximum(1000)
        self.entry_deltaZ.setSingleStep(0.2)
        self.entry_deltaZ.setValue(CONFIG.Acquisition.DZ)
        self.entry_deltaZ.setDecimals(3)
        self.entry_deltaZ.setKeyboardTracking(False)

        self.entry_NZ = QSpinBox()
        self.entry_NZ.setMinimum(1)
        self.entry_NZ.setMaximum(100)
        self.entry_NZ.setSingleStep(1)
        self.entry_NZ.setValue(1)
        self.entry_NZ.setKeyboardTracking(False)

        self.entry_dt = QDoubleSpinBox()
        self.entry_dt.setMinimum(0)
        self.entry_dt.setMaximum(12 * 3600)
        self.entry_dt.setSingleStep(1)
        self.entry_dt.setValue(0)
        self.entry_dt.setKeyboardTracking(False)

        self.entry_Nt = QSpinBox()
        self.entry_Nt.setMinimum(1)
        self.entry_Nt.setMaximum(50000)  # @@@ to be changed
        self.entry_Nt.setSingleStep(1)
        self.entry_Nt.setValue(1)
        self.entry_Nt.setKeyboardTracking(False)

        self.list_configurations = QListWidget()
        for microscope_configuration in self.configurationManager.configurations:
            self.list_configurations.addItems([microscope_configuration.name])
        self.list_configurations.setSelectionMode(
            QAbstractItemView.MultiSelection
        )  # ref: https://doc.qt.io/qt-5/qabstractitemview.html#SelectionMode-enum

        self.checkbox_withAutofocus = QCheckBox("Contrast CONFIG.AF")
        self.checkbox_withAutofocus.setChecked(
            CONFIG.MULTIPOINT_AUTOFOCUS_ENABLE_BY_DEFAULT
        )
        self.multipointController.set_af_flag(
            CONFIG.MULTIPOINT_AUTOFOCUS_ENABLE_BY_DEFAULT
        )

        self.checkbox_genFocusMap = QCheckBox("Generate focus map")
        self.checkbox_genFocusMap.setChecked(False)

        self.checkbox_withReflectionAutofocus = QCheckBox("Reflection CONFIG.AF")
        self.checkbox_withReflectionAutofocus.setChecked(
            CONFIG.MULTIPOINT_REFLECTION_AUTOFOCUS_ENABLE_BY_DEFAULT
        )

        self.multipointController.set_reflection_af_flag(
            CONFIG.MULTIPOINT_REFLECTION_AUTOFOCUS_ENABLE_BY_DEFAULT
        )
        self.btn_startAcquisition = QPushButton("Start Acquisition")
        self.btn_startAcquisition.setCheckable(True)
        self.btn_startAcquisition.setChecked(False)

        # layout
        grid_line0 = QGridLayout()
        grid_line0.addWidget(QLabel("Saving Path"))
        grid_line0.addWidget(self.lineEdit_savingDir, 0, 1)
        grid_line0.addWidget(self.btn_setSavingDir, 0, 2)

        grid_line1 = QGridLayout()
        grid_line1.addWidget(QLabel("Experiment ID"), 0, 0)
        grid_line1.addWidget(self.lineEdit_experimentID, 0, 1)

        grid_line2 = QGridLayout()
        grid_line2.addWidget(QLabel("dx (mm)"), 0, 0)
        grid_line2.addWidget(self.entry_deltaX, 0, 1)
        grid_line2.addWidget(QLabel("Nx"), 0, 2)
        grid_line2.addWidget(self.entry_NX, 0, 3)
        grid_line2.addWidget(QLabel("dy (mm)"), 0, 4)
        grid_line2.addWidget(self.entry_deltaY, 0, 5)
        grid_line2.addWidget(QLabel("Ny"), 0, 6)
        grid_line2.addWidget(self.entry_NY, 0, 7)

        grid_line2.addWidget(QLabel("dz (um)"), 1, 0)
        grid_line2.addWidget(self.entry_deltaZ, 1, 1)
        grid_line2.addWidget(QLabel("Nz"), 1, 2)
        grid_line2.addWidget(self.entry_NZ, 1, 3)
        grid_line2.addWidget(QLabel("dt (s)"), 1, 4)
        grid_line2.addWidget(self.entry_dt, 1, 5)
        grid_line2.addWidget(QLabel("Nt"), 1, 6)
        grid_line2.addWidget(self.entry_Nt, 1, 7)

        grid_af = QVBoxLayout()
        grid_af.addWidget(self.checkbox_withAutofocus)
        grid_af.addWidget(self.checkbox_genFocusMap)
        if CONFIG.SUPPORT_LASER_AUTOFOCUS:
            grid_af.addWidget(self.checkbox_withReflectionAutofocus)

        grid_line3 = QHBoxLayout()
        grid_line3.addWidget(self.list_configurations)
        # grid_line3.addWidget(self.checkbox_withAutofocus)
        grid_line3.addLayout(grid_af)
        grid_line3.addWidget(self.btn_startAcquisition)

        self.grid = QGridLayout()
        self.grid.addLayout(grid_line0, 0, 0)
        self.grid.addLayout(grid_line1, 1, 0)
        self.grid.addLayout(grid_line2, 2, 0)
        self.grid.addLayout(grid_line3, 3, 0)
        self.setLayout(self.grid)

        # add and display a timer - to be implemented
        # self.timer = QTimer()

        # connections
        self.entry_deltaX.valueChanged.connect(self.set_deltaX)
        self.entry_deltaY.valueChanged.connect(self.set_deltaY)
        self.entry_deltaZ.valueChanged.connect(self.set_deltaZ)
        self.entry_dt.valueChanged.connect(self.multipointController.set_deltat)
        self.entry_NX.valueChanged.connect(self.multipointController.set_NX)
        self.entry_NY.valueChanged.connect(self.multipointController.set_NY)
        self.entry_NZ.valueChanged.connect(self.multipointController.set_NZ)
        self.entry_Nt.valueChanged.connect(self.multipointController.set_Nt)
        self.checkbox_withAutofocus.stateChanged.connect(
            self.multipointController.set_af_flag
        )
        self.checkbox_withReflectionAutofocus.stateChanged.connect(
            self.multipointController.set_reflection_af_flag
        )
        self.checkbox_genFocusMap.stateChanged.connect(
            self.multipointController.set_gen_focus_map_flag
        )
        self.btn_setSavingDir.clicked.connect(self.set_saving_dir)
        self.btn_startAcquisition.clicked.connect(self.toggle_acquisition)
        self.multipointController.acquisitionFinished.connect(
            self.acquisition_is_finished
        )

    def set_deltaX(self, value):
        mm_per_ustep = CONFIG.SCREW_PITCH_X_MM / (
            self.multipointController.navigationController.x_microstepping
            * CONFIG.FULLSTEPS_PER_REV_X
        )  # to implement a get_x_microstepping() in multipointController
        deltaX = round(value / mm_per_ustep) * mm_per_ustep
        self.entry_deltaX.setValue(deltaX)
        self.multipointController.set_deltaX(deltaX)

    def set_deltaY(self, value):
        mm_per_ustep = CONFIG.SCREW_PITCH_Y_MM / (
            self.multipointController.navigationController.y_microstepping
            * CONFIG.FULLSTEPS_PER_REV_Y
        )
        deltaY = round(value / mm_per_ustep) * mm_per_ustep
        self.entry_deltaY.setValue(deltaY)
        self.multipointController.set_deltaY(deltaY)

    def set_deltaZ(self, value):
        mm_per_ustep = CONFIG.SCREW_PITCH_Z_MM / (
            self.multipointController.navigationController.z_microstepping
            * CONFIG.FULLSTEPS_PER_REV_Z
        )
        deltaZ = round(value / 1000 / mm_per_ustep) * mm_per_ustep * 1000
        self.entry_deltaZ.setValue(deltaZ)
        self.multipointController.set_deltaZ(deltaZ)

    def set_saving_dir(self):
        dialog = QFileDialog()
        save_dir_base = dialog.getExistingDirectory(None, "Select Folder")
        self.multipointController.set_base_path(save_dir_base)
        self.lineEdit_savingDir.setText(save_dir_base)
        self.base_path_is_set = True

    def toggle_acquisition(self, pressed):
        if self.base_path_is_set == False:
            self.btn_startAcquisition.setChecked(False)
            msg = QMessageBox()
            msg.setText("Please choose base saving directory first")
            msg.exec_()
            return
        if pressed:
            # @@@ to do: add a widgetManger to enable and disable widget
            # @@@ to do: emit signal to widgetManager to disable other widgets
            self.setEnabled_all(False)
            self.multipointController.set_selected_configurations(
                (item.text() for item in self.list_configurations.selectedItems())
            )
            self.multipointController.start_new_experiment(
                self.lineEdit_experimentID.text()
            )
            # set parameters
            self.multipointController.set_deltaX(self.entry_deltaX.value())
            self.multipointController.set_deltaY(self.entry_deltaY.value())
            self.multipointController.set_deltaZ(self.entry_deltaZ.value())
            self.multipointController.set_deltat(self.entry_dt.value())
            self.multipointController.set_NX(self.entry_NX.value())
            self.multipointController.set_NY(self.entry_NY.value())
            self.multipointController.set_NZ(self.entry_NZ.value())
            self.multipointController.set_Nt(self.entry_Nt.value())
            self.multipointController.set_af_flag(
                self.checkbox_withAutofocus.isChecked()
            )
            self.multipointController.set_reflection_af_flag(
                self.checkbox_withReflectionAutofocus.isChecked()
            )
            self.multipointController.set_base_path(self.lineEdit_savingDir.text())
            self.multipointController.run_acquisition()
        else:
            self.multipointController.request_abort_aquisition()
            self.setEnabled_all(True)

    def acquisition_is_finished(self):
        self.btn_startAcquisition.setChecked(False)
        self.setEnabled_all(True)

    def setEnabled_all(self, enabled, exclude_btn_startAcquisition=True):
        self.btn_setSavingDir.setEnabled(enabled)
        self.lineEdit_savingDir.setEnabled(enabled)
        self.lineEdit_experimentID.setEnabled(enabled)
        self.entry_deltaX.setEnabled(enabled)
        self.entry_NX.setEnabled(enabled)
        self.entry_deltaY.setEnabled(enabled)
        self.entry_NY.setEnabled(enabled)
        self.entry_deltaZ.setEnabled(enabled)
        self.entry_NZ.setEnabled(enabled)
        self.entry_dt.setEnabled(enabled)
        self.entry_Nt.setEnabled(enabled)
        self.list_configurations.setEnabled(enabled)
        self.checkbox_withAutofocus.setEnabled(enabled)
        self.checkbox_withReflectionAutofocus.setEnabled(enabled)
        self.checkbox_genFocusMap.setEnabled(enabled)
        if exclude_btn_startAcquisition is not True:
            self.btn_startAcquisition.setEnabled(enabled)

    def disable_the_start_aquisition_button(self):
        self.btn_startAcquisition.setEnabled(False)

    def enable_the_start_aquisition_button(self):
        self.btn_startAcquisition.setEnabled(True)


class MultiPointWidget2(QFrame):
    def __init__(
        self,
        navigationController,
        navigationViewer,
        multipointController,
        configurationManager=None,
        main=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.last_used_locations = None
        self.multipointController = multipointController
        self.configurationManager = configurationManager
        self.navigationController = navigationController
        self.navigationViewer = navigationViewer
        self.base_path_is_set = False
        self.location_list = np.empty((0, 3), dtype=float)
        self.add_components()
        self.setFrameStyle(QFrame.Panel | QFrame.Raised)
        self.acquisition_in_place = False

    def add_components(self):

        self.btn_setSavingDir = QPushButton("Browse")
        self.btn_setSavingDir.setDefault(False)
        self.btn_setSavingDir.setIcon(QIcon("icon/folder.png"))

        self.lineEdit_savingDir = QLineEdit()
        self.lineEdit_savingDir.setReadOnly(True)
        self.lineEdit_savingDir.setText("Choose a base saving directory")

        self.lineEdit_savingDir.setText(CONFIG.DEFAULT_SAVING_PATH)
        self.multipointController.set_base_path(CONFIG.DEFAULT_SAVING_PATH)
        self.base_path_is_set = True

        self.lineEdit_experimentID = QLineEdit()

        self.dropdown_location_list = QComboBox()
        self.btn_add = QPushButton("Add")
        self.btn_remove = QPushButton("Remove")
        self.btn_previous = QPushButton("Previous")
        self.btn_next = QPushButton("Next")
        self.btn_clear = QPushButton("Clear all")

        self.btn_load_last_executed = QPushButton("Prev Used Locations")

        self.btn_export_locations = QPushButton("Export Location List")
        self.btn_import_locations = QPushButton("Import Location List")

        self.entry_deltaX = QDoubleSpinBox()
        self.entry_deltaX.setMinimum(0)
        self.entry_deltaX.setMaximum(5)
        self.entry_deltaX.setSingleStep(0.1)
        self.entry_deltaX.setValue(CONFIG.Acquisition.DX)
        self.entry_deltaX.setDecimals(3)
        self.entry_deltaX.setKeyboardTracking(False)

        self.entry_NX = QSpinBox()
        self.entry_NX.setMinimum(1)
        self.entry_NX.setMaximum(50)
        self.entry_NX.setSingleStep(1)
        self.entry_NX.setValue(1)
        self.entry_NX.setKeyboardTracking(False)

        self.entry_deltaY = QDoubleSpinBox()
        self.entry_deltaY.setMinimum(0)
        self.entry_deltaY.setMaximum(5)
        self.entry_deltaY.setSingleStep(0.1)
        self.entry_deltaY.setValue(CONFIG.Acquisition.DX)
        self.entry_deltaY.setDecimals(3)
        self.entry_deltaY.setKeyboardTracking(False)

        self.entry_NY = QSpinBox()
        self.entry_NY.setMinimum(1)
        self.entry_NY.setMaximum(50)
        self.entry_NY.setSingleStep(1)
        self.entry_NY.setValue(1)
        self.entry_NY.setKeyboardTracking(False)

        self.entry_deltaZ = QDoubleSpinBox()
        self.entry_deltaZ.setMinimum(0)
        self.entry_deltaZ.setMaximum(1000)
        self.entry_deltaZ.setSingleStep(0.2)
        self.entry_deltaZ.setValue(CONFIG.Acquisition.DZ)
        self.entry_deltaZ.setDecimals(3)
        self.entry_deltaZ.setKeyboardTracking(False)

        self.entry_NZ = QSpinBox()
        self.entry_NZ.setMinimum(1)
        self.entry_NZ.setMaximum(100)
        self.entry_NZ.setSingleStep(1)
        self.entry_NZ.setValue(1)
        self.entry_NZ.setKeyboardTracking(False)

        self.entry_dt = QDoubleSpinBox()
        self.entry_dt.setMinimum(0)
        self.entry_dt.setMaximum(12 * 3600)
        self.entry_dt.setSingleStep(1)
        self.entry_dt.setValue(0)
        self.entry_dt.setKeyboardTracking(False)

        self.entry_Nt = QSpinBox()
        self.entry_Nt.setMinimum(1)
        self.entry_Nt.setMaximum(50000)  # @@@ to be changed
        self.entry_Nt.setSingleStep(1)
        self.entry_Nt.setValue(1)
        self.entry_Nt.setKeyboardTracking(False)

        self.list_configurations = QListWidget()
        for microscope_configuration in self.configurationManager.configurations:
            self.list_configurations.addItems([microscope_configuration.name])
        self.list_configurations.setSelectionMode(
            QAbstractItemView.MultiSelection
        )  # ref: https://doc.qt.io/qt-5/qabstractitemview.html#SelectionMode-enum

        self.checkbox_withAutofocus = QCheckBox("Contrast CONFIG.AF")
        self.checkbox_withAutofocus.setChecked(
            CONFIG.MULTIPOINT_AUTOFOCUS_ENABLE_BY_DEFAULT
        )
        self.multipointController.set_af_flag(
            CONFIG.MULTIPOINT_AUTOFOCUS_ENABLE_BY_DEFAULT
        )
        self.checkbox_withReflectionAutofocus = QCheckBox("Reflection CONFIG.AF")
        self.checkbox_withReflectionAutofocus.setChecked(
            CONFIG.MULTIPOINT_REFLECTION_AUTOFOCUS_ENABLE_BY_DEFAULT
        )
        self.multipointController.set_reflection_af_flag(
            CONFIG.MULTIPOINT_REFLECTION_AUTOFOCUS_ENABLE_BY_DEFAULT
        )
        self.btn_startAcquisition = QPushButton("Start Acquisition")
        self.btn_startAcquisition.setCheckable(True)
        self.btn_startAcquisition.setChecked(False)

        # layout
        grid_line0 = QGridLayout()
        grid_line0.addWidget(QLabel("Saving Path"))
        grid_line0.addWidget(self.lineEdit_savingDir, 0, 1)
        grid_line0.addWidget(self.btn_setSavingDir, 0, 2)
        grid_line0.addWidget(QLabel("ID"), 0, 3)
        grid_line0.addWidget(self.lineEdit_experimentID, 0, 4)

        grid_line4 = QGridLayout()
        grid_line4.addWidget(QLabel("Location List"), 0, 0)
        grid_line4.addWidget(self.dropdown_location_list, 0, 1, 1, 2)
        grid_line4.addWidget(self.btn_clear, 0, 3)

        grid_line3point5 = QGridLayout()
        grid_line3point5.addWidget(self.btn_add, 0, 0)
        grid_line3point5.addWidget(self.btn_remove, 0, 1)
        grid_line3point5.addWidget(self.btn_next, 0, 2)
        grid_line3point5.addWidget(self.btn_previous, 0, 3)
        # grid_line3point5.addWidget(self.btn_load_last_executed,0,4)

        grid_line3point75 = QGridLayout()
        grid_line3point75.addWidget(self.btn_import_locations, 0, 0)
        grid_line3point75.addWidget(self.btn_export_locations, 0, 1)

        grid_line2 = QGridLayout()
        grid_line2.addWidget(QLabel("dx (mm)"), 0, 0)
        grid_line2.addWidget(self.entry_deltaX, 0, 1)
        grid_line2.addWidget(QLabel("Nx"), 0, 2)
        grid_line2.addWidget(self.entry_NX, 0, 3)
        grid_line2.addWidget(QLabel("dy (mm)"), 0, 4)
        grid_line2.addWidget(self.entry_deltaY, 0, 5)
        grid_line2.addWidget(QLabel("Ny"), 0, 6)
        grid_line2.addWidget(self.entry_NY, 0, 7)

        grid_line2.addWidget(QLabel("dz (um)"), 1, 0)
        grid_line2.addWidget(self.entry_deltaZ, 1, 1)
        grid_line2.addWidget(QLabel("Nz"), 1, 2)
        grid_line2.addWidget(self.entry_NZ, 1, 3)
        grid_line2.addWidget(QLabel("dt (s)"), 1, 4)
        grid_line2.addWidget(self.entry_dt, 1, 5)
        grid_line2.addWidget(QLabel("Nt"), 1, 6)
        grid_line2.addWidget(self.entry_Nt, 1, 7)

        grid_af = QVBoxLayout()
        grid_af.addWidget(self.checkbox_withAutofocus)
        if CONFIG.SUPPORT_LASER_AUTOFOCUS:
            grid_af.addWidget(self.checkbox_withReflectionAutofocus)

        grid_line3 = QHBoxLayout()
        grid_line3.addWidget(self.list_configurations)
        # grid_line3.addWidget(self.checkbox_withAutofocus)
        grid_line3.addLayout(grid_af)
        grid_line3.addWidget(self.btn_startAcquisition)

        self.grid = QGridLayout()
        self.grid.addLayout(grid_line0, 0, 0)
        # self.grid.addLayout(grid_line1,1,0)
        self.grid.addLayout(grid_line4, 1, 0)
        self.grid.addLayout(grid_line3point5, 2, 0)
        self.grid.addLayout(grid_line3point75, 3, 0)
        # self.grid.addLayout(grid_line5,2,0)
        self.grid.addLayout(grid_line2, 4, 0)
        self.grid.addLayout(grid_line3, 5, 0)
        self.setLayout(self.grid)

        # add and display a timer - to be implemented
        # self.timer = QTimer()

        # connections
        self.entry_deltaX.valueChanged.connect(self.set_deltaX)
        self.entry_deltaY.valueChanged.connect(self.set_deltaY)
        self.entry_deltaZ.valueChanged.connect(self.set_deltaZ)
        self.entry_dt.valueChanged.connect(self.multipointController.set_deltat)
        self.entry_NX.valueChanged.connect(self.multipointController.set_NX)
        self.entry_NY.valueChanged.connect(self.multipointController.set_NY)
        self.entry_NZ.valueChanged.connect(self.multipointController.set_NZ)
        self.entry_Nt.valueChanged.connect(self.multipointController.set_Nt)
        self.checkbox_withAutofocus.stateChanged.connect(
            self.multipointController.set_af_flag
        )
        self.checkbox_withReflectionAutofocus.stateChanged.connect(
            self.multipointController.set_reflection_af_flag
        )
        self.btn_setSavingDir.clicked.connect(self.set_saving_dir)
        self.btn_startAcquisition.clicked.connect(self.toggle_acquisition)
        self.multipointController.acquisitionFinished.connect(
            self.acquisition_is_finished
        )

        self.btn_add.clicked.connect(self.add_location)
        self.btn_remove.clicked.connect(self.remove_location)
        self.btn_previous.clicked.connect(self.previous)
        self.btn_next.clicked.connect(self.next)
        self.btn_clear.clicked.connect(self.clear)
        self.btn_load_last_executed.clicked.connect(self.load_last_used_locations)
        self.btn_export_locations.clicked.connect(self.export_location_list)
        self.btn_import_locations.clicked.connect(self.import_location_list)

        self.dropdown_location_list.currentIndexChanged.connect(self.go_to)

        self.shortcut = QShortcut(QKeySequence(";"), self)
        self.shortcut.activated.connect(self.btn_add.click)

    def set_deltaX(self, value):
        mm_per_ustep = CONFIG.SCREW_PITCH_X_MM / (
            self.multipointController.navigationController.x_microstepping
            * CONFIG.FULLSTEPS_PER_REV_X
        )  # to implement a get_x_microstepping() in multipointController
        deltaX = round(value / mm_per_ustep) * mm_per_ustep
        self.entry_deltaX.setValue(deltaX)
        self.multipointController.set_deltaX(deltaX)

    def set_deltaY(self, value):
        mm_per_ustep = CONFIG.SCREW_PITCH_Y_MM / (
            self.multipointController.navigationController.y_microstepping
            * CONFIG.FULLSTEPS_PER_REV_Y
        )
        deltaY = round(value / mm_per_ustep) * mm_per_ustep
        self.entry_deltaY.setValue(deltaY)
        self.multipointController.set_deltaY(deltaY)

    def set_deltaZ(self, value):
        mm_per_ustep = CONFIG.SCREW_PITCH_Z_MM / (
            self.multipointController.navigationController.z_microstepping
            * CONFIG.FULLSTEPS_PER_REV_Z
        )
        deltaZ = round(value / 1000 / mm_per_ustep) * mm_per_ustep * 1000
        self.entry_deltaZ.setValue(deltaZ)
        self.multipointController.set_deltaZ(deltaZ)

    def set_saving_dir(self):
        dialog = QFileDialog()
        save_dir_base = dialog.getExistingDirectory(None, "Select Folder")
        self.multipointController.set_base_path(save_dir_base)
        self.lineEdit_savingDir.setText(save_dir_base)
        self.base_path_is_set = True

    def toggle_acquisition(self, pressed):
        if self.base_path_is_set == False:
            self.btn_startAcquisition.setChecked(False)
            msg = QMessageBox()
            msg.setText("Please choose base saving directory first")
            msg.exec_()
            return
        if pressed:
            # @@@ to do: add a widgetManger to enable and disable widget
            # @@@ to do: emit signal to widgetManager to disable other widgets

            # add the current location to the location list if the list is empty
            if len(self.location_list) == 0:
                self.add_location()
                self.acquisition_in_place = True
            self.setEnabled_all(False)
            self.multipointController.set_selected_configurations(
                (item.text() for item in self.list_configurations.selectedItems())
            )
            self.multipointController.start_new_experiment(
                self.lineEdit_experimentID.text()
            )
            # set parameters
            self.multipointController.set_deltaX(self.entry_deltaX.value())
            self.multipointController.set_deltaY(self.entry_deltaY.value())
            self.multipointController.set_deltaZ(self.entry_deltaZ.value())
            self.multipointController.set_deltat(self.entry_dt.value())
            self.multipointController.set_NX(self.entry_NX.value())
            self.multipointController.set_NY(self.entry_NY.value())
            self.multipointController.set_NZ(self.entry_NZ.value())
            self.multipointController.set_Nt(self.entry_Nt.value())
            self.multipointController.set_af_flag(
                self.checkbox_withAutofocus.isChecked()
            )
            self.multipointController.set_reflection_af_flag(
                self.checkbox_withReflectionAutofocus.isChecked()
            )
            self.multipointController.set_base_path(self.lineEdit_savingDir.text())
            self.multipointController.run_acquisition(self.location_list)
        else:
            self.multipointController.request_abort_aquisition()
            self.setEnabled_all(True)

    def load_last_used_locations(self):
        if self.last_used_locations is None or len(self.last_used_locations) == 0:
            return
        self.clear_only_location_list()

        for row in self.last_used_locations:
            x = row[0]
            y = row[1]
            z = row[2]
            if not np.any(np.all(self.location_list[:, :2] == [x, y], axis=1)):
                location_str = (
                    "x: "
                    + str(round(x, 3))
                    + " mm, y: "
                    + str(round(y, 3))
                    + " mm, z: "
                    + str(round(1000 * z, 1))
                    + " um"
                )
                self.dropdown_location_list.addItem(location_str)
                index = self.dropdown_location_list.count() - 1
                self.dropdown_location_list.setCurrentIndex(index)
                self.location_list = np.vstack((self.location_list, [[x, y, z]]))
                print(self.location_list)
                self.navigationViewer.register_fov_to_image(x, y)
            else:
                print("Duplicate values not added based on x and y.")
                # to-do: update z coordinate

    def acquisition_is_finished(self):
        if not self.acquisition_in_place:
            self.last_used_locations = self.location_list.copy()
        else:
            self.clear()
            self.acquisition_in_place = False
        self.btn_startAcquisition.setChecked(False)
        self.setEnabled_all(True)

    def setEnabled_all(self, enabled, exclude_btn_startAcquisition=True):
        self.btn_setSavingDir.setEnabled(enabled)
        self.lineEdit_savingDir.setEnabled(enabled)
        self.lineEdit_experimentID.setEnabled(enabled)
        self.entry_deltaX.setEnabled(enabled)
        self.entry_NX.setEnabled(enabled)
        self.entry_deltaY.setEnabled(enabled)
        self.entry_NY.setEnabled(enabled)
        self.entry_deltaZ.setEnabled(enabled)
        self.entry_NZ.setEnabled(enabled)
        self.entry_dt.setEnabled(enabled)
        self.entry_Nt.setEnabled(enabled)
        self.list_configurations.setEnabled(enabled)
        self.checkbox_withAutofocus.setEnabled(enabled)
        self.checkbox_withReflectionAutofocus.setEnabled(enabled)
        if exclude_btn_startAcquisition is not True:
            self.btn_startAcquisition.setEnabled(enabled)

    def disable_the_start_aquisition_button(self):
        self.btn_startAcquisition.setEnabled(False)

    def enable_the_start_aquisition_button(self):
        self.btn_startAcquisition.setEnabled(True)

    def add_location(self):
        x = self.navigationController.x_pos_mm
        y = self.navigationController.y_pos_mm
        z = self.navigationController.z_pos_mm
        if not np.any(np.all(self.location_list[:, :2] == [x, y], axis=1)):
            location_str = (
                "x: "
                + str(round(x, 3))
                + " mm, y: "
                + str(round(y, 3))
                + " mm, z: "
                + str(round(1000 * z, 1))
                + " um"
            )
            self.dropdown_location_list.addItem(location_str)
            index = self.dropdown_location_list.count() - 1
            self.dropdown_location_list.setCurrentIndex(index)
            self.location_list = np.vstack(
                (
                    self.location_list,
                    [
                        [
                            self.navigationController.x_pos_mm,
                            self.navigationController.y_pos_mm,
                            self.navigationController.z_pos_mm,
                        ]
                    ],
                )
            )
            print(self.location_list)
            self.navigationViewer.register_fov_to_image(x, y)
        else:
            print("Duplicate values not added based on x and y.")
            # to-do: update z coordinate

    def remove_location(self):
        index = self.dropdown_location_list.currentIndex()
        if index >= 0:
            self.dropdown_location_list.removeItem(index)
            x = self.location_list[index, 0]
            y = self.location_list[index, 1]
            z = self.location_list[index, 2]
            self.navigationViewer.deregister_fov_to_image(x, y)
            self.location_list = np.delete(self.location_list, index, axis=0)
            if len(self.location_list) == 0:
                self.navigationViewer.clear_slide()
            print(self.location_list)

    def next(self):
        index = self.dropdown_location_list.currentIndex()
        max_index = self.dropdown_location_list.count() - 1
        index = min(index + 1, max_index)
        self.dropdown_location_list.setCurrentIndex(index)
        x = self.location_list[index, 0]
        y = self.location_list[index, 1]
        z = self.location_list[index, 2]
        self.navigationController.move_x_to(x)
        self.navigationController.move_y_to(y)
        self.navigationController.move_z_to(z)

    def previous(self):
        index = self.dropdown_location_list.currentIndex()
        index = max(index - 1, 0)
        self.dropdown_location_list.setCurrentIndex(index)
        x = self.location_list[index, 0]
        y = self.location_list[index, 1]
        z = self.location_list[index, 2]
        self.navigationController.move_x_to(x)
        self.navigationController.move_y_to(y)
        self.navigationController.move_z_to(z)

    def clear(self):
        self.location_list = np.empty((0, 3), dtype=float)
        self.dropdown_location_list.clear()
        self.navigationViewer.clear_slide()

    def clear_only_location_list(self):
        self.location_list = np.empty((0, 3), dtype=float)
        self.dropdown_location_list.clear()

    def go_to(self, index):
        if index != -1:
            if index < len(
                self.location_list
            ):  # to avoid giving errors when adding new points
                x = self.location_list[index, 0]
                y = self.location_list[index, 1]
                z = self.location_list[index, 2]
                self.navigationController.move_x_to(x)
                self.navigationController.move_y_to(y)
                self.navigationController.move_z_to(z)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_A and event.modifiers() == Qt.ControlModifier:
            self.add_location()
        else:
            super().keyPressEvent(event)

    def _update_z(self, index, z_mm):
        self.location_list[index, 2] = z_mm
        location_str = (
            "x: "
            + str(round(self.location_list[index, 0], 3))
            + " mm, y: "
            + str(round(self.location_list[index, 1], 3))
            + " mm, z: "
            + str(round(1000 * z_mm, 1))
            + " um"
        )
        self.dropdown_location_list.setItemText(index, location_str)

    def export_location_list(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Location List", "", "CSV Files (*.csv);;All Files (*)"
        )
        if file_path:
            location_list_df = pd.DataFrame(
                self.location_list, columns=["x (mm)", "y (mm)", "z (um)"]
            )
            location_list_df["i"] = 0
            location_list_df["j"] = 0
            location_list_df["k"] = 0
            location_list_df.to_csv(file_path, index=False, header=True)

    def import_location_list(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Import Location List", "", "CSV Files (*.csv);;All Files (*)"
        )
        if file_path:
            location_list_df = pd.read_csv(file_path)
            location_list_df_relevant = None
            try:
                location_list_df_relevant = location_list_df[
                    ["x (mm)", "y (mm)", "z (um)"]
                ]
            except KeyError:
                print("Improperly formatted location list being imported")
                return
            self.clear_only_location_list()
            for index, row in location_list_df_relevant.iterrows():
                x = row["x (mm)"]
                y = row["y (mm)"]
                z = row["z (um)"]
                if not np.any(np.all(self.location_list[:, :2] == [x, y], axis=1)):
                    location_str = (
                        "x: "
                        + str(round(x, 3))
                        + " mm, y: "
                        + str(round(y, 3))
                        + " mm, z: "
                        + str(round(1000 * z, 1))
                        + " um"
                    )
                    self.dropdown_location_list.addItem(location_str)
                    index = self.dropdown_location_list.count() - 1
                    self.dropdown_location_list.setCurrentIndex(index)
                    self.location_list = np.vstack((self.location_list, [[x, y, z]]))
                    self.navigationViewer.register_fov_to_image(x, y)
                else:
                    print("Duplicate values not added based on x and y.")
            print(self.location_list)


class TrackingControllerWidget(QFrame):
    def __init__(
        self,
        trackingController,
        configurationManager,
        show_configurations=True,
        main=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.trackingController = trackingController
        self.configurationManager = configurationManager
        self.base_path_is_set = False
        self.add_components(show_configurations)
        self.setFrameStyle(QFrame.Panel | QFrame.Raised)

    def add_components(self, show_configurations):
        self.btn_setSavingDir = QPushButton("Browse")
        self.btn_setSavingDir.setDefault(False)
        self.btn_setSavingDir.setIcon(QIcon("icon/folder.png"))
        self.lineEdit_savingDir = QLineEdit()
        self.lineEdit_savingDir.setReadOnly(True)
        self.lineEdit_savingDir.setText("Choose a base saving directory")
        self.lineEdit_savingDir.setText(CONFIG.DEFAULT_SAVING_PATH)
        self.trackingController.set_base_path(CONFIG.DEFAULT_SAVING_PATH)
        self.base_path_is_set = True

        self.lineEdit_experimentID = QLineEdit()

        self.dropdown_objective = QComboBox()
        self.dropdown_objective.addItems(list(CONFIG.OBJECTIVES.keys()))
        self.dropdown_objective.setCurrentText(CONFIG.DEFAULT_OBJECTIVE)

        self.dropdown_tracker = QComboBox()
        self.dropdown_tracker.addItems(CONFIG.TRACKERS)
        self.dropdown_tracker.setCurrentText(CONFIG.DEFAULT_TRACKER)

        self.entry_tracking_interval = QDoubleSpinBox()
        self.entry_tracking_interval.setMinimum(0)
        self.entry_tracking_interval.setMaximum(30)
        self.entry_tracking_interval.setSingleStep(0.5)
        self.entry_tracking_interval.setValue(0)

        self.list_configurations = QListWidget()
        for microscope_configuration in self.configurationManager.configurations:
            self.list_configurations.addItems([microscope_configuration.name])
        self.list_configurations.setSelectionMode(
            QAbstractItemView.MultiSelection
        )  # ref: https://doc.qt.io/qt-5/qabstractitemview.html#SelectionMode-enum

        self.checkbox_withAutofocus = QCheckBox("With CONFIG.AF")
        self.checkbox_saveImages = QCheckBox("Save Images")
        self.btn_track = QPushButton("Start Tracking")
        self.btn_track.setCheckable(True)
        self.btn_track.setChecked(False)

        self.checkbox_enable_stage_tracking = QCheckBox(" Enable Stage Tracking")
        self.checkbox_enable_stage_tracking.setChecked(True)

        # layout
        grid_line0 = QGridLayout()
        tmp = QLabel("Saving Path")
        tmp.setFixedWidth(90)
        grid_line0.addWidget(tmp, 0, 0)
        grid_line0.addWidget(self.lineEdit_savingDir, 0, 1, 1, 2)
        grid_line0.addWidget(self.btn_setSavingDir, 0, 3)
        tmp = QLabel("Experiment ID")
        tmp.setFixedWidth(90)
        grid_line0.addWidget(tmp, 1, 0)
        grid_line0.addWidget(self.lineEdit_experimentID, 1, 1, 1, 1)
        tmp = QLabel("Objective")
        tmp.setFixedWidth(90)
        grid_line0.addWidget(tmp, 1, 2)
        grid_line0.addWidget(self.dropdown_objective, 1, 3)

        grid_line3 = QHBoxLayout()
        tmp = QLabel("Configurations")
        tmp.setFixedWidth(90)
        grid_line3.addWidget(tmp)
        grid_line3.addWidget(self.list_configurations)

        grid_line1 = QHBoxLayout()
        tmp = QLabel("Tracker")
        grid_line1.addWidget(tmp)
        grid_line1.addWidget(self.dropdown_tracker)
        tmp = QLabel("Tracking Interval (s)")
        grid_line1.addWidget(tmp)
        grid_line1.addWidget(self.entry_tracking_interval)
        grid_line1.addWidget(self.checkbox_withAutofocus)
        grid_line1.addWidget(self.checkbox_saveImages)

        grid_line4 = QGridLayout()
        grid_line4.addWidget(self.btn_track, 0, 0, 1, 3)
        grid_line4.addWidget(self.checkbox_enable_stage_tracking, 0, 4)

        self.grid = QVBoxLayout()
        self.grid.addLayout(grid_line0)
        if show_configurations:
            self.grid.addLayout(grid_line3)
        else:
            self.list_configurations.setCurrentRow(0)  # select the first configuration
        self.grid.addLayout(grid_line1)
        self.grid.addLayout(grid_line4)
        self.grid.addStretch()
        self.setLayout(self.grid)

        # connections - buttons, checkboxes, entries
        self.checkbox_enable_stage_tracking.stateChanged.connect(
            self.trackingController.toggle_stage_tracking
        )
        self.checkbox_withAutofocus.stateChanged.connect(
            self.trackingController.toggel_enable_af
        )
        self.checkbox_saveImages.stateChanged.connect(
            self.trackingController.toggel_save_images
        )
        self.entry_tracking_interval.valueChanged.connect(
            self.trackingController.set_tracking_time_interval
        )
        self.btn_setSavingDir.clicked.connect(self.set_saving_dir)
        self.btn_track.clicked.connect(self.toggle_acquisition)
        # connections - selections and entries
        self.dropdown_tracker.currentIndexChanged.connect(self.update_tracker)
        self.dropdown_objective.currentIndexChanged.connect(self.update_pixel_size)
        # controller to widget
        self.trackingController.signal_tracking_stopped.connect(
            self.slot_tracking_stopped
        )

        # run initialization functions
        self.update_pixel_size()
        self.trackingController.update_image_resizing_factor(
            1
        )  # to add: image resizing slider

    def slot_joystick_button_pressed(self):
        self.btn_track.toggle()
        if self.btn_track.isChecked():
            if self.base_path_is_set == False:
                self.btn_track.setChecked(False)
                msg = QMessageBox()
                msg.setText("Please choose base saving directory first")
                msg.exec_()
                return
            self.setEnabled_all(False)
            self.trackingController.start_new_experiment(
                self.lineEdit_experimentID.text()
            )
            self.trackingController.set_selected_configurations(
                (item.text() for item in self.list_configurations.selectedItems())
            )
            self.trackingController.start_tracking()
        else:
            self.trackingController.stop_tracking()

    def slot_tracking_stopped(self):
        self.btn_track.setChecked(False)
        self.setEnabled_all(True)
        print("tracking stopped")

    def set_saving_dir(self):
        dialog = QFileDialog()
        save_dir_base = dialog.getExistingDirectory(None, "Select Folder")
        self.trackingController.set_base_path(save_dir_base)
        self.lineEdit_savingDir.setText(save_dir_base)
        self.base_path_is_set = True

    def toggle_acquisition(self, pressed):
        if pressed:
            if self.base_path_is_set == False:
                self.btn_track.setChecked(False)
                msg = QMessageBox()
                msg.setText("Please choose base saving directory first")
                msg.exec_()
                return
            # @@@ to do: add a widgetManger to enable and disable widget
            # @@@ to do: emit signal to widgetManager to disable other widgets
            self.setEnabled_all(False)
            self.trackingController.start_new_experiment(
                self.lineEdit_experimentID.text()
            )
            self.trackingController.set_selected_configurations(
                (item.text() for item in self.list_configurations.selectedItems())
            )
            self.trackingController.start_tracking()
        else:
            self.trackingController.stop_tracking()

    def setEnabled_all(self, enabled):
        self.btn_setSavingDir.setEnabled(enabled)
        self.lineEdit_savingDir.setEnabled(enabled)
        self.lineEdit_experimentID.setEnabled(enabled)
        self.dropdown_tracker
        self.dropdown_objective
        self.list_configurations.setEnabled(enabled)

    def update_tracker(self, index):
        self.trackingController.update_tracker_selection(
            self.dropdown_tracker.currentText()
        )

    def update_pixel_size(self):
        objective = self.dropdown_objective.currentText()
        self.trackingController.objective = objective
        # self.internal_state.data['Objective'] = self.objective
        pixel_size_um = CONFIG.CAMERA_PIXEL_SIZE_UM[CONFIG.CAMERA_SENSOR] / (
            CONFIG.TUBE_LENS_MM
            / (
                CONFIG.OBJECTIVES[objective]["tube_lens_f_mm"]
                / CONFIG.OBJECTIVES[objective]["magnification"]
            )
        )
        self.trackingController.update_pixel_size(pixel_size_um)
        print("pixel size is " + str(pixel_size_um) + " um")

    """
        # connections
        self.checkbox_withAutofocus.stateChanged.connect(self.trackingController.set_af_flag)
        self.btn_setSavingDir.clicked.connect(self.set_saving_dir)
        self.btn_startAcquisition.clicked.connect(self.toggle_acquisition)
        self.trackingController.trackingStopped.connect(self.acquisition_is_finished)

    def set_saving_dir(self):
        dialog = QFileDialog()
        save_dir_base = dialog.getExistingDirectory(None, "Select Folder")
        self.plateReadingController.set_base_path(save_dir_base)
        self.lineEdit_savingDir.setText(save_dir_base)
        self.base_path_is_set = True

    def toggle_acquisition(self,pressed):
        if self.base_path_is_set == False:
            self.btn_startAcquisition.setChecked(False)
            msg = QMessageBox()
            msg.setText("Please choose base saving directory first")
            msg.exec_()
            return
        if pressed:
            # @@@ to do: add a widgetManger to enable and disable widget 
            # @@@ to do: emit signal to widgetManager to disable other widgets
            self.setEnabled_all(False)
            self.trackingController.start_new_experiment(self.lineEdit_experimentID.text())
            self.trackingController.set_selected_configurations((item.text() for item in self.list_configurations.selectedItems()))
            self.trackingController.set_selected_columns(list(map(int,[item.text() for item in self.list_columns.selectedItems()])))
            self.trackingController.run_acquisition()
        else:
            self.trackingController.stop_acquisition() # to implement
            pass

    def acquisition_is_finished(self):
        self.btn_startAcquisition.setChecked(False)
        self.setEnabled_all(True)

    def setEnabled_all(self,enabled,exclude_btn_startAcquisition=False):
        self.btn_setSavingDir.setEnabled(enabled)
        self.lineEdit_savingDir.setEnabled(enabled)
        self.lineEdit_experimentID.setEnabled(enabled)
        self.list_columns.setEnabled(enabled)
        self.list_configurations.setEnabled(enabled)
        self.checkbox_withAutofocus.setEnabled(enabled)
        if exclude_btn_startAcquisition is not True:
            self.btn_startAcquisition.setEnabled(enabled)
    """


class PlateReaderAcquisitionWidget(QFrame):
    def __init__(
        self,
        plateReadingController,
        configurationManager=None,
        show_configurations=True,
        main=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.plateReadingController = plateReadingController
        self.configurationManager = configurationManager
        self.base_path_is_set = False
        self.add_components(show_configurations)
        self.setFrameStyle(QFrame.Panel | QFrame.Raised)

    def add_components(self, show_configurations):
        self.btn_setSavingDir = QPushButton("Browse")
        self.btn_setSavingDir.setDefault(False)
        self.btn_setSavingDir.setIcon(QIcon("icon/folder.png"))
        self.lineEdit_savingDir = QLineEdit()
        self.lineEdit_savingDir.setReadOnly(True)
        self.lineEdit_savingDir.setText("Choose a base saving directory")
        self.lineEdit_savingDir.setText(CONFIG.DEFAULT_SAVING_PATH)
        self.plateReadingController.set_base_path(CONFIG.DEFAULT_SAVING_PATH)
        self.base_path_is_set = True

        self.lineEdit_experimentID = QLineEdit()

        self.list_columns = QListWidget()
        for i in range(CONFIG.PLATE_READER.NUMBER_OF_COLUMNS):
            self.list_columns.addItems([str(i + 1)])
        self.list_columns.setSelectionMode(
            QAbstractItemView.MultiSelection
        )  # ref: https://doc.qt.io/qt-5/qabstractitemview.html#SelectionMode-enum

        self.list_configurations = QListWidget()
        for microscope_configuration in self.configurationManager.configurations:
            self.list_configurations.addItems([microscope_configuration.name])
        self.list_configurations.setSelectionMode(
            QAbstractItemView.MultiSelection
        )  # ref: https://doc.qt.io/qt-5/qabstractitemview.html#SelectionMode-enum

        self.checkbox_withAutofocus = QCheckBox("With CONFIG.AF")
        self.btn_startAcquisition = QPushButton("Start Acquisition")
        self.btn_startAcquisition.setCheckable(True)
        self.btn_startAcquisition.setChecked(False)

        self.btn_startAcquisition.setEnabled(False)

        # layout
        grid_line0 = QGridLayout()
        tmp = QLabel("Saving Path")
        tmp.setFixedWidth(90)
        grid_line0.addWidget(tmp)
        grid_line0.addWidget(self.lineEdit_savingDir, 0, 1)
        grid_line0.addWidget(self.btn_setSavingDir, 0, 2)

        grid_line1 = QGridLayout()
        tmp = QLabel("Sample ID")
        tmp.setFixedWidth(90)
        grid_line1.addWidget(tmp)
        grid_line1.addWidget(self.lineEdit_experimentID, 0, 1)

        grid_line2 = QGridLayout()
        tmp = QLabel("Columns")
        tmp.setFixedWidth(90)
        grid_line2.addWidget(tmp)
        grid_line2.addWidget(self.list_columns, 0, 1)

        grid_line3 = QHBoxLayout()
        tmp = QLabel("Configurations")
        tmp.setFixedWidth(90)
        grid_line3.addWidget(tmp)
        grid_line3.addWidget(self.list_configurations)
        # grid_line3.addWidget(self.checkbox_withAutofocus)

        self.grid = QGridLayout()
        self.grid.addLayout(grid_line0, 0, 0)
        self.grid.addLayout(grid_line1, 1, 0)
        self.grid.addLayout(grid_line2, 2, 0)
        if show_configurations:
            self.grid.addLayout(grid_line3, 3, 0)
        else:
            self.list_configurations.setCurrentRow(0)  # select the first configuration
        self.grid.addWidget(self.btn_startAcquisition, 4, 0)
        self.setLayout(self.grid)

        # add and display a timer - to be implemented
        # self.timer = QTimer()

        # connections
        self.checkbox_withAutofocus.stateChanged.connect(
            self.plateReadingController.set_af_flag
        )
        self.btn_setSavingDir.clicked.connect(self.set_saving_dir)
        self.btn_startAcquisition.clicked.connect(self.toggle_acquisition)
        self.plateReadingController.acquisitionFinished.connect(
            self.acquisition_is_finished
        )

    def set_saving_dir(self):
        dialog = QFileDialog()
        save_dir_base = dialog.getExistingDirectory(None, "Select Folder")
        self.plateReadingController.set_base_path(save_dir_base)
        self.lineEdit_savingDir.setText(save_dir_base)
        self.base_path_is_set = True

    def toggle_acquisition(self, pressed):
        if self.base_path_is_set == False:
            self.btn_startAcquisition.setChecked(False)
            msg = QMessageBox()
            msg.setText("Please choose base saving directory first")
            msg.exec_()
            return
        if pressed:
            # @@@ to do: add a widgetManger to enable and disable widget
            # @@@ to do: emit signal to widgetManager to disable other widgets
            self.setEnabled_all(False)
            self.plateReadingController.start_new_experiment(
                self.lineEdit_experimentID.text()
            )
            self.plateReadingController.set_selected_configurations(
                (item.text() for item in self.list_configurations.selectedItems())
            )
            self.plateReadingController.set_selected_columns(
                list(
                    map(
                        int, [item.text() for item in self.list_columns.selectedItems()]
                    )
                )
            )
            self.plateReadingController.run_acquisition()
        else:
            self.plateReadingController.stop_acquisition()  # to implement
            pass

    def acquisition_is_finished(self):
        self.btn_startAcquisition.setChecked(False)
        self.setEnabled_all(True)

    def setEnabled_all(self, enabled, exclude_btn_startAcquisition=False):
        self.btn_setSavingDir.setEnabled(enabled)
        self.lineEdit_savingDir.setEnabled(enabled)
        self.lineEdit_experimentID.setEnabled(enabled)
        self.list_columns.setEnabled(enabled)
        self.list_configurations.setEnabled(enabled)
        self.checkbox_withAutofocus.setEnabled(enabled)
        self.checkbox_withReflectionAutofocus.setEnabled(enabled)
        if exclude_btn_startAcquisition is not True:
            self.btn_startAcquisition.setEnabled(enabled)

    def slot_homing_complete(self):
        self.btn_startAcquisition.setEnabled(True)


class PlateReaderNavigationWidget(QFrame):
    def __init__(self, plateReaderNavigationController, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_components()
        self.setFrameStyle(QFrame.Panel | QFrame.Raised)
        self.plateReaderNavigationController = plateReaderNavigationController

    def add_components(self):
        self.dropdown_column = QComboBox()
        self.dropdown_column.addItems([""])
        self.dropdown_column.addItems(
            [str(i + 1) for i in range(CONFIG.PLATE_READER.NUMBER_OF_COLUMNS)]
        )
        self.dropdown_row = QComboBox()
        self.dropdown_row.addItems([""])
        self.dropdown_row.addItems(
            [
                chr(i)
                for i in range(ord("A"), ord("A") + CONFIG.PLATE_READER.NUMBER_OF_ROWS)
            ]
        )
        self.btn_moveto = QPushButton("Move To")
        self.btn_home = QPushButton("Home")
        self.label_current_location = QLabel()
        self.label_current_location.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        self.label_current_location.setFixedWidth(50)

        self.dropdown_column.setEnabled(False)
        self.dropdown_row.setEnabled(False)
        self.btn_moveto.setEnabled(False)

        # layout
        grid_line0 = QHBoxLayout()
        # tmp = QLabel('Saving Path')
        # tmp.setFixedWidth(90)
        grid_line0.addWidget(self.btn_home)
        grid_line0.addWidget(QLabel("Column"))
        grid_line0.addWidget(self.dropdown_column)
        grid_line0.addWidget(QLabel("Row"))
        grid_line0.addWidget(self.dropdown_row)
        grid_line0.addWidget(self.btn_moveto)
        grid_line0.addStretch()
        grid_line0.addWidget(self.label_current_location)

        self.grid = QGridLayout()
        self.grid.addLayout(grid_line0, 0, 0)
        self.setLayout(self.grid)

        self.btn_home.clicked.connect(self.home)
        self.btn_moveto.clicked.connect(self.move)

    def home(self):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setText("Confirm your action")
        msg.setInformativeText("Click OK to run homing")
        msg.setWindowTitle("Confirmation")
        msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        msg.setDefaultButton(QMessageBox.Cancel)
        retval = msg.exec_()
        if QMessageBox.Ok == retval:
            self.plateReaderNavigationController.home()

    def move(self):
        self.plateReaderNavigationController.moveto(
            self.dropdown_column.currentText(), self.dropdown_row.currentText()
        )

    def slot_homing_complete(self):
        self.dropdown_column.setEnabled(True)
        self.dropdown_row.setEnabled(True)
        self.btn_moveto.setEnabled(True)

    def update_current_location(self, location_str):
        self.label_current_location.setText(location_str)
        row = location_str[0]
        column = location_str[1:]
        self.dropdown_row.setCurrentText(row)
        self.dropdown_column.setCurrentText(column)


class TriggerControlWidget(QFrame):
    # for synchronized trigger
    signal_toggle_live = Signal(bool)
    signal_trigger_mode = Signal(str)
    signal_trigger_fps = Signal(float)

    def __init__(self, microcontroller2):
        super().__init__()
        self.fps_trigger = 10
        self.fps_display = 10
        self.microcontroller2 = microcontroller2
        self.triggerMode = TriggerModeSetting.SOFTWARE
        self.add_components()
        self.setFrameStyle(QFrame.Panel | QFrame.Raised)

    def add_components(self):
        # line 0: trigger mode
        self.triggerMode = None
        self.dropdown_triggerManu = QComboBox()
        self.dropdown_triggerManu.addItems(
            [TriggerModeSetting.SOFTWARE, TriggerModeSetting.HARDWARE]
        )

        # line 1: fps
        self.entry_triggerFPS = QDoubleSpinBox()
        self.entry_triggerFPS.setMinimum(0.02)
        self.entry_triggerFPS.setMaximum(1000)
        self.entry_triggerFPS.setSingleStep(1)
        self.entry_triggerFPS.setValue(self.fps_trigger)

        self.btn_live = QPushButton("Live")
        self.btn_live.setCheckable(True)
        self.btn_live.setChecked(False)
        self.btn_live.setDefault(False)

        # connections
        self.dropdown_triggerManu.currentIndexChanged.connect(self.update_trigger_mode)
        self.btn_live.clicked.connect(self.toggle_live)
        self.entry_triggerFPS.valueChanged.connect(self.update_trigger_fps)

        # inititialization
        self.microcontroller2.set_camera_trigger_frequency(self.fps_trigger)

        # layout
        grid_line0 = QGridLayout()
        grid_line0.addWidget(QLabel("Trigger Mode"), 0, 0)
        grid_line0.addWidget(self.dropdown_triggerManu, 0, 1)
        grid_line0.addWidget(QLabel("Trigger FPS"), 0, 2)
        grid_line0.addWidget(self.entry_triggerFPS, 0, 3)
        grid_line0.addWidget(self.btn_live, 1, 0, 1, 4)
        self.setLayout(grid_line0)

    def toggle_live(self, pressed):
        self.signal_toggle_live.emit(pressed)
        if pressed:
            self.microcontroller2.start_camera_trigger()
        else:
            self.microcontroller2.stop_camera_trigger()

    def update_trigger_mode(self):
        self.signal_trigger_mode.emit(self.dropdown_triggerManu.currentText())

    def update_trigger_fps(self, fps):
        self.fps_trigger = fps
        self.signal_trigger_fps.emit(fps)
        self.microcontroller2.set_camera_trigger_frequency(self.fps_trigger)


class MultiCameraRecordingWidget(QFrame):
    def __init__(self, streamHandler, imageSaver, channels, main=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.imageSaver = imageSaver  # for saving path control
        self.streamHandler = streamHandler
        self.channels = channels
        self.base_path_is_set = False
        self.add_components()
        self.setFrameStyle(QFrame.Panel | QFrame.Raised)

    def add_components(self):
        self.btn_setSavingDir = QPushButton("Browse")
        self.btn_setSavingDir.setDefault(False)
        self.btn_setSavingDir.setIcon(QIcon("icon/folder.png"))

        self.lineEdit_savingDir = QLineEdit()
        self.lineEdit_savingDir.setReadOnly(True)
        self.lineEdit_savingDir.setText("Choose a base saving directory")

        self.lineEdit_experimentID = QLineEdit()

        self.entry_saveFPS = QDoubleSpinBox()
        self.entry_saveFPS.setMinimum(0.02)
        self.entry_saveFPS.setMaximum(1000)
        self.entry_saveFPS.setSingleStep(1)
        self.entry_saveFPS.setValue(1)
        for channel in self.channels:
            self.streamHandler[channel].set_save_fps(1)

        self.entry_timeLimit = QSpinBox()
        self.entry_timeLimit.setMinimum(-1)
        self.entry_timeLimit.setMaximum(60 * 60 * 24 * 30)
        self.entry_timeLimit.setSingleStep(1)
        self.entry_timeLimit.setValue(-1)

        self.btn_record = QPushButton("Record")
        self.btn_record.setCheckable(True)
        self.btn_record.setChecked(False)
        self.btn_record.setDefault(False)

        grid_line1 = QGridLayout()
        grid_line1.addWidget(QLabel("Saving Path"))
        grid_line1.addWidget(self.lineEdit_savingDir, 0, 1)
        grid_line1.addWidget(self.btn_setSavingDir, 0, 2)

        grid_line2 = QGridLayout()
        grid_line2.addWidget(QLabel("Experiment ID"), 0, 0)
        grid_line2.addWidget(self.lineEdit_experimentID, 0, 1)

        grid_line3 = QGridLayout()
        grid_line3.addWidget(QLabel("Saving FPS"), 0, 0)
        grid_line3.addWidget(self.entry_saveFPS, 0, 1)
        grid_line3.addWidget(QLabel("Time Limit (s)"), 0, 2)
        grid_line3.addWidget(self.entry_timeLimit, 0, 3)
        grid_line3.addWidget(self.btn_record, 0, 4)

        self.grid = QGridLayout()
        self.grid.addLayout(grid_line1, 0, 0)
        self.grid.addLayout(grid_line2, 1, 0)
        self.grid.addLayout(grid_line3, 2, 0)
        self.setLayout(self.grid)

        # add and display a timer - to be implemented
        # self.timer = QTimer()

        # connections
        self.btn_setSavingDir.clicked.connect(self.set_saving_dir)
        self.btn_record.clicked.connect(self.toggle_recording)
        for channel in self.channels:
            self.entry_saveFPS.valueChanged.connect(
                self.streamHandler[channel].set_save_fps
            )
            self.entry_timeLimit.valueChanged.connect(
                self.imageSaver[channel].set_recording_time_limit
            )
            self.imageSaver[channel].stop_recording.connect(self.stop_recording)

    def set_saving_dir(self):
        dialog = QFileDialog()
        save_dir_base = dialog.getExistingDirectory(None, "Select Folder")
        for channel in self.channels:
            self.imageSaver[channel].set_base_path(save_dir_base)
        self.lineEdit_savingDir.setText(save_dir_base)
        self.save_dir_base = save_dir_base
        self.base_path_is_set = True

    def toggle_recording(self, pressed):
        if self.base_path_is_set == False:
            self.btn_record.setChecked(False)
            msg = QMessageBox()
            msg.setText("Please choose base saving directory first")
            msg.exec_()
            return
        if pressed:
            self.lineEdit_experimentID.setEnabled(False)
            self.btn_setSavingDir.setEnabled(False)
            experiment_ID = self.lineEdit_experimentID.text()
            experiment_ID = (
                experiment_ID + "_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%-S.%f")
            )
            os.mkdir(os.path.join(self.save_dir_base, experiment_ID))
            for channel in self.channels:
                self.imageSaver[channel].start_new_experiment(
                    os.path.join(experiment_ID, channel), add_timestamp=False
                )
                self.streamHandler[channel].start_recording()
        else:
            for channel in self.channels:
                self.streamHandler[channel].stop_recording()
            self.lineEdit_experimentID.setEnabled(True)
            self.btn_setSavingDir.setEnabled(True)

    # stop_recording can be called by imageSaver
    def stop_recording(self):
        self.lineEdit_experimentID.setEnabled(True)
        self.btn_record.setChecked(False)
        for channel in self.channels:
            self.streamHandler[channel].stop_recording()
        self.btn_setSavingDir.setEnabled(True)


class WaveformDisplay(QFrame):

    def __init__(
        self, N=1000, include_x=True, include_y=True, main=None, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.N = N
        self.include_x = include_x
        self.include_y = include_y
        self.add_components()
        self.setFrameStyle(QFrame.Panel | QFrame.Raised)

    def add_components(self):
        self.plotWidget = {}
        self.plotWidget["X"] = PlotWidget("X", N=self.N, add_legend=True)
        self.plotWidget["Y"] = PlotWidget("X", N=self.N, add_legend=True)

        layout = QGridLayout()  # layout = QStackedLayout()
        if self.include_x:
            layout.addWidget(self.plotWidget["X"], 0, 0)
        if self.include_y:
            layout.addWidget(self.plotWidget["Y"], 1, 0)
        self.setLayout(layout)

    def plot(self, time, data):
        if self.include_x:
            self.plotWidget["X"].plot(
                time, data[0, :], "X", color=(255, 255, 255), clear=True
            )
        if self.include_y:
            self.plotWidget["Y"].plot(
                time, data[1, :], "Y", color=(255, 255, 255), clear=True
            )

    def update_N(self, N):
        self.N = N
        self.plotWidget["X"].update_N(N)
        self.plotWidget["Y"].update_N(N)


class PlotWidget(pg.GraphicsLayoutWidget):

    def __init__(self, title="", N=1000, parent=None, add_legend=False):
        super().__init__(parent)
        self.plotWidget = self.addPlot(
            title="", axisItems={"bottom": pg.DateAxisItem()}
        )
        if add_legend:
            self.plotWidget.addLegend()
        self.N = N

    def plot(self, x, y, label, color, clear=False):
        self.plotWidget.plot(
            x[-self.N :],
            y[-self.N :],
            pen=pg.mkPen(color=color, width=2),
            name=label,
            clear=clear,
        )

    def update_N(self, N):
        self.N = N


class DisplacementMeasurementWidget(QFrame):
    def __init__(
        self,
        displacementMeasurementController,
        waveformDisplay,
        main=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.displacementMeasurementController = displacementMeasurementController
        self.waveformDisplay = waveformDisplay
        self.add_components()
        self.setFrameStyle(QFrame.Panel | QFrame.Raised)

    def add_components(self):
        self.entry_x_offset = QDoubleSpinBox()
        self.entry_x_offset.setMinimum(0)
        self.entry_x_offset.setMaximum(3000)
        self.entry_x_offset.setSingleStep(0.2)
        self.entry_x_offset.setDecimals(3)
        self.entry_x_offset.setValue(0)
        self.entry_x_offset.setKeyboardTracking(False)

        self.entry_y_offset = QDoubleSpinBox()
        self.entry_y_offset.setMinimum(0)
        self.entry_y_offset.setMaximum(3000)
        self.entry_y_offset.setSingleStep(0.2)
        self.entry_y_offset.setDecimals(3)
        self.entry_y_offset.setValue(0)
        self.entry_y_offset.setKeyboardTracking(False)

        self.entry_x_scaling = QDoubleSpinBox()
        self.entry_x_scaling.setMinimum(-100)
        self.entry_x_scaling.setMaximum(100)
        self.entry_x_scaling.setSingleStep(0.1)
        self.entry_x_scaling.setDecimals(3)
        self.entry_x_scaling.setValue(1)
        self.entry_x_scaling.setKeyboardTracking(False)

        self.entry_y_scaling = QDoubleSpinBox()
        self.entry_y_scaling.setMinimum(-100)
        self.entry_y_scaling.setMaximum(100)
        self.entry_y_scaling.setSingleStep(0.1)
        self.entry_y_scaling.setDecimals(3)
        self.entry_y_scaling.setValue(1)
        self.entry_y_scaling.setKeyboardTracking(False)

        self.entry_N_average = QSpinBox()
        self.entry_N_average.setMinimum(1)
        self.entry_N_average.setMaximum(25)
        self.entry_N_average.setSingleStep(1)
        self.entry_N_average.setValue(1)
        self.entry_N_average.setKeyboardTracking(False)

        self.entry_N = QSpinBox()
        self.entry_N.setMinimum(1)
        self.entry_N.setMaximum(5000)
        self.entry_N.setSingleStep(1)
        self.entry_N.setValue(1000)
        self.entry_N.setKeyboardTracking(False)

        self.reading_x = QLabel()
        self.reading_x.setNum(0)
        self.reading_x.setFrameStyle(QFrame.Panel | QFrame.Sunken)

        self.reading_y = QLabel()
        self.reading_y.setNum(0)
        self.reading_y.setFrameStyle(QFrame.Panel | QFrame.Sunken)

        # layout
        grid_line0 = QGridLayout()
        grid_line0.addWidget(QLabel("x offset"), 0, 0)
        grid_line0.addWidget(self.entry_x_offset, 0, 1)
        grid_line0.addWidget(QLabel("x scaling"), 0, 2)
        grid_line0.addWidget(self.entry_x_scaling, 0, 3)
        grid_line0.addWidget(QLabel("y offset"), 0, 4)
        grid_line0.addWidget(self.entry_y_offset, 0, 5)
        grid_line0.addWidget(QLabel("y scaling"), 0, 6)
        grid_line0.addWidget(self.entry_y_scaling, 0, 7)

        grid_line1 = QGridLayout()
        grid_line1.addWidget(QLabel("d from x"), 0, 0)
        grid_line1.addWidget(self.reading_x, 0, 1)
        grid_line1.addWidget(QLabel("d from y"), 0, 2)
        grid_line1.addWidget(self.reading_y, 0, 3)
        grid_line1.addWidget(QLabel("N average"), 0, 4)
        grid_line1.addWidget(self.entry_N_average, 0, 5)
        grid_line1.addWidget(QLabel("N"), 0, 6)
        grid_line1.addWidget(self.entry_N, 0, 7)

        self.grid = QGridLayout()
        self.grid.addLayout(grid_line0, 0, 0)
        self.grid.addLayout(grid_line1, 1, 0)
        self.setLayout(self.grid)

        # connections
        self.entry_x_offset.valueChanged.connect(self.update_settings)
        self.entry_y_offset.valueChanged.connect(self.update_settings)
        self.entry_x_scaling.valueChanged.connect(self.update_settings)
        self.entry_y_scaling.valueChanged.connect(self.update_settings)
        self.entry_N_average.valueChanged.connect(self.update_settings)
        self.entry_N.valueChanged.connect(self.update_settings)
        self.entry_N.valueChanged.connect(self.update_waveformDisplay_N)

    def update_settings(self, new_value):
        print("update settings")
        self.displacementMeasurementController.update_settings(
            self.entry_x_offset.value(),
            self.entry_y_offset.value(),
            self.entry_x_scaling.value(),
            self.entry_y_scaling.value(),
            self.entry_N_average.value(),
            self.entry_N.value(),
        )

    def update_waveformDisplay_N(self, N):
        self.waveformDisplay.update_N(N)

    def display_readings(self, readings):
        self.reading_x.setText("{:.2f}".format(readings[0]))
        self.reading_y.setText("{:.2f}".format(readings[1]))


class LaserAutofocusControlWidget(QFrame):
    def __init__(self, laserAutofocusController, main=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.laserAutofocusController = laserAutofocusController
        self.add_components()
        self.setFrameStyle(QFrame.Panel | QFrame.Raised)

    def add_components(self):
        self.btn_initialize = QPushButton("Initialize")
        self.btn_initialize.setCheckable(False)
        self.btn_initialize.setChecked(False)
        self.btn_initialize.setDefault(False)

        self.btn_set_reference = QPushButton("Set as reference plane")
        self.btn_set_reference.setCheckable(False)
        self.btn_set_reference.setChecked(False)
        self.btn_set_reference.setDefault(False)
        if not self.laserAutofocusController.is_initialized:
            self.btn_set_reference.setEnabled(False)

        self.label_displacement = QLabel()
        self.label_displacement.setFrameStyle(QFrame.Panel | QFrame.Sunken)

        self.btn_measure_displacement = QPushButton("Measure displacement")
        self.btn_measure_displacement.setCheckable(False)
        self.btn_measure_displacement.setChecked(False)
        self.btn_measure_displacement.setDefault(False)
        if not self.laserAutofocusController.is_initialized:
            self.btn_measure_displacement.setEnabled(False)

        self.entry_target = QDoubleSpinBox()
        self.entry_target.setMinimum(-100)
        self.entry_target.setMaximum(100)
        self.entry_target.setSingleStep(0.01)
        self.entry_target.setDecimals(2)
        self.entry_target.setValue(0)
        self.entry_target.setKeyboardTracking(False)

        self.btn_move_to_target = QPushButton("Move to target")
        self.btn_move_to_target.setCheckable(False)
        self.btn_move_to_target.setChecked(False)
        self.btn_move_to_target.setDefault(False)
        if not self.laserAutofocusController.is_initialized:
            self.btn_move_to_target.setEnabled(False)

        self.grid = QGridLayout()
        self.grid.addWidget(self.btn_initialize, 0, 0, 1, 3)
        self.grid.addWidget(self.btn_set_reference, 1, 0, 1, 3)
        self.grid.addWidget(QLabel("Displacement (um)"), 2, 0)
        self.grid.addWidget(self.label_displacement, 2, 1)
        self.grid.addWidget(self.btn_measure_displacement, 2, 2)
        self.grid.addWidget(QLabel("Target (um)"), 3, 0)
        self.grid.addWidget(self.entry_target, 3, 1)
        self.grid.addWidget(self.btn_move_to_target, 3, 2)
        self.grid.setRowStretch(self.grid.rowCount(), 1)

        self.setLayout(self.grid)

        # make connections
        self.btn_initialize.clicked.connect(self.init_controller)
        self.btn_set_reference.clicked.connect(
            self.laserAutofocusController.set_reference
        )
        self.btn_measure_displacement.clicked.connect(
            self.laserAutofocusController.measure_displacement
        )
        self.btn_move_to_target.clicked.connect(self.move_to_target)
        self.laserAutofocusController.signal_displacement_um.connect(
            self.label_displacement.setNum
        )

    def init_controller(self):
        self.laserAutofocusController.initialize_auto()
        if self.laserAutofocusController.is_initialized:
            self.btn_set_reference.setEnabled(True)
            self.btn_measure_displacement.setEnabled(True)
            self.btn_move_to_target.setEnabled(True)

    def move_to_target(self, target_um):
        self.laserAutofocusController.move_to_target(self.entry_target.value())


class WellSelectionWidget(QTableWidget):

    signal_wellSelected = Signal(int, int, float)
    signal_wellSelectedPos = Signal(float, float)

    def __init__(self, format_, *args):

        if format_ == 6:
            self.rows = 2
            self.columns = 3
            self.spacing_mm = 39.2
        elif format_ == 12:
            self.rows = 3
            self.columns = 4
            self.spacing_mm = 26
        elif format_ == 24:
            self.rows = 4
            self.columns = 6
            self.spacing_mm = 18
        elif format_ == 96:
            self.rows = 8
            self.columns = 12
            self.spacing_mm = 9
        elif format_ == 384:
            self.rows = 16
            self.columns = 24
            self.spacing_mm = 4.5
        elif format_ == 1536:
            self.rows = 32
            self.columns = 48
            self.spacing_mm = 2.25

        self.format = format_

        QTableWidget.__init__(self, self.rows, self.columns, *args)
        self.setData()
        self.resizeColumnsToContents()
        self.resizeRowsToContents()
        self.setEditTriggers(QTableWidget.NoEditTriggers)
        self.cellDoubleClicked.connect(self.onDoubleClick)
        self.cellClicked.connect(self.onSingleClick)

        # size
        self.verticalHeader().setSectionResizeMode(QHeaderView.Fixed)
        self.verticalHeader().setDefaultSectionSize(int(5 * self.spacing_mm))
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Fixed)
        self.horizontalHeader().setMinimumSectionSize(int(5 * self.spacing_mm))

        self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.resizeColumnsToContents()
        self.setFixedSize(
            self.horizontalHeader().length() + self.verticalHeader().width(),
            self.verticalHeader().length() + self.horizontalHeader().height(),
        )

    def setData(self):
        """
        # cells
        for i in range(16):
            for j in range(24):
                newitem = QTableWidgetItem( chr(ord('A')+i) + str(j) )
                self.setItem(i, j, newitem)
        """
        # row header
        row_headers = []
        for i in range(16):
            row_headers.append(chr(ord("A") + i))
        self.setVerticalHeaderLabels(row_headers)

        # make the outer cells not selectable if using 96 and 384 well plates
        if self.format == 384:
            if CONFIG.NUMBER_OF_SKIP == 1:
                for i in range(self.rows):
                    item = QTableWidgetItem()
                    item.setFlags(item.flags() & ~Qt.ItemIsSelectable)
                    self.setItem(i, 0, item)
                    item = QTableWidgetItem()
                    item.setFlags(item.flags() & ~Qt.ItemIsSelectable)
                    self.setItem(i, self.columns - 1, item)
                for j in range(self.columns):
                    item = QTableWidgetItem()
                    item.setFlags(item.flags() & ~Qt.ItemIsSelectable)
                    self.setItem(0, j, item)
                    item = QTableWidgetItem()
                    item.setFlags(item.flags() & ~Qt.ItemIsSelectable)
                    self.setItem(self.rows - 1, j, item)
        elif self.format == 96:
            if CONFIG.NUMBER_OF_SKIP == 1:
                for i in range(self.rows):
                    item = QTableWidgetItem()
                    item.setFlags(item.flags() & ~Qt.ItemIsSelectable)
                    self.setItem(i, 0, item)
                    item = QTableWidgetItem()
                    item.setFlags(item.flags() & ~Qt.ItemIsSelectable)
                    self.setItem(i, self.columns - 1, item)
                for j in range(self.columns):
                    item = QTableWidgetItem()
                    item.setFlags(item.flags() & ~Qt.ItemIsSelectable)
                    self.setItem(0, j, item)
                    item = QTableWidgetItem()
                    item.setFlags(item.flags() & ~Qt.ItemIsSelectable)
                    self.setItem(self.rows - 1, j, item)

    def onDoubleClick(self, row, col):
        if (
            row >= 0 + CONFIG.NUMBER_OF_SKIP
            and row <= self.rows - 1 - CONFIG.NUMBER_OF_SKIP
        ) and (
            col >= 0 + CONFIG.NUMBER_OF_SKIP
            and col <= self.columns - 1 - CONFIG.NUMBER_OF_SKIP
        ):
            x_mm = (
                CONFIG.X_MM_384_WELLPLATE_UPPERLEFT
                + CONFIG.WELL_SIZE_MM_384_WELLPLATE / 2
                - (
                    CONFIG.A1_X_MM_384_WELLPLATE
                    + CONFIG.WELL_SPACING_MM_384_WELLPLATE * CONFIG.NUMBER_OF_SKIP_384
                )
                + col * CONFIG.WELL_SPACING_MM
                + CONFIG.A1_X_MM
                + CONFIG.WELLPLATE_OFFSET_X_MM
            )
            y_mm = (
                CONFIG.Y_MM_384_WELLPLATE_UPPERLEFT
                + CONFIG.WELL_SIZE_MM_384_WELLPLATE / 2
                - (
                    CONFIG.A1_Y_MM_384_WELLPLATE
                    + CONFIG.WELL_SPACING_MM_384_WELLPLATE * CONFIG.NUMBER_OF_SKIP_384
                )
                + row * CONFIG.WELL_SPACING_MM
                + CONFIG.A1_Y_MM
                + CONFIG.WELLPLATE_OFFSET_Y_mm
            )
            self.signal_wellSelectedPos.emit(x_mm, y_mm)
        # print('(' + str(row) + ',' + str(col) + ') doubleclicked')

    def onSingleClick(self, row, col):
        # self.get_selected_cells()
        pass

    def get_selected_cells(self):
        list_of_selected_cells = []
        for index in self.selectedIndexes():
            list_of_selected_cells.append((index.row(), index.column()))
        return list_of_selected_cells
