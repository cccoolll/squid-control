# set QT_API environment variable
import os
import glob
import argparse

os.environ["QT_API"] = "pyqt5"
import qtpy

import sys

# qt libraries
from qtpy.QtCore import *
from qtpy.QtWidgets import *
from qtpy.QtGui import *

# Add the parent directory of squid_control to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, parent_dir)

# app specific libraries
import squid_control.control.gui_hcs as gui

from configparser import ConfigParser
from squid_control.control.widgets import (
    ConfigEditorBackwardsCompatible,
    ConfigEditorForAcquisitions,
)


import glob

parser = argparse.ArgumentParser()
parser.add_argument(
    "--simulation", help="Run the GUI with simulated hardware.", action="store_true"
)
args = parser.parse_args()


def show_config(cfp, configpath, main_gui):
    config_widget = ConfigEditorBackwardsCompatible(cfp, configpath, main_gui)
    config_widget.exec_()


def show_acq_config(cfm):
    acq_config_widget = ConfigEditorForAcquisitions(cfm)
    acq_config_widget.exec_()


if __name__ == "__main__":
    cf_editor_parser = ConfigParser()
    config_files = glob.glob("." + "/" + "configuration*.ini")

    app = QApplication([])
    app.setStyle("Fusion")
    if args.simulation:
        win = gui.OctopiGUI(is_simulation=True)
    else:
        win = gui.OctopiGUI()

    acq_config_action = QAction("Acquisition Settings", win)
    acq_config_action.triggered.connect(
        lambda: show_acq_config(win.configurationManager)
    )

    file_menu = QMenu("File", win)
    file_menu.addAction(acq_config_action)


    config_action = QAction("Microscope Settings", win)
    config_action.triggered.connect(
        lambda: show_config(cf_editor_parser, config_files[0], win)
    )
    file_menu.addAction(config_action)

    try:
        csw = win.cswWindow
        if csw is not None:
            csw_action = QAction("Camera Settings", win)
            csw_action.triggered.connect(csw.show)
            file_menu.addAction(csw_action)
    except AttributeError:
        pass

    try:
        csw_fc = win.cswfcWindow
        if csw_fc is not None:
            csw_fc_action = QAction("Camera Settings (Focus Camera)", win)
            csw_fc_action.triggered.connect(csw_fc.show)
            file_menu.addAction(csw_fc_action)
    except AttributeError:
        pass

    menu_bar = win.menuBar()
    menu_bar.addMenu(file_menu)
    win.show()
    sys.exit(app.exec_())
