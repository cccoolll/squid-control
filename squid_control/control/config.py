from configparser import ConfigParser
import os
import glob
import numpy as np
from pathlib import Path
from pydantic import BaseModel

from enum import Enum
from typing import Optional, Literal, List
from squid_control.control.camera import TriggerModeSetting

import json


def conf_attribute_reader(string_value):
    """
    :brief: standardized way for reading config entries
    that are strings, in priority order
    None -> bool -> dict/list (via json) -> int -> float -> string
    REMEMBER TO ENCLOSE PROPERTY NAMES IN LISTS/DICTS IN
    DOUBLE QUOTES
    """
    actualvalue = str(string_value).strip()
    try:
        if str(actualvalue) == "None":
            return None
    except:
        pass
    try:
        if str(actualvalue) == "True" or str(actualvalue) == "true":
            return True
        if str(actualvalue) == "False" or str(actualvalue) == "false":
            return False
    except:
        pass
    try:
        actualvalue = json.loads(actualvalue)
    except:
        try:
            actualvalue = int(str(actualvalue))
        except:
            try:
                actualvalue = float(actualvalue)
            except:
                actualvalue = str(actualvalue)
    return actualvalue


def populate_class_from_dict(myclass, options):
    """
    :brief: helper function to establish a compatibility
        layer between new way of storing config and current
        way of accessing it. assumes all class attributes are
        all-uppercase, and pattern-matches attributes in
        priority order dict/list (json) -> -> int -> float-> string
    REMEMBER TO ENCLOSE PROPERTY NAMES IN LISTS IN DOUBLE QUOTES
    """
    for key, value in options:
        if key.startswith("_") and key.endswith("options"):
            continue
        actualkey = key.upper()
        actualvalue = conf_attribute_reader(value)
        setattr(myclass, actualkey, actualvalue)


class AcquisitionSetting(BaseModel):
    CROP_WIDTH: int = 3000
    CROP_HEIGHT: int = 3000
    NUMBER_OF_FOVS_PER_AF: int = 3
    IMAGE_FORMAT: str = "bmp"
    IMAGE_DISPLAY_SCALING_FACTOR: float = 0.3
    DX: float = 0.9
    DY: float = 0.9
    DZ: float = 1.5
    NX: int = 1
    NY: int = 1


class PosUpdate(BaseModel):
    INTERVAL_MS: float = 25


class MicrocontrollerDefSetting(BaseModel):
    MSG_LENGTH: int = 24
    CMD_LENGTH: int = 8
    N_BYTES_POS: int = 4


class Microcontroller2Def(Enum):
    MSG_LENGTH = 4
    CMD_LENGTH = 8
    N_BYTES_POS = 4


# class LIMIT_SWITCH_POLARITY(BaseModel):
#     ACTIVE_LOW: int = 0
#     ACTIVE_HIGH: int = 1
#     DISABLED: int = 2
#     X_HOME: int= 1
#     Y_HOME: int= 1
#     Z_HOME: int= 2


class ILLUMINATION_CODE(Enum):
    ILLUMINATION_SOURCE_LED_ARRAY_FULL = 0
    ILLUMINATION_SOURCE_LED_ARRAY_LEFT_HALF = 1
    ILLUMINATION_SOURCE_LED_ARRAY_RIGHT_HALF = 2
    ILLUMINATION_SOURCE_LED_ARRAY_LEFTB_RIGHTR = 3
    ILLUMINATION_SOURCE_LED_ARRAY_LOW_NA = 4
    ILLUMINATION_SOURCE_LED_ARRAY_LEFT_DOT = 5
    ILLUMINATION_SOURCE_LED_ARRAY_RIGHT_DOT = 6
    ILLUMINATION_SOURCE_LED_EXTERNAL_FET = 20
    ILLUMINATION_SOURCE_405NM = 11
    ILLUMINATION_SOURCE_488NM = 12
    ILLUMINATION_SOURCE_638NM = 13
    ILLUMINATION_SOURCE_561NM = 14
    ILLUMINATION_SOURCE_730NM = 15


class VolumetricImagingSetting(BaseModel):
    NUM_PLANES_PER_VOLUME: int = 20


class CmdExecutionStatus(BaseModel):
    COMPLETED_WITHOUT_ERRORS: int = 0
    IN_PROGRESS: int = 1
    CMD_CHECKSUM_ERROR: int = 2
    CMD_INVALID: int = 3
    CMD_EXECUTION_ERROR: int = 4
    ERROR_CODE_EMPTYING_THE_FLUDIIC_LINE_FAILED: int = 100


class CameraConfig(BaseModel):
    ROI_OFFSET_X_DEFAULT: int = 0
    ROI_OFFSET_Y_DEFAULT: int = 0
    ROI_WIDTH_DEFAULT: int = 3104
    ROI_HEIGHT_DEFAULT: int = 2084


class PlateReaderSetting(BaseModel):
    NUMBER_OF_ROWS: int = 8
    NUMBER_OF_COLUMNS: int = 12
    ROW_SPACING_MM: int = 9
    COLUMN_SPACING_MM: int = 9
    OFFSET_COLUMN_1_MM: int = 20
    OFFSET_ROW_A_MM: int = 20


class AFSetting(BaseModel):
    STOP_THRESHOLD: float = 0.85
    CROP_WIDTH: int = 800
    CROP_HEIGHT: int = 800
    MULTIPOINT_REFLECTION_AUTOFOCUS_ENABLE_BY_DEFAULT: bool = False
    MULTIPOINT_AUTOFOCUS_ENABLE_BY_DEFAULT: bool = False


class TrackingSetting(BaseModel):
    SEARCH_AREA_RATIO: int = 10
    CROPPED_IMG_RATIO: int = 10
    BBOX_SCALE_FACTOR: float = 1.2
    DEFAULT_TRACKER: str = "csrt"
    INIT_METHODS: list = ["roi"]
    DEFAULT_INIT_METHOD: str = "roi"


class SlidePositionSetting(BaseModel):
    LOADING_X_MM: int = 30
    LOADING_Y_MM: int = 55
    SCANNING_X_MM: int = 3
    SCANNING_Y_MM: int = 3


class OutputGainSetting(BaseModel):
    REFDIV: bool = False
    CHANNEL0_GAIN: bool = False
    CHANNEL1_GAIN: bool = False
    CHANNEL2_GAIN: bool = False
    CHANNEL3_GAIN: bool = False
    CHANNEL4_GAIN: bool = False
    CHANNEL5_GAIN: bool = False
    CHANNEL6_GAIN: bool = False
    CHANNEL7_GAIN: bool = True


class SoftwarePosLimitSetting(BaseModel):
    X_POSITIVE: float = 112.5
    X_NEGATIVE: float = 10
    Y_POSITIVE: float = 76
    Y_NEGATIVE: float = 6
    Z_POSITIVE: float = 6
    Z_NEGATIVE: float = 0.05


class FlipImageSetting(Enum):
    Horizontal = "Horizontal"
    Vertical = "Vertical"
    Both = "Both"


class BaseConfig(BaseModel):
    class Config:
        extra = "allow"  # Allow extra fields that are not defined in the model

    MicrocontrollerDef: MicrocontrollerDefSetting = MicrocontrollerDefSetting()
    VOLUMETRIC_IMAGING: VolumetricImagingSetting = VolumetricImagingSetting()
    CMD_EXECUTION_STATUS: CmdExecutionStatus = CmdExecutionStatus()
    CAMERA_CONFIG: CameraConfig = CameraConfig()
    PLATE_READER: PlateReaderSetting = PlateReaderSetting()
    AF: AFSetting = AFSetting()
    Tracking: TrackingSetting = TrackingSetting()
    SLIDE_POSITION: SlidePositionSetting = SlidePositionSetting()
    OUTPUT_GAINS: OutputGainSetting = OutputGainSetting()
    SOFTWARE_POS_LIMIT: SoftwarePosLimitSetting = SoftwarePosLimitSetting()
    Acquisition: AcquisitionSetting = AcquisitionSetting()
    USE_SEPARATE_MCU_FOR_DAC: bool = False

    BIT_POS_JOYSTICK_BUTTON: int = 0
    BIT_POS_SWITCH: int = 1

    ###########################################################
    #### machine specific configurations - to be overridden ###
    ###########################################################
    ROTATE_IMAGE_ANGLE: Optional[float] = None
    FLIP_IMAGE: Optional[FlipImageSetting] = None  #

    CAMERA_REVERSE_X: bool = False
    CAMERA_REVERSE_Y: bool = False

    DEFAULT_TRIGGER_MODE: TriggerModeSetting = TriggerModeSetting.SOFTWARE

    # note: XY are the in-plane axes, Z is the focus axis

    # change the following so that "backward" is "backward" - towards the single sided hall effect sensor
    STAGE_MOVEMENT_SIGN_X: int = 1
    STAGE_MOVEMENT_SIGN_Y: int = 1
    STAGE_MOVEMENT_SIGN_Z: int = -1
    STAGE_MOVEMENT_SIGN_THETA: int = 1

    STAGE_POS_SIGN_X: int = -1
    STAGE_POS_SIGN_Y: int = 1
    STAGE_POS_SIGN_Z: int = -1
    STAGE_POS_SIGN_THETA: int = 1

    TRACKING_MOVEMENT_SIGN_X: int = 1
    TRACKING_MOVEMENT_SIGN_Y: int = 1
    TRACKING_MOVEMENT_SIGN_Z: int = 1
    TRACKING_MOVEMENT_SIGN_THETA: int = 1

    USE_ENCODER_X: bool = False
    USE_ENCODER_Y: bool = False
    USE_ENCODER_Z: bool = False
    USE_ENCODER_THETA: bool = False

    ENCODER_POS_SIGN_X: float = 1
    ENCODER_POS_SIGN_Y: float = 1
    ENCODER_POS_SIGN_Z: float = 1
    ENCODER_POS_SIGN_THETA: float = 1

    ENCODER_STEP_SIZE_X_MM: float = 100e-6
    ENCODER_STEP_SIZE_Y_MM: float = 100e-6
    ENCODER_STEP_SIZE_Z_MM: float = 100e-6
    ENCODER_STEP_SIZE_THETA: float = 1

    FULLSTEPS_PER_REV_X: float = 200
    FULLSTEPS_PER_REV_Y: float = 200
    FULLSTEPS_PER_REV_Z: float = 200
    FULLSTEPS_PER_REV_THETA: float = 200

    # beginning of actuator specific configurations

    SCREW_PITCH_X_MM: float = 2.54
    SCREW_PITCH_Y_MM: float = 2.54
    SCREW_PITCH_Z_MM: float = 0.3

    MICROSTEPPING_DEFAULT_X: float = 256
    MICROSTEPPING_DEFAULT_Y: float = 256
    MICROSTEPPING_DEFAULT_Z: float = 256
    MICROSTEPPING_DEFAULT_THETA: float = 256

    X_MOTOR_RMS_CURRENT_MA: float = 1000
    Y_MOTOR_RMS_CURRENT_MA: float = 1000
    Z_MOTOR_RMS_CURRENT_MA: float = 500

    X_MOTOR_I_HOLD: float = 0.25
    Y_MOTOR_I_HOLD: float = 0.25
    Z_MOTOR_I_HOLD: float = 0.5

    MAX_VELOCITY_X_MM: float = 30
    MAX_VELOCITY_Y_MM: float = 30
    MAX_VELOCITY_Z_MM: float = 2

    MAX_ACCELERATION_X_MM: float = 500
    MAX_ACCELERATION_Y_MM: float = 500
    MAX_ACCELERATION_Z_MM: float = 100

    # config encoder arguments
    HAS_ENCODER_X: bool = False
    HAS_ENCODER_Y: bool = False
    HAS_ENCODER_Z: bool = False

    # enable PID control
    ENABLE_PID_X: bool = False
    ENABLE_PID_Y: bool = False
    ENABLE_PID_Z: bool = False

    # flip direction True or False
    ENCODER_FLIP_DIR_X: bool = False
    ENCODER_FLIP_DIR_Y: bool = False
    ENCODER_FLIP_DIR_Z: bool = False

    # distance for each count (um)
    ENCODER_RESOLUTION_UM_X: float = 0.05
    ENCODER_RESOLUTION_UM_Y: float = 0.05
    ENCODER_RESOLUTION_UM_Z: float = 0.1

    # end of actuator specific configurations

    SCAN_STABILIZATION_TIME_MS_X: float = 160
    SCAN_STABILIZATION_TIME_MS_Y: float = 160
    SCAN_STABILIZATION_TIME_MS_Z: float = 20
    HOMING_ENABLED_X: bool = False
    HOMING_ENABLED_Y: bool = False
    HOMING_ENABLED_Z: bool = False

    SLEEP_TIME_S: float = 0.005

    LED_MATRIX_R_FACTOR: float = 0
    LED_MATRIX_G_FACTOR: float = 0
    LED_MATRIX_B_FACTOR: float = 1

    DEFAULT_SAVING_PATH: str = "/home/tao/remote_harddisk/u2os-treatment/"

    DEFAULT_PIXEL_FORMAT: str = "MONO8"

    DEFAULT_DISPLAY_CROP: int = (
        100  # value ranges from 1 to 100 - image display crop size
    )

    CAMERA_PIXEL_SIZE_UM: dict = {
        "IMX290": 2.9,
        "IMX178": 2.4,
        "IMX226": 1.85,
        "IMX250": 3.45,
        "IMX252": 3.45,
        "IMX273": 3.45,
        "IMX264": 3.45,
        "IMX265": 3.45,
        "IMX571": 3.76,
        "PYTHON300": 4.8,
    }
    PIXEL_SIZE_ADJUSTMENT_FACTOR: float = 0.936
    OBJECTIVES: dict = {
        "2x": {"magnification": 2, "NA": 0.10, "tube_lens_f_mm": 180},
        "4x": {"magnification": 4, "NA": 0.13, "tube_lens_f_mm": 180},
        "10x": {"magnification": 10, "NA": 0.25, "tube_lens_f_mm": 180},
        "10x (Mitutoyo)": {"magnification": 10, "NA": 0.25, "tube_lens_f_mm": 200},
        "20x (Boli)": {"magnification": 20, "NA": 0.4, "tube_lens_f_mm": 180},
        "20x (Nikon)": {"magnification": 20, "NA": 0.45, "tube_lens_f_mm": 200},
        "20x": {"magnification": 20, "NA": 0.4, "tube_lens_f_mm": 180},
        "40x": {"magnification": 40, "NA": 0.6, "tube_lens_f_mm": 180},
    }
    TUBE_LENS_MM: float = 50
    CAMERA_SENSOR: str = "IMX226"
    DEFAULT_OBJECTIVE: str = "20x"
    TRACKERS: List[str] = [
        "csrt",
        "kcf",
        "mil",
        "tld",
        "medianflow",
        "mosse",
        "daSiamRPN",
    ]
    DEFAULT_TRACKER: str = "csrt"

    ENABLE_TRACKING: bool = False
    TRACKING_SHOW_MICROSCOPE_CONFIGURATIONS: bool = (
        False  # set to true when doing multimodal acquisition
    )

    SLIDE_POTISION_SWITCHING_TIMEOUT_LIMIT_S: float = 10
    SLIDE_POTISION_SWITCHING_HOME_EVERYTIME: bool = False

    SHOW_AUTOLEVEL_BTN: bool = False
    AUTOLEVEL_DEFAULT_SETTING: bool = False

    MULTIPOINT_AUTOFOCUS_CHANNEL: str = "BF LED matrix full"
    # MULTIPOINT_AUTOFOCUS_CHANNEL:str = 'BF LED matrix left half'
    MULTIPOINT_BF_SAVING_OPTION: str = "Raw"
    # MULTIPOINT_BF_SAVING_OPTION:str = 'RGB2GRAY'
    # MULTIPOINT_BF_SAVING_OPTION:str = 'Green Channel Only'

    DEFAULT_MULTIPOINT_NX: int = 1
    DEFAULT_MULTIPOINT_NY: int = 1

    ENABLE_FLEXIBLE_MULTIPOINT: bool = False

    CAMERA_SN: dict = {
        "ch 1": "SN1",
        "ch 2": "SN2",
    }  # for multiple cameras, to be overwritten in the configuration file

    ENABLE_STROBE_OUTPUT: bool = False

    Z_STACKING_CONFIG: str = "FROM CENTER"  # 'FROM BOTTOM', 'FROM TOP'

    # plate format
    WELLPLATE_FORMAT: int = 96

    # for 384 well plate
    X_MM_384_WELLPLATE_UPPERLEFT: int = 0
    Y_MM_384_WELLPLATE_UPPERLEFT: int = 0
    DEFAULT_Z_POS_MM: int = 3.970
    X_ORIGIN_384_WELLPLATE_PIXEL: int = 177  # upper left of B2
    Y_ORIGIN_384_WELLPLATE_PIXEL: int = 141  # upper left of B2
    NUMBER_OF_SKIP_384: int = 1
    A1_X_MM_384_WELLPLATE: float = 12.05
    A1_Y_MM_384_WELLPLATE: float = 9.05
    WELL_SPACING_MM_384_WELLPLATE: float = 4.5
    WELL_SIZE_MM_384_WELLPLATE: float = 3.3
    # B1 upper left corner in piexel: x = 124, y = 141
    # B1 upper left corner in mm: x = 12.13 mm - 3.3 mm/2, y = 8.99 mm + 4.5 mm - 3.3 mm/2
    # B2 upper left corner in pixel: x = 177, y = 141

    WELLPLATE_OFFSET_X_MM: float = 0  # x offset adjustment for using different plates
    WELLPLATE_OFFSET_Y_MM: float =0 # y offset adjustment for using different plates

    # for USB spectrometer
    N_SPECTRUM_PER_POINT: int = 5

    # focus measure operator
    FOCUS_MEASURE_OPERATOR: str = (
        "LAPE"  # 'GLVA' # LAPE has worked well for bright field images; GLVA works well for darkfield/fluorescence
    )

    # controller version
    CONTROLLER_VERSION: str = "Teensy"

    # How to read Spinnaker nodemaps, options are INDIVIDUAL or VALUE
    CHOSEN_READ: str = "INDIVIDUAL"

    class MCU_PINS:
        PWM1 = 5
        PWM2 = 4
        PWM3 = 22
        PWM4 = 3
        PWM5 = 23
        PWM6 = 2
        PWM7 = 1
        PWM9 = 6
        PWM10 = 7
        PWM11 = 8
        PWM12 = 9
        PWM13 = 10
        PWM14 = 15
        PWM15 = 24
        PWM16 = 25
        AF_LASER = 15

    # laser autofocus
    SUPPORT_LASER_AUTOFOCUS: bool = True
    MAIN_CAMERA_MODEL: str = "MER2-1220-32U3M"
    FOCUS_CAMERA_MODEL: str = "MER2-630-60U3M"
    FOCUS_CAMERA_EXPOSURE_TIME_MS: int = 0.2
    FOCUS_CAMERA_ANALOG_GAIN: int = 0
    LASER_AF_AVERAGING_N: int = 5
    LASER_AF_DISPLAY_SPOT_IMAGE: bool = False
    LASER_AF_CROP_WIDTH: int = 1536
    LASER_AF_CROP_HEIGHT: int = 256
    HAS_TWO_INTERFACES: bool = True
    USE_GLASS_TOP: bool = True
    SHOW_LEGACY_DISPLACEMENT_MEASUREMENT_WINDOWS: bool = False
    RUN_CUSTOM_MULTIPOINT: bool = False
    CUSTOM_MULTIPOINT_FUNCTION: str = None
    RETRACT_OBJECTIVE_BEFORE_MOVING_TO_LOADING_POSITION: bool = True
    OBJECTIVE_RETRACTED_POS_MM: float = 0.1
    CLASSIFICATION_MODEL_PATH: str = "/home/cephla/Documents/tmp/model_perf_r34_b32.pt"
    SEGMENTATION_MODEL_PATH: str = (
        "/home/cephla/Documents/tmp/model_segmentation_1073_9.pth"
    )
    CLASSIFICATION_TEST_MODE: bool = False
    USE_TRT_SEGMENTATION: bool = False
    SEGMENTATION_CROP: int = 1500
    DISP_TH_DURING_MULTIPOINT: float = 0.95
    SORT_DURING_MULTIPOINT: bool = False
    DO_FLUORESCENCE_RTP: bool = False
    ENABLE_SPINNING_DISK_CONFOCAL: bool = False
    INVERTED_OBJECTIVE: bool = False
    ILLUMINATION_INTENSITY_FACTOR: float = 0.6
    CAMERA_TYPE: Literal["Default"] = "Default"
    FOCUS_CAMERA_TYPE: Literal["Default"] = "Default"
    LASER_AF_CHARACTERIZATION_MODE: bool = False

    # limit switch
    X_HOME_SWITCH_POLARITY: int = 1
    Y_HOME_SWITCH_POLARITY: int = 1
    Z_HOME_SWITCH_POLARITY: int = 2

    # for 96 well plate
    NUMBER_OF_SKIP: int = 0
    WELL_SIZE_MM: float = 6.21
    WELL_SPACING_MM: int = 9
    A1_X_MM: float = 14.3
    A1_Y_MM: float = 11.36

    SHOW_DAC_CONTROL: bool = False
    CACHE_CONFIG_FILE_PATH: str = None
    CHANNEL_CONFIGURATIONS_PATH: str = "./focus_camera_configurations.xml"
    LAST_COORDS_PATH: str = ""

    # for check if the stage is moved
    STAGE_MOVED_THRESHOLD: float = 0.05

    # Additional field to store options
    OPTIONS: dict = {}


    def write_config_to_txt(self, output_path):
        with open(output_path, "w") as file:
            for attribute, value in self.__dict__.items():
                if isinstance(value, BaseModel):
                    file.write(f"[{attribute}]\n")
                    for sub_attribute, sub_value in value.dict().items():
                        file.write(f"{sub_attribute.lower()} = {sub_value}\n")
                else:
                    file.write(f"{attribute.lower()} = {value}\n")
                file.write("\n")

    def read_config(self, config_path):
        cached_config_file_path = None

        try:
            with open(self.CACHE_CONFIG_FILE_PATH, "r") as file:
                for line in file:
                    cached_config_file_path = line.strip()
                    break
        except FileNotFoundError:
            cached_config_file_path = None

        config_files = [config_path]
        if config_files:
            if len(config_files) > 1:
                if cached_config_file_path in config_files:
                    print(
                        "defaulting to last cached config file at "
                        + cached_config_file_path
                    )
                    config_files = [cached_config_file_path]
                else:
                    print(
                        "multiple machine configuration files found, the program will exit"
                    )
                    exit()
            print("load machine-specific configuration")
            cfp = ConfigParser()
            cfp.read(config_files[0])
            for section in cfp.sections():
                for key, value in cfp.items(section):
                    actualvalue = conf_attribute_reader(value)
                    if key.startswith("_") and key.endswith("_options"):
                        self.OPTIONS[key] = actualvalue
                    else:
                        section_upper = section.upper()
                        if hasattr(self, section_upper):
                            class_instance = getattr(self, section_upper)
                            if isinstance(class_instance, BaseModel):
                                if key.upper() in class_instance.__fields__:
                                    setattr(class_instance, key.upper(), actualvalue)
                                else:
                                    setattr(class_instance, key.upper(), actualvalue)
                            else:
                                setattr(self, section_upper, actualvalue)
                        else:
                            setattr(self, key.upper(), actualvalue)
            try:
                with open(self.CACHE_CONFIG_FILE_PATH, "w") as file:
                    file.write(str(config_files[0]))
                cached_config_file_path = config_files[0]
            except Exception as e:
                print(f"Error caching config file path: {e}")
        else:
            print(
                "configuration*.ini file not found, defaulting to legacy configuration"
            )
            config_files = glob.glob("." + "/" + "configuration*.txt")
            if config_files:
                if len(config_files) > 1:
                    print(
                        "multiple machine configuration files found, the program will exit"
                    )
                    exit()
                print("load machine-specific configuration")
                exec(open(config_files[0]).read())
            else:
                print(
                    "machine-specific configuration not present, the program will exit"
                )
                exit()
        return cached_config_file_path


CONFIG = BaseConfig()


def load_config(config_path, multipoint_function):
    global CONFIG

    config_dir = Path(os.path.abspath(__file__)).parent

    current_dir = Path(__file__).parent
    if not str(config_path).endswith(".ini"):
        config_path = current_dir / (
            "../configurations/configuration_" + str(config_path) + ".ini"
        )
    
    # Convert Path object to string if needed
    config_path = str(config_path)

    CONFIG.CACHE_CONFIG_FILE_PATH = str(config_dir / "cache_config_file_path.txt")

    CONFIG.CHANNEL_CONFIGURATIONS_PATH = str(config_dir / "u2os_fucci_illumination_configurations.xml")
    CONFIG.LAST_COORDS_PATH = str(config_dir / "last_coords.txt")

    # Check if configuration file exists
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file {config_path} not found.")
    
    print(f"Reading configuration from: {config_path}")
    
    cf_editor_parser = ConfigParser()
    cached_config_file_path = CONFIG.read_config(config_path)
    CONFIG.STAGE_POS_SIGN_X = CONFIG.STAGE_MOVEMENT_SIGN_X
    CONFIG.STAGE_POS_SIGN_Y = CONFIG.STAGE_MOVEMENT_SIGN_Y
    CONFIG.STAGE_POS_SIGN_Z = CONFIG.STAGE_MOVEMENT_SIGN_Z
    CONFIG.STAGE_POS_SIGN_THETA = CONFIG.STAGE_MOVEMENT_SIGN_THETA
    if multipoint_function:
        CONFIG.RUN_CUSTOM_MULTIPOINT = True
        CONFIG.CUSTOM_MULTIPOINT_FUNCTION = multipoint_function

    # if not (CONFIG.DEFAULT_SAVING_PATH.startswith(str(Path.home()))):
    #     CONFIG.DEFAULT_SAVING_PATH = (
    #         str(Path.home()) + "/" + CONFIG.DEFAULT_SAVING_PATH.strip("/")
    #     )

    if CONFIG.ENABLE_TRACKING:
        CONFIG.DEFAULT_DISPLAY_CROP = CONFIG.Tracking.DEFAULT_DISPLAY_CROP

    if CONFIG.WELLPLATE_FORMAT == 384:
        CONFIG.WELL_SIZE_MM = 3.3
        CONFIG.WELL_SPACING_MM = 4.5
        CONFIG.NUMBER_OF_SKIP = 1
        CONFIG.A1_X_MM = 12.05
        CONFIG.A1_Y_MM = 9.05
    elif CONFIG.WELLPLATE_FORMAT == 96:
        CONFIG.NUMBER_OF_SKIP = 0
        CONFIG.WELL_SIZE_MM = 6.21
        CONFIG.WELL_SPACING_MM = 9
        CONFIG.A1_X_MM = 14.3
        CONFIG.A1_Y_MM = 11.36
    elif CONFIG.WELLPLATE_FORMAT == 24:
        CONFIG.NUMBER_OF_SKIP = 0
        CONFIG.WELL_SIZE_MM = 15.54
        CONFIG.WELL_SPACING_MM = 19.3
        CONFIG.A1_X_MM = 17.05
        CONFIG.A1_Y_MM = 13.67
    elif CONFIG.WELLPLATE_FORMAT == 12:
        CONFIG.NUMBER_OF_SKIP = 0
        CONFIG.WELL_SIZE_MM = 22.05
        CONFIG.WELL_SPACING_MM = 26
        CONFIG.A1_X_MM = 24.75
        CONFIG.A1_Y_MM = 16.86
    elif CONFIG.WELLPLATE_FORMAT == 6:
        CONFIG.NUMBER_OF_SKIP = 0
        CONFIG.WELL_SIZE_MM = 34.94
        CONFIG.WELL_SPACING_MM = 39.2
        CONFIG.A1_X_MM = 24.55
        CONFIG.A1_Y_MM = 23.01

    # Apply additional configuration settings like wellplate_offset
    if hasattr(CONFIG, 'WELLPLATE_OFFSET_X_MM'):
        print(f"Applying wellplate offset X: {CONFIG.WELLPLATE_OFFSET_X_MM}")
        if CONFIG.WELLPLATE_FORMAT in [384, 96, 24, 12, 6]:
            CONFIG.A1_X_MM += CONFIG.WELLPLATE_OFFSET_X_MM
    
    if hasattr(CONFIG, 'WELLPLATE_OFFSET_Y_MM'):
        print(f"Applying wellplate offset Y: {CONFIG.WELLPLATE_OFFSET_Y_MM}")
        if CONFIG.WELLPLATE_FORMAT in [384, 96, 24, 12, 6]:
            CONFIG.A1_Y_MM += CONFIG.WELLPLATE_OFFSET_Y_MM

    # Write configuration to txt file after reading
    CONFIG.write_config_to_txt('config_parameters.txt')
    print("Configuration loaded and written to config_parameters.txt")

    try:
        if cached_config_file_path and os.path.exists(cached_config_file_path):
            cf_editor_parser.read(cached_config_file_path)
    except Exception as e:
        print(f"Error reading cached config: {e}")
        return False
    
    return True


# For flexible plate format:
class WELLPLATE_FORMAT_384:
    WELL_SIZE_MM = 3.3
    WELL_SPACING_MM = 4.5
    NUMBER_OF_SKIP = 1
    A1_X_MM = 12.05
    A1_Y_MM = 9.05


class WELLPLATE_FORMAT_96:
    NUMBER_OF_SKIP = 0
    WELL_SIZE_MM = 6.21
    WELL_SPACING_MM = 9
    A1_X_MM = 14.3
    A1_Y_MM = 11.36


class WELLPLATE_FORMAT_24:
    NUMBER_OF_SKIP = 0
    WELL_SIZE_MM = 15.54
    WELL_SPACING_MM = 19.3
    A1_X_MM = 17.05
    A1_Y_MM = 13.67


class WELLPLATE_FORMAT_12:
    NUMBER_OF_SKIP = 0
    WELL_SIZE_MM = 22.05
    WELL_SPACING_MM = 26
    A1_X_MM = 24.75
    A1_Y_MM = 16.86


class WELLPLATE_FORMAT_6:
    NUMBER_OF_SKIP = 0
    WELL_SIZE_MM = 34.94
    WELL_SPACING_MM = 39.2
    A1_X_MM = 24.55
    A1_Y_MM = 23.01


# For simulated camera
class SIMULATED_CAMERA:
    ORIN_X = 20
    ORIN_Y = 20
    ORIN_Z = 4
    MAGNIFICATION_FACTOR = 80


def get_microscope_configuration_data(config_section="all", include_defaults=True, is_simulation=False, is_local=False, squid_controller=None):
    """
    Get microscope configuration information in JSON format.
    
    Args:
        config_section (str): Configuration section to retrieve ('all', 'camera', 'stage', 'illumination', 'acquisition', 'limits', 'hardware', 'wellplate', 'optics', 'autofocus')
        include_defaults (bool): Whether to include default values from config.py
        is_simulation (bool): Whether the microscope is in simulation mode
        is_local (bool): Whether the microscope is in local mode
        squid_controller (object, optional): Instance of SquidController to get dynamic data. Defaults to None.
    
    Returns:
        dict: Configuration data as a dictionary
    """
    import time
    
    # Get current configuration
    config_data = {}
    
    if config_section.lower() == "all" or config_section.lower() == "camera":
        config_data["camera"] = {
            "camera_type": getattr(CONFIG, 'CAMERA_TYPE', 'Default'),
            "camera_sn": getattr(CONFIG, 'CAMERA_SN', {}),
            "camera_sensor": getattr(CONFIG, 'CAMERA_SENSOR', 'IMX226'),
            "camera_pixel_size_um": getattr(CONFIG, 'CAMERA_PIXEL_SIZE_UM', {}),
            "default_pixel_format": getattr(CONFIG, 'DEFAULT_PIXEL_FORMAT', 'MONO8'),
            "camera_reverse_x": getattr(CONFIG, 'CAMERA_REVERSE_X', False),
            "camera_reverse_y": getattr(CONFIG, 'CAMERA_REVERSE_Y', False),
            "rotate_image_angle": getattr(CONFIG, 'ROTATE_IMAGE_ANGLE', None),
            "flip_image": getattr(CONFIG, 'FLIP_IMAGE', None),
            "camera_config": {
                "roi_offset_x_default": getattr(CONFIG.CAMERA_CONFIG, 'ROI_OFFSET_X_DEFAULT', 0),
                "roi_offset_y_default": getattr(CONFIG.CAMERA_CONFIG, 'ROI_OFFSET_Y_DEFAULT', 0),
                "roi_width_default": getattr(CONFIG.CAMERA_CONFIG, 'ROI_WIDTH_DEFAULT', 3104),
                "roi_height_default": getattr(CONFIG.CAMERA_CONFIG, 'ROI_HEIGHT_DEFAULT', 2084),
            }
        }
    
    if config_section.lower() == "all" or config_section.lower() == "stage":
        config_data["stage"] = {
            "movement_signs": {
                "x": getattr(CONFIG, 'STAGE_MOVEMENT_SIGN_X', 1),
                "y": getattr(CONFIG, 'STAGE_MOVEMENT_SIGN_Y', 1),
                "z": getattr(CONFIG, 'STAGE_MOVEMENT_SIGN_Z', -1),
                "theta": getattr(CONFIG, 'STAGE_MOVEMENT_SIGN_THETA', 1),
            },
            "position_signs": {
                "x": getattr(CONFIG, 'STAGE_POS_SIGN_X', -1),
                "y": getattr(CONFIG, 'STAGE_POS_SIGN_Y', 1),
                "z": getattr(CONFIG, 'STAGE_POS_SIGN_Z', -1),
                "theta": getattr(CONFIG, 'STAGE_POS_SIGN_THETA', 1),
            },
            "screw_pitch_mm": {
                "x": getattr(CONFIG, 'SCREW_PITCH_X_MM', 2.54),
                "y": getattr(CONFIG, 'SCREW_PITCH_Y_MM', 2.54),
                "z": getattr(CONFIG, 'SCREW_PITCH_Z_MM', 0.3),
            },
            "microstepping": {
                "x": getattr(CONFIG, 'MICROSTEPPING_DEFAULT_X', 256),
                "y": getattr(CONFIG, 'MICROSTEPPING_DEFAULT_Y', 256),
                "z": getattr(CONFIG, 'MICROSTEPPING_DEFAULT_Z', 256),
                "theta": getattr(CONFIG, 'MICROSTEPPING_DEFAULT_THETA', 256),
            },
            "max_velocity_mm": {
                "x": getattr(CONFIG, 'MAX_VELOCITY_X_MM', 30),
                "y": getattr(CONFIG, 'MAX_VELOCITY_Y_MM', 30),
                "z": getattr(CONFIG, 'MAX_VELOCITY_Z_MM', 2),
            },
            "max_acceleration_mm": {
                "x": getattr(CONFIG, 'MAX_ACCELERATION_X_MM', 500),
                "y": getattr(CONFIG, 'MAX_ACCELERATION_Y_MM', 500),
                "z": getattr(CONFIG, 'MAX_ACCELERATION_Z_MM', 100),
            },
            "homing_enabled": {
                "x": getattr(CONFIG, 'HOMING_ENABLED_X', False),
                "y": getattr(CONFIG, 'HOMING_ENABLED_Y', False),
                "z": getattr(CONFIG, 'HOMING_ENABLED_Z', False),
            }
        }
    
    if config_section.lower() == "all" or config_section.lower() == "illumination":
        config_data["illumination"] = {
            "led_matrix_factors": {
                "r": getattr(CONFIG, 'LED_MATRIX_R_FACTOR', 0),
                "g": getattr(CONFIG, 'LED_MATRIX_G_FACTOR', 0),
                "b": getattr(CONFIG, 'LED_MATRIX_B_FACTOR', 1),
            },
            "illumination_intensity_factor": getattr(CONFIG, 'ILLUMINATION_INTENSITY_FACTOR', 0.6),
            "enable_strobe_output": getattr(CONFIG, 'ENABLE_STROBE_OUTPUT', False),
            "mcu_pins": {
                "pwm1": getattr(CONFIG.MCU_PINS, 'PWM1', 5),
                "pwm2": getattr(CONFIG.MCU_PINS, 'PWM2', 4),
                "pwm3": getattr(CONFIG.MCU_PINS, 'PWM3', 22),
                "pwm4": getattr(CONFIG.MCU_PINS, 'PWM4', 3),
                "pwm5": getattr(CONFIG.MCU_PINS, 'PWM5', 23),
                "af_laser": getattr(CONFIG.MCU_PINS, 'AF_LASER', 15),
            }
        }
    
    if config_section.lower() == "all" or config_section.lower() == "acquisition":
        config_data["acquisition"] = {
            "crop_width": getattr(CONFIG.Acquisition, 'CROP_WIDTH', 3000),
            "crop_height": getattr(CONFIG.Acquisition, 'CROP_HEIGHT', 3000),
            "image_format": getattr(CONFIG.Acquisition, 'IMAGE_FORMAT', 'bmp'),
            "image_display_scaling_factor": getattr(CONFIG.Acquisition, 'IMAGE_DISPLAY_SCALING_FACTOR', 0.3),
            "default_step_sizes": {
                "dx": getattr(CONFIG.Acquisition, 'DX', 0.9),
                "dy": getattr(CONFIG.Acquisition, 'DY', 0.9),
                "dz": getattr(CONFIG.Acquisition, 'DZ', 1.5),
            },
            "default_grid_sizes": {
                "nx": getattr(CONFIG.Acquisition, 'NX', 1),
                "ny": getattr(CONFIG.Acquisition, 'NY', 1),
            },
            "default_trigger_mode": str(getattr(CONFIG, 'DEFAULT_TRIGGER_MODE', 'SOFTWARE')),
            "default_saving_path": getattr(CONFIG, 'DEFAULT_SAVING_PATH', '/home/tao/remote_harddisk/u2os-treatment/'),
        }
    
    if config_section.lower() == "all" or config_section.lower() == "limits":
        config_data["limits"] = {
            "software_pos_limit": {
                "x_positive": getattr(CONFIG.SOFTWARE_POS_LIMIT, 'X_POSITIVE', 112.5),
                "x_negative": getattr(CONFIG.SOFTWARE_POS_LIMIT, 'X_NEGATIVE', 10),
                "y_positive": getattr(CONFIG.SOFTWARE_POS_LIMIT, 'Y_POSITIVE', 76),
                "y_negative": getattr(CONFIG.SOFTWARE_POS_LIMIT, 'Y_NEGATIVE', 6),
                "z_positive": getattr(CONFIG.SOFTWARE_POS_LIMIT, 'Z_POSITIVE', 6),
                "z_negative": getattr(CONFIG.SOFTWARE_POS_LIMIT, 'Z_NEGATIVE', 0.05),
            },
            "scan_stabilization_time_ms": {
                "x": getattr(CONFIG, 'SCAN_STABILIZATION_TIME_MS_X', 160),
                "y": getattr(CONFIG, 'SCAN_STABILIZATION_TIME_MS_Y', 160),
                "z": getattr(CONFIG, 'SCAN_STABILIZATION_TIME_MS_Z', 20),
            }
        }
    
    if config_section.lower() == "all" or config_section.lower() == "hardware":
        config_data["hardware"] = {
            "controller_version": getattr(CONFIG, 'CONTROLLER_VERSION', 'Teensy'),
            "microcontroller_def": {
                "msg_length": getattr(CONFIG.MicrocontrollerDef, 'MSG_LENGTH', 24),
                "cmd_length": getattr(CONFIG.MicrocontrollerDef, 'CMD_LENGTH', 8),
                "n_bytes_pos": getattr(CONFIG.MicrocontrollerDef, 'N_BYTES_POS', 4),
            },
            "support_laser_autofocus": getattr(CONFIG, 'SUPPORT_LASER_AUTOFOCUS', True),
            "enable_spinning_disk_confocal": getattr(CONFIG, 'ENABLE_SPINNING_DISK_CONFOCAL', False),
            "inverted_objective": getattr(CONFIG, 'INVERTED_OBJECTIVE', False),
            "retract_objective_before_moving": getattr(CONFIG, 'RETRACT_OBJECTIVE_BEFORE_MOVING_TO_LOADING_POSITION', True),
            "objective_retracted_pos_mm": getattr(CONFIG, 'OBJECTIVE_RETRACTED_POS_MM', 0.1),
            "use_separate_mcu_for_dac": getattr(CONFIG, 'USE_SEPARATE_MCU_FOR_DAC', False),
        }
    
    # Add wellplate configurations
    if config_section.lower() == "all" or config_section.lower() == "wellplate":
        config_data["wellplate"] = {
            "default_format": getattr(CONFIG, 'WELLPLATE_FORMAT', 96),
            "offset_x_mm": getattr(CONFIG, 'WELLPLATE_OFFSET_X_MM', 0),
            "offset_y_mm": getattr(CONFIG, 'WELLPLATE_OFFSET_Y_MM', 0),
            "formats": {
                "96_well": {
                    "well_size_mm": getattr(CONFIG, 'WELL_SIZE_MM', 6.21),
                    "well_spacing_mm": getattr(CONFIG, 'WELL_SPACING_MM', 9),
                    "a1_x_mm": getattr(CONFIG, 'A1_X_MM', 14.3),
                    "a1_y_mm": getattr(CONFIG, 'A1_Y_MM', 11.36),
                    "number_of_skip": getattr(CONFIG, 'NUMBER_OF_SKIP', 0),
                },
                "384_well": {
                    "well_size_mm": getattr(CONFIG, 'WELL_SIZE_MM_384_WELLPLATE', 3.3),
                    "well_spacing_mm": getattr(CONFIG, 'WELL_SPACING_MM_384_WELLPLATE', 4.5),
                    "a1_x_mm": getattr(CONFIG, 'A1_X_MM_384_WELLPLATE', 12.05),
                    "a1_y_mm": getattr(CONFIG, 'A1_Y_MM_384_WELLPLATE', 9.05),
                    "number_of_skip": getattr(CONFIG, 'NUMBER_OF_SKIP_384', 1),
                }
            }
        }
    
    # Add objectives configuration
    if config_section.lower() == "all" or config_section.lower() == "optics":
        config_data["optics"] = {
            "objectives": getattr(CONFIG, 'OBJECTIVES', {}),
            "default_objective": getattr(CONFIG, 'DEFAULT_OBJECTIVE', '20x'),
            "tube_lens_mm": getattr(CONFIG, 'TUBE_LENS_MM', 50),
            "pixel_size_adjustment_factor": getattr(CONFIG, 'PIXEL_SIZE_ADJUSTMENT_FACTOR', 0.936),
        }
        if squid_controller:
            try:
                squid_controller.get_pixel_size()
                pixel_size_um = squid_controller.pixel_size_xy
                config_data["optics"]["calculated_pixel_size_mm"] = pixel_size_um / 1000.0
            except Exception as e:
                config_data["optics"]["calculated_pixel_size_mm"] = f"Error: {e}"
    
    # Add autofocus configuration
    if config_section.lower() == "all" or config_section.lower() == "autofocus":
        config_data["autofocus"] = {
            "stop_threshold": getattr(CONFIG.AF, 'STOP_THRESHOLD', 0.85),
            "crop_width": getattr(CONFIG.AF, 'CROP_WIDTH', 800),
            "crop_height": getattr(CONFIG.AF, 'CROP_HEIGHT', 800),
            "multipoint_reflection_af_enable": getattr(CONFIG.AF, 'MULTIPOINT_REFLECTION_AUTOFOCUS_ENABLE_BY_DEFAULT', False),
            "multipoint_af_enable": getattr(CONFIG.AF, 'MULTIPOINT_AUTOFOCUS_ENABLE_BY_DEFAULT', False),
            "focus_measure_operator": getattr(CONFIG, 'FOCUS_MEASURE_OPERATOR', 'LAPE'),
            "multipoint_af_channel": getattr(CONFIG, 'MULTIPOINT_AUTOFOCUS_CHANNEL', 'BF LED matrix full'),
            "laser_af": {
                "main_camera_model": getattr(CONFIG, 'MAIN_CAMERA_MODEL', 'MER2-1220-32U3M'),
                "focus_camera_model": getattr(CONFIG, 'FOCUS_CAMERA_MODEL', 'MER2-630-60U3M'),
                "focus_camera_exposure_time_ms": getattr(CONFIG, 'FOCUS_CAMERA_EXPOSURE_TIME_MS', 0.2),
                "focus_camera_analog_gain": getattr(CONFIG, 'FOCUS_CAMERA_ANALOG_GAIN', 0),
                "averaging_n": getattr(CONFIG, 'LASER_AF_AVERAGING_N', 5),
                "crop_width": getattr(CONFIG, 'LASER_AF_CROP_WIDTH', 1536),
                "crop_height": getattr(CONFIG, 'LASER_AF_CROP_HEIGHT', 256),
                "has_two_interfaces": getattr(CONFIG, 'HAS_TWO_INTERFACES', True),
                "use_glass_top": getattr(CONFIG, 'USE_GLASS_TOP', True),
            }
        }
    
    # Add metadata
    config_data["metadata"] = {
        "simulation_mode": is_simulation,
        "local_mode": is_local,
        "config_section_requested": config_section,
        "include_defaults": include_defaults,
        "timestamp": time.time(),
        "config_file_path": getattr(CONFIG, 'CACHE_CONFIG_FILE_PATH', None),
        "channel_configurations_path": getattr(CONFIG, 'CHANNEL_CONFIGURATIONS_PATH', './focus_camera_configurations.xml'),
    }
    
    return {
        "success": True,
        "configuration": config_data,
        "section": config_section,
        "total_sections": len(config_data) - 1  # Exclude metadata from count
    }
