# Feature Introduction

1. **Stage Software Barrier**: The software barrier prevents the stage from moving beyond a certain area, ensuring that the stage does not collide with the microscope hardware.

    We have a function called 'is_point_in_concave_hull' that checks if a point is within the software barrier. The barrier is defined by a concave hull, which is a polygon that encloses the stage area. The file is 'edge_positions.json'. The points are in usteps instead of mm. Take X axis as a example. You can calculate the usteps from mm using this formula:

    ```
    // You can find these values in the configuration file
    CONFIG.STAGE_MOVEMENT_SIGN_X
            * int(
                Distance_mm
                / (
                    CONFIG.SCREW_PITCH_X_MM
                    / (CONFIG.MICROSTEPPING_DEFAULT_X * CONFIG.FULLSTEPS_PER_REV_X)
                )
            )
    ```
    The function 'is_point_in_concave_hull' will return True if the point is within the barrier, and False otherwise.

2. **Simulated Sample**: The simulated sample is a virtual sample that can be used for testing the microscope software without a physical sample. The simulated sample is a '.zarr' file that contains the sample data. It's size is reflects the size of microscope stage. It's handled by the 'Camera_Simulation' class in the 'camera_default.py' file.

    When user want to acquire an image. The workflow for the simulated camera is as follows:

    - The user sends a command to the microscope service.
    - The microscope service calls the functions in the 'Camera_Simulation' class. Corp the image from the simulated sample according to the current position of the stage.
    - The cropped image is returned to the user.

    
    <img style="width:auto;" src="./assets/how_simulated_sample_works.png"> 

    Some area in the stage don't have sample data. The simulated camera will return a black image in these areas. If you want to know the location of the sample data, you can check the 'docs/coordinates_of_fovs_simulated_sample.csv' file.