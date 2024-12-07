import json
from squid_control.control.config import *



# Edge positions in mm
edge_positions_mm = [
    (20, 4, 0), (20, 4, 4.5),
    (10.97, 15.33, 0), (10.97, 15.33, 4.5),
    (10.97, 76.52, 0), (10.97, 76.52, 4.5),
    (20, 78.5, 0), (20, 78.5, 4.5),
    (99.32, 78.5, 0), (99.32, 78.5, 4.5),
    (111.55, 67.52, 0), (111.55, 67.52, 4.5),
    (111.55, 15.33, 0), (111.55, 15.33, 4.5),
    (99.32, 4, 0), (99.32, 4, 4.5)
]

# Function to convert mm to microsteps (usteps)
def mm_to_usteps(x, y, z):
    usteps_x = (CONFIG.STAGE_MOVEMENT_SIGN_X
                * int(
                    x
                    / (
                        CONFIG.SCREW_PITCH_X_MM
                        / (CONFIG.MICROSTEPPING_DEFAULT_X * CONFIG.FULLSTEPS_PER_REV_X)
                    )
                ))
    usteps_y = (CONFIG.STAGE_MOVEMENT_SIGN_Y
                * int(
                    y
                    / (
                        CONFIG.SCREW_PITCH_Y_MM
                        / (CONFIG.MICROSTEPPING_DEFAULT_Y * CONFIG.FULLSTEPS_PER_REV_Y)
                    )
                ))
    usteps_z = (CONFIG.STAGE_MOVEMENT_SIGN_Z
                * int(
                    z
                    / (
                        CONFIG.SCREW_PITCH_Z_MM
                        / (CONFIG.MICROSTEPPING_DEFAULT_Z * CONFIG.FULLSTEPS_PER_REV_Z)
                    )
                ))
    return (usteps_x, usteps_y, usteps_z)

# Convert all edge positions to microsteps
edge_positions_usteps = [mm_to_usteps(x, y, z) for x, y, z in edge_positions_mm]

# Path to save the JSON file
json_file_path = 'squid-control/squid_control/control/edge_positions.json'

# Save edge positions to a JSON file
with open(json_file_path, 'w') as file:
    json.dump(edge_positions_usteps, file)

print("Edge positions in microsteps saved to:", json_file_path)
