import json

# Constants
CONFIG_SCREW_PITCH_MM_XY = 1  # Screw pitch for X and Y axes in mm
CONFIG_SCREW_PITCH_MM_Z = 0.012 * 25.4  # Screw pitch for Z axis in mm (0.012 inches to mm)
STEPS_PER_REVOLUTION = 200 * 8  # Full steps per revolution times microstepping factor

# Edge positions in mm
edge_positions_mm = [
    (20, 4, 0.1), (20, 4, 4.5),
    (10.97, 15.33, 0.1), (10.97, 15.33, 4.5),
    (10.97, 76.52, 0.1), (10.97, 76.52, 4.5),
    (20, 78.5, 0.1), (20, 78.5, 4.5),
    (99.32, 78.5, 0.1), (99.32, 78.5, 4.5),
    (111.55, 67.52, 0.1), (111.55, 67.52, 4.5),
    (111.55, 15.33, 0.1), (111.55, 15.33, 4.5),
    (99.32, 4, 0.1), (99.32, 4, 4.5)
]

# Function to convert mm to microsteps (usteps)
def mm_to_usteps(x, y, z):
    usteps_x = int(x / (CONFIG_SCREW_PITCH_MM_XY / STEPS_PER_REVOLUTION))
    usteps_y = int(y / (CONFIG_SCREW_PITCH_MM_XY / STEPS_PER_REVOLUTION))
    usteps_z = int(z / (CONFIG_SCREW_PITCH_MM_Z / STEPS_PER_REVOLUTION))
    return (usteps_x, usteps_y, usteps_z)

# Convert all edge positions to microsteps
edge_positions_usteps = [mm_to_usteps(x, y, z) for x, y, z in edge_positions_mm]

# Path to save the JSON file
json_file_path = 'squid_control\control\edge_positions.json'

# Save edge positions to a JSON file
with open(json_file_path, 'w') as file:
    json.dump(edge_positions_usteps, file)

print("Edge positions in microsteps saved to:", json_file_path)
