import csv
import os
import sys

def main():
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define the target CSV file
    csv_file = os.path.join(script_dir, "calibration_pixels.csv")
    
    if not os.path.exists(csv_file):
        print(f"Error: Could not find '{csv_file}'.")
        print(f"Make sure you run this script in the same folder as the CSV file.")
        sys.exit(1)
        
    points = []
    
    # Read the unsorted or poorly-sorted points
    print(f"Reading points from {csv_file}...")
    with open(csv_file, mode='r') as file:
        reader = csv.reader(file)
        next(reader, None)  # Extract and skip the existing header
        
        for row in reader:
            # Check if row has at least 2 elements (Pixel_X, Pixel_Y)
            if len(row) >= 2:
                try:
                    x, y = int(row[0]), int(row[1])
                    points.append((x, y))
                except ValueError:
                    print(f"Skipping invalid row: {row}")
                    
    if not points:
        print("No valid points found in the CSV.")
        sys.exit(1)
        
    if len(points) != 77:
        print(f"Warning: Expected 77 points, but found {len(points)}. World mapping might misalign.")
        
    print(f"Loaded {len(points)} points. Applying column-chunk sorting and world mapping...")
    
    # 1. Sort all points purely by X to broadly order them left-to-right
    points_by_x = sorted(points, key=lambda p: p[0])
    
    # 2. Define the real-world coordinates mapping (Corrected to Millimeters)
    # 11 columns, step size 20mm, from +100 down to -100
    world_x_vals = [100, 80, 60, 40, 20, 0, -20, -40, -60, -80, -100]
    
    # 7 rows, step size 20mm, from 410 up to 530
    world_y_vals = [410, 430, 450, 470, 490, 510, 530]
    
    # 3. Group into columns of 7, sort strictly by Y, and map world coordinates
    sorted_points_with_world = []
    points_per_column = 7
    
    col_index = 0
    for i in range(0, len(points_by_x), points_per_column):
        # Extract the 7 points that belong to this specific column
        column_chunk = points_by_x[i:i + points_per_column]
        
        # Sort these 7 points by their Y value (top-to-bottom)
        column_sorted_by_y = sorted(column_chunk, key=lambda p: p[1])
        
        # Fetch the corresponding World X coordinate for this column
        current_world_x = world_x_vals[col_index] if col_index < len(world_x_vals) else 0
        
        # Map World Y to each point in the column
        for row_index, p in enumerate(column_sorted_by_y):
            current_world_y = world_y_vals[row_index] if row_index < len(world_y_vals) else 0
            
            # Combine into a final row: [Pixel_X, Pixel_Y, World_X, World_Y]
            sorted_points_with_world.append([p[0], p[1], current_world_x, current_world_y])
            
        col_index += 1
        
    # Export the perfectly sorted and mapped points, overwriting the original CSV
    print(f"Saving mapped points back to {csv_file}...")
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # Write the new 4-column header
        writer.writerow(['Pixel_X', 'Pixel_Y', 'World_X', 'World_Y'])
        writer.writerows(sorted_points_with_world)
        
    print(f"\nSuccess! {len(sorted_points_with_world)} points have been mapped and saved in mm.")

if __name__ == "__main__":
    main()