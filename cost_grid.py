import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import Normalize

grid_size = 10 # size of the grid
previous_markers = [] # Store previous markers for removal
paused = False      # Pause flag
current_frame = 0   # Current frame index
first_key_press = True

# Generate all possible target position pairs (avoiding overlap)
target_positions = [(i, j, k, l) for i in range(grid_size) for j in range(grid_size)
                                   for k in range(grid_size) for l in range(grid_size)
                                   if (i, j) != (k, l)]  # Ensure distinct targets

# Initialize figure
fig, ax = plt.subplots(figsize=(10, 6))  # Wider figure to accommodate legend
fig.subplots_adjust(left=0.05, right=0.6)  # Adjust layout to move figure to left
distance_grid = np.zeros((grid_size, grid_size)) # Create an initial blank grid

# Display key/mouse control instructions in the figure
controls_text = (
    "Controls:\n"
    "[Mouse Click] - Pause/Resume Animation\n"
    "[Space] - Pause/Resume Animation\n\n"
    "Once animation is paused:\n"
    "[Up ↑] - Move Target 1 Forward\n"
    "[Down ↓] - Move Target 1 Backward\n"
    "[Right →] - Move Target 2 Forward\n"
    "[Left ←] - Move Target 2 Backward\n"
)

# Add text box to figure (placed at the right side)
#fig.text(0.72, 0.5, controls_text, fontsize=10, verticalalignment='center', bbox=dict(facecolor='white', alpha=0.6))
fig.text(0.60, 0.45, controls_text, fontsize=10, verticalalignment='top', 
         bbox=dict(facecolor='white', alpha=0.6))

# Define & apply fixed color range for consistent visualization across frames
norm = Normalize(vmin=0, vmax=10) 
im = ax.imshow(distance_grid, cmap='Purples', norm=norm, origin='upper', 
               extent=[-0.5, grid_size - 0.5, grid_size - 0.5, -0.5], interpolation='nearest')

# Create a list to store text elements for dynamic updates
text_elements = [[ax.text(j, i, "", ha="center", va="center", fontsize=12) 
                  for j in range(grid_size)] for i in range(grid_size)]

# Add invisible markers for the legend then add legend
legend_marker1 = ax.scatter([], [], marker='X', color='green', s=130, label="Target 1 (Green)")
legend_marker2 = ax.scatter([], [], marker='X', color='yellow', s=130, label="Target 2 (Yellow)")
ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0)

# Function to update each frame in the animation
def update(frame):
    global target_positions, previous_markers, paused, current_frame
    
    # If navigating manually, ensure current frame is used
    if not paused:
        current_frame = (current_frame + 1) % len(target_positions)  # Advance frame only when playing
            
    target1 = (target_positions[frame][0], target_positions[frame][1])
    target2 = (target_positions[frame][2], target_positions[frame][3])
    
    # Compute distance difference grid
    for i in range(grid_size):
        for j in range(grid_size):
            d1 = abs(i - target1[0]) + abs(j - target1[1])  # Distance to target1
            d2 = abs(i - target2[0]) + abs(j - target2[1])  # Distance to target2
            #distance_grid[i, j] = abs(d1 - d2)  # Absolute difference
            distance_grid[i, j] = d2 - d1  # difference
            
            text_color = 'white' if distance_grid[i, j] > 4 else 'black'
            text_elements[i][j].set_text(str(int(distance_grid[i, j])))  # Update text
            text_elements[i][j].set_color(text_color)  # Update color
        
    # Update the image
    im.set_array(np.clip(np.abs(distance_grid), 0, 10))  # Mirror negative values and cap at 10
   
    # Remove previous markers before drawing new ones
    for marker in previous_markers:
        marker.remove()
    previous_markers.clear()
    
    # Display target positions with distinct colors and a white border
    marker1 = ax.scatter(target1[1], target1[0], marker='X', color='green', s=130, edgecolors='white', linewidths=1.5)
    marker2 = ax.scatter(target2[1], target2[0], marker='X', color='yellow', s=130, edgecolors='white', linewidths=1.5)

    # Store new markers for removal in the next frame
    previous_markers.extend([marker1, marker2])

    # Re-add grid
    ax.set_xticks(np.arange(grid_size))
    ax.set_yticks(np.arange(grid_size))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks(np.arange(grid_size + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(grid_size + 1) - 0.5, minor=True)
    ax.grid(visible=True, which="minor", color="black", linestyle="-", linewidth=0.5)
    ax.set_title("Absolute Distance Difference to Targets")

    # Compute Manhattan distance between the two targets
    manhattan_distance = abs(target2[0] - target1[0]) + abs(target2[1] - target1[1])

    # Update the xlabel to display the targets and their Manhattan distance
    ax.set_xlabel(f"Target 1: {target1} | Target 2: {target2} | Target Distance: {manhattan_distance}", fontsize=12)

    # Explicitly return updated text elements so FuncAnimation redraws them
    return [im] + [text for row in text_elements for text in row]

# Function to toggle pause and handle manual frame navigation
def handle_key_press(event):
    global paused, current_frame, first_key_press
    
    # Immediately after pausing, pressing an arrow key has a disporportionate effect
    # that the following code block corrects for:
    increment = 1
    if first_key_press:
        if event.key == "up":
            increment = 3
        elif event.key == "right":
            increment = -1
        elif event.key == "left":
            increment = 3
        elif event.key == "down":
            increment = 0
        first_key_press = False

    # Based on the key pressed, update the current frame
    skip_redraw = False
    if event.key == " ":  # Space bar pauses/resumes
        paused = not paused
        first_key_press = True  # Reset flag so it triggers only after resuming
        global ani
        if paused:
            ani.event_source.stop()  # Stop animation to prevent background updates
        else:
            # Restart animation from current_frame to sync with manual navigation
            ani.frame_seq = iter(range(current_frame, len(target_positions)))  # Manually set frame sequence
            ani.event_source.start()  # Resume from the correct frame        
        skip_redraw = True
    elif paused and event.key == "right":  # Move forward one frame when paused
        current_frame = (current_frame + increment) % len(target_positions)
    elif paused and event.key == "left":  # Move backward one frame when paused
        current_frame = (current_frame - increment) % len(target_positions)
    elif paused and event.key == "up":  # Move forward by a full grid cycle
        current_frame = (current_frame + grid_size**2 - increment) % len(target_positions)
    elif paused and event.key == "down":  # Move backward by a full grid cycle
        current_frame = (current_frame - grid_size**2 + increment) % len(target_positions)
    else:
        skip_redraw = True
    
    if not skip_redraw:  # Force redraw
        ani.frame_seq = iter(range(current_frame, len(target_positions)))  # Reset animation's frame sequence
        update(current_frame)
        plt.draw()
    
# Function to toggle pause with mouse click
def toggle_pause(event):
    global paused, first_key_press
    paused = not paused
    if paused:
        ani.event_source.stop()  # Stop animation to prevent background updates
    else:
        # Restart animation from current_frame to sync with manual navigation
        ani.frame_seq = iter(range(current_frame, len(target_positions)))  # Manually set frame sequence
        ani.event_source.start()  # Resume from the correct frame        
    first_key_press = True  # Reset flag so it triggers only after resuming

# Connect keypress and mouse click events to handlers
fig.canvas.mpl_connect("key_press_event", handle_key_press)
fig.canvas.mpl_connect("button_press_event", toggle_pause)

# Create animation
ani = animation.FuncAnimation(fig, update, frames=len(target_positions), interval=500, repeat=True)

# Uncomment this line to save animation as a video file
#ani.save('optimal_transport_animation.mp4', writer='ffmpeg', fps=1, extra_args=['-preset', 'slow', '-crf', '28'])

plt.show()
