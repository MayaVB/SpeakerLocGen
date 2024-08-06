import os
import sys
import random

import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt


def critical_distance(V, T60):
    """ critical_distance determine the distance from a sound source at which the sound field transitions from 
    being dominated by the direct sound to being dominated by the reverberant sound in a given environment
    Args:
        V (_type_): room volume [m^3]
        T60 (_type_): room reverberation time [s]

    Returns:
        critical_distance: explained in the description
    """
    return 0.057*np.sqrt(V/T60)


def calculate_doa(src_pos, mic_pos):
    """
    Calculate the Direction of Arrival (DOA) from the center of the microphone array to the source positions in angles.
    
    :param src_pos: The positions of the sources [3, n_sources].
    :param mic_pos: A list of positions of the microphones [[x1, y1, z1], [x2, y2, z2], ...].
    :return: The DOA azimuth angles in degrees.
    """
    mic_pos = np.array(mic_pos)
    array_center = np.mean(mic_pos, axis=0)  # Calculate the center of the microphone array
    
    # Define the array vector as the vector from the first to the last microphone position
    array_vector = mic_pos[-1] - mic_pos[0]
    array_vector = array_vector / np.linalg.norm(array_vector)  # Normalize the array vector
    
    # Vector from array center to source
    doa_vector = src_pos - array_center[:, np.newaxis]  # src_pos shape of (3, n_sources), array_center shape (3,)
    
    # Normalize the DOA vector
    norms = np.linalg.norm(doa_vector, axis=0)  # Shape: (n_sources,) 
    doa_vector_normalized = doa_vector / norms  # Normalize each column

    # Calculate the DOA azimuth angle
    dot_product = np.dot(array_vector, doa_vector_normalized)
    azimuth_angles = np.degrees(np.arccos(np.clip(dot_product, -1.0, 1.0)))
    
    # Calculate the DOA elevation angle
    elevation = np.degrees(np.arcsin(doa_vector_normalized[2, :]))
    
    # Adjust angles to meet the desired convention (90 degrees in front, 0 and 180 degrees along the array line)
    for i in range(len(azimuth_angles)):
        cross_product = np.cross(array_vector, doa_vector_normalized[:, i])
        if cross_product[2] < 0:
            azimuth_angles[i] = 360 - azimuth_angles[i]
    
    return azimuth_angles, elevation


def generate_scenes(args):
    """
    Generates random rooms with random source and microphones positions
    :param args: from parser
    :return: A dictionary with the room size and source and microphone positions.
    """
    
    # Array size
    mics_num = args.mics_num
    mic_min_spacing = args.mic_min_spacing
    mic_max_spacing = args.mic_max_spacing
    mic_height = args.mic_height

    # Room's size
    room_len_x_min = args.room_len_x_min
    room_len_x_max = args.room_len_x_max
    aspect_ratio_min = args.aspect_ratio_min
    aspect_ratio_max = args.aspect_ratio_max
    room_len_z_min = args.room_len_z_min
    room_len_z_max = args.room_len_z_max

    # Room's T60 value
    T60 = random.choice(args.T60_options)

    # Source
    source_min_height = args.source_min_height
    source_max_height = args.source_max_height
    source_min_radius = args.source_min_radius
    source_max_radius = args.source_max_radius
    DOA_grid_lag = args.DOA_grid_lag

    margin = args.margin
    
    while True:
        src_mics_info = []

        # Draw the room's dimensions
        room_len_x = np.random.uniform(room_len_x_min, room_len_x_max)
        room_len_z = np.random.uniform(room_len_z_min, room_len_z_max)
        aspect_ratio = np.random.uniform(aspect_ratio_min, aspect_ratio_max)
        room_len_y = room_len_x * aspect_ratio
        room_dim = [*np.random.permutation([room_len_x, room_len_y]), room_len_z]

        # Generate ULA microphone positions with margin from room walls
        mic_spacing = np.random.uniform(mic_min_spacing, mic_max_spacing)
        mic_pos_start = np.random.uniform(margin, room_dim[0] - margin - (mics_num - 1) * mic_spacing)
        mic_pos_y = np.random.uniform(margin, room_dim[1] - margin)
        mic_pos_z = mic_height

        mics_pos_agg = []
        mics_pos_arr = np.zeros((mics_num, 3))
        for mic in range(mics_num):
            mic_x = mic_pos_start + mic * mic_spacing
            mic_pos = [mic_x, mic_pos_y, mic_pos_z]
            mics_pos_agg.append(mic_pos)
            mics_pos_arr[mic] = mic_pos  # Directly assign to the preallocated array

        mics_pos_arr = mics_pos_arr.T
        
        # Randomize mic array orientation
        rotation_angle = np.random.uniform(0.001, np.pi)
        rotation_matrix = np.array([
            [np.cos(rotation_angle), -np.sin(rotation_angle), 0],
            [np.sin(rotation_angle),  np.cos(rotation_angle), 0],
            [0, 0, 1]
        ])
        mics_pos_agg = [rotation_matrix.dot(np.array(mic) - np.array([room_dim[0] / 2, room_dim[1] / 2, 0])) + np.array([room_dim[0] / 2, room_dim[1] / 2, 0]) for mic in mics_pos_agg]
        mic_array_center = np.mean(mics_pos_agg, axis=0) # array center

        # Draw a source position
        eps = sys.float_info.epsilon
        src_angle = np.radians(np.arange(0.001, 180, DOA_grid_lag)+np.degrees(rotation_angle))
        src_radius = np.random.uniform(source_min_radius, source_max_radius, size=len(src_angle))

        src_angle_deg = np.degrees(src_angle)
        src_x = np.multiply(src_radius, np.cos(src_angle)) + mic_array_center[0]
        src_y = np.multiply(src_radius, np.sin(src_angle)) + mic_array_center[1]
        
        # Ensure src_z has the same size as src_x and src_y
        src_z = np.random.uniform(source_min_height, source_max_height, size=src_x.shape)
        
        src_pos = np.array([src_x, src_y, src_z])

        # Verify if all source positions are within the room dimensions considering the margin from the walls
        if np.all((margin < src_x) & (src_x < room_dim[0] - margin)) and np.all((margin < src_y) & (src_y < room_dim[1] - margin)):
            break
            

    # Compute distances between mics and source position
    src_pos_expanded = src_pos.T[:, np.newaxis, :]  # Shape: (src_pos, 1, dim)
    mics_pos_expanded = mics_pos_arr.T[np.newaxis, :, :]  # Shape: (1, num_mic, dim)
    dists = np.linalg.norm(src_pos_expanded - mics_pos_expanded, axis=2) # substruction is of size (src_pos, num_mic, dim)
       
    # Compute critical distance
    critic_dist = critical_distance(np.prod(room_dim), T60)

    # calculate DOA
    az_DOA, el_DOA = calculate_doa(src_pos, np.array(mics_pos_agg))
        
    src_mics_info.append({
        'room_dim': room_dim, 
        'src_pos': src_pos,
        'mic_pos': np.array(mics_pos_agg),
        'critic_dist': critic_dist, 
        'dists': dists, 
        'RT60': T60, 
        'DOA_az': az_DOA})
    
    return src_mics_info
        

def plot_scene(scene, save_path, index):
    """
    Plots the room shape, microphone array and speaker
    :param scene: scene parameters
    :param save_path: saving path for figure generated
    :param index: outside loop index to support multiplay plots in folder
    # NOTE :  this plot is unclear use plot_scene_interactive
    """
    room_dim = scene['room_dim']
    src_pos = scene['src_pos']
    mic_pos = scene['mic_pos']
    DOA_az = scene['DOA_az']
    
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot room dimensions
    ax.scatter(room_dim[0], room_dim[1], room_dim[2], c='b', marker='s', label='Room Dimensions')

    # Plot source position
    ax.scatter(src_pos[0][0], src_pos[0][1], src_pos[0][2], c='r', marker='o', label='Source Position')

    # Plot microphone positions
    for i, mic in enumerate(mic_pos):
        ax.scatter(mic[0], mic[1], mic[2], c='g', marker='^', label=f'Microphone {i+1} Position')

    # Set limits for the axes to ensure full view
    ax.set_xlim([0, room_dim[0]])  # Replace xmin and xmax with your desired limits
    ax.set_ylim([0, room_dim[1]])  # Replace ymin and ymax with your desired limits
    ax.set_zlim([0, room_dim[2]])  # Replace zmin and zmax with your desired limits

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Room with Source and Microphones. DOA is {:.2f} degrees'.format(DOA_az))
    ax.legend()

    # Save plot as JPEG
    plt.savefig(os.path.join(save_path, f"scene_plot_{index}.JPEG"))
    plt.close()
    

def plot_scene_interactive(scene, save_path, index):
    """
    Plots the room shape, microphone array, and speaker with azimuth angles.
    This is an HTML file for interactively moving the room shape.
    :param scene: scene parameters
    :param save_path: saving path for the figure generated
    :param index: outside loop index to support multiple plots in folder
    """
    save_path = os.path.join(save_path, 'plots')
    os.makedirs(save_path, exist_ok=True)

    room_dim = scene['room_dim']
    src_pos = scene['src_pos']
    mic_pos = scene['mic_pos']
    DOA_az = scene['DOA_az']

    fig = go.Figure()

    # Plot room dimensions (corners of the room for illustration)
    fig.add_trace(go.Scatter3d(
        x=[0, room_dim[0], room_dim[0], 0, 0, 0, room_dim[0], room_dim[0], 0, 0, room_dim[0], room_dim[0]],
        y=[0, 0, room_dim[1], room_dim[1], 0, 0, 0, room_dim[1], room_dim[1], 0, 0, room_dim[1]],
        z=[0, 0, 0, 0, 0, room_dim[2], room_dim[2], room_dim[2], room_dim[2], room_dim[2], room_dim[2], room_dim[2]],
        mode='lines',
        name='Room Dimensions'
    ))

    # Plot source positions with azimuth angles
    for i in range(src_pos.shape[1]):
        fig.add_trace(go.Scatter3d(
            x=[src_pos[0, i]],
            y=[src_pos[1, i]],
            z=[src_pos[2, i]],
            mode='markers+text',
            marker=dict(size=5, color='red'),
            text=[f'{DOA_az[i]:.2f}°'],
            textposition='top center',
            name=f'Source Position {i+1}'
        ))

    # Plot microphone positions
    for i, mic in enumerate(mic_pos):
        fig.add_trace(go.Scatter3d(
            x=[mic[0]],
            y=[mic[1]],
            z=[mic[2]],
            mode='markers',
            marker=dict(size=5, color='green'),
            name=f'Microphone {i+1} Position'
        ))

    # Set axis labels
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
        ),
        title=f'Room with Source and Microphones'
    )

    # Save plot as an interactive HTML file
    fig.write_html(os.path.join(save_path, f"scene_plot_{index}.html"))

