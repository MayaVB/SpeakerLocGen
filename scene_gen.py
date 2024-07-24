"""
Original author:            yochai_yemini
Updated and modified by:    Maya Veisman (MayaVB)

Docker images will be uploaded to Dockerhub
"""
import plotly.graph_objects as go
import numpy as np
import os
import matplotlib.pyplot as plt
from rir_gen import generate_rirs

from mpl_toolkits.mplot3d import Axes3D
from math import degrees


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


def mic_source_dist_range(room_size, T60, scene_type):
    if scene_type == 'near':
        min_dist = 0.2
        max_dist = critical_distance(np.prod(room_size), T60)
    elif (scene_type == 'far') or (scene_type == 'winning_ticket'):
        min_dist = 2*critical_distance(np.prod(room_size), T60)
        max_dist = 3
        if (min_dist >= max_dist):      # make sure that min_dist <= max_dist
            min_dist = 1.5*critical_distance(np.prod(room_size), T60)
    elif scene_type == 'random':
        min_dist = 0.2
        max_dist = 3
    else:
        raise ValueError("scene_type must be one of {'near', 'far', 'random', 'winning_ticket'}")
    return min_dist, max_dist


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


def generate_scenes(scenes_num, scene_type, files_num):
    """
    Generates random rooms with random source and microphones positions
    :param scenes_num: How many scenes (=rooms with source-microphones setups) to generate.
    :param scene_type: 'near', 'far', 'winning_ticket' or 'random'.
    :return: A dictionary with the room size and source and microphone positions.
    """
    
    # Array size
    mics_num = 5            # default: 5
    mic_min_spacing = 0.03  # minimum spacing between microphones in the array (default: 0.03)
    mic_max_spacing = 0.08  # maximum spacing between microphones in the array (default: 0.08)
    mic_height = 0.5        # Array height from floor (default: 0.5)

    # Room's size
    room_len_x_min = 4
    room_len_x_max = 7
    aspect_ratio_min = 1
    aspect_ratio_max = 1.5
    room_len_z_min = 2.3
    room_len_z_max = 2.9

    # Room's T60 value
    T60_options = [0.2, 0.4, 0.6, 0.8]	    # Time for the RIR to reach 60dB of attenuation [s]
    np.random.shuffle(T60_options)
    T60 = T60_options[0]

    # Source
    source_min_height = 1.5                 # source miminum height (default: 1.5)
    source_max_height = 2                   # source maximun height (default: 2)
    source_min_radius = 1.5                 # source localization/walking minimum radius (default: 1.5)
    source_max_radius = 1.7                 # source localization/walking maximum radius (default: 1.7)
    DOA_grid_lag = 5                        # degrees

    margin = 0.5                            # Margin distance between the source/mics to the walls
    
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
        rotation_angle = np.random.uniform(0, np.pi)
        rotation_matrix = np.array([
            [np.cos(rotation_angle), -np.sin(rotation_angle), 0],
            [np.sin(rotation_angle),  np.cos(rotation_angle), 0],
            [0, 0, 1]
        ])
        mics_pos_agg = [rotation_matrix.dot(np.array(mic) - np.array([room_dim[0] / 2, room_dim[1] / 2, 0])) + np.array([room_dim[0] / 2, room_dim[1] / 2, 0]) for mic in mics_pos_agg]
        mic_array_center = np.mean(mics_pos_agg, axis=0) # array center

        # Draw a source position
        src_angle = np.radians(np.arange(30, 165, DOA_grid_lag)+np.degrees(rotation_angle))
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
    CURRENTLY DOSENT SUPPORT MULTIPY SRC LOCATION- TBD
    Plots the room shape, microphone array and speaker
    :param scene: scene parameters
    :param save_path: saving path for figure generated
    :param index: outside loop index to support multiplay plots in folder
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

