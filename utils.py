import h5py
import os
import soundfile as sf


def save_data_h5py(rev_signals, scene, scene_idx, src_pos_index, speaker_out_dir=[]):
    """
    Save wav data and label DOA in two groups, this is done for each scene and for each source position
    :param rev_signals: reverberent signals
    :param scene: scene parameters
    :param scene_idx: scene index
    :param src_pos_index: source position index
    :param speaker_out_dir: directory where the dataset is stored (defults is home dir)
    """
    # os.makedirs(speaker_out_dir, exist_ok=True)
    dataset_path = os.path.join(speaker_out_dir, 'RevMovingSrcDataset.h5')

    with h5py.File(dataset_path, 'a') as h5f:
        group_name = f"sceneIndx_{scene_idx}_roomDim_{scene['room_dim']}_rt60_{scene['RT60']}_srcPosInd_{src_pos_index}"
        grp = h5f.create_group(group_name)
        grp.create_dataset(f'signals_', data=rev_signals)
        grp.create_dataset(f'speaker_DOA_', data=scene['DOA_az'][src_pos_index])
        grp.create_dataset(f'reverberation_time_', data=scene['RT60'])
        grp.create_dataset(f'mic_positions', data=scene['mic_pos'])
                
        # Increment the dataset length
        if 'data_len' in h5f.attrs:
            h5f.attrs['data_len'] += 1
        else:
            h5f.attrs['data_len'] = 1
        

def save_wavs(rev_signals, file, speaker_out_dir, fs):
    for j, sig in enumerate(rev_signals, 1):
        out_file_name = '_ch{}'.format(j) + '.wav' 
        out_file_path = os.path.join(speaker_out_dir, out_file_name)
        sf.write(out_file_path, sig, fs)


def round_numbers(list_in):
    return list(map(lambda x: round(x, 4), list_in))


def write_scene_to_file(scenes, file_name):
    """
    print scene info to a txt file
    :param scenes: list of dict containing info about the scenario generated
    :param file_name: file name to save the info
    """
    with open (file_name, 'w') as f:
        for scene in scenes:
            f.write(f"Room size: {round_numbers(scene['room_dim'])}\n")
            f.write(f"Reverberation time: {round(scene['RT60'])}\n")
            f.write(f"Critical distance: {round(scene['critic_dist'], 4)}\n")
            pos = round_numbers(scene['src_pos'][0])
            f.write(f"Source position: {pos}\n")
            mics_num = len(scene['mic_pos'])
            for i in range(mics_num):
                pos = round_numbers(scene['mic_pos'][i])
                dist = round_numbers(scene['dists'][i])
                f.write(f"Mic{i} pos\t: {pos}, dist:{dist}\n")
            f.write('\n\n\n')
            f.flush()
    return


def read_hdf5_file(file_path):
    data_dict = {}
    
    with h5py.File(file_path, 'r') as h5f:
        # List all groups
        group_keys = list(h5f.keys())
        print("Groups in HDF5 file:", group_keys)
        
        for group in group_keys:
            print(f"Group: {group}")
            data_dict[group] = {}
            grp = h5f[group]
            
            dataset_keys = list(grp.keys())
            print(f"Datasets in {group}:", dataset_keys)
            
            for dataset in dataset_keys:
                data = grp[dataset][:]
                print(f"Dataset: {dataset}, Data shape: {data.shape}")
                data_dict[group][dataset] = data
                
                # Example: print the first few samples
                print(f"First few samples of {dataset}: {data[:10]}")
    
    return data_dict