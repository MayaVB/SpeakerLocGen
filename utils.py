import h5py

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