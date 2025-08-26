import h5py


def print_hdf5_structure(name, obj):
    print(f"{name}: {obj.shape} {obj.dtype}")
    
if __name__ == "__main__":
    f = h5py.File('matrix.h5', 'r')
    f.visititems(print_hdf5_structure)
    f.close()