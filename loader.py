import h5py
import numpy as np
import copy

class DataLoader:
    '''
    
    '''
    def __init__(self, field_name):
        self.field_name = copy.deepcopy(field_name)

    def iter(self, h5py_file):
        keys = list(h5py_file.keys())
        if len(keys) > 0:
            if isinstance(h5py_file[keys[0]], h5py.Group):
                for key in keys:
                    yield from self.iter(h5py_file[key])
            else:
                tmp = {}
                for field in self.field_name:
                    tmp[field] = np.array(h5py_file[field])
                yield tmp

    def read(self, file_name):
        file = h5py.File(file_name, 'r')
        data = {x:[] for x in self.field_name}
        for items in self.iter(file):
            for field in self.field_name:
                data[field].append(items[field])
        return data

def read_h5py_data(file_name):
    loader = DataLoader(['atNUM', 'atXYZ', 'eDFTB'])
    data = loader.read(file_name)
    pos = [x[0,:,:] for x in data['atXYZ']]
    en = np.concatenate(data['eDFTB'])
    return data['atNUM'], pos, en

