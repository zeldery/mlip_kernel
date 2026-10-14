'''


'''

from loader import read_h5py_data
from kernel_script.src.mlip_kernel.mol_kernel import GaussianRidgeRegression
from qmllib.representations import generate_fchl19
from cMBDF import get_cmbdf, get_convolutions

def main():
    input_file = ''
    model_file = ''
    predict_file = ''
    data = read_h5py_data(input_file)
    numbers = data['atNUM']
    positions = data['atXYZ']
    energies = data['eDFTB']
    descriptor_list = []
    n = len(numbers)
    conv = get_convolutions()
    for i in range(n):
        descriptor = get_cmbdf(numbers[i], positions[i], conv)
        descriptor_list.append(descriptor)
    model = GaussianRidgeRegression()
    model.fit_batch(data[''])

if __name__ == '__main__':
    main()