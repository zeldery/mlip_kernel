'''

'''

from qmllib.representations import generate_fchl19
from cMBDF import get_cmbdf, get_convolutions
from loader import read_h5py_data
from kernel_script.src.mlip_kernel.mol_kernel import GaussianRidgeRegression
from tqdm import tqdm

def main():
    atomic_list, position_list, energy = read_h5py_data('../qm7x_hc_opt.hdf5')
    conv = get_convolutions()
    # cmbdf = get_cmbdf(atomic_list[0], position_list[0], conv)
    # fchl = generate_fchl19(atomic_list[0], position_list[0], [1, 6])

    cmbdf_list = []
    for atomic, position in tqdm(zip(atomic_list, position_list), 'Structure:'):
        cmbdf_list.append(get_cmbdf(atomic, position, conv))
    model = GaussianRidgeRegression(1e-9, 1e-3)
    model.fit_batch(cmbdf_list, energy)

if __name__ == '__main__':
    main()