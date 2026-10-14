'''
Main kernel module to handle molecule-wise and element-wise kernel method
Author: Nguyen Thien Phuc Tu
Last update: 2026-10-13
'''

import torch

def gaussian_kernel_single(x, gamma):
    '''
    Generic Gaussian kernel
    Use the relations:
        ||x-y||^2 = ||x||^2 + ||y||^2 - 2xy^T
    with broadcasting for ||x|| and ||y|| to save memory on [Nx, Ny, feature]
    Input:
        x: [N, feature]
    Output:
        [N, N]
    '''
    x_sq = torch.einsum('ij,ij->i', x, x)
    x_xt = torch.einsum('ij,kj->ik', x, x)
    diff = x_sq[:,None] + x_sq[None,:] - 2*x_xt
    return torch.exp(-gamma*diff)

def gaussian_kernel_pair(x, y, gamma):
    '''
    Generic Gaussian kernel for two different matrix
    Use the relations:
        ||x-y||^2 = ||x||^2 + ||y||^2 - 2xy^T
    with broadcasting for ||x|| and ||y|| to save memory on [Nx, Ny, feature]
    Input:
        x: [Nx, feature]
        y: [Ny, feature]
    Output:
        [Nx, Ny]
    '''
    x_sq = torch.einsum('ij,ij->i', x, x)
    y_sq = torch.einsum('ij,ij->i', y, y)
    xy = torch.einsum('ij,kj->ik', x, y)
    diff = x_sq[:,None] + y_sq[None, :] - 2*xy
    return torch.exp(-gamma*diff)

def convert_atomic_kernel(kernel, n_atoms, dim=0):
    '''
    Convert the dimension dim of kernel from atomic to molecule
    n_atoms: list of number of atoms in each molecule
    '''
    n_mol = n_atoms.shape[0]
    index = torch.repeat_interleave(torch.arange(n_mol, device=kernel.device), n_atoms)
    if dim == 0:
        results = torch.zeros(n_mol, kernel.shape[1], dtype=kernel.dtype, device=kernel.device)
    elif dim == 1:
        results = torch.zeros(kernel.shape[0], n_mol, dtype=kernel.dtype, device=kernel.device)
    else:
        raise ValueError(f'Invalid dim valuation of {dim}')
    results.index_add_(dim, index, kernel)
    return results

def construct_molecular_gaussian_kernel(elements, )


    

def gaussian_molecular_kernel(elements, gamma, zx_list, x_list, zy_list=None, y_list=None):
    '''
    Compute the Molecular Kernel
    input:
        z1, z2[optional]: list of [Natom,]
        x1, x2[optional]: list of [Natom, Nfeature]
    '''
    assert len(zx_list) == len(x_list)
    if zy_list is None or y_list is None:
        n_mol = len(x_list)
        final_kernel = np.zeros((n_mol, n_mol))
        for ele in elements:
            x_tmp = []
            n_tmp = []
            for i in range(n_mol):
                index = zx_list[i]==ele
                n_tmp.append(index.sum())
                x_tmp.append(x_list[i][index])
            x_concat = np.concatenate(x_tmp, axis=0)
            kernel = gaussian_kernel(gamma, x_concat)
            final_kernel += convert_kernel(kernel, n_tmp, n_tmp)
        return final_kernel
    else:
        n_mol_x = len(x_list)
        n_mol_y = len(y_list)
        final_kernel = np.zeros((n_mol_x, n_mol_y))
        for ele in elements:
            x_tmp = []
            n_tmp_x = []
            for i in range(n_mol_x):
                index = zx_list[i]==ele
                n_tmp_x.append(index.sum())
                x_tmp.append(x_list[i][index])
            x_concat = np.concatenate(x_tmp, axis=0)
            y_tmp = []
            n_tmp_y = []
            for j in range(n_mol_y):
                index = zy_list[j]==ele
                n_tmp_y.append(index.sum())
                y_tmp.append(y_list[j][index])
            y_concat = np.concatenate(y_tmp, axis=0)
            kernel = gaussian_kernel(gamma, x_concat, y_concat)
            final_kernel += convert_kernel(kernel, n_tmp_x, n_tmp_y)
        return final_kernel

class GaussianRidgeRegression:
    '''
    
    '''
    def __init__(self, llambda, gamma, elements, batch_size=200):
        self.llambda = llambda
        self.gamma = gamma
        self.elements = elements.copy()
        self.batch_size = batch_size

    def fit(self, z_list, x_list, y):
        '''
        Input:
            x_list: list of [Natom, Nfeature]

        '''
        n_mol = len(x_list)
        kernel = gaussian_molecular_kernel(self.elements, self.gamma, z_list, x_list)
        self.coef = np.linalg.solve(kernel+self.llambda*np.eye(n_mol), y)
        self.x_train = x_list
        self.z_train = z_list
        return self

    def fit_batch(self, z_list, x_list, y):
        '''
        Input:
            x_list: list of [Natom, Nfeature]
        Use the batch to not constructing full Natom x Natom kernel
        Still have full molecular kernel
        '''
        n_mol = len(x_list)
        result_kernel = np.zeros((n_mol, n_mol))
        batch_index = list(range(0, n_mol, self.batch_size))
        batch_index.append(n_mol)
        for i in range(len(batch_index)-1):
            x_slice = slice(batch_index[i], batch_index[i+1])
            for j in range(i, len(batch_index)-1):
                y_slice = slice(batch_index[j], batch_index[j+1])
                kernel = gaussian_molecular_kernel(self.elements, self.gamma, 
                                                   z_list[x_slice], x_list[x_slice],
                                                   z_list[y_slice], x_list[y_slice])
                result_kernel[x_slice, y_slice] = kernel
                if i!=j:
                    result_kernel[y_slice, x_slice] = kernel.transpose()
        self.coef = np.linalg.solve(result_kernel + self.llambda*np.eye(n_mol), y)
        self.x_train = x_list
        self.z_train = z_list
        return self

    def predict(self, z, x):
        '''
        Input:
            x: [Natom, Nfeature]
        '''
        kernel = gaussian_molecular_kernel(self.elements, self.gamma, 
                                           self.z_train, self.x_train, [z], [x])
        return (kernel[:,0]*self.coef).sum()

    def predict_batch(self, z_list, x_list):
        '''
        Input:
            x_list: list of [Natom, Nfeature]
        '''
        n_mol = len(x_list)
        kernel = np.zeros((n_mol, len(self.x_train)))
        batch_index = list(range(0, n_mol, self.batch_size))
        batch_index.append(n_mol)
        batch_index_train = list(range(0, len(self.x_train), self.batch_size))
        batch_index_train.append(len(self.x_train))
        for i in range(len(batch_index)-1):
            slice_test = slice(batch_index[i], batch_index[i+1])
            for j in range(len(batch_index_train)-1):
                slice_train = slice(batch_index_train[j], batch_index_train[j+1])
                kernel_slice = gaussian_molecular_kernel(self.elements, self.gamma,
                                                   z_list[slice_test], x_list[slice_test],
                                                   self.z_train[slice_train], self.x_train[slice_train])
                kernel[slice_test, slice_train] = kernel_slice
        return np.matmul(kernel, self.coef)
    
    def save(self, file_name):
        '''
        The training set is ragged, so it is stored flat:
        all atoms concatenated, plus the number of atom of each molecule
        This keeps every entry rectangular, so no pickle is needed to read it back
        '''
        data = {}
        data['elements'] = np.asarray(self.elements)
        data['llambda'] = self.llambda
        data['gamma'] = self.gamma
        data['batch_size'] = self.batch_size
        data['n_atom'] = np.array([len(z) for z in self.z_train])
        data['x_train'] = np.concatenate(self.x_train, axis=0)
        data['z_train'] = np.concatenate(self.z_train, axis=0)
        data['coef'] = self.coef
        np.savez(file_name, **data)

    def load(self, file_name):
        data = np.load(file_name)
        self.elements = data['elements'].tolist()
        self.llambda = data['llambda'].item()
        self.gamma = data['gamma'].item()
        self.batch_size = data['batch_size'].item()
        offset = np.cumsum(data['n_atom'])[:-1]
        self.x_train = np.split(data['x_train'], offset, axis=0)
        self.z_train = np.split(data['z_train'], offset)
        self.coef = data['coef']
        return self
