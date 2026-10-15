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
    Generic Gaussian kernel for two different set of atoms
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

def convert_kernel(kernel, n_atoms, dim=0):
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

def construct_molecular_kernel(atomic_numbers, descriptors, gamma):
    '''
    Construct the molecular gaussian kernel
    Input:
        atomic_numbers: [Nmol, Natom] masked with -1
        descriptors: [Nmol, Natom, Nfeature] masked with 0
    Output:
        kernel: [Nmol, Nmol]
    '''
    elements = atomic_numbers.unique(sorted=True)
    if elements[0] == -1: # Eliminate -1 masking
        elements = elements[1:]
    results = torch.zeros(atomic_numbers.shape[0], atomic_numbers.shape[0],
                          dtype=descriptors.dtype, device=descriptors.device)
    for ele in elements:
        atomic_index = (atomic_numbers == ele)
        n_atoms = atomic_index.sum(dim=1)
        x = descriptors[atomic_index,:] # Become [Natom_total, feature]
        kernel_tmp = gaussian_kernel_single(x, gamma)
        kernel_tmp = convert_kernel(kernel_tmp, n_atoms, dim=0)
        kernel_tmp = convert_kernel(kernel_tmp, n_atoms, dim=1)
        results += kernel_tmp
    return results

def construct_molecular_kernel_pair(atomic_numbers1, descriptors1, 
                                             atomic_numbers2, descriptors2, gamma):
    '''
    Construct the molecular gaussian kernel for 2 different set of molecule
    Input:
        atomic_numbers1 : [Nmol1, Natom1] masked with -1
        atomic_numbers2 : [Nmol2, Natom2] masked with -1
        descriptors1: [Nmol1, Natom1, feature]
        descriptors2: [Nmol2, Natom2, feature]
    Output:
        kernel: [Nmol1, Nmol2]
    '''
    elements1 = atomic_numbers1.unique(sorted=True)
    elements2 = atomic_numbers2.unique(sorted=True)
    elements = elements1[torch.isin(elements1, elements2)].unique(sorted=True)
    if elements[0] == -1: # Eliminate -1
        elements = elements[1:]
    results = torch.zeros(atomic_numbers1.shape[0], atomic_numbers2.shape[0],
                          dtype=descriptors1.dtype, device=descriptors1.device)
    for ele in elements:
        atomic_index1 = (atomic_numbers1 == ele)
        n_atoms1 = atomic_index1.sum(dim=1)
        x = descriptors1[atomic_index1,:]
        atomic_index2 = (atomic_numbers2 == ele)
        n_atoms2 = atomic_index2.sum(dim=1)
        y = descriptors2[atomic_index2,:]
        kernel_tmp = gaussian_kernel_pair(x, y, gamma)
        kernel_tmp = convert_kernel(kernel_tmp, n_atoms1, dim=0)
        kernel_tmp = convert_kernel(kernel_tmp, n_atoms2, dim=1)
        results += kernel_tmp
    return results

def construct_atomic_kernel(atomic_numbers, descriptors, gamma):
    '''
    Construct the atomic kernel model
    Used for energy prediction, but should be concat to predict energy, force and Hessian
    Input:
        atomic_numbers: [Nmol, Natoms]
        descriptors: [Nmol, Natoms, feature]
    Output:
        kernel: [Nmol, Natoms]
    '''
    elements = atomic_numbers.unique(sorted=True)
    if elements[0] == -1: # Eliminate -1 masking
        elements = elements[1:]
    atomic_flat = atomic_numbers[atomic_numbers!=-1] # Flat atomic_numbers, for determine columns
    results= torch.zeros(atomic_numbers.shape[0], atomic_flat.shape[0],
                         dtype=descriptors.dtype, device=descriptors.device)
    for ele in elements:
        atomic_index = (atomic_numbers == ele)
        n_atoms = atomic_index.sum(dim=1)
        x = descriptors[atomic_index,:]
        kernel_tmp = gaussian_kernel_single(x, gamma)
        kernel_tmp = convert_kernel(kernel_tmp, n_atoms, dim=0)
        col = (atomic_flat == ele).nonzero().squeeze(1) # Where to put the results
        results[:,col] += kernel_tmp
    return results

def construct_atomic_kernel_pair(atomic_numbers1, descriptors1, 
                                 atomic_numbers2, descriptors2, gamma):
    '''
    Construct the atomic kernel model, for deployment
    Used for energy prediction, but should be concat to predict energy, force and Hessian
    Input:
        atomic_numbers1: [Nmol1, Natoms1]
        descriptors1: [Nmol1, Natoms1, feature]
        atomic_numbers2: [Nmol2, Natoms2]
        descriptors2: [Nmol2, Natoms2, feature]
    Output:
        kernel: [Nmol1, Natoms2]
    '''
    elements1 = atomic_numbers1.unique(sorted=True)
    elements2 = atomic_numbers2.unique(sorted=True)
    elements = elements1[torch.isin(elements1, elements2)].unique(sorted=True)
    if elements[0] == -1: # Eliminate -1
        elements = elements[1:]
    atomic_flat2 = atomic_numbers2[atomic_numbers2!=-1]
    results = torch.zeros(atomic_numbers1.shape[0], atomic_flat2.shape[0],
                          dtype=descriptors1.dtype, device=descriptors1.device)
    for ele in elements:
        atomic_index1 = (atomic_numbers1 == ele)
        n_atoms1 = atomic_index1.sum(dim=1)
        x = descriptors1[atomic_index1,:]
        atomic_index2 = (atomic_numbers2 == ele)
        y = descriptors2[atomic_index2,:]
        kernel_tmp = gaussian_kernel_pair(x, y, gamma)
        kernel_tmp = convert_kernel(kernel_tmp, n_atoms1, dim=0)
        col = (atomic_flat2 == ele).nonzero().squeeze(1) # Where to put the results
        results[:,col] += kernel_tmp
    return results

# class GaussianRidgeRegression:
#     '''
    
#     '''
#     def __init__(self, llambda, gamma, elements, batch_size=200):
#         self.llambda = llambda
#         self.gamma = gamma
#         self.elements = elements.copy()
#         self.batch_size = batch_size

#     def fit(self, z_list, x_list, y):
#         '''
#         Input:
#             x_list: list of [Natom, Nfeature]

#         '''
#         n_mol = len(x_list)
#         kernel = gaussian_molecular_kernel(self.elements, self.gamma, z_list, x_list)
#         self.coef = np.linalg.solve(kernel+self.llambda*np.eye(n_mol), y)
#         self.x_train = x_list
#         self.z_train = z_list
#         return self

#     def fit_batch(self, z_list, x_list, y):
#         '''
#         Input:
#             x_list: list of [Natom, Nfeature]
#         Use the batch to not constructing full Natom x Natom kernel
#         Still have full molecular kernel
#         '''
#         n_mol = len(x_list)
#         result_kernel = np.zeros((n_mol, n_mol))
#         batch_index = list(range(0, n_mol, self.batch_size))
#         batch_index.append(n_mol)
#         for i in range(len(batch_index)-1):
#             x_slice = slice(batch_index[i], batch_index[i+1])
#             for j in range(i, len(batch_index)-1):
#                 y_slice = slice(batch_index[j], batch_index[j+1])
#                 kernel = gaussian_molecular_kernel(self.elements, self.gamma, 
#                                                    z_list[x_slice], x_list[x_slice],
#                                                    z_list[y_slice], x_list[y_slice])
#                 result_kernel[x_slice, y_slice] = kernel
#                 if i!=j:
#                     result_kernel[y_slice, x_slice] = kernel.transpose()
#         self.coef = np.linalg.solve(result_kernel + self.llambda*np.eye(n_mol), y)
#         self.x_train = x_list
#         self.z_train = z_list
#         return self

#     def predict(self, z, x):
#         '''
#         Input:
#             x: [Natom, Nfeature]
#         '''
#         kernel = gaussian_molecular_kernel(self.elements, self.gamma, 
#                                            self.z_train, self.x_train, [z], [x])
#         return (kernel[:,0]*self.coef).sum()

#     def predict_batch(self, z_list, x_list):
#         '''
#         Input:
#             x_list: list of [Natom, Nfeature]
#         '''
#         n_mol = len(x_list)
#         kernel = np.zeros((n_mol, len(self.x_train)))
#         batch_index = list(range(0, n_mol, self.batch_size))
#         batch_index.append(n_mol)
#         batch_index_train = list(range(0, len(self.x_train), self.batch_size))
#         batch_index_train.append(len(self.x_train))
#         for i in range(len(batch_index)-1):
#             slice_test = slice(batch_index[i], batch_index[i+1])
#             for j in range(len(batch_index_train)-1):
#                 slice_train = slice(batch_index_train[j], batch_index_train[j+1])
#                 kernel_slice = gaussian_molecular_kernel(self.elements, self.gamma,
#                                                    z_list[slice_test], x_list[slice_test],
#                                                    self.z_train[slice_train], self.x_train[slice_train])
#                 kernel[slice_test, slice_train] = kernel_slice
#         return np.matmul(kernel, self.coef)
    
#     def save(self, file_name):
#         '''
#         The training set is ragged, so it is stored flat:
#         all atoms concatenated, plus the number of atom of each molecule
#         This keeps every entry rectangular, so no pickle is needed to read it back
#         '''
#         data = {}
#         data['elements'] = np.asarray(self.elements)
#         data['llambda'] = self.llambda
#         data['gamma'] = self.gamma
#         data['batch_size'] = self.batch_size
#         data['n_atom'] = np.array([len(z) for z in self.z_train])
#         data['x_train'] = np.concatenate(self.x_train, axis=0)
#         data['z_train'] = np.concatenate(self.z_train, axis=0)
#         data['coef'] = self.coef
#         np.savez(file_name, **data)

#     def load(self, file_name):
#         data = np.load(file_name)
#         self.elements = data['elements'].tolist()
#         self.llambda = data['llambda'].item()
#         self.gamma = data['gamma'].item()
#         self.batch_size = data['batch_size'].item()
#         offset = np.cumsum(data['n_atom'])[:-1]
#         self.x_train = np.split(data['x_train'], offset, axis=0)
#         self.z_train = np.split(data['z_train'], offset)
#         self.coef = data['coef']
#         return self
