'''
cMBDF Descriptor calculation module
Rewrite from dkhan42's repos (github.com/dkhan42/cMBDF) using Claude

This is an AI rewrite of the code. Use with caution
'''

import math

import torch
from torch import nn
from combinenet.sf import create_triple_index
from combinenet.utils import create_double_index_batch

# (period, group, van-der-Waals radius / Angstrom) per nuclear charge.
# Radii are Bondi as tabulated by ASE, completed with Alvarez (2013),
# Dalton Trans. 42, 8617 for the transition metals Bondi does not cover.
ELEMENT_DATA = {
     1: (1,  1, 1.20),      # H
     2: (1, 18, 1.40),      # He
     3: (2,  1, 1.82),      # Li
     4: (2,  2, 1.53),      # Be
     5: (2, 13, 1.92),      # B
     6: (2, 14, 1.70),      # C
     7: (2, 15, 1.55),      # N
     8: (2, 16, 1.52),      # O
     9: (2, 17, 1.47),      # F
    10: (2, 18, 1.54),      # Ne
    11: (3,  1, 2.27),      # Na
    12: (3,  2, 1.73),      # Mg
    13: (3, 13, 1.84),      # Al
    14: (3, 14, 2.10),      # Si
    15: (3, 15, 1.80),      # P
    16: (3, 16, 1.80),      # S
    17: (3, 17, 1.75),      # Cl
    18: (3, 18, 1.88),      # Ar
    19: (4,  1, 2.75),      # K
    20: (4,  2, 2.31),      # Ca
    21: (4,  3, 2.15),      # Sc
    22: (4,  4, 2.11),      # Ti
    23: (4,  5, 2.07),      # V
    24: (4,  6, 2.06),      # Cr
    25: (4,  7, 2.05),      # Mn
    26: (4,  8, 2.04),      # Fe
    27: (4,  9, 2.00),      # Co
    28: (4, 10, 1.63),      # Ni
    29: (4, 11, 1.40),      # Cu
    30: (4, 12, 1.39),      # Zn
    31: (4, 13, 1.87),      # Ga
    32: (4, 14, 2.11),      # Ge
    33: (4, 15, 1.85),      # As
    34: (4, 16, 1.90),      # Se
    35: (4, 17, 1.85),      # Br
    36: (4, 18, 2.02),      # Kr
    37: (5,  1, 3.03),      # Rb
    38: (5,  2, 2.49),      # Sr
    39: (5,  3, 2.32),      # Y
    40: (5,  4, 2.23),      # Zr
    41: (5,  5, 2.18),      # Nb
    42: (5,  6, 2.17),      # Mo
    43: (5,  7, 2.16),      # Tc
    44: (5,  8, 2.13),      # Ru
    45: (5,  9, 2.10),      # Rh
    46: (5, 10, 1.63),      # Pd
    47: (5, 11, 1.72),      # Ag
    48: (5, 12, 1.58),      # Cd
    49: (5, 13, 1.93),      # In
    50: (5, 14, 2.17),      # Sn
    51: (5, 15, 2.06),      # Sb
    52: (5, 16, 2.06),      # Te
    53: (5, 17, 1.98),      # I
    54: (5, 18, 2.16),      # Xe
}


def charge_scaling(z):
    '''A(Z) = log(P + 1) G, eq. 28, with P the period and G the group.'''
    z = int(z)
    if z not in ELEMENT_DATA:
        raise KeyError('no period/group/vdW data for Z = %d; extend ELEMENT_DATA' % z)
    period, group, _ = ELEMENT_DATA[z]
    return math.log(period + 1.0)*group


def gaussian_width(z_list):
    '''sigma(Z_1..Z_nu-1) = sum_j Z_j rvdW(Z_j) / sum_j Z_j, eq. 30.'''
    return (sum(float(z)*ELEMENT_DATA[int(z)][2] for z in z_list)
            / sum(float(z) for z in z_list))


def next_fast_len(n):
    '''
    Smallest integer >= n with no prime factor above 11, as scipy.fft.next_fast_len.
    '''
    n = max(int(n), 1)
    while True:
        m = n
        for p in (2, 3, 5, 7, 11):
            while m % p == 0:
                m //= p
        if m == 1:
            return n
        n += 1


def hermite(x, m):
    '''
    Physicists' Hermite polynomial H_m(x):
        H_0 = 1,  H_1 = 2x,  H_{k+1} = 2x H_k - 2k H_{k-1}
    '''
    h_prev = torch.ones_like(x)
    if m == 0:
        return h_prev
    h = 2.0*x
    for k in range(1, m):
        h_prev, h = h, 2.0*x*h - 2.0*k*h_prev
    return h


def gaussian_derivative(u, sigma, m):
    '''
    f_m of eq. 7, the m-th derivative of the normal PDF N(0, sigma):
        (-1)^m / (sqrt(2 pi) sigma (sqrt(2) sigma)^m) H_m(u/(sqrt(2) sigma)) exp(-u^2/(2 sigma^2))
    '''
    scale = math.sqrt(2.0)*sigma
    x = u/scale
    prefactor = ((-1.0)**m)/(math.sqrt(2.0*math.pi)*sigma*scale**m)
    return prefactor*hermite(x, m)*torch.exp(-x*x)


def radial_weights(r, alpha, n_order):
    '''g_n2(r): exp(-alpha (n+1) r), then 1/(r+1)^(2n+3), n = 0 .. n_order-1.'''
    return ([torch.exp(-alpha*(n + 1.0)*r) for n in range(n_order)]
            + [1.0/(r + 1.0)**(2*n + 3) for n in range(n_order)])


def angular_weights(theta, n_order):
    '''
    g_n3(theta): cos((2n+1)t) - cos((2n+1)(t+pi)), then the same with sin.
    Both reduce to twice the plain harmonic, since 2n+1 is odd.
    '''
    k = [2*n + 1 for n in range(n_order)]
    return ([torch.cos(c*theta) - torch.cos(c*(theta + math.pi)) for c in k]
            + [torch.sin(c*theta) - torch.sin(c*(theta + math.pi)) for c in k])


def _convolutions(weights, offsets, sigma, order, n_level, step):
    '''
    H^{nm}(x) = (-1)^m (g_n * f_m)(x), eqs. 8-10 and 14-16, on the grid
    x_k = k step that the weights are sampled on, with level s holding the
    s-th derivative in x, (-1)^s H^{n(m+s)} (eq. 33).

    weights: list of (n,) sampled at t = 0 .. (n-1) step
    offsets: (2n,) at u = -n step .. (n-1) step
    Returns [n_level, n, len(weights) (order+1)], weighting function first,
    then m.
    '''
    n = weights[0].shape[0]
    size = 3*n - 1
    # full convolution index n + k is x = k step: the 'same' slice of
    # cMBDF.py starts one step earlier
    spectra = [torch.fft.rfft(gaussian_derivative(offsets, sigma, d), size)
               for d in range(order + n_level)]
    out = torch.zeros((n_level, n, len(weights)*(order + 1)), dtype=torch.float64)
    for w, weight in enumerate(weights):
        spectrum = torch.fft.rfft(weight, size)
        h = [(-1.0)**d*step*torch.fft.irfft(spectrum*s, size)[n:2*n] for d, s in enumerate(spectra)]
        for m in range(order + 1):
            for s in range(n_level):
                out[s, :, w*(order + 1) + m] = (-1.0)**s*h[m + s]
    return out


class _GridLookup(torch.autograd.Function):
    '''
    table[level][group, nearest grid point of x], [..., n_feature].  The
    derivative in x is the same read one level up (backward and forward
    mode), and zero past the last level.
    '''
    generate_vmap_rule = True

    @staticmethod
    def forward(x, group, table, step, level):
        index = torch.floor(x/step + 0.5).long().clamp(0, table.shape[2] - 1)
        return table[level][group, index]

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, group, table, step, level = inputs
        ctx.save_for_backward(x, group, table)
        ctx.save_for_forward(x, group, table)
        ctx.step, ctx.level = step, level

    @staticmethod
    def _slope(ctx):
        x, group, table = ctx.saved_tensors
        if ctx.level + 1 >= table.shape[0]:
            return x.new_zeros(x.shape + table.shape[-1:])
        return _GridLookup.apply(x, group, table, ctx.step, ctx.level + 1)

    @staticmethod
    def backward(ctx, grad_output):
        return (grad_output*_GridLookup._slope(ctx)).sum(-1), None, None, None, None

    @staticmethod
    def jvp(ctx, x_tangent, *_):
        return _GridLookup._slope(ctx)*x_tangent[..., None]


class _SafeArccos(torch.autograd.Function):
    '''
    arccos with d theta / d cos = -1/max(sin theta, epsilon), the epsilon of
    cMBDF.py's gradient code, so collinear triplets stay finite.
    '''
    generate_vmap_rule = True

    @staticmethod
    def forward(cosine, epsilon):
        return torch.arccos(cosine)

    @staticmethod
    def setup_context(ctx, inputs, output):
        cosine, epsilon = inputs
        ctx.save_for_backward(cosine)
        ctx.save_for_forward(cosine)
        ctx.epsilon = epsilon

    @staticmethod
    def _slope(ctx):
        cosine, = ctx.saved_tensors
        # clamp sin^2 before the square root, so second derivatives are not NaN
        return -1.0/torch.sqrt((1.0 - cosine*cosine).clamp_min(ctx.epsilon**2))

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output*_SafeArccos._slope(ctx), None

    @staticmethod
    def jvp(ctx, cosine_tangent, _):
        return _SafeArccos._slope(ctx)*cosine_tangent


class CMBDF(nn.Module):
    '''
    The cMBDF module: pre-evaluated grids of eqs. 10 and 16 stored with their
    parameters, and the descriptor computed from them.

    Like SymmetryFunction in combinenet it takes element indices, 0 .. n-1 in
    the order of element_list and -1 for padding, not nuclear charges;
    combinenet.utils.create_element_encoder(element_list)[atomic_numbers]
    converts one to the other.

    The parameters are plain attributes, and dump/write store only those; the
    grids (about 200 MB in float64 for H, C, N, O) are rebuilt by set and load
    in about a second.  The grids are buffers, so .to(device) and
    .float()/.double() move them with the module; set and load build them on
    the CPU in float64, the precision to keep for second derivatives.
    '''
    def __init__(self):
        super().__init__()
        self.initialized = False

    def set(self, element_list, cut_off=10.0, order=4, n_order=4, alpha=1.5, angular_sigma=0.5,
            radial_step=0.0008, angular_step=0.0002, n_level=3):
        '''
        element_list:  nuclear charges, in the order of the element index
        cut_off:       Angstrom
        order:         m = 0 .. order, the derivative orders
        n_order:       n = 0 .. n_order-1, per weighting-function family
        alpha:         radial decay, 1/Angstrom
        angular_sigma: angular Gaussian width, radians; None takes eq. 30 as written
        radial_step:   radial grid spacing, Angstrom
        angular_step:  angular grid spacing, radians
        n_level:       grids kept per channel, value and derivatives (2 = forces, 3 = Hessians)
        Feature length 2 (nu-1) (order+1) n_order: 80 at the defaults, 40 at the
        paper's n_order = 2.
        '''
        self.element_list = [int(z) for z in element_list]
        self.cut_off = float(cut_off)
        self.order = int(order)
        self.n_order = int(n_order)
        self.alpha = float(alpha)
        self.angular_sigma = None if angular_sigma is None else float(angular_sigma)
        self.radial_step = float(radial_step)
        self.angular_step = float(angular_step)
        self.n_level = int(n_level)
        self._build()
        self.initialized = True

    def read(self, file_name):
        try:
            state_dict = torch.load(file_name, map_location=torch.device('cpu'), weights_only=True)
        except:
            raise IOError(f'Cannot load the state dictionary from {file_name}')
        self.load(state_dict)

    def write(self, file_name):
        torch.save(self.dump(), file_name)

    def load(self, data):
        self.set(data['element_list'], data['cut_off'], data['order'], data['n_order'], data['alpha'],
                 data['angular_sigma'], data['radial_step'], data['angular_step'], data['n_level'])

    def dump(self):
        # plain numbers, so .float() cannot round them; the grids are rebuilt from these
        return {'element_list': self.element_list.copy(), 'cut_off': self.cut_off, 'order': self.order,
                'n_order': self.n_order, 'alpha': self.alpha, 'angular_sigma': self.angular_sigma,
                'radial_step': self.radial_step, 'angular_step': self.angular_step, 'n_level': self.n_level}

    def _build(self):
        '''
        The grids from the parameters, on the CPU in float64, registered as
        non-persistent buffers so that .to() moves them but dump leaves them out.
        '''
        f64 = torch.float64
        elements, cut_off, order, n_order, n_level = self.element_list, self.cut_off, self.order, self.n_order, self.n_level
        n_elem = len(elements)
        pair_index = torch.zeros((n_elem, n_elem), dtype=torch.long)
        pairs = []
        for i in range(n_elem):
            for j in range(i, n_elem):
                pair_index[i, j] = pair_index[j, i] = len(pairs)
                pairs.append((elements[i], elements[j]))

        # radial grid r in [0, cut_off), as cMBDF.py; angular grid theta in [0, pi]
        n_r = next_fast_len(int(cut_off/self.radial_step))
        step_r = cut_off/n_r
        rgrid = torch.arange(n_r, dtype=f64)*step_r
        rgrid2 = (torch.arange(2*n_r, dtype=f64) - n_r)*step_r
        n_a = next_fast_len(int(math.pi/self.angular_step)) + 1
        step_a = math.pi/(n_a - 1)
        agrid = torch.arange(n_a, dtype=f64)*step_a
        agrid2 = (torch.arange(2*n_a, dtype=f64) - n_a)*step_a

        rw = radial_weights(rgrid, self.alpha, n_order)
        aw = angular_weights(agrid, n_order)
        radial = torch.stack([_convolutions(rw, rgrid2, gaussian_width([z]), order, n_level, step_r)
                              for z in elements], dim=1)
        angular = torch.stack([_convolutions(aw, agrid2, gaussian_width(p) if self.angular_sigma is None
                                             else self.angular_sigma, order, n_level, step_a)
                               for p in pairs], dim=1)
        # radial [n_level, n_elem, n_r, n_feature_r], angular [n_level, n_pair, n_a, n_feature_a]
        self.register_buffer('radial', radial, persistent=False)
        self.register_buffer('angular', angular, persistent=False)
        self.register_buffer('a_scaling', torch.tensor([charge_scaling(z) for z in elements], dtype=f64),
                             persistent=False)
        self.register_buffer('pair_index', pair_index, persistent=False)
        self.rstep, self.astep = step_r, step_a

    @property
    def n_feature(self):
        return self.radial.shape[-1] + self.angular.shape[-1]

    def compute(self, atomic_index, positions):
        '''
        Descriptor of one structure.
        atomic_index: [n_atoms], element index, -1 for no-use atoms
        positions:    [n_atoms, 3], Angstrom
        Return [n_atoms, n_feature]
        '''
        return self.batch_compute(atomic_index.unsqueeze(0), positions.unsqueeze(0))[0]

    def batch_compute(self, atomic_index, positions):
        '''
        Descriptor of a batch, eq. 31.
        atomic_index: [n_structure, n_atoms], element index, -1 for no-use atoms
        positions:    [n_structure, n_atoms, 3], Angstrom
        Return [n_structure, n_atoms, n_feature], zero on no-use atoms and
        differentiable in positions up to second order (backward and forward mode).
        '''
        n_structure, n_atoms = atomic_index.shape
        element = atomic_index.flatten().clamp(min=0)  # no-use atoms are never in a pair
        a_scale = self.a_scaling[element]
        pos = positions.reshape(-1, 3)

        # Pairs i < j inside the cut-off, as flat indices into the batch
        index = create_double_index_batch(atomic_index, positions, self.cut_off)
        vector = pos[index[1]] - pos[index[0]]
        distance = vector.norm(2, -1)

        # Two-body, eq. 8: P2[i] = sum_j A(Z_j) H2[Z_j](R_ij); a pair adds to
        # both atoms, each with the other atom's prefactor and grid
        to_first = a_scale[index[1], None]*_GridLookup.apply(distance, element[index[1]], self.radial, self.rstep, 0)
        to_second = a_scale[index[0], None]*_GridLookup.apply(distance, element[index[0]], self.radial, self.rstep, 0)
        radial_aev = pos.new_zeros((n_structure*n_atoms, to_first.shape[-1]))
        radial_aev = radial_aev.index_add(0, index[0], to_first).index_add(0, index[1], to_second)

        # Three-body, eq. 15: P3[i] = sum_{j != k} A3(Z_j, Z_k)/(R_ij R_ik R_jk)^2 H3(theta_jik),
        # twice the sum over the pairs of neighbours that create_triple_index lists
        angular_aev = pos.new_zeros((n_structure*n_atoms, self.angular.shape[-1]))
        if index.shape[1] > 1:
            central_atom, pair_index, flip = create_triple_index(index)
            pair_vector = vector.index_select(0, pair_index.view(-1)).view(2, -1, 3)
            pair_vector = pair_vector*torch.where(flip, -1.0, 1.0).unsqueeze(-1)  # from the central atom
            other_atom = torch.where(flip, index[0][pair_index], index[1][pair_index])
            rij2, rik2 = (pair_vector*pair_vector).sum(-1)
            vjk = pair_vector[1] - pair_vector[0]
            rjk2 = (vjk*vjk).sum(-1)
            cosine = ((pair_vector[0]*pair_vector[1]).sum(-1)/torch.sqrt(rij2*rik2)).clamp(-1.0, 1.0)
            theta = _SafeArccos.apply(cosine, 1e-8)
            weight = 2.0*torch.sqrt(a_scale[other_atom[0]]*a_scale[other_atom[1]])/(rij2*rik2*rjk2)
            group = self.pair_index[element[other_atom[0]], element[other_atom[1]]]
            angular_term = _GridLookup.apply(theta, group, self.angular, self.astep, 0)
            angular_aev = angular_aev.index_add(0, central_atom, weight[:, None]*angular_term)
        return torch.cat([radial_aev, angular_aev], dim=-1).view(n_structure, n_atoms, -1)

    def forward(self, atomic_index, positions):
        return self.batch_compute(atomic_index, positions)
