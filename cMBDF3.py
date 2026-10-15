'''
cMBDF: convolutional many-body distribution functional representation, in PyTorch.

A port of cMBDF2.py from numpy/numba to torch, computing the same descriptor:
    D. Khan and O. A. von Lilienfeld,
    "Generalized convolutional many body distribution functional representations",
    arXiv:2409.20471v2 [physics.chem-ph]; PNAS 122(41), e2415662122 (2025).
Equation numbers refer to the arXiv v2 preprint, and every choice documented in
cMBDF2.py (charge scaling, element resolved widths, ATM scaling, odd harmonics,
no smooth cutoff, the two ambiguities) is kept unchanged.

What is different from cMBDF2.py:
  - Molecules are computed in padded batches, atomic_numbers [B, N] masked
    with -1 and coords [B, N, 3], the layout kernel.py uses, on CPU or GPU.
  - The descriptor is differentiable with autograd.  The H functions are still
    looked up on a grid, which on its own is piecewise constant, so the lookup
    is a custom autograd function whose derivative is the grid of the next
    Hermite degree, dH^{nm}/dx = -H^{n(m+1)} (eq. 31).  That is the same
    analytic rule cMBDF2.py uses for its explicit gradients, so
    torch.autograd.grad through the descriptor reproduces cMBDF2's dmat, and
    forces of any model built on it come out of loss.backward() directly.
  - Grids are kept for n_extra Hermite degrees beyond the descriptor's own
    (default 2), so second derivatives are analytic as well.  Past the last
    stored degree the derivative is zero.
  - The three-body sum runs over ordered pairs j != k directly instead of
    j < k with a factor of 2; the two are the same sum.
'''

import math

import torch

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
    '''
    A(Z) = log(P + 1) G, eq. 26, with P the period and G the group number.
    '''
    z = int(z)
    if z not in ELEMENT_DATA:
        raise KeyError('no period/group/vdW data for Z = %d; extend ELEMENT_DATA' % z)
    period, group, _ = ELEMENT_DATA[z]
    return math.log(period + 1.0)*group


def many_body_scaling(z_list):
    '''
    A_nu(Z_1..Z_nu-1) = (prod_j A(Z_j))^(1/(nu-1)), eq. 27.
    The product runs over the nu-1 neighbours only; the central atom is not
    included.
    '''
    return math.prod(charge_scaling(z) for z in z_list)**(1.0/len(z_list))


def gaussian_width(z_list):
    '''
    sigma(Z_1..Z_nu-1) = sum_j Z_j rvdW(Z_j) / sum_j Z_j, eq. 28.
    '''
    return (sum(float(z)*ELEMENT_DATA[int(z)][2] for z in z_list)
            / sum(float(z) for z in z_list))


def next_fast_len(n):
    '''
    Smallest integer >= n with no prime factor above 11, as scipy.fft.next_fast_len,
    so the grids come out the same size as in cMBDF2.py.
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


def radial_weighting(n, r, alpha=1.5, kind=0):
    '''
    Two-body weighting functions g^2_n(r), eq. 24.
        kind 0: exp(-alpha (n+1) r)
        kind 1: 1/(r+1)^(2n+3)
    '''
    if kind == 0:
        return torch.exp(-alpha*(n + 1.0)*r)
    return 1.0/((r + 1.0)**(2*n + 3))


def angular_weighting(n, theta, kind=0):
    '''
    Three-body weighting functions g^3_n(theta), eq. 25.
        kind 0: cos((2n+1)t) - cos((2n+1)(t+pi))
        kind 1: sin((2n+1)t) - sin((2n+1)(t+pi))
    Both are written out as in the paper.  Because 2n+1 is odd they reduce to
    2cos((2n+1)t) and 2sin((2n+1)t); even harmonics would vanish identically,
    which is why only odd ones appear.
    '''
    k = 2*n + 1
    if kind == 0:
        return torch.cos(k*theta) - torch.cos(k*(theta + math.pi))
    return torch.sin(k*theta) - torch.sin(k*(theta + math.pi))


def hermite(x, m):
    '''
    Physicists' Hermite polynomial H_m(x) by the three-term recurrence
        H_0 = 1,  H_1 = 2x,  H_{k+1} = 2x H_k - 2k H_{k-1}
    (numpy.polynomial.hermite.hermval in cMBDF2.py).
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
    m-th derivative of the normal PDF N(0, sigma) with respect to its argument,
    i.e. the function f_m of eq. 7 evaluated at u = r - R_ij.

    d^m/du^m N(0,sigma)(u)
        = (-1)^m / (sqrt(2 pi) sigma (sqrt(2) sigma)^m) H_m(u/(sqrt(2) sigma))
          exp(-u^2 / (2 sigma^2))
    '''
    scale = math.sqrt(2.0)*sigma
    x = u/scale
    prefactor = ((-1.0)**m)/(math.sqrt(2.0*math.pi)*sigma*scale**m)
    return prefactor*hermite(x, m)*torch.exp(-x*x)


def _correlate_on_grid(weight, derivative, step, n_grid):
    '''
    Evaluate  K(R) = integral dt weight(t) derivative(t - R)  for every R on the
    same grid the weighting function is sampled on.

    weight:     (n_grid,)   sampled at t = 0, step, .. (n_grid-1) step
    derivative: (2 n_grid,) sampled at u = -n_grid step .. (n_grid-1) step

    This is the cross-correlation of eq. 4 / eq. 13, done as a zero-padded FFT
    convolution with the reversed derivative (scipy's fftconvolve in cMBDF2.py).
    '''
    n = weight.shape[-1] + derivative.shape[-1] - 1
    full = torch.fft.irfft(torch.fft.rfft(weight, n)*torch.fft.rfft(derivative.flip(-1), n), n)
    return step*full[n_grid - 1:2*n_grid - 1]


def _circular_correlate(weight, derivative):
    '''
    Periodic counterpart of _correlate_on_grid, for angular_mode='circular'
    (the circular convolution of eq. 16).  Both inputs are sampled on one full
    period.
    '''
    n = weight.shape[-1]
    spectrum = torch.fft.rfft(weight)*torch.conj(torch.fft.rfft(derivative))
    return torch.fft.irfft(spectrum, n)


class Convolutions:
    '''
    Pre-evaluated H functions of eqs. 10 and 16, stored on fixed grids.

    Because sigma depends on the chemical identity (eq. 28) there is one radial
    grid per element and one angular grid per unordered element pair.

    Attributes:
        elements    [n_elem] sorted nuclear charges covered
        rtables     [n_extra+1, n_elem, n_feature_r, n_r]  H^{nm}_2 per element;
                    rtables[s] is the same at Hermite degree m+s, used for the
                    s-th derivative through eq. 32
        atables     [n_extra+1, n_pair, n_feature_a, n_a]  H^{nm}_3 per element pair
        a_scaling   [n_elem] A(Z) of eq. 26
        pair_index  [n_elem, n_elem] index into the pair axis
    rconvs, rconvs_next, aconvs, aconvs_next give rtables[0], rtables[1],
    atables[0], atables[1] under cMBDF2.py's names.
    '''

    def __init__(self, elements, rcut, order, n_order, alpha, rstep, astep,
                 angular_mode, angular_sigma, normalize, sigma_scale, n_extra=2,
                 dtype=torch.float64, device=None):
        self.elements = torch.tensor(sorted(int(z) for z in elements), dtype=torch.long)
        self.rcut = float(rcut)
        self.order = int(order)
        self.n_order = int(n_order)
        self.alpha = float(alpha)
        self.angular_mode = angular_mode
        self.angular_sigma = angular_sigma
        self.normalize = bool(normalize)
        self.sigma_scale = float(sigma_scale)
        self.n_extra = int(n_extra)
        self._build(rstep, astep)
        self.to(device=device, dtype=dtype)

    def _build(self, rstep, astep):
        # tables are always built in float64 on the CPU, then moved by to()
        f64 = torch.float64
        elements = self.elements.tolist()
        n_elem = len(elements)
        self.a_scaling = torch.tensor([charge_scaling(z) for z in elements], dtype=f64)

        self.pair_index = torch.zeros((n_elem, n_elem), dtype=torch.long)
        pairs = []
        for i in range(n_elem):
            for j in range(i, n_elem):
                self.pair_index[i, j] = len(pairs)
                self.pair_index[j, i] = len(pairs)
                pairs.append((elements[i], elements[j]))

        # radial grids: r in [0, rcut), offsets u in [-rcut, rcut)
        n_r = next_fast_len(int(round(self.rcut/rstep)))
        self.rstep = self.rcut/n_r
        r_grid = torch.arange(n_r, dtype=f64)*self.rstep
        u_grid = (torch.arange(2*n_r, dtype=f64) - n_r)*self.rstep

        # angular grid: theta in [0, pi] inclusive, so theta = pi lands on the
        # last point exactly and no index can run past the end
        n_a = next_fast_len(int(round(math.pi/astep))) + 1
        self.astep = math.pi/(n_a - 1)
        theta_grid = torch.arange(n_a, dtype=f64)*self.astep
        theta_offsets = (torch.arange(2*n_a, dtype=f64) - n_a)*self.astep

        n_level = self.n_extra + 1
        n_deriv = self.order + n_level    # extra degrees for eq. 32 and beyond
        n_weight = 2*self.n_order
        n_feature = n_weight*(self.order + 1)

        self.rtables = torch.zeros((n_level, n_elem, n_feature, n_r), dtype=f64)
        for e, z in enumerate(elements):
            sigma = self.sigma_scale*gaussian_width([z])
            derivatives = [gaussian_derivative(u_grid, sigma, m) for m in range(n_deriv)]
            stack = torch.zeros((n_weight, n_deriv, n_r), dtype=f64)
            for kind in range(2):
                for n in range(self.n_order):
                    weight = radial_weighting(n, r_grid, self.alpha, kind)
                    for m in range(n_deriv):
                        stack[kind*self.n_order + n, m] = _correlate_on_grid(
                            weight, derivatives[m], self.rstep, n_r)
            for s in range(n_level):
                self.rtables[s, e] = self._flatten(stack, s)

        self.atables = torch.zeros((n_level, len(pairs), n_feature, n_a), dtype=f64)
        for p, (z1, z2) in enumerate(pairs):
            if self.angular_sigma is None:
                sigma = self.sigma_scale*gaussian_width([z1, z2])
            else:
                sigma = float(self.angular_sigma)
            stack = torch.zeros((n_weight, n_deriv, n_a), dtype=f64)
            if self.angular_mode == 'circular':
                period = torch.cat((theta_grid, theta_grid[1:-1].flip(0)))
                offsets = (torch.arange(len(period), dtype=f64) - len(period)//2)*self.astep
                derivatives = [gaussian_derivative(offsets, sigma, m) for m in range(n_deriv)]
            else:
                derivatives = [gaussian_derivative(theta_offsets, sigma, m) for m in range(n_deriv)]
            for kind in range(2):
                for n in range(self.n_order):
                    weight = angular_weighting(n, theta_grid, kind)
                    for m in range(n_deriv):
                        if self.angular_mode == 'circular':
                            extended = torch.cat((weight, weight[1:-1].flip(0)))
                            rolled = torch.roll(derivatives[m], -(len(extended)//2))
                            stack[kind*self.n_order + n, m] = self.astep*_circular_correlate(
                                extended, rolled)[:n_a]
                        else:
                            stack[kind*self.n_order + n, m] = _correlate_on_grid(
                                weight, derivatives[m], self.astep, n_a)
            for s in range(n_level):
                self.atables[s, p] = self._flatten(stack, s)

        if self.normalize:
            # each channel is divided by its own maximum, and every derivative
            # level by that same constant, or eq. 32 no longer differentiates
            # what eq. 9 sums
            for tables in (self.rtables, self.atables):
                scale = tables[0].abs().amax(dim=-1, keepdim=True)
                scale[scale == 0.0] = 1.0
                tables /= scale

        self._build_lookup()

    def _build_lookup(self):
        # nuclear charge -> element index, -1 for anything not covered
        self.z_to_index = torch.full((int(self.elements.max()) + 1,), -1, dtype=torch.long)
        self.z_to_index[self.elements] = torch.arange(len(self.elements))

    def _flatten(self, stack, shift):
        '''
        Collapse [weighting function, derivative order] into the single feature
        axis of eq. 29, ordered kind-major then n then m.  shift=s selects
        degree m+s throughout.
        '''
        n_weight, _, n_grid = stack.shape
        return stack[:, shift:shift + self.order + 1].reshape(n_weight*(self.order + 1), n_grid)

    def to(self, device=None, dtype=None):
        for name in ('rtables', 'atables', 'a_scaling'):
            setattr(self, name, getattr(self, name).to(device=device, dtype=dtype))
        for name in ('pair_index', 'z_to_index', 'elements'):
            setattr(self, name, getattr(self, name).to(device=device))
        return self

    @property
    def rconvs(self):
        return self.rtables[0]

    @property
    def rconvs_next(self):
        return self.rtables[1]

    @property
    def aconvs(self):
        return self.atables[0]

    @property
    def aconvs_next(self):
        return self.atables[1]

    @property
    def n_feature(self):
        return self.rtables.shape[2] + self.atables.shape[2]

    def element_index(self, atomic_numbers):
        '''
        Map nuclear charges to element indices.  Padding (-1) maps to 0, which is
        harmless because padded atoms are masked out everywhere they are used.
        '''
        z = atomic_numbers.to(self.z_to_index.device)
        real = z >= 0
        too_large = real & (z >= len(self.z_to_index))
        index = self.z_to_index[z.clamp(0, len(self.z_to_index) - 1)]
        missing = real & ((index < 0) | too_large)
        if missing.any():
            raise KeyError('elements %s not covered by these convolutions (have %s)'
                           % (z[missing].unique().tolist(), self.elements.tolist()))
        return torch.where(real, index, torch.zeros_like(index))

    def save(self, file_name):
        torch.save({'elements': self.elements.cpu(), 'rcut': self.rcut, 'order': self.order,
                    'n_order': self.n_order, 'alpha': self.alpha, 'rstep': self.rstep,
                    'astep': self.astep, 'normalize': self.normalize,
                    'angular_mode': self.angular_mode, 'angular_sigma': self.angular_sigma,
                    'sigma_scale': self.sigma_scale, 'n_extra': self.n_extra,
                    'rtables': self.rtables.cpu(), 'atables': self.atables.cpu(),
                    'a_scaling': self.a_scaling.cpu(), 'pair_index': self.pair_index.cpu()},
                   file_name)

    @classmethod
    def load(cls, file_name, dtype=torch.float64, device=None):
        data = torch.load(file_name)
        self = cls.__new__(cls)
        for key in ('elements', 'rcut', 'order', 'n_order', 'alpha', 'rstep', 'astep',
                    'normalize', 'angular_mode', 'angular_sigma', 'sigma_scale', 'n_extra',
                    'rtables', 'atables', 'a_scaling', 'pair_index'):
            setattr(self, key, data[key])
        self._build_lookup()
        return self.to(device=device, dtype=dtype)


def get_convolutions(elements, rcut=10.0, order=4, n_order=2, alpha=1.5,
                     rstep=0.0008, astep=0.0002, angular_mode='linear',
                     angular_sigma=None, normalize=False, sigma_scale=1.0,
                     n_extra=2, dtype=torch.float64, device=None):
    '''
    Pre-evaluate the H functions of eqs. 10 and 16 on fixed grids.

    elements: nuclear charges that will be encountered (the grids are element
    resolved, eq. 28).

    order    m, the derivative order          -> a factor (order+1) per block
    n_order  n, the weighting function order  -> 2 n channels per block
    n_extra  Hermite degrees kept beyond the descriptor's own, i.e. how many
             orders of positional derivative are analytic (1 = forces only)
    Feature vector length is 2 (nu-1) (order+1) n_order = 40 at the paper's
    nu = 3, m = 4, n = 2.
    '''
    return Convolutions(elements, rcut, order, n_order, alpha, rstep, astep,
                        angular_mode, angular_sigma, normalize, sigma_scale,
                        n_extra, dtype, device)


class _GridLookup(torch.autograd.Function):
    '''
    H(x) read from tables[level][group, :, nearest grid point of x].

    Forward is the nearest-point lookup of cMBDF2.py.  Backward is
    dH^{nm}/dx = -H^{n(m+1)} (eq. 31), i.e. the same lookup one level up, so
    derivatives of every order up to n_extra are analytic.  The lookup itself
    is not differentiable with respect to the tables.
    '''
    generate_vmap_rule = True

    @staticmethod
    def forward(x, group, tables, level, step):
        index = torch.floor(x/step + 0.5).long().clamp(0, tables.shape[-1] - 1)
        return tables[level][group, :, index]

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, group, tables, level, step = inputs
        ctx.save_for_backward(x, group, tables)
        ctx.level = level
        ctx.step = step

    @staticmethod
    def backward(ctx, grad_output):
        x, group, tables = ctx.saved_tensors
        if ctx.level + 1 >= tables.shape[0]:
            return torch.zeros_like(x), None, None, None, None
        h_next = _GridLookup.apply(x, group, tables, ctx.level + 1, ctx.step)
        return -(grad_output*h_next).sum(-1), None, None, None, None


class _SafeArccos(torch.autograd.Function):
    '''
    arccos of a cosine clamped to [-1, 1], with d theta / d cos =
    -1/max(sin theta, epsilon): the epsilon guard of cMBDF2.py, so collinear
    triplets give a large but finite gradient instead of inf or NaN.
    '''
    generate_vmap_rule = True

    @staticmethod
    def forward(cosine, epsilon):
        return torch.arccos(cosine.clamp(-1.0, 1.0))

    @staticmethod
    def setup_context(ctx, inputs, output):
        cosine, epsilon = inputs
        ctx.save_for_backward(cosine)
        ctx.epsilon = epsilon

    @staticmethod
    def backward(ctx, grad_output):
        cosine, = ctx.saved_tensors
        clamped = cosine.clamp(-1.0, 1.0)
        # clamp sin^2 before the square root, not sin after it: sqrt has an
        # infinite slope at 0, which would put 0*inf = NaN into second
        # derivatives at collinear (and masked j = k) triplets
        sine = torch.sqrt((1.0 - clamped*clamped).clamp_min(ctx.epsilon**2))
        return -grad_output/sine, None


def cmbdf_batch(atomic_numbers, coords, convs, epsilon=1e-8):
    '''
    Atomic cMBDF representation of a padded batch of molecules, eq. 29.

    Input:
        atomic_numbers: [B, N] masked with -1
        coords:         [B, N, 3], padded rows are ignored
        convs:          Convolutions from get_convolutions
    Output:
        [B, N, n_feature], zero on padded atoms, differentiable with respect to
        coords.

    Memory goes as B N^3 n_feature_a for the three-body term; generate_cmbdf
    splits large sets into batches.
    '''
    dtype = convs.rtables.dtype
    device = convs.rtables.device
    atomic_numbers = atomic_numbers.to(device)
    coords = coords.to(device=device, dtype=dtype)
    n_atom = atomic_numbers.shape[1]

    real = atomic_numbers >= 0
    elem = convs.element_index(atomic_numbers)                  # [B, N]
    a_scale = convs.a_scaling[elem]*real                        # A(Z_j), 0 on padding

    # v[b, i, j] = R_j - R_i, distances made safe (1.0) wherever the pair is not
    # used, so no 0/0 reaches the gradient through masked entries
    v = coords[:, None, :, :] - coords[:, :, None, :]
    d2 = (v*v).sum(-1)
    not_self = ~torch.eye(n_atom, dtype=torch.bool, device=device)
    pair = real[:, :, None] & real[:, None, :] & not_self & (d2 > 0.0)
    d2_safe = torch.where(pair, d2, torch.ones_like(d2))
    dist = torch.sqrt(d2_safe)
    in_cut = pair & (dist < convs.rcut)

    # two-body, eqs. 4-9: P2[i] = sum_j A(Z_j) H2[Z_j](R_ij); grid and prefactor
    # both belong to the neighbour j
    group = elem[:, None, :].expand(-1, n_atom, -1)
    h2 = _GridLookup.apply(dist, group, convs.rtables, 0, convs.rstep)   # [B, N, N, Fr]
    weight2 = a_scale[:, None, :]*in_cut
    p2 = (weight2[..., None]*h2).sum(2)

    # three-body, eq. 15: P3[i] = sum_{j != k} A_3(Z_j, Z_k)/(R_ij R_ik R_jk)^2
    # H3[Z_j, Z_k](theta_jik), theta centred on i
    dot = torch.einsum('bijc,bikc->bijk', v, v)
    cosine = dot/(dist[:, :, :, None]*dist[:, :, None, :])
    theta = _SafeArccos.apply(cosine, epsilon)
    triplet = in_cut[:, :, :, None] & in_cut[:, :, None, :] & pair[:, None, :, :]
    atm = 1.0/(d2_safe[:, :, :, None]*d2_safe[:, :, None, :]*d2_safe[:, None, :, :])
    scaling = torch.sqrt(a_scale[:, :, None]*a_scale[:, None, :])     # [B, j, k]
    weight3 = scaling[:, None, :, :]*atm*triplet
    group3 = convs.pair_index[elem[:, :, None], elem[:, None, :]]      # [B, j, k]
    group3 = group3[:, None, :, :].expand(-1, n_atom, -1, -1)
    h3 = _GridLookup.apply(theta, group3, convs.atables, 0, convs.astep)  # [B, N, N, N, Fa]
    p3 = (weight3[..., None]*h3).sum((2, 3))

    return torch.cat((p2, p3), dim=-1)*real[..., None]


def get_cmbdf(charges, coords, convs, pad=None, gradients=False, epsilon=1e-8):
    '''
    Atomic cMBDF representation of one molecule, as cMBDF2.get_cmbdf.

    Input:
        charges: [Natom]
        coords:  [Natom, 3]
        convs:   Convolutions from get_convolutions
    Output:
        [pad, n_feature], and with gradients=True also [pad, n_feature, pad, 3]
        holding d P[i, f] / d R[a, c], from autograd.
    '''
    charges = torch.as_tensor(charges).long()
    coords = torch.as_tensor(coords, dtype=convs.rtables.dtype, device=convs.rtables.device)
    size = charges.shape[0]
    if pad is None:
        pad = size
    if pad < size:
        raise ValueError('pad = %d is smaller than the molecule (%d atoms)' % (pad, size))

    def representation(r):
        return cmbdf_batch(charges[None], r[None], convs, epsilon)[0]

    mat = coords.new_zeros((pad, convs.n_feature))
    mat[:size] = representation(coords).detach()
    if not gradients:
        return mat
    dmat = coords.new_zeros((pad, convs.n_feature, pad, 3))
    dmat[:size, :, :size] = torch.autograd.functional.jacobian(representation, coords,
                                                               vectorize=True)
    return mat, dmat


def pad_molecules(nuclear_charges, coords, pad=None):
    '''
    Ragged lists of molecules -> atomic_numbers [B, N] (-1 padded) and
    coords [B, N, 3] (zero padded), the layout of cmbdf_batch and kernel.py.
    '''
    sizes = [len(q) for q in nuclear_charges]
    if pad is None:
        pad = max(sizes)
    atomic_numbers = torch.full((len(sizes), pad), -1, dtype=torch.long)
    positions = torch.zeros((len(sizes), pad, 3), dtype=torch.float64)
    for b, (q, r) in enumerate(zip(nuclear_charges, coords)):
        q = torch.as_tensor(q).long()
        r = torch.as_tensor(r, dtype=torch.float64)
        if q.shape[0] != r.shape[0]:
            raise ValueError('charges and coordinates length mismatch for molecule %d' % b)
        atomic_numbers[b, :len(q)] = q
        positions[b, :len(q)] = r
    return atomic_numbers, positions


def generate_cmbdf(nuclear_charges, coords, convs=None, pad=None, gradients=False,
                   batch_size=64, progress_bar=False, **kwargs):
    '''
    cMBDF for a list of molecules.

    convs=None builds the convolutions from the elements found in the data
    (kwargs go to get_convolutions).  pad=None returns one tensor per molecule
    at its own atom count; pass an integer for a single padded tensor.
    batch_size molecules are computed together, padded to the largest of them.
    '''
    charges = [torch.as_tensor(q).long() for q in nuclear_charges]
    if convs is None:
        convs = get_convolutions(torch.cat(charges).unique().tolist(), **kwargs)

    if gradients:
        iterator = zip(charges, coords)
        if progress_bar:
            from tqdm import tqdm
            iterator = tqdm(list(iterator))
        results = [get_cmbdf(q, r, convs, pad=pad, gradients=True) for q, r in iterator]
        reps = [item[0] for item in results]
        grads = [item[1] for item in results]
        if pad is None:
            return reps, grads
        return torch.stack(reps), torch.stack(grads)

    starts = range(0, len(charges), batch_size)
    if progress_bar:
        from tqdm import tqdm
        starts = tqdm(starts)
    reps = []
    for s in starts:
        z, r = pad_molecules(charges[s:s + batch_size], coords[s:s + batch_size])
        with torch.no_grad():
            out = cmbdf_batch(z, r, convs)
        for b, q in enumerate(charges[s:s + batch_size]):
            reps.append(out[b, :len(q)])
    if pad is None:
        return reps
    mat = reps[0].new_zeros((len(reps), pad, convs.n_feature))
    for b, rep in enumerate(reps):
        mat[b, :rep.shape[0]] = rep
    return mat
