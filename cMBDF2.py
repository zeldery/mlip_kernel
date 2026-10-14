'''
cMBDF: convolutional many-body distribution functional representation.

Implemented directly from
    D. Khan and O. A. von Lilienfeld,
    "Generalized convolutional many body distribution functional representations",
    arXiv:2409.20471v2 [physics.chem-ph]; PNAS 122(41), e2415662122 (2025).

All equation numbers in the comments refer to the arXiv v2 preprint.  Where the
reference implementation in cMBDF.py disagrees with the paper, the paper is
followed here.  The deviations that are corrected, relative to cMBDF.py:

  eqs. 26, 27  charge scaling A(Z) = log(P+1)G over the nu-1 *neighbours*,
               not sqrt(Z_i Z_j) / cbrt(Z_i Z_j Z_k) over all nu atoms.
  eqs. 1,3,28  element specific Gaussian widths sigma(Z) from van-der-Waals
               radii, so the convolution grid is per element (2-body) and per
               element pair (3-body) instead of one shared grid.
  eq. 15       three-body Axilrod-Teller-Muto scaling 1/(Rij Rik Rjk)^2.
               cMBDF.py uses this in generate_data but replaces it with an
               exponential in generate_data_with_gradients.
  eq. 25       angular weighting functions are the cos *and* sin types at odd
               harmonics (2n+1), not cos(n*theta) for n = 1..nAs.
  eqs. 31, 32  gradients raise the Hermite degree by one analytically, instead
               of finite-differencing the convolution grid with np.gradient.
  eqs. 5, 7    Hermite polynomials from the exact m-th derivative of the
               Gaussian at any degree; cMBDF.py's hard-coded table has wrong
               constant terms at even degrees 2 and 4 and stops at degree 5.

Deliberate deviations from cMBDF.py that are *not* in the paper, and are
therefore dropped here: the smooth cutoff 0.5(cos(pi r/rcut)+1) applied in
generate_data_with_gradients, the cos2*cos3 factor on the first angular
channel, and the per-channel max-normalisation of the convolution grids
(available as normalize=True, off by default).

Scope: many-body orders nu = 2 and 3, i.e. the 40-dimensional "cMBDF" of the
paper at m = 4, n = 2.  The pseudo four-body functionals of eqs. 17-22 are not
implemented here.

Two points where the paper is ambiguous, and what is done about it:
  - eq. 28 defines sigma from van-der-Waals radii "alongside all distributions
    (eqs. 3, 12, 18)", so the *angular* Gaussian of eq. 12 also takes a width in
    Angstrom units applied to an argument in radians.  That is implemented as
    written; it makes the angular Gaussian very broad.  Pass angular_sigma to
    override it with an explicit width in radians.
  - eq. 13 defines the 3-body functional as an integral over [0, pi] while
    eq. 16 calls the same object a circular convolution.  The definition is
    followed (angular_mode='linear'); 'circular' is available.
'''

import numpy as np
import numba as nb
from numpy.polynomial.hermite import hermval
from scipy.signal import fftconvolve
from scipy.fft import next_fast_len

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
    return np.log(period + 1.0)*group


def many_body_scaling(z_list):
    '''
    A_nu(Z_1..Z_nu-1) = (prod_j A(Z_j))^(1/(nu-1)), eq. 27.
    The product runs over the nu-1 neighbours only; the central atom is not
    included.
    '''
    values = np.array([charge_scaling(z) for z in z_list])
    return float(np.prod(values)**(1.0/len(values)))


def gaussian_width(z_list):
    '''
    sigma(Z_1..Z_nu-1) = sum_j Z_j rvdW(Z_j) / sum_j Z_j, eq. 28.
    '''
    z_array = np.array([float(z) for z in z_list])
    radii = np.array([ELEMENT_DATA[int(z)][2] for z in z_list])
    return float((z_array*radii).sum()/z_array.sum())


def radial_weighting(n, r, alpha=1.5, kind=0):
    '''
    Two-body weighting functions g^2_n(r), eq. 24.
        kind 0: exp(-alpha (n+1) r)
        kind 1: 1/(r+1)^(2n+3)
    '''
    if kind == 0:
        return np.exp(-alpha*(n + 1.0)*r)
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
        return np.cos(k*theta) - np.cos(k*(theta + np.pi))
    return np.sin(k*theta) - np.sin(k*(theta + np.pi))


def gaussian_derivative(u, sigma, m):
    '''
    m-th derivative of the normal PDF N(0, sigma) with respect to its argument,
    i.e. the function f_m of eq. 7 evaluated at u = r - R_ij.

    d^m/du^m N(0,sigma)(u)
        = (-1)^m / (sqrt(2 pi) sigma (sqrt(2) sigma)^m) H_m(u/(sqrt(2) sigma))
          exp(-u^2 / (2 sigma^2))

    with H_m the physicists' Hermite polynomial, evaluated here at arbitrary
    degree rather than from a hard-coded table.
    '''
    scale = np.sqrt(2.0)*sigma
    x = u/scale
    coef = np.zeros(m + 1)
    coef[m] = 1.0
    prefactor = ((-1.0)**m)/(np.sqrt(2.0*np.pi)*sigma*scale**m)
    return prefactor*hermval(x, coef)*np.exp(-x*x)


def _correlate_on_grid(weight, derivative, step, n_grid):
    '''
    Evaluate  K(R) = integral dt weight(t) derivative(t - R)  for every R on the
    same grid the weighting function is sampled on.

    weight:     (n_grid,)   sampled at t = 0, step, .. (n_grid-1) step
    derivative: (2 n_grid,) sampled at u = -n_grid step .. (n_grid-1) step

    This is the cross-correlation of eq. 4 / eq. 13.  The paper writes it as a
    convolution (eqs. 8, 16); the two agree up to the (-1)^m that eqs. 9 and 15
    carry in front, because Gaussian derivatives have definite parity.  Taking
    the correlation directly gives the functional value with no sign bookkeeping.
    '''
    full = fftconvolve(weight, derivative[::-1])
    return step*full[n_grid - 1:2*n_grid - 1]


def _circular_correlate(weight, derivative):
    '''
    Periodic counterpart of _correlate_on_grid, for angular_mode='circular'
    (the circular convolution of eq. 16).  Both inputs are sampled on one full
    period.
    '''
    n = len(weight)
    spectrum = np.fft.rfft(weight)*np.conjugate(np.fft.rfft(derivative))
    return np.fft.irfft(spectrum, n)


class Convolutions:
    '''
    Pre-evaluated H functions of eqs. 10 and 16, stored on fixed grids.

    Because sigma depends on the chemical identity (eq. 28) there is one radial
    grid per element and one angular grid per unordered element pair, rather
    than the single shared grid of cMBDF.py.

    Attributes:
        elements    [n_elem] sorted nuclear charges covered
        rconvs      [n_elem, n_feature_r, n_r]  H^{nm}_2 per element
        rconvs_next [n_elem, n_feature_r, n_r]  the same at Hermite degree m+1,
                    used for gradients through eq. 32
        aconvs      [n_pair, n_feature_a, n_a]  H^{nm}_3 per element pair
        aconvs_next [n_pair, n_feature_a, n_a]
        a_scaling   [n_elem] A(Z) of eq. 26
        pair_index  [n_elem, n_elem] index into the pair axis
    '''

    def __init__(self, elements, rcut, order, n_order, alpha, rstep, astep,
                 angular_mode, angular_sigma, normalize, sigma_scale):
        self.elements = np.asarray(sorted(int(z) for z in elements), dtype=np.int64)
        self.rcut = float(rcut)
        self.order = int(order)
        self.n_order = int(n_order)
        self.alpha = float(alpha)
        self.angular_mode = angular_mode
        self.angular_sigma = angular_sigma
        self.normalize = bool(normalize)
        self.sigma_scale = float(sigma_scale)
        self._build(rstep, astep)

    def _build(self, rstep, astep):
        n_elem = len(self.elements)
        index_of = {int(z): i for i, z in enumerate(self.elements)}
        self.a_scaling = np.array([charge_scaling(z) for z in self.elements])

        self.pair_index = np.zeros((n_elem, n_elem), dtype=np.int64)
        pairs = []
        for i in range(n_elem):
            for j in range(i, n_elem):
                self.pair_index[i, j] = len(pairs)
                self.pair_index[j, i] = len(pairs)
                pairs.append((self.elements[i], self.elements[j]))

        # radial grids: r in [0, rcut), offsets u in [-rcut, rcut)
        n_r = next_fast_len(int(round(self.rcut/rstep)))
        self.rstep = self.rcut/n_r
        self.n_r = n_r
        r_grid = np.arange(n_r)*self.rstep
        u_grid = (np.arange(2*n_r) - n_r)*self.rstep

        # angular grid: theta in [0, pi] inclusive, so theta = pi lands on the
        # last point exactly and no index can run past the end
        n_a = next_fast_len(int(round(np.pi/astep))) + 1
        self.n_a = n_a
        self.astep = np.pi/(n_a - 1)
        theta_grid = np.arange(n_a)*self.astep
        theta_offsets = (np.arange(2*n_a) - n_a)*self.astep

        n_deriv = self.order + 2          # one extra degree for eq. 32
        n_weight = 2*self.n_order

        self.rconvs = np.zeros((n_elem, n_weight*(self.order + 1), n_r))
        self.rconvs_next = np.zeros_like(self.rconvs)
        for e, z in enumerate(self.elements):
            sigma = self.sigma_scale*gaussian_width([z])
            derivatives = [gaussian_derivative(u_grid, sigma, m) for m in range(n_deriv)]
            stack = np.zeros((n_weight, n_deriv, n_r))
            for kind in range(2):
                for n in range(self.n_order):
                    weight = radial_weighting(n, r_grid, self.alpha, kind)
                    for m in range(n_deriv):
                        stack[kind*self.n_order + n, m] = _correlate_on_grid(
                            weight, derivatives[m], self.rstep, n_r)
            self.rconvs[e] = self._flatten(stack, 0)
            self.rconvs_next[e] = self._flatten(stack, 1)

        self.aconvs = np.zeros((len(pairs), n_weight*(self.order + 1), n_a))
        self.aconvs_next = np.zeros_like(self.aconvs)
        for p, (z1, z2) in enumerate(pairs):
            if self.angular_sigma is None:
                sigma = self.sigma_scale*gaussian_width([z1, z2])
            else:
                sigma = float(self.angular_sigma)
            stack = np.zeros((n_weight, n_deriv, n_a))
            if self.angular_mode == 'circular':
                period = np.concatenate((theta_grid, theta_grid[-2:0:-1]))
                offsets = (np.arange(len(period)) - len(period)//2)*self.astep
                derivatives = [gaussian_derivative(offsets, sigma, m) for m in range(n_deriv)]
            else:
                derivatives = [gaussian_derivative(theta_offsets, sigma, m) for m in range(n_deriv)]
            for kind in range(2):
                for n in range(self.n_order):
                    weight = angular_weighting(n, theta_grid, kind)
                    for m in range(n_deriv):
                        if self.angular_mode == 'circular':
                            extended = np.concatenate((weight, weight[-2:0:-1]))
                            rolled = np.roll(derivatives[m], -(len(extended)//2))
                            stack[kind*self.n_order + n, m] = self.astep*_circular_correlate(
                                extended, rolled)[:n_a]
                        else:
                            stack[kind*self.n_order + n, m] = _correlate_on_grid(
                                weight, derivatives[m], self.astep, n_a)
            self.aconvs[p] = self._flatten(stack, 0)
            self.aconvs_next[p] = self._flatten(stack, 1)

        if self.normalize:
            # each channel is divided by its own maximum, so the matching
            # derivative grid must be divided by that same constant, not by its
            # own maximum, or eq. 32 no longer differentiates what eq. 9 sums
            for array, derivative in ((self.rconvs, self.rconvs_next),
                                      (self.aconvs, self.aconvs_next)):
                scale = np.max(np.abs(array), axis=-1, keepdims=True)
                scale[scale == 0.0] = 1.0
                array /= scale
                derivative /= scale

        self.index_of = index_of

    def _flatten(self, stack, shift):
        '''
        Collapse [weighting function, derivative order] into the single feature
        axis of eq. 29, ordered kind-major then n then m.  shift=1 selects
        degree m+1 throughout, which is eq. 31's -H^{n(m+1)} without the sign.
        '''
        n_weight, _, n_grid = stack.shape
        out = np.zeros((n_weight*(self.order + 1), n_grid))
        for w in range(n_weight):
            for m in range(self.order + 1):
                out[w*(self.order + 1) + m] = stack[w, m + shift]
        return out

    @property
    def n_feature(self):
        return self.rconvs.shape[1] + self.aconvs.shape[1]

    def element_index(self, charges):
        try:
            return np.array([self.index_of[int(z)] for z in charges], dtype=np.int64)
        except KeyError as exc:
            raise KeyError('element %s not covered by these convolutions (have %s)'
                           % (exc, self.elements.tolist()))

    def save(self, file_name):
        np.savez(file_name, elements=self.elements, rcut=self.rcut, order=self.order,
                 n_order=self.n_order, alpha=self.alpha, rstep=self.rstep,
                 astep=self.astep, normalize=self.normalize,
                 angular_mode=self.angular_mode,
                 angular_sigma=(-1.0 if self.angular_sigma is None else self.angular_sigma),
                 sigma_scale=self.sigma_scale,
                 rconvs=self.rconvs, rconvs_next=self.rconvs_next,
                 aconvs=self.aconvs, aconvs_next=self.aconvs_next,
                 a_scaling=self.a_scaling, pair_index=self.pair_index)

    @classmethod
    def load(cls, file_name):
        data = np.load(file_name)
        self = cls.__new__(cls)
        self.elements = data['elements']
        self.rcut = data['rcut'].item()
        self.order = data['order'].item()
        self.n_order = data['n_order'].item()
        self.alpha = data['alpha'].item()
        self.rstep = data['rstep'].item()
        self.astep = data['astep'].item()
        self.normalize = bool(data['normalize'].item())
        self.angular_mode = str(data['angular_mode'].item())
        sigma = data['angular_sigma'].item()
        self.angular_sigma = None if sigma < 0 else sigma
        self.sigma_scale = data['sigma_scale'].item()
        self.rconvs = data['rconvs']
        self.rconvs_next = data['rconvs_next']
        self.aconvs = data['aconvs']
        self.aconvs_next = data['aconvs_next']
        self.a_scaling = data['a_scaling']
        self.pair_index = data['pair_index']
        self.n_r = self.rconvs.shape[-1]
        self.n_a = self.aconvs.shape[-1]
        self.index_of = {int(z): i for i, z in enumerate(self.elements)}
        return self


def get_convolutions(elements, rcut=10.0, order=4, n_order=2, alpha=1.5,
                     rstep=0.0008, astep=0.0002, angular_mode='linear',
                     angular_sigma=None, normalize=False, sigma_scale=1.0):
    '''
    Pre-evaluate the H functions of eqs. 10 and 16 on fixed grids.

    elements: nuclear charges that will be encountered.  Unlike cMBDF.py the
    grids are element resolved (eq. 28), so this list is required.

    order    m, the derivative order          -> a factor (order+1) per block
    n_order  n, the weighting function order  -> 2 n channels per block
    Feature vector length is 2 (nu-1) (order+1) n_order = 40 at the paper's
    nu = 3, m = 4, n = 2.
    '''
    return Convolutions(elements, rcut, order, n_order, alpha, rstep, astep,
                        angular_mode, angular_sigma, normalize, sigma_scale)


@nb.jit(nopython=True, cache=True)
def _two_body(elem_index, coords, a_scaling, rconvs, rcut, rstep):
    '''
    P^{nm}_2[i] = sum_j A(Z_j) H^{nm}_2(R_ij), eqs. 4-9.

    The sum runs over every neighbour j != i, and both the prefactor A(Z_j) and
    the grid (through sigma(Z_j)) belong to the neighbour, so the i and j
    contributions of a pair differ and are accumulated separately.
    '''
    size = len(elem_index)
    n_feature = rconvs.shape[1]
    n_grid = rconvs.shape[2]
    out = np.zeros((size, n_feature))
    for i in range(size):
        for j in range(i + 1, size):
            diff = coords[i] - coords[j]
            distance = np.sqrt(diff[0]*diff[0] + diff[1]*diff[1] + diff[2]*diff[2])
            if distance == 0.0 or distance >= rcut:
                continue
            index = int(distance/rstep + 0.5)
            if index >= n_grid:
                index = n_grid - 1
            ei = elem_index[i]
            ej = elem_index[j]
            wi = a_scaling[ei]
            wj = a_scaling[ej]
            for f in range(n_feature):
                out[i, f] += wj*rconvs[ej, f, index]
                out[j, f] += wi*rconvs[ei, f, index]
    return out


@nb.jit(nopython=True, cache=True)
def _three_body(elem_index, coords, a_scaling, pair_index, aconvs, rcut, astep):
    '''
    P^{nm}_3[i] = sum_jk A_3(Z_j,Z_k) / (R_ij R_ik R_jk)^2 H^{nm}_3(theta_ijk),
    eq. 15.

    theta_ijk is centred on i.  The sum over jk is over ordered pairs, so the
    j < k loop below contributes twice; everything in the summand is symmetric
    under j <-> k.
    '''
    size = len(elem_index)
    n_feature = aconvs.shape[1]
    n_grid = aconvs.shape[2]
    out = np.zeros((size, n_feature))
    for i in range(size):
        for j in range(size):
            if j == i:
                continue
            vij = coords[j] - coords[i]
            rij = np.sqrt(vij[0]*vij[0] + vij[1]*vij[1] + vij[2]*vij[2])
            if rij == 0.0 or rij >= rcut:
                continue
            for k in range(j + 1, size):
                if k == i:
                    continue
                vik = coords[k] - coords[i]
                rik = np.sqrt(vik[0]*vik[0] + vik[1]*vik[1] + vik[2]*vik[2])
                if rik == 0.0 or rik >= rcut:
                    continue
                vjk = coords[k] - coords[j]
                rjk = np.sqrt(vjk[0]*vjk[0] + vjk[1]*vjk[1] + vjk[2]*vjk[2])
                if rjk == 0.0:
                    continue
                cosine = (vij[0]*vik[0] + vij[1]*vik[1] + vij[2]*vik[2])/(rij*rik)
                if cosine > 1.0:
                    cosine = 1.0
                elif cosine < -1.0:
                    cosine = -1.0
                index = int(np.arccos(cosine)/astep + 0.5)
                if index >= n_grid:
                    index = n_grid - 1
                scaling = np.sqrt(a_scaling[elem_index[j]]*a_scaling[elem_index[k]])
                atm = (rij*rik*rjk)**2
                weight = 2.0*scaling/atm
                p = pair_index[elem_index[j], elem_index[k]]
                for f in range(n_feature):
                    out[i, f] += weight*aconvs[p, f, index]
    return out


@nb.jit(nopython=True, cache=True)
def _two_body_gradient(elem_index, coords, a_scaling, rconvs_next, rcut, rstep):
    '''
    grad_Ra P^{nm}_2[i] = -sum_j A(Z_j) H^{n(m+1)}_2(R_ij) grad_Ra R_ij,
    eqs. 32 and 34.  The minus sign is eq. 31's dH^{nm}/dx = -H^{n(m+1)}.
    '''
    size = len(elem_index)
    n_feature = rconvs_next.shape[1]
    n_grid = rconvs_next.shape[2]
    out = np.zeros((size, n_feature, size, 3))
    for i in range(size):
        for j in range(i + 1, size):
            diff = coords[i] - coords[j]
            distance = np.sqrt(diff[0]*diff[0] + diff[1]*diff[1] + diff[2]*diff[2])
            if distance == 0.0 or distance >= rcut:
                continue
            index = int(distance/rstep + 0.5)
            if index >= n_grid:
                index = n_grid - 1
            unit = diff/distance                     # grad_Ri R_ij
            ei = elem_index[i]
            ej = elem_index[j]
            wi = a_scaling[ei]
            wj = a_scaling[ej]
            for f in range(n_feature):
                term_i = -wj*rconvs_next[ej, f, index]
                term_j = -wi*rconvs_next[ei, f, index]
                for c in range(3):
                    out[i, f, i, c] += term_i*unit[c]
                    out[i, f, j, c] -= term_i*unit[c]
                    out[j, f, j, c] -= term_j*unit[c]
                    out[j, f, i, c] += term_j*unit[c]
    return out


@nb.jit(nopython=True, cache=True)
def _three_body_gradient(elem_index, coords, a_scaling, pair_index, aconvs,
                         aconvs_next, rcut, astep, epsilon):
    '''
    grad_Ra P^{nm}_3[i] = sum_jk A_3 [ H^{nm}_3 grad_Ra f_ijk
                                       - f_ijk H^{n(m+1)}_3 grad_Ra theta_ijk ],
    eq. 35, with f_ijk = 1/(R_ij R_ik R_jk)^2.
    '''
    size = len(elem_index)
    n_feature = aconvs.shape[1]
    n_grid = aconvs.shape[2]
    out = np.zeros((size, n_feature, size, 3))
    grad_f = np.zeros((3, 3))
    grad_theta = np.zeros((3, 3))
    for i in range(size):
        for j in range(size):
            if j == i:
                continue
            vij = coords[j] - coords[i]
            rij = np.sqrt(vij[0]*vij[0] + vij[1]*vij[1] + vij[2]*vij[2])
            if rij == 0.0 or rij >= rcut:
                continue
            for k in range(j + 1, size):
                if k == i:
                    continue
                vik = coords[k] - coords[i]
                rik = np.sqrt(vik[0]*vik[0] + vik[1]*vik[1] + vik[2]*vik[2])
                if rik == 0.0 or rik >= rcut:
                    continue
                vjk = coords[k] - coords[j]
                rjk = np.sqrt(vjk[0]*vjk[0] + vjk[1]*vjk[1] + vjk[2]*vjk[2])
                if rjk == 0.0:
                    continue
                cosine = (vij[0]*vik[0] + vij[1]*vik[1] + vij[2]*vik[2])/(rij*rik)
                if cosine > 1.0:
                    cosine = 1.0
                elif cosine < -1.0:
                    cosine = -1.0
                index = int(np.arccos(cosine)/astep + 0.5)
                if index >= n_grid:
                    index = n_grid - 1

                atm = (rij*rik*rjk)**2
                f_ijk = 1.0/atm
                scaling = 2.0*np.sqrt(a_scaling[elem_index[j]]*a_scaling[elem_index[k]])

                # grad f_ijk = -2 f_ijk ( grad R_ij / R_ij + .. ), rows i, j, k
                uij = vij/rij                        # grad_Rj R_ij
                uik = vik/rik
                ujk = vjk/rjk
                for c in range(3):
                    di = -uij[c]/rij - uik[c]/rik
                    dj = uij[c]/rij - ujk[c]/rjk
                    dk = uik[c]/rik + ujk[c]/rjk
                    grad_f[0, c] = -2.0*f_ijk*di
                    grad_f[1, c] = -2.0*f_ijk*dj
                    grad_f[2, c] = -2.0*f_ijk*dk

                # grad theta = -(1 - cos^2)^-1/2 grad cos
                sine = np.sqrt(1.0 - cosine*cosine)
                if sine < epsilon:
                    sine = epsilon
                for c in range(3):
                    dcos_j = vik[c]/(rij*rik) - cosine*vij[c]/(rij*rij)
                    dcos_k = vij[c]/(rij*rik) - cosine*vik[c]/(rik*rik)
                    grad_theta[1, c] = -dcos_j/sine
                    grad_theta[2, c] = -dcos_k/sine
                    grad_theta[0, c] = (dcos_j + dcos_k)/sine

                for f in range(n_feature):
                    p = pair_index[elem_index[j], elem_index[k]]
                    h_value = aconvs[p, f, index]
                    h_next = aconvs_next[p, f, index]
                    for c in range(3):
                        out[i, f, i, c] += scaling*(h_value*grad_f[0, c]
                                                    - f_ijk*h_next*grad_theta[0, c])
                        out[i, f, j, c] += scaling*(h_value*grad_f[1, c]
                                                    - f_ijk*h_next*grad_theta[1, c])
                        out[i, f, k, c] += scaling*(h_value*grad_f[2, c]
                                                    - f_ijk*h_next*grad_theta[2, c])
    return out


def get_cmbdf(charges, coords, convs, pad=None, gradients=False, epsilon=1e-8):
    '''
    Atomic cMBDF representation of one molecule, eq. 29.

    Input:
        charges: [Natom]
        coords:  [Natom, 3]
        convs:   Convolutions from get_convolutions
    Output:
        [pad, n_feature], and with gradients=True also [pad, n_feature, pad, 3]
        holding d P[i, f] / d R[a, c].
    '''
    charges = np.asarray(charges)
    coords = np.ascontiguousarray(np.asarray(coords, dtype=np.float64))
    size = len(charges)
    if pad is None:
        pad = size
    if pad < size:
        raise ValueError('pad = %d is smaller than the molecule (%d atoms)' % (pad, size))
    elem_index = convs.element_index(charges)

    n_r = convs.rconvs.shape[1]
    mat = np.zeros((pad, convs.n_feature))
    mat[:size, :n_r] = _two_body(elem_index, coords, convs.a_scaling,
                                 convs.rconvs, convs.rcut, convs.rstep)
    mat[:size, n_r:] = _three_body(elem_index, coords, convs.a_scaling,
                                   convs.pair_index, convs.aconvs,
                                   convs.rcut, convs.astep)
    if not gradients:
        return mat

    dmat = np.zeros((pad, convs.n_feature, pad, 3))
    dmat[:size, :n_r, :size, :] = _two_body_gradient(
        elem_index, coords, convs.a_scaling, convs.rconvs_next,
        convs.rcut, convs.rstep)
    dmat[:size, n_r:, :size, :] = _three_body_gradient(
        elem_index, coords, convs.a_scaling, convs.pair_index, convs.aconvs,
        convs.aconvs_next, convs.rcut, convs.astep, epsilon)
    return mat, dmat


def generate_cmbdf(nuclear_charges, coords, convs=None, pad=None, gradients=False,
                   n_jobs=1, progress_bar=False, **kwargs):
    '''
    cMBDF for a list of molecules.

    convs=None builds the convolutions from the elements found in the data.
    pad=None returns one array per molecule at its own atom count, which is what
    the atomic sum kernels want; pass an integer for a single padded array.
    '''
    charges = [np.asarray(q) for q in nuclear_charges]
    for i, (q, r) in enumerate(zip(charges, coords)):
        if q.shape[0] != np.asarray(r).shape[0]:
            raise ValueError('charges and coordinates length mismatch for molecule %d' % i)
    if convs is None:
        elements = np.unique(np.concatenate(charges))
        convs = get_convolutions(elements, **kwargs)

    def one(q, r):
        return get_cmbdf(q, r, convs, pad=pad, gradients=gradients)

    if n_jobs == 1:
        iterator = zip(charges, coords)
        if progress_bar:
            from tqdm import tqdm
            iterator = tqdm(list(iterator))
        results = [one(q, r) for q, r in iterator]
    else:
        from joblib import Parallel, delayed
        iterator = list(zip(charges, coords))
        if progress_bar:
            from tqdm import tqdm
            iterator = tqdm(iterator)
        results = Parallel(n_jobs=n_jobs)(delayed(one)(q, r) for q, r in iterator)

    if gradients:
        reps = [item[0] for item in results]
        grads = [item[1] for item in results]
        if pad is None:
            return reps, grads
        return np.asarray(reps), np.asarray(grads)
    if pad is None:
        return results
    return np.asarray(results)
