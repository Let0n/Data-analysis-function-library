# Customized Colormap
from matplotlib.colors import LinearSegmentedColormap as LSC
cm_nl = LSC.from_list('gray', [(0.0, '0.1'), (0.2, '0.8'), (0.6, 'lightyellow'), (1.0, 'gold')])
figure(figsize=(3, 1), dpi=300)
imshow([linspace(0, 1, 256), linspace(0, 1, 256)], cmap=cm_nl, aspect=5)
gca().set_axis_off()

import numpy as np
import matplotlib as mpl

# Physics
def dynes(v, gap, gamma, g0):
    '''Dynes gap function'''
    return g0*abs(np.real((v+1j*gamma)/np.sqrt((v+1j*gamma)**2-gap**2)))

def minus_diff_fermi(w, V, T):
    '''negative derivative of Fermi function. w, V in volts'''
    k_B = 8.617e-5 #eV/K
    x = (w-V)/(2*k_B*T)
    expmax = np.round(np.log(sys.float_info.max)) #max exponent to avoid overflow
    nonzero = abs(x) < expmax
    sech = np.zeros_like(x)
    sech[nonzero] = 1./np.cosh(x[nonzero])
    return 1/(4*k_B*T)*sech**2

def didv(w, A, V, T, dos):
    '''Thermally broadening of DOS as dI/dV. dos and w are same-length 1d arrays'''
    '''
    Example:
    w = linspace(15, -15, 3001)*1e-3 # must be wider than bias+2kT/e
    bias = linspace(-10, 10, 1001)*1e-3
    tmp_dos = dynes(w, 1.4e-3, 0.1e-3, 1.0)
    tmp_didv = zeros_like(bias)
    for ix in range(len(bias)):
    tmp_didv[ix] = didv(w, 1.0, bias[ix], 4.0, tmp_dos)
    '''
    return A*np.trapz(dos*minus_diff_fermi(w, V, T), x=w)/np.trapz(minus_diff_fermi(w, V, T), x=w)

def tbroad_dynes(v, delta, gamma, amp, T):
    '''Thermally broadening dynes'''
    k_B = 8.617e-5
    if not len(v):
        print('v must be an 1d array, not scalar!')
        return np.zeros_like(v)
    else:
        dw = min(abs(np.diff(v)[0])/10, k_B*T/10)
        w = np.arange(v.min()-10*k_B*T, v.max()+10*k_B*T, dw) # enlarged energy vector for integrals
        dos = dynes(w, delta, gamma, amp)
        didv = np.zeros_like(v)
        for ix in range(len(v)):
            didv[ix] = np.trapz(dos*minus_diff_fermi(w, v[ix], T), x=w)/np.trapz(minus_diff_fermi(w, v[ix], T), x=w)
        return didv

def maki(E, d, B, gamma, b, z, max_iter=1000, abs_tol=1e-8, spin=None, initial='random'):
    '''Maki model for SC under field w/ depairing and SO scattering'''
    h = 5.79e-5*B
    def u_plus_minus(E, d, h, gamma, b, z, max_iter=1000, abs_tol=1e-8):
        up, un = 2*rand(len(E))-1 + (2*rand(len(E))-1) * 1j, 2*rand(len(E))-1 + (2*rand(len(E))-1) * 1j
        e = E + 1j * gamma
        for m in range(max_iter):
            up_m = (e-h)/d + z*up/sqrt(1-up**2) + b*(un-up)/sqrt(1-un**2)
            resp = sum(abs(up_m-up))
            up = copy(up_m)
            un_m = (e+h)/d + z*up/sqrt(1-un**2) + b*(up-un)/sqrt(1-up**2)
            resn = sum(abs(un_m-un))
            un = copy(un_m)
    #         print('Iteration %d, residual %.3e'%(m+1, resp+resn))
            if resp + resn < abs_tol:
                break 
            if m+1 == max_iter:
                    print('Reaches maximal number of iterations, residual = %.3E.'%(resp+resn))
        return up, un
    
    if b and z:
        results = zeros((len(E), 4), dtype=complex128)
        y0 = 2*rand(4)-1 +( 2*rand(4)-1) * 1j # random initial guess
        if initial == 'in_phase':
        # approximated in-phase solution
            up1, un1 = u_plus_minus(E, d, h, gamma, 0, z, max_iter, abs_tol) # b=0
            up2, un2 = u_plus_minus(E, d, h, gamma, b, 0, max_iter, abs_tol) # z=0
            up0, un0 = up1+up2, un1+un2 # sum is in phase
        
        for k in range(len(E)):
            e = E[k] + 1j * gamma
            if initial == 'in_phase':
                y0 = [-0.5*(up0[k]/sqrt(1-up0[k]**2)+un0[k]/sqrt(1-un0[k]**2)), -0.5/h*(up0[k]/sqrt(1-up0[k]**2)-un0[k]/sqrt(1-un0[k]**2)), 
                 0.5/d*(1/sqrt(1-up0[k]**2)+1/sqrt(1-un0[k]**2)), -0.5/h/d*(1/sqrt(1-up0[k]**2)-1/sqrt(1-un0[k]**2))]
            F = lambda y: array([y[0] + e*y[1] + h**2*y[3] - d*z*(y[0]*y[1]-h**2*y[2]*y[3]),
                                 y[1] - y[2] + e*y[3] + d*z*(y[1]*y[2]-y[0]*y[3]) - 2*b*d*(y[1]*y[2]+y[0]*y[3]),
                                 y[0]**2 + h**2 * y[2]**2 - d**2*(y[1]**2+h**2*y[3]**2) + 1,
                                 y[0]*y[2] + d**2*y[1]*y[3]], dtype=complex128)
            J = lambda y: array([[1-d*z*y[1], e-d*z*y[0], d*z*h**2*y[3], h**2+h**2*d*z*y[2]],
                                 [-d*z*y[3]-2*b*d*y[3], 1+d*z*y[2]-2*b*d*y[2], -1+d*z*y[1]-2*b*d*y[1], e-d*z*y[0]-2*b*d*y[0]],
                                 [2*y[0], -2*d**2*y[1], 2*h**2*y[2], -2*d**2*h**2*y[3]],
                                 [y[2], d**2*y[3], y[0], d**2*y[1]]], dtype=complex128)
            for m in range(max_iter):
                ym = y0 - linalg.inv(J(y0)) @ F(y0)
                if sum(abs(ym-y0)) < abs_tol:
                    y = copy(ym)
                    break
                if m+1 == max_iter:
                    print('E = %.2fmeV reaches maximal number of iterations, residual = %.3e'%(E[k]*1e3, sum(abs(ym-y0))))
                y0 = copy(ym)
            y = copy(ym)
            results[k] = y
        up = -1/d * (results[:, 0] + h * results[:, 2]) / (results[:, 1] - h * results[:, 3])
        un = -1/d * (results[:, 0] - h * results[:, 2]) / (results[:, 1] + h * results[:, 3])
    else:
        up, un = u_plus_minus(E, d, h, gamma, b, z, max_iter, abs_tol)
    rho_p = 0.5 * abs(real((up)/sqrt((up)**2-1)))
    rho_n = 0.5 * abs(real((un)/sqrt((un)**2-1)))
    rho = rho_p + rho_n
    if spin == 'up':
        return rho_p
    elif spin == 'down':
        return rho_n
    elif spin == 'both':
        return rho, rho_p, rho_n
    else:
        # default only return total DOS
        return rho

def tbmodel(w, t, Gamma, Delta_0, N_kmesh=1024):
    '''5-component tight-binding model for BSCCO'''
    k = np.linspace(0, pi, N_kmesh)
    kx, ky = np.meshgrid(k, k)
    Delta_k = Delta_0 * (np.cos(kx) - np.cos(ky))/2
    xi = 2*t[1]*(np.cos(kx)+np.cos(ky)) + 4*t[2]*np.cos(kx)*cos(ky) + 2*t[3]*(np.cos(2*kx)+np.cos(2*ky))\
    + 4*t[4]*(np.cos(2*kx)*np.cos(ky)+np.cos(kx)*np.cos(2*ky)) + 4*t[5]*np.cos(2*kx)*np.cos(2*ky) - t[0]
    # t[0]: chemical potential
    A = np.zeros((len(w), N_kmesh, N_kmesh))
    dos = np.zeros(len(w))
    for iw in range(len(w)):
        A[iw] = -1./pi * np.imag(1./(w[iw] - xi + 1j*Gamma - abs(Delta_k)**2/(w[iw]+xi+1j*Gamma)))
        dos[iw] = sum(A[iw])
    return A, dos # spectral function and density of states

def deconvSCtip(v, g, delta_t, gamma_t, T, eps=None, alpha=1.0,  method='Twomey', dynes='imagE'):
    '''
    Deconvolute an SIS spectrum based from known tip gap and broadening at finite temperature.
    Input:
        v - voltage array
        g - measured SIS spectrum
        delta_t, gamma_t - known tip gap and broadening
        T - temperature
        eps - energy array for deconvolution; if None, then will copy v
        alpha - damping factor to smooth the result (not used for 'pinv' method)
        method - 'Twomey', 'NNLS', or 'pinv', method to be used for deconvolution. 
                 'Twomey': approximate solution of Fredholm Integral Equation of the first kind (DOI: 10.1145/321150.321157)
                 'nnls': non-negative least square solver with damping factor alpha
                 'pinv': (pseudo-)inverse (not recommended)
        dynes - Dynes function to be used. 'imagE' adds broadening to the energy, while 'imagD' adds broadening to Delta
    Output:
        LDOS - sample density of states (approximation)
        eps - energy array
        reconv - reconvoluted spectrum (for validation)
        Residual will be printed out.
    '''
    def makeH(dim) :    
        '''Make H matrix for estimating Fredholm equations as in (Twomey 1963).'''
        # create (symmetric) diagonal vectors
        d2 = numpy.ones(dim - 2)
        d1 = numpy.concatenate(([-2], numpy.repeat(-4, dim - 3), [-2]))
        d0 = numpy.concatenate(([1, 5], numpy.repeat(6, dim - 4), [5, 1]))
        # make matrix with diagonals
        Hmat = numpy.diag(d1, 1) + numpy.diag(d2, 2)
        Hmat = Hmat + Hmat.T
        numpy.fill_diagonal(Hmat, d0)
        return Hmat
    
    v = asarray(v)
    dv = abs(diff(v)[0]) # must be equally distace array
    vv = linspace(v.min()-len(v)*dv, v.max()+len(v)*dv, len(v)*3)
    # extrapolate g to minimize edge effects, use quantity[len(v):2*len(v)] to recover
    gg = hstack((ones_like(v)*g[0], g, ones_like(v)*g[-1])) 
    trim = False
    if eps is None:
        eps = copy(vv) # then x will be a square matrix
        trim = True #trim output LDOS later
    En_g, eV_g = np.meshgrid(eps, vv)
    x = En_g + eV_g
    if dynes == 'imagE':
        rho_t = sign(x)*real((x+1j*gamma_t)/sqrt((x+1j*gamma_t)**2 - delta_t**2))
    elif dynes == 'imagD':
        rho_t = sign(x)*real(x/sqrt(x**2 - (delta_t+1j*gamma_t)**2))
    else:
        # experimental: 
        rho_t = sign(x)*real(x/sqrt(x**2+1j*gamma_t*x*2 - delta_t**2))
    drho_t = np.gradient(rho_t, axis=0)/dv
    k_B = 8.617e-5 #eV/K
    fer = 1./(1+exp(En_g/k_B/T))
    fer_V = 1./(1+exp(x/k_B/T))
    dfer_V = np.gradient(fer_V, axis=0)/dv
    dfer_V[np.isnan(dfer_V)] = 0
    A = (drho_t * (fer - fer_V) + rho_t * dfer_V) * diff(eps)[0]
    if method == 'Twomey':
        H = makeH(len(eps))
        LDOS = np.linalg.solve(A.T @ A + alpha * H , A.T @ gg)
    elif method == 'nnls':
        LDOS, _ = scipy.optimize.nnls(vstack((A, alpha*eye(A.shape[1]))), concatenate((gg, zeros_like(eps))))
#         LDOS /= diff(eps)[0]
    elif method == 'pinv':
        Ainv = scipy.linalg.pinv(A)
        LDOS = np.flipud(Ainv @ gg)
    else:
        raise ValueError("Choose one method among 'Twomey', 'NNLS', and 'pinv'.")
    reconv_full = A @ LDOS
    reconv = reconv_full[len(v):2*len(v)]
    res = sqrt(sum((reconv-g)**2))
    print('Residual %.3f'%res, end='\r')
    if trim:
        LDOS = LDOS[len(v):2*len(v)]
        eps = copy(v)
    return LDOS, eps, reconv
    
# Math

def res_amp(w, amp, w0, Q):
    '''Lorentzian amplitude of frequency response sweep'''
    return amp/np.sqrt(1+(2*Q*(w-w0)/w0)**2)
def res_phase(w, t0, w0, Q):
    '''Lorentzian phase of frequency response sweep, in degree'''
    return t0 + np.arctan2(1, 2*Q*(w-w0)/w0)/np.pi*180

def ABCD2S(A, B, C, D, Z0=50.0):
    '''Convert ABCD parameters to S parameters'''
    S11 = (A+B/Z0-C*Z0-D)/(A+B/Z0+C*Z0+D)
    S12 = 2*(A*D-B*C)/(A+B/Z0+C*Z0+D)
    S21 = 2/(A+B/Z0+C*Z0+D)
    S22 = (-A+B/Z0-C*Z0+D)/(A+B/Z0+C*Z0+D)
    return S11, S12, S21, S22

def gauss2dkernal(A, sigma):
    '''2D Gaussian kernal for smoothing'''
    col, row = A.shape
    return np.outer(scipy.signal.gaussian(col, sigma), scipy.signal.gaussian(row, sigma))

def corr2d_fft(A, B, mode='full'):
    '''2D crosscorrelation based on fftconvolve'''
    return scipy.signal.fftconvolve(A, B[::-1, ::-1], mode=mode)

# Misc


def cleanuphdf5():
'''clean up unclosed hdf5 files'''
    import gc, h5py
    for obj in gc.get_objects():   # Browse through ALL objects
        if isinstance(obj, h5py.File):   # Just HDF5 files
            try:
                obj.close()
            except:
                pass # Was already closed

class Spec(object):
    def __init__(self):
        pass
def loadspec(fname, printheader=False):
    '''Read noise spectra from Zurich lock-in'''
    def readkey(obj, key):
        value = squeeze(obj[key][:].tolist()) #squeeze dimensions of MATLAB's matrix (mininal 2-D)
        if value.ndim is 0:
            val = float(value) # scalar
        elif value.ndim is 1:
            val = value
        elif value.ndim is 2:
            val = value
        elif value.ndim is 3:
            val = value[:, :, 0] # take first set if multiple sets of measurements are stored
        return val
    # init instance and header (dict)
    self = Spec()
    self.header = {}
    try:
        f = scipy.io.loadmat(fname, matlab_compatible=True) # before MATLAB v7.3
        hdf5 = False
        opt = dict((key, squeeze(f['opt'])[key].tolist()) for key in squeeze(f['opt']).dtype.names)
    except:
        import h5py
        f = h5py.File(fname, 'r') # hdf5 f format, MATLAB v7.3 and later
        hdf5 = True
        opt = f['opt']
    # read setup conditions
    self.header['R_tunnel'] = readkey(f, 'R_tunnel')
    self.header['divider'] = readkey(f, 'divider')
    self.bias = np.hstack((0, abs(readkey(f, 'biassetp')), -abs(readkey(f, 'biassetp'))))/self.header['divider']   
    self.current = np.hstack((0, abs(readkey(f,'current_setp')), -abs(readkey(f,'current_setp'))))
    # read options into header
    for key, value in opt.items():
        if not key == 'filter':
            self.header[key] = float(squeeze(value))
    # read spectral data
    snc = f['snc']
    self.freq = readkey(snc, 'freq')
    self.intnoise = readkey(snc, 'int') # integrated noise
    self.rawspectra = readkey(snc, 'r')
    if hdf5: 
        self.rawspectra = self.rawspectra.T
        f.close()  
    # filter compensation
    self.filter = 1.5426
#     def scale(order): return np.sqrt(np.power(2., 1./order)-1) # tau/(2*pi*f_3db) of n-th order RC filter
#     if 'filter' in opt:
#         self.filter = squeeze(opt['filter'][:].tolist())
#     else:
#         self.filter = abs(np.power(1./(1.0+1j*(self.freq-self.header['f0'])/self.header['BW']*scale(self.header['order'])),\
#                                self.header['order']))
    self.psd = (self.rawspectra/self.filter)**2
    if printheader:
        for k, v in self.header.items():
            print(k, v)
    return self
