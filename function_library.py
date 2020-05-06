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
    return A*np.trapz(dos*minus_diff_fermi(w, V, T), x=w)/np.trapz(minus_diff_fermi(w, en[ix], T), x=w)

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


# Math

def res_amp(w, amp, w0, Q):
    '''Lorentzian amplitude of frequency response sweep'''
    return amp/np.sqrt(1+(2*Q*(w-w0)/w0)**2)
def res_phase(w, t0, w0, Q):
    '''Lorentzian phase of frequency response sweep, in degree'''
    return t0 + np.arctan2(1, 2*Q*(w-w0)/w0)/np.pi*180

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