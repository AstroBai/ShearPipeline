import warnings
warnings.filterwarnings("ignore")
import numpy as np
from astropy.io import fits
import os
import sys
import pyccl as ccl

class CosmicShear:
    
    def __init__(self, k_max=1, ell_max=5000):
        file_path = "../data/KiDS_Legacy_cosmic_shear_data_release/KiDS_Legacy_xipm.fits"
        hdu = fits.open(file_path)
        #---------------------------------------
        # get n(z)
        nz_source = hdu[4].data
        self.z_value = nz_source.field('Z_MID')  # z_value here
        fields = ['BIN1', 'BIN2', 'BIN3', 'BIN4', 'BIN5', 'BIN6']
        nz_source_value = []
        for field in fields:
            nz_source_value.append(nz_source.field(field))    
        self.nz_source_value = np.array(nz_source_value) 
        #---------------------------------------
        # get data
        self.data_vector_xi = np.load('../data/KiDS_Legacy_cosmic_shear_data_release/KiDS_Legacy_no_cut_shear_data_vector.npy')
        Cov =  np.load('../data/KiDS_Legacy_cosmic_shear_data_release/KiDS_Legacy_no_cut_shear_cov_matrix.npy') 
        self.C = Cov         
        self.inv_C = np.linalg.inv(Cov)
        theta_arcmin_p = np.load('../data/KiDS_Legacy_cosmic_shear_data_release/KiDS_Legacy_no_cut_shear_theta_p.npy')   
        self.theta_deg_p = theta_arcmin_p / 60
        theta_arcmin_m = np.load('../data/KiDS_Legacy_cosmic_shear_data_release/KiDS_Legacy_no_cut_shear_theta_m.npy')   
        self.theta_deg_m = theta_arcmin_m / 60
        
        os.environ["OMP_NUM_THREADS"] = "1"
        self.fremu_nl = ccl.FREmu(k_max=k_max)
        self.data_vector_p = None
        self.data_vector_m = None
        
        self.ell_max = ell_max
        
    def set_cosmo(self, Om=0.3, Ob=0.05, h=0.7, ns=0.96, As=2.1e-9, mnu=0.06, fR0=-1e-5, dz1=0.0, dz2=0.0, dz3=0.0, dz4=0.0, dz5=0.0, dz6=0.0, m1=0.0, m2=0.0, m3=0.0, m4=0.0, m5=0.0, m6=0.0, A_IA=0.0, eta_IA=0.0):
        Onu = mnu / 93.14 / h**2
        self.cosmo = ccl.Cosmology(Omega_c=Om-Ob-Onu, Omega_b=Ob, h=h, n_s=ns, A_s=As, m_nu=mnu, extra_parameters={"fR0":fR0}, mass_split='sum', matter_power_spectrum=self.fremu_nl)  
        self.dzs = np.array([dz1,dz2,dz3,dz4,dz5,dz6])
        self.ms = np.array([m1,m2,m3,m4,m5,m6])
        self.A_IA = A_IA
        self.eta_IA = eta_IA

    def data_vector(self):
    # here, parameters are passed from frlss theory
        dzs = self.dzs
        ms = self.ms
        cosmo = self.cosmo
        ell_before_100 = np.arange(2, 100, 1)  # Step of 1 before 100
        ell_after_100 = np.arange(100, self.ell_max, 20)  # Step of 10 after 100
        ell = np.concatenate((ell_before_100, ell_after_100))
        ell = np.arange(2, self.ell_max, 1)
        data_vector_p = []
        data_vector_m = []
        theta_index = 0
        for i in range(6):
            for j in range(6 - i):
                j = j + i
                z_true1 = self.z_value - dzs[i]
                z_true2 = self.z_value - dzs[j]
                nz_true1 = self.nz_source_value[i]
                valid_idx = z_true1 > 0     # z must be positive
                z_true1 = z_true1[valid_idx]
                nz_true1 = nz_true1[valid_idx]
                # cut z at 3
                valid_idx = z_true1 < 3
                z_true1 = z_true1[valid_idx]
                nz_true1 = nz_true1[valid_idx]
                
                nz_true2 = self.nz_source_value[j]
                valid_idx = z_true2 > 0     # z must be positive
                z_true2 = z_true2[valid_idx]
                nz_true2 = nz_true2[valid_idx]
                # cut z at 3
                valid_idx = z_true2 < 3
                z_true2 = z_true2[valid_idx]
                nz_true2 = nz_true2[valid_idx]
                
                lens1 = ccl.WeakLensingTracer(cosmo, dndz=(z_true1,nz_true1), use_A_ia=False)
                lens2 = ccl.WeakLensingTracer(cosmo, dndz=(z_true2, nz_true2), use_A_ia=False)
                cls_fremu = ccl.angular_cl(cosmo, lens1, lens2, ell, limber_integration_method='spline')
                xi_plus_fremu = ccl.correlation(cosmo, ell=ell, C_ell=cls_fremu, 
            theta=self.theta_deg_p, 
                                        type='GG+', method='fftlog')
                xi_plus_fremu_obs = (1+ms[i]) * (1+ms[j]) * xi_plus_fremu
                xi_minus_fremu = ccl.correlation(cosmo, ell=ell, C_ell=cls_fremu, 
            theta=self.theta_deg_m, 
                                            type='GG-', method='fftlog')
                xi_minus_fremu_obs = (1+ms[i]) * (1+ms[j]) * xi_minus_fremu
                data_vector_p.append(xi_plus_fremu_obs)
                data_vector_m.append(xi_minus_fremu_obs)
                theta_index += 1

        self.data_vector_p = np.concatenate(data_vector_p)  
        self.data_vector_m = np.concatenate(data_vector_m)
        self.data_vector_xi = np.concatenate((self.data_vector_p, self.data_vector_m))
        

from matplotlib import pyplot as plt
        
cs_1 = CosmicShear(k_max=1, ell_max=5000)
cs_1.set_cosmo()
cs_05 = CosmicShear(k_max=0.5, ell_max=5000)
cs_05.set_cosmo()

cs_1.data_vector()
cs_05.data_vector()


plt.figure(figsize=(16,4))        
plt.plot(cs_1.data_vector_xi, label='k_max=1')       
plt.plot(cs_05.data_vector_xi, label='k_max=0.5')       
plt.legend()
plt.xlabel('Data index')
plt.ylabel('xi (observed)')
plt.yscale('log')
plt.tight_layout()
plt.savefig('xi_DES_no_cut.png')

sigma_array = np.sqrt(np.diag(cs_1.C))

plt.figure(figsize=(16,4))         
plt.plot((cs_1.data_vector_xi-cs_05.data_vector_xi)/sigma_array, label='delta/sigma')
plt.axhline(y=0.05, color='r', linestyle='--')
plt.axhline(y=-0.05, color='r', linestyle='--')

plt.legend()
plt.xlabel('Data index')
plt.ylabel('Error')
plt.tight_layout()        
plt.savefig('error_DES_no_cut.png')

# kick out index that error is larger than 5%
safe_idx = np.where(abs(cs_1.data_vector_xi-cs_05.data_vector_xi)/sigma_array < 0.05)[0]
print(safe_idx)
# Group by bins and keep last continuous points in each bin
bins = safe_idx // 20
filtered_idx = []

for bin_num in range(20):
    # Get points in this bin
    bin_points = safe_idx[bins == bin_num]
    
    # Find last continuous sequence
    last_seq = []
    for i in reversed(range(len(bin_points))):
        if i == len(bin_points)-1 or bin_points[i] - 1 == bin_points[i-1]:
            last_seq.append(bin_points[i])
        else:
            break
    
    if last_seq:
        filtered_idx.extend(sorted(last_seq))

filtered_idx = np.array(filtered_idx)
print(filtered_idx)
np.save('safe_idx_des_no_cut.npy', filtered_idx)