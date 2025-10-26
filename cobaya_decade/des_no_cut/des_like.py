import warnings
warnings.filterwarnings("ignore")
import numpy as np
from astropy.io import fits
import os
import pyccl as ccl
from cobaya.likelihood import Likelihood
from tqdm import tqdm
from fremu import FREmu

class DES(Likelihood):
    
    def initialize(self):
        file_path = "../data/DES_3x2pt/2pt_NG_final_2ptunblind_02_26_21_wnz_maglim_covupdate.fits"
        hdu = fits.open(file_path)
        #---------------------------------------
        # get n(z)
        nz_source = hdu[6].data
        self.z_value = nz_source.field('Z_MID')  # z_value here
        fields = ['BIN1', 'BIN2', 'BIN3', 'BIN4']
        nz_source_value = []
        for field in fields:
            nz_source_value.append(nz_source.field(field))    
        self.nz_source_value = np.array(nz_source_value) 
        #---------------------------------------
        # get data
        self.data_vector_xi = np.load('../data/DES_3x2pt/DES_no_cut_shear_data_vector.npy')
        Cov =  np.load('../data/DES_3x2pt/DES_no_cut_shear_cov_matrix.npy')          
        self.inv_C = np.linalg.inv(Cov)
        theta_arcmin_p = np.load('../data/DES_3x2pt/DES_no_cut_shear_theta_p.npy')   
        self.theta_deg_p = theta_arcmin_p / 60
        theta_arcmin_m = np.load('../data/DES_3x2pt/DES_no_cut_shear_theta_m.npy')   
        self.theta_deg_m = theta_arcmin_m / 60
        
        os.environ["OMP_NUM_THREADS"] = "1"
        self.fremu_nl = ccl.FREmu()
        
        # get scale cuts
        self.safe_idx = np.load('../scale_cuts/safe_idx_des.npy')
        
    def get_requirements(self):
        return {'omegam': None, 'omegab': None, 'H0': None, 'ns': None, 'As': None, 'mnu': None, 'logfR0': None, 'logMc': None, 'eta_b': None} 
        
    def logp(self,**params_values):
        try:
        # here, parameters are passed from frlss theory
            pp = self.provider
            Om = pp.get_param('omegam')
            Ob = pp.get_param('omegab')
            h = pp.get_param('H0') / 100
            ns = pp.get_param('ns')
            As = pp.get_param('As')
            mnu = pp.get_param('mnu')
            logfR0 = pp.get_param('logfR0')
            A_IA = params_values['A_IA_des']
            eta_IA = params_values['eta_IA_des']
            logMc = pp.get_param('logMc')
            eta_b = pp.get_param('eta_b')
            dz1 = params_values['dz1_des']
            dz2 = params_values['dz2_des']
            dz3 = params_values['dz3_des']
            dz4 = params_values['dz4_des']
            m1 = params_values['m1_des']
            m2 = params_values['m2_des']
            m3 = params_values['m3_des']
            m4 = params_values['m4_des']
            
            fR0 = -10 ** logfR0
            Onu = mnu/ 93.14 / h**2
            bcm = ccl.baryons.BaryonsSchneider15(log10Mc=logMc, eta_b=eta_b, k_s=55.0)
            cosmo = ccl.Cosmology(Omega_c=Om-Ob-Onu, Omega_b=Ob, h=h, n_s=ns, A_s=As, m_nu=mnu, extra_parameters={"fR0":fR0}, mass_split='sum', matter_power_spectrum=self.fremu_nl, baryonic_effects=bcm)  
            #ell = np.arange(2, 5000)
            ell_before_100 = np.arange(2, 100, 1)  # Step of 1 before 100
            ell_after_100 = np.arange(100, 5000, 20)  # Step of 10 after 100
            ell = np.concatenate((ell_before_100, ell_after_100))
            dzs = np.array([dz1,dz2,dz3,dz4])
            ms = np.array([m1,m2,m3,m4])
            A_IAs = A_IA * ((1+self.z_value)/1.62) ** eta_IA
            data_vector_p = []
            data_vector_m = []
            theta_index = 0
            for i in range(4):
                for j in range(4 - i):
                    j = j + i
                    z_true1 = self.z_value - dzs[i]
                    z_true2 = self.z_value - dzs[j]
                    
                    nz_true1 = self.nz_source_value[i]
                    valid_idx = z_true1 > 0     # z must be positive
                    z_true1 = z_true1[valid_idx]
                    nz_true1 = nz_true1[valid_idx]
                    
                    nz_true2 = self.nz_source_value[j]
                    valid_idx = z_true2 > 0     # z must be positive
                    z_true2 = z_true2[valid_idx]
                    nz_true2 = nz_true2[valid_idx]
                    
                    lens1 = ccl.WeakLensingTracer(cosmo, dndz=(z_true1,nz_true1), use_A_ia=True, ia_bias=(self.z_value, A_IAs))
                    lens2 = ccl.WeakLensingTracer(cosmo, dndz=(z_true2,nz_true2), use_A_ia=True, ia_bias=(self.z_value, A_IAs))
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
            
 
            data_vector_p = np.concatenate(data_vector_p)  
            data_vector_m = np.concatenate(data_vector_m)
            predict_vector =   np.concatenate([data_vector_p,data_vector_m])
            diff = predict_vector - self.data_vector_xi
            diff = diff[self.safe_idx]
            inv_C = self.inv_C[self.safe_idx][:,self.safe_idx]
            chi2 = np.dot(diff, np.dot(inv_C, diff))
            if np.isnan(chi2):
                return -np.inf
            return -chi2 / 2
        except Exception as e:
            return -np.inf
        