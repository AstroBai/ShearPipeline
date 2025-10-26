import warnings
warnings.filterwarnings("ignore")
import numpy as np
from astropy.io import fits
import os
import pyccl_backup as ccl
from cobaya.likelihood import Likelihood
from tqdm import tqdm
from fremu import fremu

class DES(Likelihood):
    
    def initialize(self):
        file_path = "../data/DES_3x2pt/2pt_NG_final_2ptunblind_02_26_21_wnz_maglim_covupdate.fits"
        hdu = fits.open(file_path)
        #---------------------------------------
        # get n(z)
        nz_source = hdu[6].data
        self.z_value_source = nz_source.field('Z_MID')  # z_value here
        fields = ['BIN1', 'BIN2', 'BIN3', 'BIN4']
        nz_source_value = []
        for field in fields:
            nz_source_value.append(nz_source.field(field))    
        self.nz_source_value = np.array(nz_source_value) 
        
        nz_lens = hdu[7].data
        self.z_value_lens = nz_lens.field('Z_MID')  # z_value here
        fields = ['BIN1', 'BIN2', 'BIN3', 'BIN4']
        nz_lens_value = []
        for field in fields:
            nz_lens_value.append(nz_lens.field(field))    
        self.nz_lens_value = np.array(nz_lens_value) 
        #---------------------------------------
        # get data      
        theta_arcmin_p = np.load('../data/DES_3x2pt/DES_shear_theta_p.npy')   
        self.theta_deg_p = theta_arcmin_p / 60
        theta_arcmin_m = np.load('../data/DES_3x2pt/DES_shear_theta_m.npy')   
        self.theta_deg_m = theta_arcmin_m / 60
        theta_arcmin_gamma = np.load('../data/DES_3x2pt/DES_gamma_theta.npy')   
        self.theta_deg_gamma = theta_arcmin_gamma / 60
        theta_arcmin_w = np.load('../data/DES_3x2pt/DES_w_theta.npy')   
        self.theta_deg_w = theta_arcmin_w / 60
        Cov =  np.load('../data/DES_3x2pt/DES_3by2_cov_matrix.npy') 
        self.inv_C = np.linalg.inv(Cov)
        self.data_vector = np.load('../data/DES_3x2pt/DES_3by2_data_vector.npy')
        self.fremu_nl = ccl.FREmu()
        
        
        
    def get_requirements(self):
        return {'omegam': None, 'omegab': None, 'H0': None, 'ns': None, 'As': None, 'mnu': None, 'logfR0': None} 
        
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
            logMc = params_values['logMc_des']
            eta_b = params_values['eta_b_des']
            dz1 = params_values['dz1_des']
            dz2 = params_values['dz2_des']
            dz3 = params_values['dz3_des']
            dz4 = params_values['dz4_des']
            m1 = params_values['m1_des']
            m2 = params_values['m2_des']
            m3 = params_values['m3_des']
            m4 = params_values['m4_des']
            dz1_l = params_values['dz1_l_des']
            dz2_l = params_values['dz2_l_des']
            dz3_l = params_values['dz3_l_des']
            dz4_l = params_values['dz4_l_des']
            sl1 = params_values['sl1_des']
            sl2 = params_values['sl2_des']
            sl3 = params_values['sl3_des']
            sl4 = params_values['sl4_des']
            c1 = params_values['magl_1_des']
            c2 = params_values['magl_2_des']
            c3 = params_values['magl_3_des']
            c4 = params_values['magl_4_des']
            b1 = params_values['b1_des']
            b2 = params_values['b2_des']
            b3 = params_values['b3_des']
            b4 = params_values['b4_des']
            fR0 = -10 ** logfR0
            Onu = mnu/ 93.14 / h**2
            bcm = ccl.baryons.BaryonsSchneider15(log10Mc=logMc, eta_b=eta_b, k_s=55.0)
            cosmo = ccl.Cosmology(Omega_c=Om-Ob-Onu, Omega_b=Ob, h=h, n_s=ns, A_s=As, m_nu=mnu, extra_parameters={"fR0":fR0}, mass_split='sum', matter_power_spectrum=self.fremu_nl,baryonic_effects=bcm)
            ell_before_100 = np.arange(2, 100, 1)  # Step of 1 before 100
            ell_after_100 = np.arange(100, 5000, 20)  # Step of 10 after 100
            ell = np.concatenate((ell_before_100, ell_after_100))
            dzs = np.array([dz1,dz2,dz3,dz4])
            ms = np.array([m1,m2,m3,m4])
            dzl = np.array([dz1_l,dz2_l,dz3_l,dz4_l])
            sls = np.array([sl1,sl2,sl3,sl4])
            cs = np.array([c1,c2,c3,c4])
            bs = np.array([b1,b2,b3,b4])
            A_IAs = A_IA * ((1+self.z_value_source)/1.62) ** eta_IA
            #================================================================
            #xi
            data_vector_p = []
            data_vector_m = []
            theta_index = 0
            for i in range(4):
                for j in range(4 - i):
                    j = j + i
                    z_s_true1 = self.z_value_source - dzs[i]
                    z_s_true2 = self.z_value_source - dzs[j]
                    
                    nz_s_true1 = self.nz_source_value[i]
                    valid_idx = z_s_true1 > 0
                    z_s_true1 = z_s_true1[valid_idx]
                    nz_s_true1 = nz_s_true1[valid_idx]
                    
                    nz_s_true2 = self.nz_source_value[j]
                    valid_idx = z_s_true2 > 0
                    z_s_true2 = z_s_true2[valid_idx]
                    nz_s_true2 = nz_s_true2[valid_idx]
                    source1 = ccl.WeakLensingTracer(cosmo, dndz=(z_s_true1, nz_s_true1), use_A_ia=True, ia_bias=(self.z_value_source, A_IAs))
                    source2 = ccl.WeakLensingTracer(cosmo, dndz=(z_s_true2, nz_s_true2), use_A_ia=True, ia_bias=(self.z_value_source, A_IAs))
                    cls_fremu = ccl.angular_cl(cosmo, source1, source2, ell, limber_integration_method='spline')
                    xi_plus_fremu = ccl.correlation(cosmo, ell=ell, C_ell=cls_fremu, 
                theta=self.theta_deg_p[theta_index, :][self.theta_deg_p[theta_index, :] != 0], 
                                            type='GG+', method='fftlog')
                    xi_plus_fremu_obs = (1+ms[i]) * (1+ms[j]) * xi_plus_fremu
                    xi_minus_fremu = ccl.correlation(cosmo, ell=ell, C_ell=cls_fremu, 
                theta=self.theta_deg_m[theta_index, :][self.theta_deg_m[theta_index, :] != 0], 
                                             type='GG-', method='fftlog')
                    xi_minus_fremu_obs = (1+ms[i]) * (1+ms[j]) * xi_minus_fremu
                    data_vector_p.append(xi_plus_fremu_obs)
                    data_vector_m.append(xi_minus_fremu_obs)
                    theta_index += 1
            data_vector_p = np.concatenate(data_vector_p)  
            data_vector_m = np.concatenate(data_vector_m)
            #================================================================
            #gamma
            data_vector_gamma = []
            theta_index = 0
            for i in range(4):
                for j in range(4):
                    z_l_true1 = self.z_value_lens 
                    z_l_true1 = sls[i] * (z_l_true1 - np.mean(z_l_true1)) + np.mean(z_l_true1) - dzl[i]
                    z_s_true2 = self.z_value_source - dzs[j]
                    bz = np.ones_like(self.z_value_lens)*bs[i]
                    cz = np.ones_like(self.z_value_lens)*cs[i]
                    nz_l_true1 = sls[i] * self.nz_lens_value[i]
                    valid_idx = z_l_true1 > 0     # z must be positive
                    z_l_true1 = z_l_true1[valid_idx]
                    nz_l_true1 = nz_l_true1[valid_idx]
                    nz_s_true2 = self.nz_source_value[j]
                    valid_idx = z_s_true2 > 0
                    z_s_true2 = z_s_true2[valid_idx]
                    nz_s_true2 = nz_s_true2[valid_idx]
                    lens1 = ccl.NumberCountsTracer(cosmo, dndz=(z_l_true1, nz_l_true1),bias=(self.z_value_lens,bz),mag_bias=(self.z_value_lens,cz),has_rsd=False)
                    source2 = ccl.WeakLensingTracer(cosmo, dndz=(z_s_true2, nz_s_true2), use_A_ia=True, ia_bias=(self.z_value_source, A_IAs))
                    cls_fremu = ccl.angular_cl(cosmo, lens1, source2, ell, limber_integration_method='spline')
                    gamma = ccl.correlation(cosmo, ell=ell, C_ell=cls_fremu, 
                theta=self.theta_deg_gamma[theta_index, :][self.theta_deg_gamma[theta_index, :] != 0], 
                                            type='NG', method='fftlog')
                    gamma_fremu_obs = (1+ms[j]) * gamma
                    data_vector_gamma.append(gamma_fremu_obs)
                    theta_index += 1
            data_vector_gamma = np.concatenate(data_vector_gamma)
            #================================================================
            #w
            data_vector_w = []
            theta_index = 0
            for i in range(4):
                z_l_true1 = self.z_value_lens 
                z_l_true1 = sls[i] * (z_l_true1 - np.mean(z_l_true1)) + np.mean(z_l_true1) - dzl[i]
                bz1 = np.ones_like(self.z_value_lens)*bs[i]
                cz1 = np.ones_like(self.z_value_lens)*cs[i]
                nz_true1 = sls[i] * self.nz_lens_value[i]
                valid_idx = z_l_true1 > 0     # z must be positive
                z_l_true1 = z_l_true1[valid_idx]
                nz_true1 = nz_true1[valid_idx]
                lens1 = ccl.NumberCountsTracer(cosmo, dndz=(z_l_true1, nz_true1),bias=(self.z_value_lens,bz1),mag_bias=(self.z_value_lens,cz1),has_rsd=True)
                cls_fremu = ccl.angular_cl(cosmo, lens1, lens1, ell, limber_integration_method='spline')
                w = ccl.correlation(cosmo, ell=ell, C_ell=cls_fremu, 
                theta=self.theta_deg_w[theta_index, :][self.theta_deg_w[theta_index, :] != 0], type='NN', method='fftlog')
                data_vector_w.append(w)
                theta_index += 1
            data_vector_w = np.concatenate(data_vector_w)
            #======================================================================
            #FINAL
            predict_vector = np.concatenate([data_vector_p,data_vector_m,data_vector_gamma,data_vector_w])
            diff = predict_vector - self.data_vector
            chi2 = np.dot(diff, np.dot(self.inv_C, diff))
            if np.isnan(chi2):
                return -np.inf
            return -chi2 / 2
        except Exception as e:
            return -np.inf
        
