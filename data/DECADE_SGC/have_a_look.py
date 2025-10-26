import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import seaborn as sns # type: ignore
from matplotlib.colors import LogNorm

#=====================================================================================================
# Fonts settings
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams.update({'font.size': 30})
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

plt.rcParams['xtick.major.size'] = 15
plt.rcParams['xtick.major.width'] = 2.5
plt.rcParams['xtick.minor.size'] = 7.5
plt.rcParams['xtick.minor.width'] = 1.5

plt.rcParams['ytick.major.size'] = 15
plt.rcParams['ytick.major.width'] = 2.5
plt.rcParams['ytick.minor.size'] = 7.5
plt.rcParams['ytick.minor.width'] = 1.5

plt.rcParams['axes.unicode_minus'] = False 
plt.rcParams['axes.linewidth'] = 2 
plt.rcParams['lines.linewidth'] = 2

#=====================================================================================================
# Reading Fits
file_path = "2pt_decade_sgc_20250218.fits"
hdu = fits.open(file_path)
hdu.info()
print(hdu[2].data.names)
#=====================================================================================================
# nz
plt.figure(figsize=(16,16))
plt.subplot(2,1,1)
nz_source = hdu[4].data
#print(nz_source.names)
z = nz_source.field('Z_MID')
fields = ['BIN1', 'BIN2', 'BIN3', 'BIN4']
for field in fields:
    plt.plot(z,nz_source.field(field),label=field)
plt.xlim(0, 2.5) 
plt.xlabel('z')
plt.ylabel('Source n(z)')    
plt.legend() 
    

#=====================================================================================================
# xi
cov_mat = hdu[1].data
xi_p = hdu[2].data
xi_m = hdu[3].data
theta = xi_p.field('ANG')
theta = theta[:20]     
np.save('DECADE_SGC_shear_theta_p.npy',theta)
np.save('DECADE_SGC_shear_theta_m.npy',theta)
#print(np.count_nonzero(theta_reduced_p) + np.count_nonzero(theta_reduced_m))
print(cov_mat.shape)

#=====================================================================================================
# Data generation - shear
data_vector_xi = np.concatenate((xi_p.field('VALUE'),xi_m.field('VALUE')))
print(data_vector_xi.shape)
np.save('DECADE_SGC_shear_data_vector.npy',data_vector_xi)
np.save('DECADE_SGC_shear_cov_matrix.npy',cov_mat)
hdu.close()