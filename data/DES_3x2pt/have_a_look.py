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
file_path = "2pt_NG_final_2ptunblind_02_26_21_wnz_maglim_covupdate.fits"
hdu = fits.open(file_path)
#hdu.info()
#print(hdu[2].data.names)
#=====================================================================================================
# nz
plt.figure(figsize=(16,16))
plt.subplot(2,1,1)
nz_source = hdu[6].data
#print(nz_source.names)
z = nz_source.field('Z_MID')
fields = ['BIN1', 'BIN2', 'BIN3', 'BIN4']
for field in fields:
    plt.plot(z,nz_source.field(field),label=field)
plt.xlim(0, 2.5) 
plt.xlabel('z')
plt.ylabel('Source n(z)')    
plt.legend() 
    
plt.subplot(2,1,2)
nz_lens = hdu[7].data
z = nz_lens.field('Z_MID')
fields = ['BIN1', 'BIN2', 'BIN3', 'BIN4', 'BIN5','BIN6']
for field in fields:
    plt.plot(z,nz_lens.field(field),label=field)    
plt.xlim(0, 2.5) 
plt.xlabel('z')
plt.ylabel('Lens n(z)')     
plt.legend()  
plt.tight_layout()    
plt.savefig('./outfiles/DES_nz.png')    

#=====================================================================================================
# xi
Cov = hdu[1].data
errors = np.sqrt(np.diag(Cov))
plt.figure(figsize=(30,25))
xi_plus = hdu[2].data
#print(xi_plus)
bin1s = xi_plus.field('BIN1')
bin2s = xi_plus.field('BIN2')
theta = xi_plus.field('ANG')
theta = theta[:20]
value = xi_plus.field('VALUE')
i = 0
theta_reduced_p = np.zeros((10,20))
theta_index = 0
for bin1 in range(4):
    for bin2 in range(4 - bin1):
        index = bin1 * 5 + bin2 + 1
        plt.subplot(5,5,index)
        plt.errorbar(theta, value[20*i:20*(i+1)], yerr=errors[20*i:20*(i+1)], fmt='o', color='k')
        i += 1
        plt.xscale('log')
        plt.yscale('log') 
        plt.xlabel(r'$\theta(\mathrm{arcmin})$')
        plt.ylabel(r'$\xi_{+}$')
        plt.legend([f'{bin1 + 1},{bin2+bin1+ 1}'])
        if bin1+1 == 1 and bin2+bin1+1 == 4:
            plt.fill_between([0,5],1e-7, 1e-4, facecolor='grey', alpha=0.5) 
            theta_ = theta[theta>5]
            theta_reduced_p[theta_index,:len(theta_)] = theta_
        elif bin1+1 == 1 and bin2+bin1+1 == 1:  
            plt.fill_between([0,2],1e-7, 1e-4, facecolor='grey', alpha=0.5)  
            theta_ = theta[theta>2]
            theta_reduced_p[theta_index,:len(theta_)] = theta_  
        elif bin1+1 == 4 and bin2+bin1+1 == 4:
            plt.fill_between([0,5],1e-7, 1e-4, facecolor='grey', alpha=0.5) 
            theta_ = theta[theta>5]
            theta_reduced_p[theta_index,:len(theta_)] = theta_   
        else:
            plt.fill_between([0,6],1e-7, 1e-4, facecolor='grey', alpha=0.5)                                    
            theta_ = theta[theta>6]
            theta_reduced_p[theta_index,:len(theta_)] = theta_   
        plt.ylim(1e-7, 1e-4)
        theta_index += 1
xi_plus = hdu[3].data
bin1s = xi_plus.field('BIN1')
bin2s = xi_plus.field('BIN2')
theta = xi_plus.field('ANG')
theta = theta[:20]
value = xi_plus.field('VALUE')
i = 0        
theta_reduced_m = np.zeros((10,20))
theta_index = 0
for bin1 in range(4):
    for bin2 in range(4 - bin1):
        index = bin1 * 5 + bin2 + 1
        plt.subplot(5,5,25-index+1)
        plt.errorbar(theta, value[20*i:20*(i+1)], yerr=errors[200+20*i:200+20*(i+1)], fmt='o', color='k')
        i += 1
        plt.xscale('log')
        plt.yscale('log') 
        plt.xlabel(r'$\theta(\mathrm{arcmin})$')
        plt.ylabel(r'$\xi_{-}$')    
        plt.legend([f'{bin1 + 1},{bin2+bin1+ 1}'])   
        if bin1+1 == 4 and bin2+bin1+1 == 4:
            plt.fill_between([0,60],1e-7, 1e-4, facecolor='grey', alpha=0.5) 
            theta_ = theta[theta>60]
            theta_reduced_m[theta_index,:len(theta_)] = theta_  
        elif bin1+1 == 2 and bin2+bin1+1 == 2:  
            plt.fill_between([0,60],1e-7, 1e-4, facecolor='grey', alpha=0.5)  
            theta_ = theta[theta>60]
            theta_reduced_m[theta_index,:len(theta_)] = theta_  
        elif bin1+1 == 1 and bin2+bin1+1 == 2:  
            plt.fill_between([0,60],1e-7, 1e-4, facecolor='grey', alpha=0.5)  
            theta_ = theta[theta>60]
            theta_reduced_m[theta_index,:len(theta_)] = theta_  
        elif bin1+1 == 1 and bin2+bin1+1 == 3:  
            plt.fill_between([0,60],1e-7, 1e-4, facecolor='grey', alpha=0.5)    
            theta_ = theta[theta>60]
            theta_reduced_m[theta_index,:len(theta_)] = theta_          
        elif bin1+1 == 1 and bin2+bin1+1 == 1:
            plt.fill_between([0,25],1e-7, 1e-4, facecolor='grey', alpha=0.5)   
            theta_ = theta[theta>25]
            theta_reduced_m[theta_index,:len(theta_)] = theta_  
        elif bin1+1 == 1 and bin2+bin1+1 == 4:  
            plt.fill_between([0,50],1e-7, 1e-4, facecolor='grey', alpha=0.5)     
            theta_ = theta[theta>50]
            theta_reduced_m[theta_index,:len(theta_)] = theta_   
        else:
            plt.fill_between([0,80],1e-7, 1e-4, facecolor='grey', alpha=0.5)  
            theta_ = theta[theta>80]
            theta_reduced_m[theta_index,:len(theta_)] = theta_    
        plt.ylim(1e-7, 1e-4)
        theta_index += 1
        
plt.tight_layout()    
plt.savefig('./outfiles/DES_xi.png')        
#np.save('DES_shear_theta_p.npy',theta_reduced_p)
#np.save('DES_shear_theta_m.npy',theta_reduced_m)
#print(np.count_nonzero(theta_reduced_p) + np.count_nonzero(theta_reduced_m))
print(theta,'xi')
#=====================================================================================================
# gamma
plt.figure(figsize=(30,20))
gamma = hdu[4].data
print(gamma.names)
print(gamma)
print(hdu[5].header,'HEADER')

bin1s = gamma.field('BIN1')
bin2s = gamma.field('BIN2')
theta = gamma.field('ANG')

theta = theta[:20]
print(theta,'gamma')
value = gamma.field('VALUE')
i = 0
theta_reduced = np.zeros((24,20))
theta_index = 0
for bin1 in range(6):
    for bin2 in range(4):
        index = bin2 * 6 + bin1 + 1
        ax = plt.subplot(4,6,index)
        plt.scatter(theta,value[20*i:20*(i+1)],c='k')
        i += 1
        plt.xscale('log')
        plt.yscale('log') 
        plt.legend([f'{bin1 + 1},{bin2+1}'])
        
        if bin1+1 == 1:
            plt.fill_between([0,25],1e-7, 1e-2, facecolor='grey', alpha=0.5) 
            theta_ = theta[theta>25]
            theta_reduced[theta_index,:len(theta_)] = theta_  
        elif bin1+1 == 2:  
            plt.fill_between([0,20],1e-7, 1e-2, facecolor='grey', alpha=0.5)  
            theta_ = theta[theta>20]
            theta_reduced[theta_index,:len(theta_)] = theta_  
        elif bin1+1 == 3:  
            plt.fill_between([0,12],1e-7, 1e-2, facecolor='grey', alpha=0.5)  
            theta_ = theta[theta>12]
            theta_reduced[theta_index,:len(theta_)] = theta_  
        elif bin1+1 == 4:  
            plt.fill_between([0,11],1e-7, 1e-2, facecolor='grey', alpha=0.5)    
            theta_ = theta[theta>11]
            theta_reduced[theta_index,:len(theta_)] = theta_          
        else:
            plt.fill_between([0,225],1e-7, 1e-2, facecolor='grey', alpha=0.5)  
            theta_ = theta[theta>230]
            theta_reduced[theta_index,:len(theta_)] = theta_    
    
        plt.ylim(1e-6, 1e-2)
        if bin1 == 0:
            plt.ylabel(r'$\gamma_t$') 
        else:
            ax.yaxis.set_tick_params(labelleft=False) 
        if bin2 == 3:
            plt.xlabel(r'$\theta(\mathrm{arcmin})$')
        else:
            ax.xaxis.set_tick_params(labelbottom=False) 
        theta_index += 1    
            
print(theta_reduced)        
np.save('DES_gamma_theta.npy',theta_reduced)    
plt.subplots_adjust(wspace=0, hspace=0)             
plt.savefig('./outfiles/DES_gamma.png')
            
#=====================================================================================================
# w
plt.figure(figsize=(40,8))
w = hdu[5].data
print(w.names)
print(w)
bin1s = w.field('BIN1')
bin2s = w.field('BIN2')
theta = w.field('ANG')
theta = theta[:20]
print(theta,'w')
value = w.field('VALUE')
i = 0
theta_reduced = np.zeros((6,20))
theta_index = 0
for bin1 in range(6):
    for bin2 in range(1):
        index = bin2 * 6 + bin1 + 1
        ax = plt.subplot(1,6,index)
        plt.scatter(theta,value[20*i:20*(i+1)],c='k')
        i += 1
        plt.xscale('log')
        plt.yscale('log') 
        plt.legend([f'{bin1 + 1},{bin2+1}'])
        
        if bin1+1 == 1:
            plt.fill_between([0,32],1e-5, 1e-1, facecolor='grey', alpha=0.5) 
            theta_ = theta[theta>32]
            theta_reduced[theta_index,:len(theta_)] = theta_  
        elif bin1+1 == 2:  
            plt.fill_between([0,25],1e-5, 1e-1, facecolor='grey', alpha=0.5)  
            theta_ = theta[theta>25]
            theta_reduced[theta_index,:len(theta_)] = theta_  
        elif bin1+1 == 3:  
            plt.fill_between([0,17],1e-5, 1e-1, facecolor='grey', alpha=0.5)  
            theta_ = theta[theta>17]
            theta_reduced[theta_index,:len(theta_)] = theta_  
        elif bin1+1 == 4:  
            plt.fill_between([0,15],1e-5, 1e-1, facecolor='grey', alpha=0.5)    
            theta_ = theta[theta>15]
            theta_reduced[theta_index,:len(theta_)] = theta_          
        else:
            plt.fill_between([0,225],1e-5, 1e-1, facecolor='grey', alpha=0.5)  
            theta_ = theta[theta>230]
            theta_reduced[theta_index,:len(theta_)] = theta_    
        
        plt.ylim(1e-4, 1e-1)
        if bin1 == 0:
            plt.ylabel(r'$w$') 
        else:
            ax.yaxis.set_tick_params(labelleft=False) 
        if bin2 == 0:
            plt.xlabel(r'$\theta(\mathrm{arcmin})$')
        else:
            ax.xaxis.set_tick_params(labelbottom=False)     
        theta_index+=1            
print(np.count_nonzero(theta_reduced))         
np.save('DES_w_theta.npy',theta_reduced)         
plt.subplots_adjust(wspace=0, hspace=0) 
plt.savefig('./outfiles/DES_w.png')

#=====================================================================================================
# Cov
Cov = hdu[1].data
#print(Cov.shape)
def pooling(matrix, pool_size):
    pooled_matrix = np.zeros((matrix.shape[0] // pool_size, matrix.shape[1] // pool_size))
    for i in range(0, matrix.shape[0], pool_size):
        for j in range(0, matrix.shape[1], pool_size):
            pooled_matrix[i//pool_size, j//pool_size] = np.mean(matrix[i:i+pool_size, j:j+pool_size])
    return pooled_matrix

pool_size = 1
pooled_matrix = pooling(Cov, pool_size)
nonzero_min = np.min(pooled_matrix[pooled_matrix > 0])
plt.figure(figsize=(16, 14))
sns.heatmap(pooled_matrix, cmap='rocket', annot=False,xticklabels=False, yticklabels=False, norm=LogNorm(vmin=nonzero_min, vmax=np.max(pooled_matrix)))
plt.title('Covariance Matrix')
#plt.xlabel('Index')
#plt.ylabel('Index')
plt.tight_layout()    
plt.savefig('./outfiles/DES_cov.png')  


#=====================================================================================================
# Data generation - shear
xi_plus_data = hdu[2].data
xip = xi_plus_data.field('VALUE')
xi_minus_data = hdu[3].data
xim = xi_minus_data.field('VALUE')
xi = np.concatenate([xip,xim])
print(xi.shape,'!!')
indices_xi = np.concatenate((np.arange(0,20), 
                          np.arange(24,40),
                          np.arange(44,60),
                          np.arange(63,80),
                          np.arange(84,100),
                          np.arange(104,120),
                          np.arange(124,140),
                          np.arange(144,160),
                          np.arange(164,180),
                          np.arange(183,200),
                          np.arange(210,220),
                          np.arange(234,240),
                          np.arange(254,260),
                          np.arange(273,280),
                          np.arange(294,300),
                          np.arange(315,320),
                          np.arange(335,340),
                          np.arange(355,360),
                          np.arange(375,380),
                          np.arange(394,400)))

data_vector_xi = xi[indices_xi]
cov_mat = Cov[np.ix_(indices_xi, indices_xi)]
print(data_vector_xi.shape)
print(cov_mat.shape)

#np.save('DES_shear_data_vector.npy',data_vector_xi)
#np.save('DES_shear_cov_matrix.npy',cov_mat)

plt.figure(figsize=(16, 14))
sns.heatmap(cov_mat, cmap='rocket', annot=False,xticklabels=False, yticklabels=False, norm=LogNorm(vmin=nonzero_min, vmax=np.max(pooled_matrix)))
plt.title('Covariance Matrix')
#plt.xlabel('Index')
#plt.ylabel('Index')
plt.tight_layout()    
plt.savefig('./outfiles/DES_cov_reduced.png') 


#=========================================================
#gamma data vector


gamma_data = hdu[4].data
gamma = gamma_data.field('VALUE')
print(gamma.shape,'!!')
indices_gamma = np.concatenate((np.arange(10,20), 
                          np.arange(30,40),
                          np.arange(50,60),
                          np.arange(70,80),  #
                          
                          np.arange(89,100),
                          np.arange(109,120),
                          np.arange(129,140),
                          np.arange(149,160),  #
                          
                          np.arange(167,180),
                          np.arange(187,200),
                          np.arange(207,220),
                          np.arange(227,240),  #
                          
                          np.arange(246,260),
                          np.arange(266,280),
                          np.arange(286,300),
                          np.arange(306,320)))
data_vector_gamma = gamma[indices_gamma]
np.save('DES_gamma_data_vector.npy',data_vector_gamma)
#=========================================================
#w data vector


w_data = hdu[5].data
w = w_data.field('VALUE')
print(w.shape,'!!')
indices_w = np.concatenate((np.arange(11,20), 
                          np.arange(30,40),
                          np.arange(48,60),
                          np.arange(68,80)))

data_vector_w = w[indices_w]
np.save('DES_w_data_vector.npy',data_vector_w)

#========================================================
#3x2pt covmat

indices_cov = np.concatenate((indices_xi,indices_gamma+len(xi),indices_w+len(xi)+len(gamma)))
cov_3by2 = Cov[np.ix_(indices_cov,indices_cov)]
print(cov_3by2.shape,'!!')
np.save('DES_3by2_cov_matrix.npy',cov_3by2)
vector_3by2 = np.concatenate((xi[indices_xi],gamma[indices_gamma],w[indices_w]))
print(vector_3by2.shape,'!!')
np.save('DES_3by2_data_vector.npy',vector_3by2)
hdu.close()