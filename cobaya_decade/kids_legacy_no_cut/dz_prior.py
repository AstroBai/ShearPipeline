import numpy as np

# Load covariance once
cov_path = "/home/jbai/KiDS_Legacy_MG/ShearPipeline/data/KiDS_Legacy_cosmic_shear_data_release/Nz_covariance.txt"
cov = np.loadtxt(cov_path)
inv_cov = np.linalg.inv(cov)

def dz_prior(dz1_kids, dz2_kids, dz3_kids, dz4_kids, dz5_kids, dz6_kids):
    dzs = np.array([dz1_kids, dz2_kids, dz3_kids, dz4_kids, dz5_kids, dz6_kids])
    chi2 = np.dot(dzs, np.dot(inv_cov, dzs))
    return -0.5 * chi2
