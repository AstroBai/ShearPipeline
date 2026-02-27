# plot cobaya chains KiDS_Legacy_MG/ShearPipeline/cobaya_decade/kids_fix_mnu_chains using getdist
import getdist
import matplotlib.pyplot as plt
import numpy as np
from getdist import plots, loadMCSamples

def plot_kids_chains(chains_path, output_path):
    # Load the MCMC samples from the chain file
    # Use loadMCSamples instead of MCSamples constructor
    samples = loadMCSamples(file_root=chains_path, settings={'ignore_rows': 0.3})  # Adjust ignore_rows as needed

    # Create a triangle plot for the parameters of interest
    g = plots.get_subplot_plotter()
    g.triangle_plot(samples, ['omegam', 'H0', 'logA', 'logfR0'], filled=True)

    # Save the plot
    plt.savefig(output_path)
    plt.close()

if __name__ == "__main__":
    chains_path = 'kids_fix_mnu_chains/test'  # Path to the chain file (without .txt extension)
    output_path = 'kids_fix_mnu_triangle_plot.png'  # Path to save the triangle plot
    plot_kids_chains(chains_path, output_path)