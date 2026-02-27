Chains and data files from the KiDS-Legacy cosmic shear analysis 

Please cite the following papers when using this data:
  - Wright et al. 2025 (A&A, 686, A170)    [KiDS DR5 data release paper]
  - Wright et al. 2025 (A&A, 703, A158)    [KiDS-Legacy cosmic shear analysis]
  - Stölzner et al. 2025 (A&A, 702, A169)  [KiDS-Legacy consistency and joint constraints with external probes]
  - Reischke et al. 2025 (A&A, 699, A124)  [KiDS-Legacy covariance]
  - Wright et al. 2025 (A&A, 703, A144)    [KiDS Legacy redshift calibration]
and include the following acknowledgement in your paper:
  "Based on observations made with ESO Telescopes at the La Silla Paranal Observatory under programme
   IDs 179.A-2004, 177.A-3016, 177.A-3017, 177.A-3018, 298.A-5015."

This repository contains two subfolders:
  - chains_and_config_files: 
    This folder contains four subfolders:
      - cosebis: COSEBIs (Wright et al. 2025)
      - bandpowers: Band powers (Wright et al. 2025)
      - xipm: binned two-point correlation functions (Wright et al. 2025)
      - cosebis_plus_external: COSEBIs + DES Y3 Cosmic shear + DESI Y1 BAO + Pantheon+ SN (Stölzner et al. 2025)
      Note: Fiducial constraints were inferred from COSEBIs and band powers. The PCFs are provided for completeness and for comparison with previous studies.

    Each of these subfolders contains four files: 
      - Fiducial chain generated with CosmoSIS. See the header for column names.
      - pipeline.ini: for running the fiducial chain with CosmoSIS
      - values.ini: CosmoSIS sampling parameters
      - priors.ini: priors for the IA and dz parameters

  - data:
    This folder contains three fits files (one for each statistic) and the dz covariance in ascii format. 
    These data are required for running the pipeline.ini files in chains_and_config_files.
    The fits files contain several extensions:
        - En, Bn: COSEBIs E- and B modes
        - PeeE, PeeB: Band powers E- and B modes
        - xiP, xiM: xi_plus and xi_minus 2PCFs
        - COVMAT: Covariance matrix of COSEBIs E- and B modes, Band powers E- and B modes, or xi_plus and xi_minus 2PCFs
        - NZ_SOURCE: Redshift distribution per tomographic bin
    Additionally, the cosmosis_xipm subdirectory contains 2PCF measurements from treecorr, which are required for modelling the xipm theory predictions.
    Note: the COSEBIs datavector contains 20 modes per tomographic bin combination. In the cosmic shear analysis, we made use of the 
          first six modes (as defined in the scale_cuts module) since the addition of modes 7-20 did not result in any improvement
          of constraining power, returning constraints identical to those measured in the 6 mode case.


Note: The fiducial cosmic shear analysis features a mass-dependent intrinsic alignment model (see sects. 2.4.4 and appendix B in Wright et al.) 
      with a correlated Gaussian prior between the mean halo mass of early-type galaxies per tomographic bin
      and a correlated prior between the amplitude of intrinsic alignments of red galaxies and the slope of the mass scaling of intrinsic galaxy alignments. 
      These are provided in: https://github.com/AngusWright/CosmoPipe/tree/master/ia_models/mass_dependent_ia
      This repository contains the mean and prior file in ascii format and the CosmoSIS module for computing the theory prediction.

Running the fiducial chains requires installing the following packages:
https://github.com/cosmosis-developers/cosmosis
https://github.com/cosmosis-developers/cosmosis-standard-library
https://github.com/maricool/2pt_stats
https://github.com/KiDS-WL/CosmoPowerCosmosis
Additionally, you need to clone the following repository (no installation required for running the chain)
https://github.com/AngusWright/CosmoPipe

To run the chain you need to set the following variables in the pipeline.ini:
  - output_folder: full path to the chain output folder
  - config_folder: full path to the relevant subfolder within chains_and_config_files
  - data_folder: full path to the data folder
  - csl_path: full path to the cosmosis standard library
  - 2pt_stats_path: full path to the 2pt_stats repository
  - cosmopowercosmosis_path: full path to the CosmoPowerCosmosis repository
  - cosmopipe_path: full path to the CosmoPipe repository


