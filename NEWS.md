# fdaPDE 2.0

## New features

1) smooth regression for manifold and volumetric domains 
2) smooth regression with areal data 
3) functional smooth PCA (SF-PCA algorithm) : function smooth.FEM.FPCA
4) Stochastic GCV computation has been added: parameter 'GCVmethod' can be used both regression and FPCA, can be either 'Stochastic' or 'Exact'.
5) Kfold cross-validation is available in FPCA algorithm.

## Deprecated functions and name changes

1) smooth.FEM.PDE.basis and smooth.FEM.PDE.sv.basis are deprecated, use smooth.FEM.basis instead.
2) The usage of 'MESH' in classes and function names has been deprecated, now use 'mesh'.
3) Meuse dataset has been removed. New datasets are provided.
4) The parameter CPP_CODE has been removed, the R-only functions (whose names began with R_) have been removed.

## Bug fixes
A bug in R/C++ indexes conversion in the space-varying regression has been fixed.
