# Content
```
vibtcr
│   README.md
│   ... 
│   
└───data/
│   │   alpha-beta-splits/ (all TCR data, split in two disjoint sets: alpha+beta, beta-only)
│   │   ergo2-paper/ (data used in the ERGO II paper - contains VDJDB and McPAS)
│   │   mhc/ (the NetMHCIIpan-4.0 data)
│   │   nettcr2-paper/ (data used in the NetTCR2.0 paper - contains IEDB, VDJDB and MIRA)
│   │   vdjdb/ (complete VDJdb data from 5th of September)
│   
└───notebooks/
│   │   notebooks.classification/ (TCR-peptide experiments with AVIB/MVIB)
|   │   notebooks.mouse/ (identifying mouse TCRs as suitable OOD dataset)
│   │   notebooks.ood/ (out-of-distribution detection experiments with AVIB)
│   │   notebooks.regression/ (peptide-MHC BA regression with AVIB)
│   
└───tcrmodels/ (Python package which wraps SOTA ML-based TCR models)
│   
└───vibtcr/ (Python package which implements MVIB and AVIB for TCR-peptide interaction prediction)
```

# tcrmodels
`tcrmodels` wraps ML-based TCR prediction models.
So far, it includes:
* [ERGO II](https://github.com/IdoSpringer/ERGO-II)
* [NetTCR2.0](https://github.com/mnielLab/NetTCR-2.0)

### Install `tcrmodels`
```
cd tcrmodels
pip install .
pip install torch===1.4.0 torchvision===0.5.0 -f https://download.pytorch.org/whl/torch_stable.html
```

`tcrmodels` requires Python 3.6

# vibtcr
`vibtcr` is a Python packages which implements the Mutlimodal Variational Information 
Bottleneck (MVIB) and the Attentive Variational Information Bottleneck (AVIB), with a focus on
TCR-peptide interaction prediction.

### Install `vibtcr`
```
cd vibtcr
pip install .
```
Remark: `vibtcr` requires a different version of PyTorch than `tcrmodels`. It's recommended to install them in different environments.
