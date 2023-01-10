# vibtcr
`vibtcr` is a Python package which implements the Mutlimodal Variational Information 
Bottleneck (MVIB) and the Attentive Variational Information Bottleneck (AVIB), with a focus on
TCR-peptide interaction prediction.

![architecture](architecture.png?raw=true "AVIB architecture")

### Install `vibtcr`
```
cd vibtcr
pip install .
```
Remark: `vibtcr` requires a different version of PyTorch than `tcrmodels`. It's recommended to install them in different environments.

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
`tcrmodels` wraps state-of-the-art ML-based TCR prediction models.
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

### ERGO II
Springer I, Tickotsky N and Louzoun Y (2021), Contribution of T Cell Receptor Alpha and Beta CDR3, MHC Typing, V and J Genes to Peptide Binding Prediction. Front. Immunol. 12:664514. DOI: https://doi.org/10.3389/fimmu.2021.664514

### NetTCR-2.0
Montemurro, A., Schuster, V., Povlsen, H.R. et al. NetTCR-2.0 enables accurate prediction of TCR-peptide binding by using paired TCRα and β sequence data. Commun Biol 4, 1060 (2021). DOI: https://doi.org/10.1038/s42003-021-02610-3

# License
For `vibtcr`, we provide a non-commercial license, see LICENSE.txt
