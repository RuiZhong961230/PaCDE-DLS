# PaCDE-DLS
Parameter adaptive competitive differential evolution with local search

## Abstract
This paper introduces an efficient variant of Competitive Differential Evolution (CDE), named Parameter Adaptive CDE with Dynamic Local Search (PaCDE-DLS), to overcome the limitations of DE, including high-level parameter dependency and problem structure dependency. Two primary components are integrated into PaCDE-DLS to enhance the performance: The Success History Adaptation (SHA) mechanism adjusts control parameters based on past successful evolutionary trajectories, and the Dynamic Local Search (DLS) operator refines candidate solutions in the later optimization stage to achieve a higher convergence accuracy. We implement rigorous experiments in CEC2017, CEC2020, CEC2022, and seven engineering problems to benchmark PaCDE-DLS against ten state-of-the-art optimizers. Additionally, ablation studies further highlight the independent contributions of the SHA mechanism and DLS operator. Moreover, We extend PaCDE-DLS to an ensemble learning framework for potato plant disease diagnosis, where top-3 high-accuracy deep learning models are fused using a PaCDE-DLS-optimized soft voting scheme. Experimental results demonstrate that PaCDE-DLS-Ensemble performed well and achieved an accuracy of 99.768%, precision of 99.772%, recall of 99.768%, and F1 score of 99.767%, which underscores the broad applicability of PaCDE-DLS in real-world scenarios. The source code of this research can be downloaded from https://github.com/RuiZhong961230/PaCDE-DLS.

## Citation
@article{Zhong:25,  
title = {Parameter adaptive competitive differential evolution with local search},  
journal = {Applied Intelligence},  
pages={1-61},  
volume={56},  
year={2026},  
publisher={Springer},  
doi = {https://doi.org/10.1007/s10489-026-07141-0 },  
}

## Datasets and Libraries
CEC benchmarks and Engineering problems are provided by opfunu==1.0.0 and enoppy==0.1.1 libraries, respectively. The potato plant disease dataset is downloaded from Kaggle https://www.kaggle.com/datasets/hafiznouman786/potato-plant-diseases-data.

## Contact
If you have any questions, please don't hesitate to contact zhongrui[at]iic.hokudai.ac.jp.
