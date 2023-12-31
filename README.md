# NeRVI: Compressive Neural Representation of Visualization Images for Communicating Volume Visualization Results
This repository contains the PyTorch implementation for paper "NeRVI: Compressive Neural Representation of Visualization Images for Communicating
Volume Visualization Results".

# Prerequisites
* Linux
* CUDA >= 10.0
* Python >= 3.7
* Numpy
* Pytorch >= 1.0

# How to run the code
* First, change the directory path and iso values of the data in dataio.py.
* Second, set the parameter settings (e.g., batch size, lr etc.) in main.py.
* For training, set the 'train' as train and the 'approach' as CNN in the main.py.
* For testing, set the 'train' as inf.

# Citation
@article{gu2023nervi,<br/>
  title={NeRVI: Compressive neural representation of visualization images for communicating volume visualization results},<br/>
  author={Gu, Pengfei and Chen, Danny Z and Wang, Chaoli},<br/>
  journal={Computers \& Graphics},<br/>
  volume={116},<br/>
  pages={216--227},<br/>
  year={2023},<br/>
  publisher={Elsevier}<br/>
}
# Acknowledgements
This research was supported in part by the U.S. National Sci3 ence Foundation through grants CNS-1629914, DUE-1833129, IIS-1955395, IIS-2101696, and OAC-2104158, and the U.S.
Department of Energy through grant DE-SC0023145. The authors thank Ziang Tong for developing the visual interface.
