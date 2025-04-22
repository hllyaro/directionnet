# DirectionNet

This repository contains the TensorFlow implementation of the model introduced in the CVPR 2020 paper:

**Wide-Baseline Relative Camera Pose Estimation with Directional Learning**  
Kefan Chen, Noah Snavely, Ameesh Makadia  
IEEE Conference on Computer Vision and Pattern Recognition (CVPR 2020).  
[arXiv](https://arxiv.org/abs/2106.03336)
[Original repo](https://github.com/arthurchen0518/DirectionNet/tree/main?tab=readme-ov-file)

## Requirements

To run the code, perform the following actions:

- Create a conda environment with Python 3.6.* (recommended):
- `conda create -n <your_env_name> python=3.6.9`
- Activate conda environment:
- `conda activate py369`
- Run the following commands:
- `pip install tensorflow`
- `pip install tensorflow==1.15.*`
- `pip install tensorflow-graphics`
- `pip install tensorflow-probability==0.7.0`


```bash
pip install -r requirements.txt
```
If you plan to use GPU acceleration, make sure you have the appropriate versions of CUDA and cuDNN installed. These libraries are required for TensorFlow to utilize your GPU effectively.

- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
- [cuDNN Library](https://developer.nvidia.com/cudnn)


### TODOs:

- At the moment, I am investigating dataset generation


## Citation
```bibtex
@InProceedings{Chen_2021_CVPR,
  author    = {Chen, Kefan and Snavely, Noah and Makadia, Ameesh},
  title     = {Wide-Baseline Relative Camera Pose Estimation With Directional Learning},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month     = {June},
  year      = {2021},
  pages     = {3258-3268}
}
