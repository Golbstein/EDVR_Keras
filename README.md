# EDVR in Keras
EDVR: Video Restoration with Enhanced Deformable Convolutional Networks

Keras implementation of **"Video Restoration with Enhanced Deformable Convolutional Networks" CVPR, 2019**: [EDVR](https://arxiv.org/abs/1905.02716)
[Based on author's Pytorch implementation](https://github.com/xinntao/EDVR)

# Model Architecture
![alt text](https://github.com/Golbstein/EDVR_Keras/blob/master/assets/arch.JPG)

# TODO:
**Help needed in translation of authors' DFA module to Keras\Tensorflow**: [DCN_V2](https://github.com/xinntao/EDVR/tree/master/codes/models/modules/DCNv2)

- [x] PreDeblur Module
- [ ] Alignment with Pyramid, Cascading and Deformable Convolution (PCD) Module
- [x] Temporal Spatial Attention (TSA) Fusion Module
- [x] Reconstruction Module
- [x] Pixel shuffle layer (Subpixel)
- [ ] Support changing frame shape (currently supports only fixed shape)


# Results
![alt text](https://github.com/Golbstein/EDVR_Keras/blob/master/assets/res.JPG)


* ## Dependencies
  - **Python 3.6**
  - **Keras>2.2.x**
