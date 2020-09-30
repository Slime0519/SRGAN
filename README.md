# SRGAN

## Description
This repository contains my implementation RefSR method proposed in [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/pdf/1609.04802)

![](/Description_image/comparison_image.png)

### Generator
![](/Description_image/Generator.png)
    
Generator is also called SRResNet.    
It has 5 Residual blocks (simpler version), and reconstruct image by pixelshuffler(a.k.a subpixel convolutional layer)
 proposed in [ “Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network (CVPR 2016)](https://arxiv.org/pdf/1609.05158.pdf)
  

### Discriminator
![](/Description_image/Discriminator.png)
   
Discriminator has ordinary structure like traditional Discriminator of GAN.<br>
It has Sequential ( Conv + BN + LeakyReLU ) blocks and downsample input image alternately.   

### Perceptual loss
Proposed in [Perceptual Losses for Real-Time Style Transfer and Super-Resolution ](https://arxiv.org/pdf/1603.08155) paper<br>
![](/Description_image/VGGLoss.png)<br>
To improve perceptual quality of result images, SRGAN use perceptual loss(a.k.a VGG loss).
This function compare two feature map extracted intermediate layer in pretrained VGG-19 network, we can easily implement this function. 
I used pytorch internal library and extract feature map easily.<br>

<table style="padding:10px; margin-left:auto; margin-right:auto" align="center">
  <tr>
      <p><td colspan = "3" span style="color:white" align = "center"><b>Extracted features</td></p>
  </tr>
  <tr>
      <td>Original image</td>
      <td colspan = "2" align="center"><img src="/Description_image/original_input.png" align="center" width = 150px height = 150px ></td>
  </tr>
  <tr>
    <td>VGG22</td>
    <td><img src="/Description_image/VGG22_1.png" align="center" width = 150px height = 150px></td>
    <td><img src="/Description_image/VGG22_2.png" align="center" width = 150px height = 150px></td>
    
   <!--<td><img src="./Scshot/trip_end.png" align="right" alt="4" width =  279px height = 496px></td>-->
  </tr>
  <tr>
    <td>VGG54</td>
    <td><img src="/Description_image/VGG54-1.png" align="center" width = 150px height = 150px></td>
    <td><img src="/Description_image/VGG54-2.png" align="center" width = 150px height = 150px></td>
  </tr>
</table>
<br>
As above table, VGG22 extract low-level features of image, VGG54 extract high-level features of image. I use VGG54, which perform better to generate result.


## Dataset
| Dataset name | usage               | link                                                                   |
|--------------|---------------------|------------------------------------------------------------------------|
| CUFED        | Training/Validation | https://drive.google.com/open?id=1hGHy36XcmSZ1LtARWmGL5OK1IUdWJi3I     |
| CUFED5       | Test                | https://drive.google.com/file/d/1Fa1mopExA9YGG1RxrCZZn7QFTYXLx6ph/view |


## References
1. ["Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network", CVPR 2017](https://arxiv.org/pdf/1609.04802)
2. ["Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network", CVPR 2016](https://arxiv.org/pdf/1609.05158.pdf)
3. ["Perceptual Losses for Real-Time Style Transfer and Super-Resolution", CVPR 2016 ](https://arxiv.org/pdf/1603.08155)
