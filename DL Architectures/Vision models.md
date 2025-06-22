
# 🧠 Computer Vision Tasks and Models Overview

## **Computer Vision Tasks**
1. **Image Classification**
2. **Object Detection**: Localization + Classification
3. **Image Segmentation**: Segmentation + Classification
   - Semantic Segmentation
   - Instance Segmentation
   - Panoptic Segmentation
4. **Landmark Detection / Face Recognition**
5. **OCR (Optical Character Recognition)**: Text Extraction
6. **Visual Tracking**
7. **Action Recognition / Pose Estimation**
8. **3D Construction**

---

## **What is CNN?**
CNN (Convolutional Neural Networks) are partially connected networks that reduce the number of parameters and speed up training.

### **Layers in CNN**
1. **Convolution Layer**: Applies learnable filters to extract features.
2. **ReLU Layer**: Introduces non-linearity.
3. **Max Pooling Layer**: Reduces dimensionality.
4. **Fully Connected Layer**: Connects all neurons from previous layers.

### **Benefits of ReLU**
- **Non-Linearity**: Learns complex relationships.
- **Sparse Outputs**: Reduces computation.
- **Computational Efficiency**: Faster than other activations.
- **Easy to Optimize**: Aids optimizer convergence.

**Max Pooling Formula:**  
`(n + 2p - f) / s + 1`

---

## **Image Classification Architectures**

### 🔹 Basic CNN Models
- **AlexNet-8 (2012)**
- **VGGNet-16, VGGNet-19 (2014)**
- **Inception-v1, v2, v3, v4 (GoogleNet, 2014)**
- **ResNet (2015)**
- **ConvNext (2022)**

### 🔹 Advanced CNN Models
- **DenseNet (2016)**
- **SqueezeNet (2016)**
- **ShuffleNet (2017)**
- **MobileNet (2017)**
- **Xception (2017)**
- **EfficientNet (2019)**, **EfficientNet-V2 (2021)**

### 🔹 Neural Architecture Search Models
- **NASNet (2018)**
- **MNASNet (2019)**

### 🔹 Vision Transformer (ViT)
- **Vision Transformer (2020)**
- **DEVit (2020)**
- **Swin Transformer (2021)**

### **Comparison of Models**
| Model                    | Best Use Case                         |
|--------------------------|----------------------------------------|
| CNN                      | Small datasets, quick implementation   |
| Advanced CNNs            | High accuracy, transfer learning       |
| Neural Architecture Search | Computational optimization          |
| Vision Transformer       | Large datasets, rich relationships     |

---

## **Tasks in Image Classification**
- Binary Classification
- Multi-Class Classification
- Multi-Label Classification
- Facial Recognition (Haar Cascade)
- Image Retrieval
- Image Tagging

---

## **Object Detection**

### 🔹 Single-Stage Detectors
- YOLO (v1–v8)
- YOLO-NAS (2023)
- SSD (2016)
- DSSD (2016)
- RetinaNet (2017)
- EfficientDet (2020)

### 🔹 Two-Stage Detectors
- R-CNN (2014)
- Fast R-CNN (2015)
- Faster R-CNN (2015)
- Mask R-CNN (2017)
- Cascade R-CNN (2018)

### 🔹 Transformer-Based Detectors
- DETR (2020)
- Deformable DETR (2021)
- Swin Transformer (2021)

### 🔹 Real-Time Detectors
- YOLO (v1–v8)
- SSD (2016)
- EfficientDet (2020)
- RT-DETR (2023)

---

## **Image Segmentation Models**

### 🔹 CNN-Based Segmentation
- FCN (2015)
- U-Net (2015)
- SegNet (2016)
- PSPNet (2017)

### 🔹 Region-Based Segmentation
- Mask R-CNN (2017)
- DeepLab v1–v3+ (2015–2018)

### 🔹 Transformer-Based Segmentation
- SETR (2021)
- Swin-Unet (2021)

### 🔹 3D Segmentation
- VoxResNet (2016)
- V-Net (2016)
- 3D U-Net (2016)

### 🔹 Real-Time Segmentation
- ENet (2016)
- BiSeNet (2018)
- Fast-SCNN (2019)

### **Segmentation Model Comparison**
| Model              | Best Use Case                        |
|--------------------|--------------------------------------|
| CNN-Based          | Efficient 2D segmentation, medical imaging |
| Region-Based       | High accuracy for complex objects    |
| Transformer-Based  | Large datasets, contextual understanding |
| 3D Segmentation    | MRI, CT scan analysis                |
