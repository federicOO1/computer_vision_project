#  License Plate Detection and Recognition

This project addresses the full pipeline of **license plate detection and recognition** using deep learning. We experimented with multiple models—both classical and modern—to evaluate their effectiveness and generalization capabilities across different datasets.

---

##  Dataset

- The dataset used is based on the **CCPD (Chinese City Parking Dataset)**, which contains real-world vehicle images with annotated license plates.
- We used a **meta version of CCPD** including images from different subsets (`Base`, `Challenge`, etc.).
- **Download link:** <https://github.com/detectRecog/CCPD>

---

##  Detection

We tested two detection approaches:

### 1.  Baseline CNN  
A simple convolutional model trained for binary classification to predict bounding boxes. It served only as an initial reference and performed poorly in generalization.

### 2.  YOLOv5  
We implemented **YOLOv5** for robust and real-time license plate detection. YOLOv5 works by dividing the image into grids and predicting bounding boxes and class probabilities in a single forward pass.

> **Input** → Convolutional backbone → Feature Pyramid → Bounding box & confidence output

YOLOv5 gave significantly better detection results and was used to crop license plate regions from images for the recognition stage.

---

##  Recognition

Once the license plates were cropped using YOLOv5, we focused on the **recognition** step.

###  Baseline OCR

We first tested basic OCR tools like:
- **Tesseract**
- **EasyOCR**

These tools were used **without fine-tuning** and produced **poor results**, especially under varied lighting, occlusion, and angles.

---

###  Deep Learning Models

We then trained and evaluated several deep learning models on the cropped plate images, applying **data augmentation** to increase robustness:

####  Data Augmentation used:
- **Color jitter**
- **Small random rotations**
- **Random erasing**

---

### 1.  Holistic CNN  
A lightweight CNN with a shared feature extractor and **multiple classification heads** for each character position.

### 2.  CRNN (with CTC Loss)  
A model combining CNN for feature extraction and RNN for sequential modeling. Trained using **CTC Loss** to handle variable-length license plates.

### 3.  LPRNet (with CTC Loss)  
A deeper architecture that performs well in structured environments but suffers from poor generalization on more complex images.

Although these models performed well on the **Base dataset**, they **struggled to generalize** to other test sets (e.g., `Challenge`, `Weather`, etc.).

---

##  Final Model: PDLPRNet

The final and most successful model was **PDLPRNet**, composed of:

- **IGFE (Improved Global Feature Extractor)**: Enhances global understanding of the input plate.
- **Encoder**: Captures deep representations and positional info.
- **Parallel Decoder**: Predicts all license plate characters in parallel, improving both speed and accuracy.

PDLPRNet achieved **good performance** and **better generalization** across different datasets.

- **Training**: On ~100k images from the **Base dataset**
- **Testing**: On the rest of CCPD subsets
- **Loss**: Label-smoothing cross-entropy
- **Optimizer**: Adam with learning rate scheduler and warmup

> **Note:** Final recognition performance still heavily depends on the **quality of YOLOv5 detection**, especially on challenging datasets.

---

## Results

- Used Metrics: **Sequence Accuracy** and **Character Accuracy**
- PDLPRNet outperformed all previous models in both accuracy and generalization.
- Recognition performance was best when detection IOU ≥ 0.7.

---

## Future Work

- Improve detection accuracy for more challenging datasets.
- Train recognition model with **multi-domain data**.
- Apply **end-to-end joint training** (YOLO + recognition).
- Add more robust **augmentation** techniques.

---

## References

- Wang, Z., Xu, L., Li, X., et al.  
  *Towards End-to-End License Plate Detection and Recognition: A Large Dataset and Baseline*  
  [paper link](https://openaccess.thecvf.com/content_ECCV_2018/papers/Zhenbo_Xu_Towards_End-to-End_License_ECCV_2018_paper.pdf)
