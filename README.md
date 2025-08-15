
# 🧠 Deep Learning-Based Automated Brain Tumor Segmentation from MRI Images Using a U-Net Convolutional Neural Network

A deep learning project for automatic brain tumor segmentation from **MRI scans** using **the U-Net architecture**.
Developed as part of my Final Year Project (PFE) in **Biomedical Engineering**, this work integrates **medical imaging**, **artificial intelligence**, and **healthcare innovation**.

---

🚀 Overview

Accurate segmentation of brain tumors from MRI scans is essential for **diagnosis**, **treatment planning**, and follow-up in **neuro-oncology**.
This project implements a U-Net convolutional neural network, trained on public MRI datasets and fine-tuned using clinical MRI data from **the Oncology and Hematology Center of the Mohammed VI University Hospital in Marrakech**.

An interactive Streamlit application called **NeuroSeg** is provided, enabling real-time testing and visualization of the model’s segmentation results.

---

## 📚 Dataset

The model was trained using the **LGG MRI Segmentation Dataset** provided by [Mateusz Buda on Kaggle](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation).  
This dataset comprises 110 cases of lower-grade glioma (LGG) patients, including MRI scans with manual segmentation masks for FLAIR abnormalities.  
The dataset was sourced from The Cancer Imaging Archive (TCIA) and is publicly available for research purposes.

In addition, the model was fine-tuned on clinical MRI data from the Oncology and Hematology Center of the Mohammed VI University Hospital in Marrakech.

⚠️ The hospital data is private and cannot be shared publicly.
Number of images: 196 MRI scans

Number of masks: 196 corresponding segmentation masks

Number of patients: 10

Tumor type: Glioblastoma

This combination of public and clinical data improves model performance and ensures that the system generalizes well to real-world cases.

---

## 📌 Highlights

- **Architecture**: U-Net optimized for medical image segmentation.
- **Performance**: High Dice similarity score for precise tumor boundary detection.
- **Preprocessing**: Image normalization, resizing, and augmentation.
- **Fine-tuning**: Model trained and refined for better generalization.
- **Deployment**: Streamlit-based web app for quick demos.
- **External Model Storage**: Pre-trained weights hosted on Google Drive.

---

## 📊 Results

| Metric       | Score |
|--------------|-------|
| Dice Score   | 0.8443  |
| Loss          | 0.557  |
**Example outputs:**

 ![](Results.jpg) 



