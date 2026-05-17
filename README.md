# SkinCare AI — CNN-Based Preliminary Skin Disease Classification

## Project Overview

SkinCare AI is a Deep Learning and Computer Vision project designed to classify dermatological images into selected skin disease categories.

The project focuses on preliminary image classification using Convolutional Neural Networks (CNNs) and Transfer Learning. Several models were tested and compared, including a simple CNN, an improved CNN with regularization techniques, and MobileNetV2.

The best-performing model was MobileNetV2 using Transfer Learning.

> **Important note:** This project is developed for educational and preliminary classification purposes only. It does not replace a professional medical diagnosis by a dermatologist.

---

## Problem Statement

Skin disease classification from images is a challenging task because different dermatological conditions may share similar visual patterns such as redness, inflammation, plaques, or texture changes.

Manual diagnosis requires medical expertise. This project proposes a Deep Learning-based approach to assist in the preliminary classification of skin images.

The goal is to classify an input image into one of the following categories:

- Acne
- Eczema
- Psoriasis
- Rosacea

---

## Dataset

The dataset used in this project is the **Skin Disease Dataset** from Kaggle.

Original dataset link:  
https://www.kaggle.com/datasets/pacificrm/skindiseasedataset

The original dataset contains multiple skin disease classes. After visual inspection and project scoping, four classes were selected:

```text
Acne
Eczema
Psoriasis
Rosacea
```

The `Unknown_Normal` class was excluded because it contained several non-skin images such as objects, rooms, and food, which could introduce noise into the training process.

---

## Project Structure

```text
SkinCareAI/
│
├── README.md
├── requirements.txt
│
├── notebooks/
│   ├── drafts/
│   │   ├── 01_data_exploration.ipynb
│   │   ├── 02_simple_cnn.ipynb
│   │   ├── 03_cnn_dropout_batchnorm.ipynb
│   │   └── 04_mobilenetv2_transfer_learning.ipynb
│   │
│   └── final/
│       └── SkinCareAI_Final_Notebook.ipynb
│
├── models/
│   ├── mobilenetv2_phase1_best_model.keras
│   └── class_indices.json
│
├── app/
│   └── app.py
│
├── results/
│   ├── figures/
│   └── metrics/
│
└── presentation/
    └── SkinCareAI_Presentation.pptx
```

---

## Deep Learning Pipeline

The project follows the following pipeline:

```text
Dataset loading
↓
Class selection
↓
Image preprocessing
↓
Train / validation / test preparation
↓
Model training
↓
Model evaluation
↓
Model comparison
↓
Best model selection
↓
Streamlit application integration
```

---

## Data Preprocessing

The selected images were resized to:

```text
224 × 224 pixels
```

For MobileNetV2, the specific preprocessing function `preprocess_input` was used, as required by the pre-trained MobileNetV2 architecture.

The dataset was divided into:

- Training set
- Validation set
- Test set

The test set was used only for final model evaluation.

---

## Models Tested

Several models were implemented and evaluated during the experimentation phase.

| Model | Test Accuracy | Observation |
|---|---:|---|
| Simple CNN | 51.88% | Baseline model, overfitting observed |
| CNN with Dropout and Batch Normalization | 24.23% | Poor generalization, biased predictions |
| MobileNetV2 Transfer Learning — Phase 1 | 70.65% | Best model |
| MobileNetV2 Fine-Tuning | 67% | Slightly lower than Phase 1 |

---

## Final Model

The final selected model is:

```text
MobileNetV2 Transfer Learning — Phase 1
```

MobileNetV2 was used as a pre-trained CNN model. During the first transfer learning phase, the convolutional base was frozen and only the newly added classification layers were trained on the selected skin disease dataset.

This approach allowed the model to benefit from pre-trained visual features learned from ImageNet while adapting the final classifier to the dermatological image classes.

---

## Final Results

The final MobileNetV2 model achieved the following performance on the test set:

| Metric | Value |
|---|---:|
| Test Accuracy | 70.65% |
| Test Loss | 0.7024 |
| Macro F1-score | 0.72 |
| Weighted F1-score | 0.70 |

### Classification Report

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Acne | 0.81 | 0.77 | 0.79 | 65 |
| Eczema | 0.65 | 0.81 | 0.73 | 112 |
| Psoriasis | 0.68 | 0.51 | 0.58 | 88 |
| Rosacea | 0.81 | 0.75 | 0.78 | 28 |

The model performed well on Acne, Eczema, and Rosacea. Psoriasis remained the most difficult class, probably because it shares visual similarities with other inflammatory skin conditions such as Eczema.

---

## Results Interpretation

The Simple CNN showed signs of overfitting, as the training accuracy was higher than the validation and test accuracy.

The improved CNN with Dropout and Batch Normalization did not improve performance. It mainly predicted one class and failed to correctly classify some categories, which indicates poor generalization.

MobileNetV2 achieved the best results because it uses pre-trained visual features learned from a large dataset. This confirms that Transfer Learning is more suitable for this dermatological image classification task than training a CNN from scratch on a limited dataset.

Fine-tuning was also tested, but it did not improve the overall performance. Therefore, the MobileNetV2 Phase 1 model was selected as the final model.

---

## Streamlit Application

A Streamlit application can be used to perform inference on new skin images.

The application allows the user to:

- Upload a skin image
- Display the uploaded image
- Predict the skin disease class
- Display the confidence score
- Show a medical disclaimer

Example output:

```text
Predicted class: Eczema
Confidence: 82.45%
```

---

## How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/your-username/SkinCareAI.git
cd SkinCareAI
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit application

```bash
streamlit run app/app.py
```

---

## Requirements

Main libraries used in this project:

```text
tensorflow
keras
numpy
pandas
matplotlib
seaborn
scikit-learn
pillow
streamlit
```

---

## Limitations

This project has some limitations:

- The dataset size is limited compared to real clinical datasets.
- Some skin diseases share similar visual characteristics.
- The model is not a medical diagnostic tool.
- Prediction quality may be affected by image quality, lighting, angle, and skin tone variations.
- The system should be validated on larger and more diverse datasets before any real-world use.

---

## Future Work

Future improvements may include:

- Using a larger and cleaner dermatological dataset
- Adding more skin disease classes
- Testing other architectures such as EfficientNet or Vision Transformers
- Improving model explainability with Grad-CAM
- Deploying the Streamlit application online
- Adding a confidence threshold for uncertain predictions

---

## Author

**Souha Guezguez**  
Data Science and Artificial Intelligence Student  
École Polytechnique de Sousse

---

## Disclaimer

This project is for educational purposes only.  
It does not provide medical advice and does not replace a diagnosis by a qualified dermatologist.
