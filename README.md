# ISIC Kaggle Competition Github

This repository contains the Kaggle notebooks used for my submission in the [ISIC 2024 Kaggle Competition](https://www.kaggle.com/competitions/isic-2024-challenge).

## Dataset
The provided dataset consists of labeled images of skin lesions, along with additional metadata. This dataset contains 401,059 images, with 400,666 negative (benign) samples and 393 positive (malignant) samples.

## Evaluation
Submissions are evaluated using partial area under the ROC curve above 80% true positive rate. This is because in clinical practice, true positive rate below a certain threshold (here 80%) is unacceptable, as cancer will too often go undiagnosed.

## Current Model Architecture
The current model's architecture has two main components:
1. **EfficientNet:** The first component of this model is an Convolutional Neural Network using the EfficientNet B0 architecture, trained for 4 epochs. This model predicts the class of each sample based purely on the images provided.
2. **Catboost & LightGBM Ensemble:** The second component of this architecture is an ensemble of CatBoost and LightGBM gradient boosting machines. These models are trained on the metadata associated with each image, as well as the EfficientNet predictions.

Notebooks for both of these can be found in this repository.

## Dataset Imbalance
The dataset contains a severe class imbalance, with 400,666 negative samples and only 393 positive samples. To prevent bias, three methods are employed:
1. **Data Augmentation:** The existing images were augmented using random resizing, cropping, flipping, rotations, and color jitters before deep learning models were trained on the images. This helped to prevent overfitting when coupled with oversampling.
2. **Oversampling:** In an attempt to prevent bias towards one class, the image classification models were trained using oversampling, such that they were presented with the same number of images for both classes.
3. **Bagging:** The LightGBM model utilized bagging to present the model with a more balanced number of each class.
4. **Weighted Loss Function:** The CatBoost model was provided weights to account for class imbalance in the loss function.

## Models Tested
In addition to EfficientNet, multiple other deep learning models were trained. However, the baseline performance of EfficientNet was significantly better than these models, causing it to be selected.
1. **EfficientNet B0:** Baseline pAUC: 0.120
2. **MobileNet:** Baseline pAUC: 0.090
3. **ResNet:** Baseline pAUC: 0.050
4. **DenseNet:**
5. **Vision Transformers:** Currently being trained and tested. I plan to try to add a transformer-based model for diversity of methods.

## Note on Overfitting
Initially after adding EfficientNet predictions to the LightGBM & CatBoost ensemble, I saw a decrease in model performance. I realized this was because EfficientNet had been slightly overfit on the training data.
Generally, slight overfitting is fine, since in most cases validation loss is still decreasing, albeit at a slower rate. However, I noticed that in this case, EfficientNet predictions for training data (the same data
which the gradient boosting machines are trained on) were more accurate than for testing data, causing an over-reliance on these predictions, and decreasing model performance on the test data.
To remedy this issue, I used an earlier epoch where overfitting had not yet occured, and saw significant performance improvements.
