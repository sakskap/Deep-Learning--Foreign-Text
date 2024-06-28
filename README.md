# **Chinese News Title Classification**

This project aims to classify Chinese news titles into ten different categories using deep learning models. The models were trained and evaluated on a dataset of Chinese news titles, with a focus on improving accuracy and generalization through hyperparameter tuning and ensemble methods.

## **Table of Contents**

- [**Dataset**](#dataset)
- [**Preprocessing**](#preprocessing)
- [**Model Architecture**](#model-architecture)
- [**Training**](#training)
- [**Evaluation**](#evaluation)
- [**Results**](#results)
- [**Hyperparameter Tuning**](#hyperparameter-tuning)
- [**Conclusion**](#conclusion)
- [**Future Work**](#future-work)
- [**Usage**](#usage)

## **Dataset**

The dataset consists of Chinese news titles labeled into ten categories:
- **Entertainment**
- **Sports**
- **Real Estate**
- **Automotive**
- **Education**
- **Technology**
- **Military**
- **World**
- **Agriculture**
- **Esports**

## **Preprocessing**

1. **Tokenization**: Chinese text was tokenized using Jieba.
2. **Vectorization**: The tokenized text was converted into numerical features using TF-IDF vectorization.
3. **Handling Missing Values**: Missing values in the `keyword` column were filled with a placeholder.

## **Model Architecture**

Three different models were created with varying architectures to explore their performance:

1. **Model 1**: 
   - Dense layer with **128 units**
   - Dropout layer with **50% rate**
   - Dense layer with **64 units**
   - Dropout layer with **50% rate**
   - Output layer with **10 units** (softmax activation)

2. **Model 2**: 
   - Dense layer with **256 units**
   - Dropout layer with **50% rate**
   - Dense layer with **128 units**
   - Dropout layer with **50% rate**
   - Output layer with **10 units** (softmax activation)

3. **Model 3**: 
   - Dense layer with **64 units**
   - Dropout layer with **50% rate**
   - Dense layer with **32 units**
   - Dropout layer with **50% rate**
   - Output layer with **10 units** (softmax activation)

## **Training**

The models were trained using the following parameters:
- **Loss function**: Sparse categorical crossentropy
- **Optimizer**: Adam
- **Metrics**: Accuracy
- **Epochs**: 1 (for quick comparison)
- **Batch size**: 32

## **Evaluation**

The models were evaluated on a validation set, and their accuracy was recorded. An ensemble model was also created by averaging the predictions of the three individual models.

## **Results**

- **Model 1 Accuracy**: 82.03%
- **Model 2 Accuracy**: 84.02%
- **Model 3 Accuracy**: 76.24%
- **Ensemble Model Accuracy**: 83.96%

## **Hyperparameter Tuning**

Hyperparameter tuning was performed to optimize the model's performance. The following parameters were tuned:
- Number of units in dense layers
- Dropout rates
- Learning rates

## **Conclusion**
The project successfully developed and evaluated multiple models for classifying Chinese news titles. However, the models struggled to generalize well to unseen test data, indicating areas for further improvement.

## **Future Work**
Future improvements could involve:

- Exploring advanced architectures such as transformer-based models.
- Applying cross-validation and more extensive hyperparameter tuning.
- Enhancing data preprocessing techniques and data augmentation.

