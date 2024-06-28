Chinese News Title Classification
This project aims to classify Chinese news titles into ten different categories using deep learning models. The models were trained and evaluated on a dataset of Chinese news titles, with a focus on improving accuracy and generalization through hyperparameter tuning and ensemble methods.

Table of Contents
Dataset
Preprocessing
Model Architecture
Training
Evaluation
Results
Hyperparameter Tuning
Conclusion
Future Work
Usage
Dataset
The dataset consists of Chinese news titles labeled into ten categories:

Entertainment
Sports
Real Estate
Automotive
Education
Technology
Military
World
Agriculture
Esports
Preprocessing
Tokenization: Chinese text was tokenized using Jieba.
Vectorization: The tokenized text was converted into numerical features using TF-IDF vectorization.
Handling Missing Values: Missing values in the keyword column were filled with a placeholder.
Model Architecture
Three different models were created with varying architectures to explore their performance:

Model 1:

Dense layer with 128 units
Dropout layer with 50% rate
Dense layer with 64 units
Dropout layer with 50% rate
Output layer with 10 units (softmax activation)
Model 2:

Dense layer with 256 units
Dropout layer with 50% rate
Dense layer with 128 units
Dropout layer with 50% rate
Output layer with 10 units (softmax activation)
Model 3:

Dense layer with 64 units
Dropout layer with 50% rate
Dense layer with 32 units
Dropout layer with 50% rate
Output layer with 10 units (softmax activation)
Training
The models were trained using the following parameters:

Loss function: Sparse categorical crossentropy
Optimizer: Adam
Metrics: Accuracy
Epochs: 1 (for quick comparison)
Batch size: 32
Evaluation
The models were evaluated on a validation set, and their accuracy was recorded. An ensemble model was also created by averaging the predictions of the three individual models.

Results
Model 1 Accuracy: 82.03%
Model 2 Accuracy: 84.02%
Model 3 Accuracy: 76.24%
Ensemble Model Accuracy: 83.96%
Hyperparameter Tuning
Hyperparameter tuning was performed to optimize the model's performance. The following parameters were tuned:

Number of units in dense layers
Dropout rates
Learning rates
Example of tuning with Keras Tuner:

python
Copy code
import keras_tuner as kt

def build_model(hp):
    model = Sequential()
    model.add(Dense(hp.Int('units', min_value=32, max_value=512, step=32), activation='relu', input_dim=X_train.shape[1]))
    model.add(Dropout(hp.Float('dropout', min_value=0.2, max_value=0.5, step=0.1)))
    model.add(Dense(hp.Int('units', min_value=32, max_value=512, step=32), activation='relu'))
    model.add(Dropout(hp.Float('dropout', min_value=0.2, max_value=0.5, step=0.1)))
    model.add(Dense(len(label_encoder.classes_), activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

tuner = kt.RandomSearch(build_model, objective='val_accuracy', max_trials=10, executions_per_trial=2)
tuner.search(X_train, y_train, epochs=5, validation_data=(X_val, y_val))

best_model = tuner.get_best_models(num_models=1)[0]
Conclusion
The project successfully developed and evaluated multiple models for classifying Chinese news titles. However, the models struggled to generalize well to unseen test data, indicating areas for further improvement.

Future Work
Future improvements could involve:

Exploring advanced architectures such as transformer-based models.
Applying cross-validation and more extensive hyperparameter tuning.
Enhancing data preprocessing techniques and data augmentation.
Usage
To use this project, follow these steps:

Clone the repository:

bash
Copy code
git clone https://github.com/yourusername/chinese-news-classification.git
cd chinese-news-classification
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Train the models and make predictions:

python
Copy code
python train.py
python predict.py
Evaluate the results and generate the submission file:

python
Copy code
python evaluate.py
For more details, refer to the individual scripts and notebooks in the repository.
