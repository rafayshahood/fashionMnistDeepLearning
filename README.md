
# Fashion MNIST Deep Learning Project

## Introduction
This project focuses on classifying fashion products from the Fashion MNIST dataset using deep learning techniques. The dataset contains 70,000 grayscale images in 10 categories, with each image being 28x28 pixels. Our goal is to develop a model that can accurately classify these images into their respective categories.

## Data Preprocessing
- The data is normalized to ensure the pixel values range between 0 and 1, improving the model's training efficiency.
- Labels are one-hot encoded to match the output format expected by the deep learning model.

## Developing a Model
The project starts with establishing a baseline accuracy that any model should exceed. Given the equal distribution of classes in the dataset, a naive common-sense baseline is set at 10%. The initial aim is to develop the smallest possible model that performs better than this baseline. This involves selecting appropriate loss functions, optimizers, and designing a model architecture that is suitable for multi-class single-label classification.

## Final Model
The final model consists of multiple dense layers with ReLU activation and a softmax activation for the output layer. It incorporates dropout layers to reduce overfitting and uses L2 regularization. The model is trained with an Adam optimizer and a learning rate of 0.001.

## Results
- The final model achieved an accuracy of approximately 87.2% on the test set, significantly surpassing the baseline accuracy.
- The improvement in performance demonstrates the effectiveness of the chosen model architecture and training parameters.

## Conclusion
The project successfully develops a deep learning model that generalizes well to unseen data from the Fashion MNIST dataset. It highlights the importance of data preprocessing, thoughtful model architecture design, and the selection of training parameters in building effective deep learning models.

## Installation and Usage
To replicate this project:
1. Ensure you have Python and TensorFlow installed.
2. Clone the repository and open the `fasionMnistDL.ipynb` notebook in Jupyter Notebook or JupyterLab.
3. Execute each cell to preprocess the data, build and train the model, and evaluate its performance on the test data.
