# Fruits and Vegetables Classification using CNN
# Introduction
This project demonstrates the application of Convolutional Neural Networks (CNN) for classifying images of fruits and vegetables. The model is trained on a dataset containing images of various fruits and vegetables, and it is designed to identify the type of fruit or vegetable in an image.

# Dataset
The dataset used in this project consists of images of fruits and vegetables, divided into training, validation, and test sets. The dataset is structured into different directories for each class of fruit or vegetable.
https://www.kaggle.com/datasets/muhriddinmuxiddinov/fruits-and-vegetables-dataset


# Data Preprocessing
Loading and Visualizing Data
The dataset is loaded using TensorFlow's image_dataset_from_directory function. The images are resized to 180x180 pixels, and batches of 32 images are created for training, validation, and testing. The class names are extracted from the dataset directory structure.

# Displaying Sample Images
To get an idea of the dataset, a few sample images from the training set are displayed along with their labels. This helps in understanding the variety and structure of the dataset.

# Model Building
# Model Architecture
The model architecture is built using a sequential CNN. It includes the following layers:

Rescaling Layer: Normalizes pixel values by scaling them to the range [0, 1].
Convolutional Layers: Extracts features from the images using convolution operations with ReLU activation.
MaxPooling Layers: Reduces the spatial dimensions of the feature maps.
Flatten Layer: Converts the 2D feature maps into a 1D vector.
Dropout Layer: Reduces overfitting by randomly dropping a fraction of the input units.
Dense Layers: Performs classification using fully connected layers with ReLU and Softmax activations.
Compiling the Model
The model is compiled using the Adam optimizer, Sparse Categorical Crossentropy loss function, and accuracy as the evaluation metric.

# Training the Model
The model is trained for 25 epochs using the training dataset. The validation dataset is used to monitor the model's performance and prevent overfitting.

# Model Evaluation
The trained model's performance is evaluated using the validation dataset. Accuracy and loss curves are plotted to visualize the training and validation performance over the epochs.

# Predicting and Saving the Model
An image is loaded and preprocessed for prediction. The model predicts the class of the fruit or vegetable in the image, and the predicted class along with the accuracy is printed. The model is then saved for future use.