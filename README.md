# Artificial Neural Network for Churn Prediction

This repository contains a Python implementation of an Artificial Neural Network (ANN) for predicting customer churn based on a dataset of customer demographics, account information, and service usage metrics.

## Table of Contents
- [Project Description](#project-description)
- [Dataset Information](#dataset-information)
- [Model Overview](#model-overview)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Project Description
This project implements an Artificial Neural Network (ANN) model to predict customer churn. The goal is to identify customers who are likely to stop using a service based on their historical data. Churn prediction is critical for companies to maintain customer retention.

The project uses the `Churn_Modelling.csv` dataset, which contains features such as customer demographics and banking activity to predict whether a customer will churn or not.

## Dataset Information
The dataset used in this project, `Churn_Modelling.csv`, contains 14 columns and 10,000 rows. Key columns include:
- **CustomerId**: Unique identifier for each customer
- **Gender**: Gender of the customer
- **Age**: Age of the customer
- **Tenure**: Number of years the customer has been with the company
- **Balance**: Account balance of the customer
- **Exited**: Target variable (1 if the customer churned, 0 otherwise)

The dataset can be found in this repository and is loaded in the notebook for model training.

## Model Overview
The ANN is built using Keras with TensorFlow as the backend. The network architecture consists of the following:
1. **Input Layer**: Takes customer features as input
2. **Two Hidden Layers**: Fully connected layers with 6 neurons each and the ReLU activation function
3. **Output Layer**: A single neuron with the sigmoid activation function to predict churn probability

The model is trained using binary cross-entropy as the loss function, with the Adam optimizer. The final model is evaluated based on accuracy and other metrics such as the confusion matrix and AUC-ROC.

## Installation
To run the model, you need to have Python and the following libraries installed:
- TensorFlow
- Keras
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

You can install the dependencies using the following command:
```bash
pip install -r requirements.txt
```

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ann-churn-prediction.git
   ```
2. Navigate to the project directory:
   ```bash
   cd ann-churn-prediction
   ```
3. Run the Jupyter notebook to train and evaluate the model:
   ```bash
   jupyter notebook artificial_neural_network.ipynb
   ```

Make sure to load the `Churn_Modelling.csv` file when prompted by the notebook.

## Results
The model is evaluated using accuracy, precision, recall, and AUC-ROC. Detailed results, including the confusion matrix and plots of the training process, are available in the Jupyter notebook.

## Contributing
Feel free to contribute to this project by forking the repository and submitting pull requests.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

---
