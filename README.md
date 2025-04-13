# Tensorflow
Overview
This notebook introduces fundamental concepts of TensorFlow, including tensor operations, activation functions, loss functions, and gradient computation. It also demonstrates how to build and train a simple neural network using gradient descent for regression tasks. The notebook is designed for beginners to understand TensorFlow's core functionalities and its application in machine learning.

Features
1. Tensor Operations
Scalar, Vector, Matrix, and Tensor Creation:

Demonstrates the creation of tensors with varying dimensions.

Example:

python
scalar = tf.constant(7)
vector = tf.constant([10, 10])
matrix = tf.constant([[10, 7], [7, 10]])
tensor = tf.constant([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
Broadcasting:

Explains element-wise operations with broadcasting.

Example:

python
tensor_a + tensor_b
2. Gradient Computation
Uses TensorFlow's GradientTape to compute gradients.

Example:

python
with tf.GradientTape() as tape:
    y = x**2
dy_dx = tape.gradient(y, x)
3. Activation Functions
Demonstrates common activation functions (sigmoid, tanh, relu) using TensorFlow's Keras API.

Example:

python
sigmoid = tf.keras.activations.sigmoid(x)
tanh = tf.keras.activations.tanh(x)
relu = tf.keras.activations.relu(x)
4. Loss Functions
Calculates Mean Squared Error (MSE) and Binary Cross-Entropy loss.

Example:

python
mse_fn = tf.keras.losses.MeanSquaredError()
mse = mse_fn(y_true, y_pred)

cross_entropy = tf.keras.losses.binary_crossentropy(y_true, y_pred)
5. Simple Neural Network Implementation
Generates synthetic data for a regression task:

y
=
2
x
+
1
+
noise
y=2x+1+noise

Builds a simple feedforward neural network using Keras Sequential API:

Single dense layer with linear activation.

Trains the model using stochastic gradient descent (SGD) optimizer and MSE loss.

Dependencies
The following Python libraries are required:

tensorflow: For tensor operations and neural network implementation.

numpy: For numerical computations.

matplotlib: For visualizing data and training metrics.

Installation
To install the required libraries, run:

bash
pip install tensorflow numpy matplotlib
Usage Instructions
Clone or download the notebook file to your local machine.

Open the notebook in Jupyter Notebook or Google Colab.

Run all cells sequentially to execute the analysis.

Key Sections
Tensor Operations
Demonstrates the creation of tensors and basic operations like addition and broadcasting.

Gradient Computation
Uses TensorFlow's GradientTape for automatic differentiation to compute gradients.

Activation Functions and Loss Functions
Explains commonly used activation functions and loss functions in machine learning.

Neural Network Training
Data Generation:

Creates synthetic data for regression tasks.

Example:

python
X = np.random.rand(100,1)
y = 2*X + 1 + noise
Model Definition:

Single dense layer with linear activation for regression.

Example:

python
model = Sequential([Dense(1, input_shape=(1,), activation='linear')])
Training:

Trains the model using SGD optimizer and MSE loss for 
100
100 epochs.

Example:

python
model.fit(X, y, epochs=100)
Prediction:

Makes predictions on new data points.

Example:

python
y_pred = model.predict(X_new)
Observations
TensorFlow simplifies mathematical operations on tensors with automatic broadcasting.

Gradient computation is efficiently handled by GradientTape.

The simple neural network accurately predicts linear relationships in the synthetic dataset.

Future Improvements
Extend the neural network to handle more complex datasets or nonlinear relationships.

Explore additional activation functions like Leaky ReLU or Swish.

Implement regularization techniques (e.g., L2 regularization or dropout) to improve model robustness.

License
This project is open-source and available under the MIT License.
