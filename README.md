# MNIST Digit Recognizer — Neural Network from Scratch

A 2-layer neural network built with **only NumPy** (no TensorFlow, PyTorch, or Keras) to classify handwritten digits from the [Kaggle Digit Recognizer](https://www.kaggle.com/competitions/digit-recognizer) competition, based on the classic MNIST dataset.

## Overview

This project implements forward propagation, backpropagation, and gradient descent manually to train a simple feedforward network that classifies 28x28 grayscale images of handwritten digits (0–9).

**Dev set accuracy:** ~87% after 500 iterations of gradient descent.

## Architecture

- **Input layer:** 784 units (28x28 flattened pixel values, normalized to [0, 1])
- **Hidden layer:** 10 units, ReLU activation
- **Output layer:** 10 units, softmax activation (one per digit class)

```
Input (784) → Dense (10, ReLU) → Dense (10, Softmax) → Prediction
```

## Data

- `train.csv` — 42,000 labeled images (`label` + 784 pixel columns)
- `test.csv` — 28,000 unlabeled images for submission
- Training data is split into a 1,000-example **dev set** and a 41,000-example **training set**
- Pixel values are normalized by dividing by 255

## Project Structure

| Step | Description |
|---|---|
| `init_params()` | Randomly initializes weights (small, scaled by 0.01) and zero biases |
| `forward_prop()` | Computes activations through the network |
| `backward_prop()` | Computes gradients via backpropagation |
| `update_params()` | Applies gradient descent update to weights/biases |
| `gradient_descent()` | Runs the full training loop, printing accuracy every 10 iterations |
| `make_predictions()` | Runs forward prop and returns predicted digit classes |
| `test_prediction()` | Visualizes a single prediction against its true label |

## Usage

1. Place `train.csv` and `test.csv` in the expected input path (defaults to Kaggle's `/kaggle/input/competitions/digit-recognizer/`).
2. Run all cells in order. Training runs for 500 iterations at a learning rate of 0.10:

```python
W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.10, 500)
```

3. Check dev set accuracy:

```python
dev_predictions = make_predictions(X_dev, W1, b1, W2, b2)
accuracy = get_acc(dev_predictions, Y_dev)
print(f"Dev Accuracy: {accuracy * 100:.1f}%")
```



## Results

| Metric | Value |
|---|---|
| Dev accuracy | ~87.0% |
| Iterations | 500 |
| Learning rate | 0.10 |

## Possible Improvements

- Add a second hidden layer or increase hidden layer width
- Tune the learning rate and iteration count
- Add L2 regularization or dropout to reduce overfitting
- Try mini-batch gradient descent instead of full-batch
- Experiment with different weight initialization schemes (e.g., He initialization for ReLU layers)

## Requirements

- Python 3
- `numpy`
- `pandas`
- `matplotlib`
