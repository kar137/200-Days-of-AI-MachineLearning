# Multiclass Classification

## Introduction

Previously, we discussed logistic regression for binary classification. But what about problems with more than two classes?  
E.g., handwritten digit classification (MNIST) or classifying the Iris dataset.

## Binary vs Multiclass

Binary classification deals with two classes, while multiclass handles three or more.

Examples:
- **Binary**: Cats vs Dogs
- **Multiclass**: Classifying A, B, C

---

## One-vs-All (OvA / One-vs-Rest)

This method splits a multiclass problem into multiple binary classification problems.  
For each class, train a binary logistic regression model that predicts the probability of the sample belonging to that class vs all others.

### For class A:
Predict:  
$$
P(A|x) = h^{(1)}_w(x)
$$

### For class B:
Predict:  
$$
P(B|x) = h^{(2)}_w(x)
$$

### For class C:
Predict:  
$$
P(C|x) = h^{(3)}_w(x)
$$

### Prediction:
$$
\hat{y} = \arg\max_i h^{(i)}_w(x)
$$

**Note**:  
This method may suffer from class imbalance in each binary task.

---

## One-vs-One (OvO)

Train a binary classifier for each pair of classes.  
For \( N \) classes, you need:

$$
\frac{N(N - 1)}{2}
$$

classifiers.

### Prediction:
Each classifier votes; the class with the most votes is predicted.

**Issue**:  
Ties can occur if classes receive the same number of votes.

---

## Softmax Function

A direct approach to multiclass classification.

### Score Calculation:
$$
s_k(x) = w_k^T x
$$

Where \( w_k \) is the weight vector for class \( k \).

### Softmax:
$$
\hat{P}_k = \frac{\exp(s_k(x))}{\sum_{j=1}^K \exp(s_j(x))}
$$

### Prediction:
$$
\hat{y} = \arg\max_k \hat{P}_k = \arg\max_k (w_k^T x)
$$

---

## Cost Function: Cross Entropy

Generalized form of binary cross-entropy for multiple classes.

### For one example:
$$
\text{Cost}(\hat{y}, y) = -\sum_{k=1}^K y_k \log(\hat{y}_k)
$$

Where:
- \( y \) is a one-hot encoded true label
- \( \hat{y} \) is the predicted probability vector

### For \( m \) examples:
$$
J(W) = -\frac{1}{m} \sum_{i=1}^m \sum_{k=1}^K y_k^{(i)} \log(\hat{y}_k^{(i)})
$$

---

## Gradient Descent Optimization

Cross-entropy is convex, so we can use gradient descent.

### Gradient for class \( k \):
$$
\nabla_{w_k} J(W) = \frac{1}{m} \sum_{i=1}^m (\hat{y}_k^{(i)} - y_k^{(i)}) x^{(i)}
$$

### Update Rule:
$$
w_k := w_k - \alpha \nabla_{w_k} J(W)
$$

Where \( \alpha \) is the learning rate.

---

## Key Takeaways

- Use OvA and OvO methods to adapt binary classifiers for multiclass problems.
- Softmax is a direct approach to multiclass classification.
- Cross-entropy generalizes binary logistic regression loss.
- Gradient descent updates each class's weights based on prediction error.
