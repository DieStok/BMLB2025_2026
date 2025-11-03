# Note on how to read this:
This is a completely delineated list of all content covered in the lectures and coding labs. It may look daunting, but if you understand the lecture materials, can explain how, e.g. gradient descent for multivariate linear regression and K-means clustering and PCA works step-by-step, you are probably quite close to getting a good grade. Note that all lecture content can be on the exam, as can (the logic/pseudocode of) implementations, as can be scikit-learn pipelines (to the extent covered in the lectures on Day 6 and the Project). 

# Day 1

## <lecture_1> 🎓

- Can describe the difference between supervised and unsupervised machine learning  
- Can describe the difference between classification and regression  
- Can explain GWAS as (many) linear regressions on variants (“a bunch of linear regressions in a trenchcoat”)  
- Can write down the formula for univariate linear regression:

  $$
  \theta_{0} + \theta_{1} \cdot x^{(i)}
  $$

- Can describe what samples and features are in data for Machine Learning  
- Can describe how we learn parameters from training data (using a cost function and gradient descent)  
- Can explain what a cost function measures  
- Can write down the cost function for linear regression (mean-squared error):

  $$
  J(\theta) = \frac{1}{2m}\sum_{i=1}^{m}\big(y_{\text{pred}}^{(i)} - y_{\text{real}}^{(i)}\big)^2
  $$
  where  
  $$
  y_{\text{pred}}^{(i)} = \theta_{0} + \theta_{1} \cdot x^{(i)}
  $$

- Can describe how trying different values for linear regression parameters and seeing what the MSE is shows you what the optimal parameters are  
- Can explain that just trying all possible parameter combinations is impossible  
- Can explain that gradient descent is the solution: take tiny steps towards the minimum of an unknown cost function  
- Can describe how to calculate a gradient:

  $$
  \Delta y / \Delta x
  $$

  and take a small step towards the minimum:

  $$
  \text{current\_param} - \alpha \cdot \text{gradient}
  $$

- Can describe that iteratively taking small steps towards the minimum of the cost function will eventually yield good parameters  
- Can explain why we need to use partial derivatives rather than a single gradient  
- Can explain that taking the partial derivative of a function just means treating all parameters except one as constants  
- Can calculate partial derivatives of univariate/multivariate linear regression cost function  
- Can use the chain rule to calculate partial derivatives. Example (for MSE):

  $$
  \frac{1}{2m}\sum_{i=1}^{m}\big(\theta_{0} + \theta_{1}\cdot x^{(i)} - y_{\text{real}}^{(i)}\big)^2
  $$

  Partial derivative w.r.t. $\theta_{0}$:

  $$
  \frac{1}{m}\sum_{i=1}^{m}\big(\theta_{0} + \theta_{1}\cdot x^{(i)} - y_{\text{real}}^{(i)}\big)
  $$

- Can describe that $\alpha$ or $\eta$ is a hyperparameter that governs step size in gradient descent  
- Can describe that hyperparameters are not optimized by the ML algorithm itself  

---

## <coding_lab_1 (Day 1 short 1)> 💻

- Can implement the MSE function for univariate linear regression  
- Explain what the 3D surface plot of the univariate linear regression cost function means  
- Can implement the gradient descent function for univariate linear regression  
- Explain what a contour plot of a cost function shows  
- Can explain that since $\theta_{1}$’s partial derivative is multiplied with the feature value while $\theta_{0}$’s is not, normalizing features allows equal step sizes  

---

## <lecture_2> 🎓

- Can explain how to extend univariate to multivariate regression:

  $$
  \theta_{0} + \theta_{1} \cdot x_{1}^{(i)} + \theta_{2} \cdot x_{2}^{(i)}
  $$

  or more generally:

  $$
  \hat{y}^{(i)} = \theta_{0} + \sum_{j=1}^{n} \theta_{j} \cdot x_{j}^{(i)}
  $$

- Can write down partial derivatives for multivariate regression:

  $$
  J(\boldsymbol{\theta}) = \frac{1}{2m}\sum_{i=1}^{m}\big(\hat{y}^{(i)} - y_{\text{real}}^{(i)}\big)^2
  $$

  $$
  \frac{\partial J}{\partial \theta_{0}} = \frac{1}{m}\sum_{i=1}^{m}\big(\hat{y}^{(i)} - y_{\text{real}}^{(i)}\big)
  $$

  $$
  \frac{\partial J}{\partial \theta_{j}} = \frac{1}{m}\sum_{i=1}^{m}\big(\hat{y}^{(i)} - y_{\text{real}}^{(i)}\big)x_{j}^{(i)} \quad \text{for } j=1,\dots,n
  $$

- Can explain polynomial features (raising inputs to powers)  
- Can explain why standardizing features helps gradient descent converge  
- Can explain overfitting vs underfitting, bias vs variance  
- Can explain and draw examples of high-bias and high-variance fits  
- Can explain train/validation/test splits, k-fold cross-validation, and learning curves  
- Can explain why k-fold CV is preferred over a single split  
- Can explain the learning curve differences for high bias vs high variance  

---

## <coding_lab_2 (Day 1 short 2)> 💻

- Can add polynomial features  
- Can explain scaling differences:

  Z-score:
  $$
  x' = \frac{x - \mu}{\sigma}
  $$
  Range scaling:
  $$
  x' = a + \frac{(x - x_{\min})}{(x_{\max} - x_{\min})}(b - a)
  $$

- Can implement multivariate linear regression and gradient descent for any number of parameters  

---

## <lecture_3> 🎓

- Can explain why we use linear algebra for ML  
- Can perform matrix and vector operations  
- Can determine when matrices can be multiplied (e.g., $5\times3$ with $3\times1$ → $5\times1$)  
- Example:

  $$
  A =
  \begin{bmatrix}
  1 & 2 & 3 \\
  0 & -1 & 4 \\
  2 & 0 & 1 \\
  -3 & 5 & 2 \\
  4 & 1 & -2
  \end{bmatrix},
  \quad
  \mathbf{x} =
  \begin{bmatrix}
  1 \\ 2 \\ -1
  \end{bmatrix}
  $$

  $$
  A\mathbf{x} =
  \begin{bmatrix}
  2 \\ -6 \\ 1 \\ 5 \\ 8
  \end{bmatrix}
  $$

- Can explain non-commutativity of matrix multiplication  
- Can cast linear regression as matrix–vector multiplication  

---

## <computer_lab_3 (Day 1 afternoon)> 💻

- Can explain 9-fold CV on polynomial regression  
- Can do linear algebra by hand and verify with NumPy  
- Can implement regression hypothesis, MSE, and gradient descent using NumPy  

---

# Day 2

## <lecture_4> 🎓

- Can explain why plain linear regression fails for classification  
- Can define logistic regression:

  $$
  h_\theta(x) = g(\theta^\top x), \quad g(z) = \frac{1}{1 + e^{-z}}
  $$

- Can interpret $h_\theta(x)$ as $p(y=1 \mid x;\theta)$ and use $1 - h_\theta(x)$ for $y=0$  
- Can explain linear and nonlinear decision boundaries  
- Can explain why MSE is poor for logistic regression and define the logistic loss:

  $$
  \text{Cost}(x) =
  \begin{cases}
    -\log(h_\theta(x)) & \text{if } y=1,\\
    -\log(1-h_\theta(x)) & \text{if } y=0
  \end{cases}
  $$

  Dataset average:
  $$
  J(\theta)=\frac{1}{m}\sum_{i=1}^m[-y\log h_\theta(x^{(i)}) - (1-y)\log(1-h_\theta(x^{(i)}))]
  $$

- Gradient of $J(\theta)$:

  $$
  \frac{\partial J}{\partial \theta_j}
  = \frac{1}{m}\sum_{i=1}^m(h_\theta(x^{(i)}) - y^{(i)})x_j^{(i)}
  $$

- Gradient descent update:

  $$
  \theta_j \leftarrow \theta_j - \alpha \cdot \frac{1}{m}\sum_{i=1}^m(h_\theta(x^{(i)}) - y^{(i)})x_j^{(i)}
  $$

---

## <computer_lab_4 (Day 2 short 1)> 💻

- Implement sigmoid, hypothesis, cost, and gradient descent for logistic regression  
- Plot sigmoid-based decision boundaries  

---

## <lecture_5> 🎓

- One-vs-rest multiclass logistic regression:

  $$
  \hat{y} = \arg\max_i P(y=i\mid x;\theta^{(i)})
  $$

- Define TP, TN, FP, FN; sensitivity (TPR) and specificity (TNR)  
- Explain threshold trade-offs, ROC curves, and AUC  
- Explain precision–recall and when to prefer PRC  
- Write L2-regularized objective (excluding $\theta_0$):

  $$
  J(\theta)=\frac{1}{m}\sum_{i=1}^m[-y\log h_\theta(x^{(i)}) - (1-y)\log(1-h_\theta(x^{(i)}))]+\frac{\lambda}{2m}\sum_{j=1}^n\theta_j^2
  $$

- Explain bias–variance effects  
- Explain nested cross-validation (outer folds for generalization, inner for hyperparameter search)  

---

## <computer_lab_5 (Day 2 short 2)> 💻

- Implement L2 regularization  
- Handle missing data (EDA, cleaning, KNN imputation)  
- Implement regularized logistic regression for multiple λ values  
- Implement ROC curve construction manually  

---

## <lecture_6> 🎓

- Explain self-supervised learning  
- Explain neural networks and single neurons  
- Explain forward pass via linear algebra  
- Explain one-hot encoding for multiclass  
- Explain lack of inductive bias in MLPs and how CNNs/GNNs add it  

---

## <computer_lab_6 (Day 2 afternoon)> 💻

- Implement multiclass logistic regression  
- Predict test labels  
- Implement nested cross-validation  
- Implement a basic feedforward neural network
