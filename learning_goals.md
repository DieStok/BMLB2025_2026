# Day1

## <lecture_1> 🎓

- Can describe the difference between supervised and unsupervised machine learning
- Can describe the difference between classification and regression
- Can explain GWAS as (many) linear regressions on variants (“a bunch of linear regressions in a trenchcoat”)
- Can write down the formula for univariate linear regression (\(\theta_{0} + \theta_{1} \cdot x^{(i)}\))
- Can describe what samples and features are in data for Machine Learning
- Can describe how we learn parameters from training data (using a cost function and gradient descent)
- Can explain what a cost function measures
- Can write down the cost function for linear regression (the mean-squared error, \(\frac{1}{2m}\sum_{i=1}^{m}\big(y_{\text{pred}}^{(i)} - y_{\text{real}}^{(i)}\big)^2\); where \(y_{\text{pred}}^{(i)} = \theta_{0} + \theta_{1} \cdot x^{(i)}\) for univariate linear regression)
- Can describe how trying different values for linear regression parameters and seeing what the MSE is shows you what the optimal parameters are.
- Can explain that just trying all possible parameter combinations is impossible.
- Can explain that gradient descent is the solution: take tiny steps towards the minimum of an unknown cost function (unknown == we don't know the full cost function for every possible parameter combo, so we don't know what it fully 'looks like')
- Can describe how to calculate a gradient (\(\Delta y / \Delta x\)) and take a small step towards the minimum \( \text{current\_param} - \alpha \cdot \text{gradient} \)
- Can describe that iteratively taking small steps towards the minimum of the cost function will eventually yield good parameters (given that there is a global minimum (as in linear regression) or many almost-as-good global minima (as in deep learning))
- Can explain why we need to use partial derivatives rather than a single gradient
- Can explain that taking the partial derivative of a function just means treating all parameters except the one you are taking the partial derivative of as numbers, thereby getting only the slope of the function w.r.t. that parameter.
- Can calculate partial derivatives of univariate/multivariate linear regression cost function.
- Can use the chain rule to calculate partial derivatives. For instance, for MSE: partial derivative of \(\frac{1}{2m}\sum_{i=1}^{m}\big(\theta_{0} + \theta_{1}\cdot x^{(i)} - y_{\text{real}}^{(i)}\big)^2\) with respect to \(\theta_{0}\) = derivative of \(x^2\) \(\rightarrow\) \(2x\) \(\times\) derivative of \(\theta_{0} + \theta_{1}\cdot x^{(i)} - y_{\text{real}}^{(i)}\) w.r.t. \(\theta_{0}\) which is just \(x^{1}\) \(\rightarrow\) \(1 \times x^{0} = 1 \times 1 = 1\). So it just comes down to \(\frac{1}{m}\sum_{i=1}^{m}\big(\theta_{0} + \theta_{1}\cdot x^{(i)} - y_{\text{real}}^{(i)}\big)\).
- Can describe that $\alpha$ or $\eta$ is a hyperparameter that governs the size of the steps you take in gradient descent.
- Can describe that hyperparameters are not optimized by the ML training algorithm but need to be picked beforehand to train the ML algorithm. 

## <coding_lab_1 (Day 1 short 1)> 💻

- Can implement the MSE function for univariate linear regression in Python code themselves
- Explain what the 3D surface plot of the univariate linear regression cost function means
- Can implement the gradient descent function for univariate linear regression themselves
- Explain what a contour plot of a cost function shows
- Can explain that since \(\theta_{1}\)'s partial derivative is multiplied with the feature value while \(\theta_{0}\)'s is not, normalizing the feature values allows gradient descent to take more equal steps.

## <lecture_2> 🎓

- Can explain how to extend univariate regression to multidimensional regression (e.g. \(\theta_{0} + \theta_{1} \cdot x_{1}^{(i)} + \theta_{2} \cdot x_{2}^{(i)}\))
  - More generally, for \(n\) features: \(\hat{y}^{(i)} = \theta_{0} + \sum_{j=1}^{n} \theta_{j} \cdot x_{j}^{(i)}\).

- Can write down the partial derivatives of multivariate regression  
  - With cost \(J(\boldsymbol{\theta}) = \frac{1}{2m}\sum_{i=1}^{m}\big(\hat{y}^{(i)} - y_{\text{real}}^{(i)}\big)^2\), where \(\hat{y}^{(i)} = \theta_{0} + \sum_{j=1}^{n}\theta_{j} x_{j}^{(i)}\):
    - \(\displaystyle \frac{\partial J}{\partial \theta_{0}} = \frac{1}{m}\sum_{i=1}^{m}\big(\hat{y}^{(i)} - y_{\text{real}}^{(i)}\big)\)
    - \(\displaystyle \frac{\partial J}{\partial \theta_{j}} = \frac{1}{m}\sum_{i=1}^{m}\big(\hat{y}^{(i)} - y_{\text{real}}^{(i)}\big)\,x_{j}^{(i)} \quad \text{for } j=1,\dots,n\)

- Can explain how to fit more complex functions (power functions/polynomials) by simply raising input features to some power and fitting multivariate regression to these modified features

- Can explain why standardizing features (bringing them to the same scale) makes sure gradient descent converges well (similar step size in all partial derivatives)

- Can explain that the goal of ML is not to fit the training data perfectly, but rather to use the training data to learn a function that generalises well to unseen examples

- Can explain the difference between underfitting, overfitting, and a good (enough) fit.

- Can explain and draw a fit with high variance: overfit/varies a lot if input data changes slightly

- Can explain and draw a fit with high bias: underfit/can't fit the data well due to insufficient expressivity or too strict assumptions (e.g. the bias that a function is linear while it is quadratic)

- Can explain how to estimate how well the model will generalise (train-validation-test split; learning curves; regularisation)

- Can explain exactly how k-fold cross-validation is done in practice

- Can explain why k-fold cross-validation is prefered over a single train-validation-test split

- Can explain what you would do after you are sure your model generalises well (train a final model on all training data, since usually model performance improves with training data set size; you only do the cross-validation to test whether your model will perform well on unseen data, afterwards train the best model you can)

- Can explain how learning curves are constructed by training models on different downsamplings of the data.

- Can explain what differentiates a learning curve of a model that has high bias from one with high variance

- Can draw learning curves for a high bias and high variance scenario. 

- Can explain that in the case of high variance, adding more training data usually works and makes the model better


## <coding_lab_2 (Day 1 short 2)> 💻

- Can add some polynomial features to fit more complex regressions

- Can explain how scaling to a Z-score differs from scaling to a set range (range scaling, e.g. to between \([-1, 1]\))
  - Z-score: \(x' = \dfrac{x - \mu}{\sigma}\)
  - Range scaling to \([a,b]\): \(x' = a + \dfrac{(x - x_{\min})}{(x_{\max} - x_{\min})}\,(b-a)\)

- Can program a multivariate/multi-variable linear regression function in Python that works for any amount of parameters

- Can modify the gradient descent function to work for linear regression with any amount of parameters

- Can optimize the parameters of a multivariate linear regression using gradient descent. 

## <lecture_3> 🎓

- Can explain why we use linear algebra to run ML algorithms (fast, parallel calculations)
- Can explain that deep learning, especially, is just a bunch of matrix multiplications in a trenchcoat [relevant xkcd](https://xkcd.com/1838) (note: this becomes more clear on day 3)
- Can add vectors and matrices together
- Can perform scalar multiplication of vectors and matrices
- Can transpose vectors and matrices
- Can perform matrix-vector and matrix-matrix multiplication
- Knows when two matrices can be multiplied together and what the shape of the result is (e.g. $5 \times 3$ matrix times $3 \times 1$ matrix can be multiplied, yields $5 \times 1$ vector). So: number of columns in matrix $A$ matches number of rows in matrix $B$.
  - **Small example matrix-vector multiplication**
    (if this does not render for you in markdown, no worries. It's just an example of what you should be able to do.)
    Let

    $$
    A =
    \begin{bmatrix}
    1 & 2 & 3 \\
    0 & -1 & 4 \\
    2 & 0 & 1 \\
    -3 & 5 & 2 \\
    4 & 1 & -2
    \end{bmatrix} \in \mathbb{R}^{5\times 3},
    \quad
    \mathbf{x} =
    \begin{bmatrix}
    1\\
    2\\
    -1
    \end{bmatrix}\in \mathbb{R}^{3\times 1}.
    $$

    Then $A\mathbf{x}\in \mathbb{R}^{5\times 1}$ is

    $$A\mathbf{x} =
    \begin{bmatrix}
    1\cdot 1 + 2\cdot 2 + 3\cdot(-1) \\
    0\cdot 1 + (-1)\cdot 2 + 4\cdot(-1) \\
    2\cdot 1 + 0\cdot 2 + 1\cdot(-1) \\
    (-3)\cdot 1 + 5\cdot 2 + 2\cdot(-1) \\
    4\cdot 1 + 1\cdot 2 + (-2)\cdot(-1)
    \end{bmatrix}
    =
    \begin{bmatrix}
    1 + 4 - 3 \\
    0 - 2 - 4 \\
    2 + 0 - 1 \\
    -3 + 10 - 2 \\
    4 + 2 + 2
    \end{bmatrix}
    =
    \begin{bmatrix}
    2\\
    -6\\
    1\\
    5\\
    8
    \end{bmatrix}.
    $$
- Can explain what it means that matrix-matrix multiplication is non-commutative
- Can explain how we can cast multivariate linear regression as a matrix vector multiplication.
- Can perform this multiplication.

## <computer_lab_3 (Day 1 afternoon)> 💻

- Can explain what's happening when 9-fold cross-validation is applied to polynomial linear regression.
- Can do simple linear algebra multiplications by hand using pen and paper
- Can check the answers of the above using numpy matrix multiplication functionality
- Can implement the linear regression hypothesis function for any number of features/parameters using linear algebra and numpy
- Can implement the mean-squared error function using linear algebra and numpy
- Can implement the gradient descent function using linear algebra and numpy


# Day 2

## <lecture_4> 🎓 

- Can explain why a linear-regression-then-threshold approach is problematic for classification (predictions can go below 0 or above 1; breaks on many datasets).
- Can define logistic regression as using a sigmoid-transformed linear function for classification: \( h_\theta(x) = g(\theta^\top x) \) with \( g(z) = \frac{1}{1 + e^{-z}} \).
- Can describe key properties of the sigmoid: approaches 0 as \( z \to -\infty \), approaches 1 as \( z \to \infty \), and equals 0.5 at \( z = 0 \).
- Can interpret \( h_\theta(x) \) as a probability \( p(y=1 \mid x; \theta) \) and use \( p(y=0 \mid x; \theta) = 1 - h_\theta(x) \). Can connect this to log-odds.
- Can work through a concrete linear decision boundary example (e.g., \( \theta = [-3, 1, 1] \Rightarrow x_1 + x_2 = 3 \)).
- Can explain how adding polynomial features enables non-linear decision boundaries (e.g., using \( x_1^2, x_2^2 \) to form circular boundaries).
- Can state why MSE is a poor cost for logistic regression (non-convex and low bandwidth) and explain the convex alternative.
- Can write the logistic (cross-entropy) loss per example in both cases and the dataset-averaged objective:  
  \[
  \text{Cost}(x) =
  \begin{cases}
    -\log\big(h_\theta(x)\big) & \text{if } y=1,\\[2pt]
    -\log\big(1 - h_\theta(x)\big) & \text{if } y=0,
  \end{cases}
  \quad
  J(\theta)=\frac{1}{m}\sum_{i=1}^m \text{Cost}\big(x^{(i)}\big).
  \]
  \[
  \text{Equivalent compact form: } \text{Cost}(x) = -\,y \log h_\theta(x) - (1-y)\log\big(1-h_\theta(x)\big).
  \]
- Can reason about the penalty behavior of the logistic loss (e.g., predicting 0 when \( y=1 \) drives cost to \(\infty\); MSE would penalize far less).
- Can write the gradient of \( J(\theta) \) for logistic regression and explain its similarity to linear regression (same form with a different hypothesis):  
  \[
  \frac{\partial J}{\partial \theta_j}
  = \frac{1}{m}\sum_{i=1}^m\big(h_\theta(x^{(i)}) - y^{(i)}\big)\,x_j^{(i)}.
  \]
- Can perform gradient descent parameter updates for all parameters \( \theta_j \):  
  \[
  \theta_j \leftarrow \theta_j - \alpha \cdot \frac{1}{m}\sum_{i=1}^m\big(h_\theta(x^{(i)}) - y^{(i)}\big)\,x_j^{(i)}.
  \]

## <computer_lab_4 (Day 2 short 1)> 💻

- Can implement the sigmoid function
- Can assign classification labels using a random threshold on a sigmoid plot
- Can implement the logistic regression hypothesis function (just a sigmoid wrapper around the linear regression hypothesis function)
- Can implement the logistic regression cost function
- Can change a gradient descent implementation to use the logistic regression hypothesis and cost functions
- Can explain why the decision function for 1D data following a sigmoid function approaches a sign function
- Can create a basic plot to show the decision boundary of a logistic regression

## <lecture_5> 🎓

- Can explain one-vs-rest multiclass logistic regression: train \(k\) binary classifiers \(h_{\theta^{(i)}}(x)\) and predict \(\arg\max_i P(y=i\mid x;\theta^{(i)})\).
- Can articulate why accuracy can be misleading (e.g., trivial always-positive predictor) and define TP, TN, FP, FN alongside sensitivity (TPR) and specificity (TNR).
- Can adjust the decision threshold to favor sensitivity or specificity and explain the trade-off.
- Can mechanistically explain the ROC curve (i.e. explain how it is built; TPR vs. FPR across thresholds)
- Can explain how to compare classifiers using ROC AUC (0.5 random, 1.0 perfect).
- Can explain why ROC AUC can be over-optimistic on imbalanced data and when to use the precision–recall AUC PRC instead; can define precision and recall.
- Can write the L2-regularized objective (do not penalize \(\theta_0\)) and explain bias–variance effects (adds bias, reduces variance) for linear/logistic models.
- Can identify hyperparameters (e.g., \(\lambda\) for regularization, \(\alpha\) for learning rate) and explain why simple CV model selection can leak validation information.
- Can mechanistically explain nested cross-validation (outer folds for generalization estimate; inner folds for hyperparameter search) and its computational cost in full detail; can extend to multiple hyperparameters.

## <computer_lab_5 (Day 2 short 2)> 💻

- Can implement L2 regularization into the cost function calculation 
- Can answer exploratory data analysis questions (specifically concerning NaNs and untrustworthy values) about a dataset
- Can clean up data after pinpointing untrustworthy/unusable values
- Can use K-NN imputation to impute missing values
- Can implement regularised logistic regression training for multiple values of lambda (==strength of the regularisation; == strength of biasing towards small parameter values;)
- Can implement and mechanistically understand the construction of an ROC curve (cycling through decision thresholds, making class predictions for each threshold, comparing this to the true labels, calculating TPR and FPR, plotting a point on the graph)

## <lecture_6> 🎓
- Can explain why we use neural networks (proven performance, no feature engineering, universal approximation theorem)
- Can explain why computational neural networks are not truly like networks of neurons in brains at all
- Can explain what a single neuron/unit calculates (weighted sum of its inputs + a bias, then passed through a nonlinear activation function)
- Can explain a neural network diagram (input features, intermediate layers, outputs)
- Can explain how to do a forward pass through a neural network with linear algebra
- Knows that for multi-class neural network predictions the labels are now a one-hot vector, with 1 for the correct class, and 0 for all others. 

## <computer_lab_6 (Day 2 afternoon)> 💻
- Can implement multiclass logistic regression yourself
- Can use the learned parameters to predict the most probable class on some unseen test data
- Can implement nested-cross validation using scikit-learn
- Can implement a basic feedforward pass through a neural network using matrix multiplications
