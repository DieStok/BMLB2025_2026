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

## <coding_lab_1> 💻

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


## <coding_lab_2> 💻

- Can add some polynomial features to fit more complex regressions

- Can explain how scaling to a Z-score differs from scaling to a set range (range scaling, e.g. to between \([-1, 1]\))
  - Z-score: \(x' = \dfrac{x - \mu}{\sigma}\)
  - Range scaling to \([a,b]\): \(x' = a + \dfrac{(x - x_{\min})}{(x_{\max} - x_{\min})}\,(b-a)\)

- Can program a multivariate/multi-variable linear regression function in Python that works for any amount of parameters

- Can modify the gradient descent function to work for linear regression with any amount of parameters

- Can optimize the parameters of a multivariate linear regression using gradient descent. 
