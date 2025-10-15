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
- Can use the chain rule to calculate partial derivatives. For instance, for MSE: partial derivative of \(\frac{1}{2m}\sum_{i=1}^{m}\big(\theta_{0} + \theta_{1}\cdot x^{(i)} - y_{\text{real}}^{(i)}\big)^2\) with respect to \(\theta_{0}\) = derivative of \(x^2\) \(\rightarrow\) \(2x\) \(\times\) derivative of \(\theta_{0} + \theta_{1}\cdot x^{(i)} - y_{\text{real}}^{(i)}\) w.r.t. \(\theta_{0}\) which is just \(x^{1}\) \(\rightarrow\) \(1 \times x^{0} = 1 \times 1 = 1\). So it just comes down to \(\frac{1}{m}\sum_{i=1}^{m}\big(\theta_{0} + \theta_{1}\cdot x^{(i)} - y_{\text{real}}^{(i)}\big)\). **NOTE: ALL \(x^{(i)}\), \(y_{\text{pred}}^{(i)}\) and \(y_{\text{real}}^{(i)}\) have superscript indices \((i)\).**
- Can describe that alpha or eta is a hyperparameter that governs the size of the steps you take in gradient descent.
- Can describe that hyperparameters are not optimized by the ML training algorithm but need to be picked beforehand to train the ML algorithm. 

## <coding lab_1> 💻

- Can implement the MSE function for univariate linear regression in Python code themselves
- Explain what the 3D surface plot of the univariate linear regression cost function means
- Can implement the gradient descent function for univariate linear regression themselves
- Explain what a contour plot of a cost function shows
- Can explain that since \(\theta_{1}\)'s partial derivative is multiplied with the feature value while \(\theta_{0}\)'s is not, normalizing the feature values allows gradient descent to take more equal steps.
