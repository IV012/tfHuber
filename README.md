# tfHuber

Python implementation of **T**uning-**F**ree **Huber** Estimation and Regression

## Description

This package implements the Huber mean estimator, Huber covariance matrix estimation and adaptive Huber regression estimators efficiently. For all these methods, the robustification parameter *&tau;* is calibrated via a tuning-free principle.

Specifically, for Huber regression, assume the observed data vectors (*Y*, *X*) follow a linear model *Y = &theta;<sub>0</sub> + X &theta; + &epsilon;*, where *Y* is an *n*-dimensional response vector, *X* is an *n* &times; *d* design matrix, and *&epsilon;* is an *n*-vector of noise variables whose distributions can be asymmetric and/or heavy-tailed. The package computes the standard Huber's *M*-estimator when *d < n* and the Huber-Lasso estimator when *d > n*. The vector of coefficients *&theta;* and the intercept term *&theta;<sub>0</sub>* are estimated successively via a two-step procedure. See [Wang et al., 2020](https://www.math.ucsd.edu/~wez243/tfHuber.pdf) for more details of the two-step tuning-free framework.

## Requirement  
```
numpy
setuptools
wheel
```

## Functions

There are four functions in this package: 

* `mean(X, grad=True, tol=1e-5, max_iter=500)`: Huber mean estimation. Return a tuple of mean, $\tau$ and the number of iteration.  
*X*: A 1-d array.  
*grad*: Using gradient descent or weighed least square to optimize the mean, default *True*  
*tol*: Tolerance of the error, default *1e-5*.  
*max_iter*: Maximum times of iteration, default *500*.
* `cov(X, type="element", pairwise=False, tol=1e-5, max_iter=500)`: Huber covariance matrix estimation. Return a 2d covariance matrix.  
*X*: A 2-d array.  
*type*: If set to `"element"`, apply adaptive huber M-estimation; or if set to `"spectrum"`, apply spectrum-wise truncated estimation. Default `"element" `  
*pairwise*: Pairwise covariance or difference based covariance. Default *false*.   
*tol*: Tolerance of the error, default *1e-5*.  
*max_iter*: Maximum times of iteration, default *500*.
* `one_step_reg(X, Y, grad=True, tol=1e-5, max_iter=500
two_step_reg(X, Y, grad=True, tol=1e-5, constTau=1.345, max_iter=500)`   
One or two step adaptive Huber regression. Return a tuple of coefficients, $\tau$ and the number of iteration.   
*X, Y*: Arrays of data.  
*grad*: Using gradient descent or weighed least square to optimize the mean, default *True*.  
*tol*: Tolerance of the error, default *1e-5*.  
*constTau*: Default 1.345. Used only in two-step method.  
*max_iter*: Maximum times of iteration, default *500*.  
* `cvlasso(X, Y, lSeq=0, nlambda=30, constTau=2.5, phi0=0.001, gamma=1.5, tol=0.001, nfolds=3)`: K-fold cross validated Huber-lasso regression. Return a tuple of coefficients, $tau$, the number of iteration and minimun of $\lambda$.   
*X, Y*: Arrays of data.  
*lSeq*: A list of Lasso parameter $\lambda$. If not set, automatically find a range of $\lambda$ to be cross validated.  
*nlambda*: The number of $\lambda$ used for validation.  
*constTau, phi0, gamma*: Some parameters.  
*tol*: Tolerance of the error, default *0.001*.  
*nfolds*: Number of folds to be cross validated.




## Examples 


We present an example of adaptive Huber methods. Here we generate data from a linear model *Y = X &theta; + &epsilon;*, where *&epsilon;* follows a normal distribution, and estimate the intercept and coefficients by tuning-free Huber regression.

```python
import numpy
import tfhuber
X = np.random.uniform(-1.5, 1.5, (10000, 10))
Y = intercept + np.dot(X, beta) + np.random.normal(0, 1, 10000)

mu, tau, iteration = tf.mean(Y, grad=True, tol=1e-5, max_iter=500)
cov = tf.cov(X, method=1, tol=1e-5, max_iter=500)

theta, tau, iteration = tf.one_step_reg(X, Y, grad=True, tol=1e-5, max_iter=500)
theta, tau, iteration = tf.two_step_reg(X, Y, grad=True, tol=1e-5, consTau=1.345, max_iter=500)

theta, tau, iteration, lam = tf.cvlasso(X, Y) 
```
Simulation result can be viewed in this [colab notebook](https://colab.research.google.com/drive/1XyBMNHog_RqFo3dkoQENt7wT2yMVEIdQ?usp=sharing).  

## License
GPL (>= 3)

## Author(s)

Yifan Dai <yifandai@yeah.net>, Qiang Sun <qsun.ustc@gmail.com>

Description and algorithms refer to [Xiaoou Pan's page](https://github.com/XiaoouPan/tfHuber).

## References

Guennebaud, G. and Jacob B. and others. (2010). Eigen v3. [Website](http://eigen.tuxfamily.org)

Ke, Y., Minsker, S., Ren, Z., Sun, Q. and Zhou, W.-X. (2019). User-friendly covariance estimation for heavy-tailed distributions. *Statis. Sci.* **34** 454-471, [Paper](https://projecteuclid.org/euclid.ss/1570780979)

Pan, X., Sun, Q. and Zhou, W.-X. (2019). Nonconvex regularized robust regression with oracle properties in polynomial time. Preprint. [Paper](https://arxiv.org/abs/1907.04027)

Sanderson, C. and Curtin, R. (2016). Armadillo: A template-based C++ library for linear algebra. *J. Open Source Softw.* **1** 26. [Paper](http://conradsanderson.id.au/pdfs/sanderson_armadillo_joss_2016.pdf)

Sun, Q., Zhou, W.-X. and Fan, J. (2020). Adaptive Huber regression. *J. Amer. Stat. Assoc.* **115** 254-265. [Paper](https://doi.org/10.1080/01621459.2018.1543124)

Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. *J. R. Stat. Soc. Ser. B. Stat. Methodol.* **58** 267–288. [Paper](https://www.jstor.org/stable/2346178?seq=1#metadata_info_tab_contents)

Wang, L., Zheng, C., Zhou, W. and Zhou, W.-X. (2020). A new principle for tuning-free Huber regression. *Stat. Sinica* to appear. [Paper](https://www.math.ucsd.edu/~wez243/tfHuber.pdf)