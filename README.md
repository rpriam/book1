## "Linear and Deep Models Basics with Pytorch, Numpy, and Scikit-Learn" <br />
## Files for the computer book in deep learning with statistical background <br /> 
## Amazon kdp paper back  - 2023 <br />
## ISBN-13 :979-8371441577 <br />

 <br />

  
**Main Document (ebook) with 247 pages (27-12-2022)**   [book_pytorch_scikit_learn_numpy.pdf](./text/book_pytorch_scikit_learn_numpy.pdf) <br />
  
 <hr> 
 <br />
 <br />

 **Available Files in this repository**  <br />

 Datasets, main file .py and notebooks .pynb at [./notebooks](./notebooks) <br />
 <br />
 <br />
 <br />
 

**Main Features** <br />

- Theory for the linear models and implementation with pytorch and scikit-learn  <br />

- Practice of deep learning with pytorch for feedforward neural networks <br />

- Many examples and exercices to practice and understand further the contents <br />

- Very large datasets 450000 and 11000000 on a home computer with a few gigabytes  <br />

- Step by step for theory & code (require only minimum knowledge in python and maths)  <br />

- Learn the basics without compromise before consolidate towards more advanced models <br />

 <br />
 <br />

**Abstract**  <br />

This book is an introduction to computational statistics for the generalized linear models (glm) and to machine learning with the python language. Extensions of the glm with nonlinearities come from hidden layer(s) within a neural network for linear and nonlinear regression or classification. This allows to present side by side classical statistics and current deep learning. The loglikelihoods and the corresponding loss functions are explained. The gradient and hessian matrix are discussed and implemented for these linear and nonlinear models. Several methods are implemented from scratch with numpy for prediction (linear, logistic, poisson regressions) and for reduction (principal component analysis, random projection). The gradient descent, newton-raphson, natural gradient and l-fbgs algorithms are implemented. The datasets in stake are with 10 to 10^7 rows, and are tabular such that images or texts are vectorized. The data are stored in a compressed format (memmap or hdf5) and loaded by chunks for several case studies with pytorch or scikit-learn. Pytorch is presented for training with minibatches via a generic implementation for studying with computer programs. Scikit-learn is presented for processing large datasets via the partial fit, after the small examples. Sixty exercises are proposed at the end of the chapters with selected solutions to go beyond the contents. <br />
 <br />
 <br />
 

**Chapters** <br />

1. Introduction <br />

    Polynomial regression  <br />
    Error on a train sample  <br />
    Error on a test sample  <br />

2. Linear models with numpy and scikit-learn ([chapter02_book.ipynb](./notebooks/chapter02_book.ipynb)) <br />

    Theory for linear regression  <br />
    Theory for logistic regression <br />
    Loglikelihood and loss function <br />
    Analytical expression of the derivatives  <br />
    implementation with numpy <br />
    Implementation with Scikit-Learn <br />

3. First-order training of linear models ([chapter03_book.ipynb](./notebooks/chapter03_book.ipynb)) <br />

    Algorithm with one datum and with one minibatch <br />
    Implementation of the algorithms with numpy <br />
    Implementation of the algorithms with pytorch <br />

4. Neural networks for (deep) glm ([chapter04_book.ipynb](./notebooks/chapter04_book.ipynb)) <br />

    Presentation of the different loss functions from pytorch <br />
    Generic implementation of the algorithms with pytorch <br />
    Example of nonlinear frontier with a small dataset <br />

5. Lasso selection for (deep) glm ([chapter05_book.ipynb](./notebooks/chapter05_book.ipynb)) <br />

    Penalization of the regression for sparse solution <br />
    Implementation with pytorch for a neural network <br />
    Selection of the hyperparameters (grid and bayesian) <br />

6. Hessian and covariance for (deep) glm ([chapter06_book.ipynb](./notebooks/chapter06_book.ipynb)) <br />

    Notion of variance of the parameters <br />
    Implementation with statsmodels for linear models <br />
    Implementation with pytorch for a neural network <br />

7. Second-order training of (deep) glm ([chapter07_book.ipynb](./notebooks/chapter07_book.ipynb)) <br />

    Expression of the update for 1st-order for poisson regression <br />
    Expression of the update for 2nd-order for poisson regression <br />
    Implementation of gradient descent for the poisson regression <br />
    Implementation of newton-raphson and natural gradient with numpy <br />
    Implementation of l-fbgs algorithm with pytorch for deep regressions <br />
    Notion of quality of the estimation for comparison </br>

8. Autoencoder compared to ipca and t-sne ([chapter08_book.ipynb](./notebooks/chapter08_book.ipynb)) <br />

    Introduction to the algebra for principal component analysis </br>
    Implementation step by step for principal component analysis </br>
    Implementation with scikit-Learn of pca and (non)linear autoencoders </br>
    Implementation of t-sne with python from two modules </br>
    Implementation of random projection for large datasets </br>
    Notion of quality of the visualization for comparison </br>

9. Solution to selected exercices ([chapter09_book.ipynb](./notebooks/chapter09_book.ipynb)) <br />

    Several solutions for large datasets with scikit-learn <br />
    Several solutions for neural networks with pytorch <br />
 <br />
 <br />


**About the author**

Rodolphe Priam has a diploma in practical applied statistics and a phD related to data sciences. He is ﬁrst author, main author or co-author of several communications on model inference in statistics and computer journals. He has served as a lecturer in French universities during four and half years from 2002 to 2004 and 2006 to 2008, he has taught applied statistics, probabilities and computing (statistical inference, queueing theory, langage r, langage java, etc) for more than trenty courses and lectures at universities and engineering schools.

----------------------------------------------------------------------

---------------------------------------------------------------------


