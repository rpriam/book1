## Data files and companion jupyter notebooks for the new book: <br />

## "Linear and Deep Models Basics with Pytorch, Numpy, and Scikit-Learn" <br />
 <br />
 <br />
 
 **Notebooks .pynb and main file utils.py available at (notebooks)(./notebooks), more available in the document**<br />
 <br />
 <br />

**Printed book with corrected exercices is available at at amazon local places** <br />
     [.com](https://www.amazon.de/dp/B0BRDGQND1), 
     [.de](https://www.amazon.de/dp/B0BRDGQND1), 
     [.fr](https://www.amazon.fr/dp/B0BRDGQND1), 
     [.es](https://www.amazon.es/dp/B0BRDGQND1), 
     [.it](https://www.amazon.it/dp/B0BRDGQND1), 
     [.nl](https://www.amazon.nl/dp/B0BRDGQND1), 
     [.pl](https://www.amazon.es/pl/B0BRDGQND1), 
     [.se](https://www.amazon.es/se/B0BRDGQND1), 
     [.ca](https://www.amazon.ca/dp/B0BRDGQND1),
     [.uk](https://www.amazon.uk/dp/B0BRDGQND1) <br />

[![Cover book](https://github.com/rpriam/book1/blob/main/cover.png)](https://www.amazon.com/dp/B0BRDGQND1) 

Published from Amazon kdp (December 27, 2022) <br />
ASIN : B0BRDGQND1 <br />
Language : English <br />
Paperback : 283 pages <br />
ISBN-13 : 979-8371441577 <br />
<br />
<br />

**Main Features** <br />

- Theory for the linear models and implementation with pytorch and scikit-learn  <br />

- Practice of deep learning with pytorch for feedforward neural networks <br />

- Many examples and exercices to practice and understand better the contents <br />

- Very large datasets 450000 and 11000000 on a home computer with a few gigabytes  <br />

- Step by step for theory & code (require only minimum knowledge in python and maths)  <br />

- Learn the basics without compromise before consolidate towards more advanced models <br />


**Abstract**  <br />

This book is an introduction to computational statistics for the generalized linear models (glm) and to machine learning with the python language. Extensions of the glm with nonlinearities come from hidden layer(s) within a neural network for linear and nonlinear regression or classification. This allows to present side by side classical statistics and current deep learning. The loglikelihoods and the corresponding loss functions are explained. The gradient and hessian matrix are discussed and implemented for these linear and nonlinear models. Several methods are implemented from scratch with numpy for prediction (linear, logistic, poisson regressions) and for reduction (principal component analysis, random projection). The gradient descent, newton-raphson, natural gradient and l-fbgs algorithms are implemented. The datasets in stake are with 10 to 10^7 rows, and are tabular such that images or texts are vectorized. The data are stored in a compressed format (memmap or hdf5) and loaded by chunks for several case studies with pytorch or scikit-learn. Pytorch is presented for training with minibatches via a generic implementation for studying with computer programs. Scikit-learn is presented for processing large datasets via the partial fit, after the small examples. Sixty exercises are proposed at the end of the chapters with selected solutions to go beyond the contents. <br />


**Chapter** <br />

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



**About the author**

Rodolphe Priam has a diploma in practical applied statistics and a phD related to data sciences. He is ﬁrst author, main author or co-author of several communications on model inference in statistics and computer journals. He has served as a lecturer in French universities during four and half years from 2002 to 2004 and 2006 to 2008, he has taught applied statistics, probabilities and computing (statistical inference, queueing theory, langage r, langage java, etc) for more than trenty courses and lectures at universities and engineering schools. He was a research assistant in England during two years and a half from 2010 to 2012, he worked on sampling and statistical programming for surveys. He was hired as a statistician engineer at an hospital during one full year from 2019 to 2020, he has worked on biostatistical programming for medical student projects.

----------------------------------------------------------------------

---------------------------------------------------------------------


