---
title: "The Optimal Bayes Classifier"
description: "Theory and practice"
featured: True
orderInSeries: 1
draft: true
pubDate: "Oct 01 2024"
tags:
  - Probability
  - Bayes Classifier
  - Bayes Theorem
  - R
  - Python
---

## Introduction
When we speak about classification models, we are looking for the best model, well, the Bayes Classifier is the best classificator in the world, it uses all information about the probability distribution to predict between two classes, the most probably. In this case, we assume that $X$ and $Y$ have joint distribution, called $P_{X,Y}$ and $\{{X_i, Y_i}\}_{i=1}^{n}$ is a random sample of $P_{X,Y}$, here train and test data are independent draws from the same distribution.

Supose that you want classify a test subject if have or not stress based on brain activity, so we have.

- $X$: "Average brain activity in Amygdala".
- $Y$: 1 if subject have strees (red) 0 if not (green).

![Image1](~/assets/blog_images/example1.png)

**figure 1:** Example of stress classification model

As show in **figure 1**, higest values of $Y$ are correlated with stress, in this example, our test subjet is classificated as stress. If you notice, we can view our probability distribution as $P(Y=1 | X)$ and $P(Y=0 | X)$ as shown in **figure 2**

![Image1](~/assets/blog_images/example2.png)

**figure 2:** Example of stress classification model, probabilities

Thus, we we can say that for a given $X$, $f(X)$ is the most likely label.

$$
\begin{equation}
f(X) = \argmax_{Y=y}P(Y=y | X=x)
\end{equation}
$$

## The Bayes' Theorem
Now, lets to use the Bayes Theorem to rewrite our classification function (eq 1) in therms of likehood an marginal distributions, lets start with bayes.

$$
\begin{equation}
P(Y=y|X=x) = \frac{P(X=x|Y=y)P(Y=y)}{P(X=x)}
\end{equation}
$$

Using *(2)* in *(1)*, we have

$$
\begin{equation}
f(X) = \argmax_{Y=y}P(X=x|Y=y)P(Y=y)
\end{equation}
$$

Now, we can consider the appropiate probability models for the therms $P(Y=y)$ and the conditional distribution of the inputs $P(X=x|Y=y)$. 

In a classification model, its natural think that $Y \sim Bernoulli(\theta)$. in this case, i've to work with $P(X=x|Y=y) \sim Normal(\mu_y, \sigma_y)$, note that de conditional distribution's parameters may be different for $Y=0$ and $Y=1$.

## Decision Boundary

The decision Boundary occurs when $P(Y=1|X=x)$ = $P(Y=0|X=x)$. Here we can compute de ratio:

$$
\frac{P(Y=1|X=x)}{P(Y=0|X=x)} = 1
$$
$$
\begin{equation}
\frac{P(X=x|Y=1)P(Y=1)}{P(X=x|Y=0)P(Y=0)} = 1
\end{equation}
$$

## Example

Suppose that, in the stress classification case, we khow that:
- $Y \sim Berniolli(\theta = 0.3)$
- $X|Y=0 \sim Normal(\mu_0 = 5, \sigma_0 =1 )$ 
- $X|Y=1 \sim Normal(\mu_1 = 7, \sigma_1 =1 )$ 

```r
theta <- 0.9

y <- rbinom(1000, 1, theta)
n1 <- sum(y == 1)
n2 <- sum(y == 0)
## Distribution for x given y=0
mu0 <- 15
sigma0 <- 3
## Distribution for x given y=1
mu1 <- 5
sigma1 <- 3

## Create a Simulated Dataframe
xy0 <- rnorm(n2, mu0, sigma0)
xy1 <- rnorm(n1, mu1, sigma1)
sim_df <- data.frame(rbind(cbind(xy0, 0), cbind(xy1, 1)))
colnames(sim_df) <- c("x", "y")
```