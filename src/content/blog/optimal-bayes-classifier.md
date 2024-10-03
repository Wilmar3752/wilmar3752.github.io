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

When discussing classification models, we seek the most effective one. The Bayes Classifier is often considered the best classifier available, as it utilizes all information about the probability distribution to predict the most probable class. Here, we assume that $X$ and $Y$ have a joint distribution, denoted as $P_{X,Y}$, and $\{X_i, Y_i\}_{i=1}^{n}$ represents a random sample from $P_{X,Y}$. In this context, the training and testing data are independent draws from the same distribution.

Suppose you want to classify whether a test subject has stress based on brain activity. In this case, we define:

- $X$: Average brain activity in the Amygdala.
- $Y$: 1 if the subject has stress (red), and 0 if not (green).

![Image1](~/assets/blog_images/example1.png)

**Figure 1:** Example of a stress classification model.

As shown in **Figure 1**, higher values of $Y$ are correlated with stress. In this example, our test subject is classified as experiencing stress. We can view our probability distribution as $P(Y=1 | X)$ and $P(Y=0 | X)$, as illustrated in **Figure 2**.

![Image1](~/assets/blog_images/example2.png)

**Figure 2:** Example of stress classification model probabilities.

Thus, we can say that for a given $X$, $f(X)$ is the most likely label:

$$
\begin{equation}
f(X) = \arg\max_{Y=y}P(Y=y | X=x)
\end{equation}
$$

## The Bayes' Theorem

Now, let's use Bayes' Theorem to rewrite our classification function (eq 1) in terms of likelihood and marginal distributions. We start with Bayes' Theorem:

$$
\begin{equation}
P(Y=y|X=x) = \frac{P(X=x|Y=y)P(Y=y)}{P(X=x)}
\end{equation}
$$

Using (2) in (1), we have:

$$
\begin{equation}
f(X) = \arg\max_{Y=y}P(X=x|Y=y)P(Y=y)
\end{equation}
$$

Next, we can consider the appropriate probability models for the terms $P(Y=y)$ and the conditional distribution of the inputs $P(X=x|Y=y)$.

In a classification model, it is natural to assume that $Y \sim \text{Bernoulli}(\theta)$. In this case, we model $P(X=x|Y=y) \sim \text{Normal}(\mu_y, \sigma_y)$, noting that the parameters of the conditional distribution may differ for $Y=0$ and $Y=1$.

## Decision Boundary

The decision boundary occurs when $P(Y=1|X=x) = P(Y=0|X=x)$. Here, we can compute the ratio:

$$
\frac{P(Y=1|X=x)}{P(Y=0|X=x)} = 1
$$

This leads to:

$$
\begin{equation}
\frac{P(X=x|Y=1)P(Y=1)}{P(X=x|Y=0)P(Y=0)} = 1
\end{equation}
$$

## Example

In the context of stress classification, suppose we know that:

- $Y \sim \text{Bernoulli}(\theta = 0.3)$
- $X|Y=0 \sim \text{Normal}(\mu_0 = 5, \sigma_0 = 1)$ 
- $X|Y=1 \sim \text{Normal}(\mu_1 = 7, \sigma_1 = 1)$ 

### The Classifier

In this case, the formula for equation 3 is:

$$
f(X) = \arg\max_{Y=y}\frac{1}{\sigma_y\sqrt{2\pi}} \exp\left(-\frac{1}{2}\left(\frac{x-\mu_y}{\sigma_y}\right)^2\right) \theta^{y}(1-\theta)^{1-y}; \text{  } y=0,1
$$

If one test subject has $X=6$ in their brain activity, you can substitute the given values into the classifier and obtain $\hat{y} = 0$, meaning the algorithm predicts that the subject does not have stress.

### Decision Boundary

$$
\frac{\frac{1}{\sigma_1\sqrt{2\pi}} \exp\left(-\frac{1}{2}\left(\frac{x-\mu_1}{\sigma_1}\right)^2\right) \cdot \theta}{\frac{1}{\sigma_0\sqrt{2\pi}} \exp\left(-\frac{1}{2}\left(\frac{x-\mu_0}{\sigma_0}\right)^2\right) \cdot (1-\theta)} = 1
$$

This is the case where $\sigma_1 = \sigma_0 = \sigma$. After performing the algebra, you can show that:

$$
x = \dfrac{1}{2}\left[ \mu_0 + \mu_1 - \log\frac{\theta}{1-\theta} \right]
$$

Using the provided data:

$$
x = \dfrac{1}{2}\left[ 5 + 7 - \log\frac{0.3}{1-0.3} \right] \approx 6.424
$$

Finally, our classification problem is illustrated in **Figure 3**.

![Image1](~/assets/blog_images/simulation.png)

**Figure 3:** Classification model.

## Conclusions

You can prove that the Bayes classifier is optimal; however, it never captures the true probability distribution. Some models that I plan to explore in the future have been developed to address this issue.

As future work, we can find the decision boundary when $\sigma_1 \neq \sigma_0$ and then work with multivariate distributions for $X|Y$.

## R Code
```r
library(ggplot2)

# Distribution for y
theta <- 0.3

y <- rbinom(100000, 1, theta)
n1 <- sum(y == 1)
n2 <- sum(y == 0)

# Distribution for x given y=0
mu0 <- 5
sigma0 <- 1

# Distribution for x given y=1
mu1 <- 7
sigma1 <- 1

# Create a simulated dataframe
xy0 <- rnorm(n2, mu0, sigma0)
xy1 <- rnorm(n1, mu1, sigma1)
sim_df <- data.frame(rbind(cbind(xy0, 0), cbind(xy1, 1)))
colnames(sim_df) <- c("x", "y")

# The calculation for y given x distribution
ygx <- function(y, x) {
  if (y == 0) {
    mu <- mu0
    sigma <- sigma0
  } else if (y == 1) {
    mu <- mu1
    sigma <- sigma1
  }
  fx <- dnorm(x, mu, sigma) * (theta^y) * (1 - theta)^(1 - y)
  return(fx)
}

# Calculate argmax
argmax <- function(x) {
  if (ygx(1, x) < ygx(0, x)) {
    return(0)
  } else {
    return(1)
  }
}

sim_df["y_pred"] <- apply(sim_df["x"], FUN = argmax, MARGIN = 1)

head(sim_df)

# Confusion matrix
table(sim_df$y, sim_df$y_pred)

objective_function <- function(x) {
  abs(ygx(y = 1, x = x) - ygx(y = 0, x = x))
}

result <- optim(par=(mu0+mu1)/2, fn = objective_function, method='L-BFGS-B', lower=min(mu0, mu1), upper=max(mu0, mu1))
result

ygx(1, result$par) / ygx(0, result$par)

hist(sim_df$x)

# Open a PNG graphics device
# Plot the graph
plot(x=seq(-0, 10, by=0.1), y=ygx(y=1, seq(-0, 10, by=0.1)), type='l', 
     xlab='Value for x', ylab='P(Y|X)', 
     col='red', 
     main="Optimal Bayes Classifier Simulation",
     ylim=c(0, 0.5))
points(x=seq(-0, 10, by=0.1), y=ygx(y=0, seq(-0, 10, by=0.1)), type='l', col="green")
abline(v=result$par, lty=2, col="blue")

# Add a legend
legend("topleft", legend=c("y=
