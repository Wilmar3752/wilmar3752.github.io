---
title: "The Optimal Bayes Classifier"
description: "Theory and practice"
featured: True
orderInSeries: 1
draft: true
pubDate: "Nov 13 2023"
tags:
  - Probability
  - Bayes Classifier
  - Bayes Theorem
  - R
  - Python
---
When we speak about classification models, we are looking for the best model, well, the Bayes Classifier is the best classificator in the world, it uses all information about the probability distribution to predict between two classes, the most probably. In this case, we assume that $X$ and $Y$ have joint distribution, called $P_{X,Y}$ and $\{{X_i, Y_i}\}_{i=1}^{n}$ is a random sample of $P_{X,Y}$, here train and test data are independent draws from the same distribution.

Supose that you want classify a test subject if have or not stress based on brain activity, so we have.

- $X$: "Average brain activity in Amygdala".
- $Y$: 1 if subject have strees (red) 0 if not (green).

![Image1](~/assets/blog_images/example1.png)

**figure 1** Example of stress classification model

As show in figure 1, higest values of $Y$ are correlated with stress, in this example, our test subjet is classificated as stress. If you notice, we have a mixture of two probability distributions, say $P(Y=1 | X) \sim $ distribution 1 and $P(Y=0 | X) \sim $ distribution 2

![Image1](~/assets/blog_images/example2.png)
