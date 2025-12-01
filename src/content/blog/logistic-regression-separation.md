---
title: "El Problema de Separación en la Regresión Logística"
description: "Causas, consecuencias y soluciones al problema de separación completa y cuasicompleta"
featured: True
draft: true
pubDate: "Dec 01 2025"
tags:
  - Logistic Regression
  - Maximum Likelihood
  - Firth Penalization
  - R
  - Statistics
---

## Introducción

La regresión logística es uno de los modelos lineales generalizados más utilizados debido a su amplia gama de aplicaciones. Puede ser utilizado para conocer la probabilidad de que ocurra un suceso dado un conjunto de covariables, e incluso es utilizado como algoritmo de Machine Learning para realizar clasificaciones binarias (diagnóstico clínico, spam en correos electrónicos y reconocimiento de imágenes, entre otras).

Sin embargo, en la experiencia práctica, existen casos donde los algoritmos para realizar estimaciones de máxima verosimilitud no convergen y muchos investigadores no tienen claras las causas del problema ni cómo solucionarlo (Allison, 2008).

Según Albert & Anderson (1984), este problema tiene dos causas principales: la **separación completa** y la **separación cuasicompleta**, las cuales causan que, al no converger los algoritmos de estimación por máxima verosimilitud, algunas de las estimaciones o todas sean infinitas. Con separación completa, las salidas para cada sujeto en el conjunto de datos pueden ser perfectamente predichas, mientras que con separación cuasicompleta esto sucede para un subconjunto de sujetos.

Muchos han sido los esfuerzos por resolver este problema. En particular, el trabajo de Heinze & Schemper (2002) es uno de los más referenciados y consiste en realizar una modificación a la función score de la regresión logística utilizando un método que fue originalmente desarrollado por Firth (1992) para reducir el sesgo en las estimaciones de máxima verosimilitud.

## Regresión Logística

Para los modelos lineales generalizados, partimos del predictor lineal:

$$
\eta_i = \beta_0 + \beta_1 x_{i1} + \ldots + \beta_k x_{ik} = \mathbf{x}_i^T \boldsymbol{\beta}
$$

El cual se relaciona con $\mu_i$ mediante $g(\cdot)$, esta función $g(\cdot)$ es llamada función de enlace y está relacionada con la distribución de la variable de respuesta $y$ y su representación en la familia exponencial de dispersión. De la cual se deduce que:

$$
\eta_i = \mathbf{x}_i^T \boldsymbol{\beta} = \log\left(\frac{\pi_i}{1-\pi_i}\right)
$$

Y finalmente se obtiene:

$$
\begin{equation}
\pi_i = \frac{\exp(\eta_i)}{1+\exp(\eta_i)} = \frac{\exp(\mathbf{x}_i^T \boldsymbol{\beta})}{1+\exp(\mathbf{x}_i^T \boldsymbol{\beta})}
\end{equation}
$$

### Estimación por Máxima Verosimilitud

Tengamos en cuenta el problema de regresión logística para datos no agrupados donde la variable de respuesta $y$ representa la ocurrencia de un evento $A$ de interés:

$$
y_i = \begin{cases}
0 & \text{Si no ocurre } A \\
1 & \text{Si ocurre } A
\end{cases}
$$

Esto implica que $y_i \sim \text{Bernoulli}(\pi_i)$, recordemos que $E(y_i) = \pi_i$. Si llamamos $f_i(y_i)$ a la función de masa de probabilidad (fmp) de $y_i$ tenemos:

$$
f_i(y_i) = \pi_i^{y_i}(1-\pi_i)^{1-y_i}
$$

Para una muestra aleatoria $y_1, \ldots, y_n$, la función de verosimilitud $L(\cdot)$ está definida por:

$$
L(\boldsymbol{\pi} \mid y_1, \ldots, y_n) = \prod_{i=1}^{n} \pi_i^{y_i}(1-\pi_i)^{1-y_i} = \exp\left[\sum_{i=1}^{n} y_i \log\left(\frac{\pi_i}{1-\pi_i}\right) + \sum_{i=1}^{n} \log(1-\pi_i)\right]
$$

Si $l(\cdot) = \log(L(\cdot))$ es la log verosimilitud, se tiene:

$$
l(\boldsymbol{\pi} \mid y_1, \ldots, y_n) = \sum_{i=1}^{n} y_i \log\left(\frac{\pi_i}{1-\pi_i}\right) + \sum_{i=1}^{n} \log(1-\pi_i)
$$

Finalmente, si reemplazamos los resultados de (1) en la ecuación anterior, se obtiene:

$$
l(\boldsymbol{\beta} \mid y_1, \ldots, y_n) = \sum_{i=1}^{n} y_i \mathbf{x}_i^T \boldsymbol{\beta} - \sum_{i=1}^{n} \log(1+\exp(\mathbf{x}_i^T \boldsymbol{\beta}))
$$

### La Función Score

Para encontrar los estimadores de máxima verosimilitud, necesitamos derivar la log verosimilitud con respecto a $\boldsymbol{\beta}$ e igualarla a cero. La derivada parcial de $l(\boldsymbol{\beta})$ con respecto a $\beta_j$ es:

$$
\frac{\partial l(\boldsymbol{\beta})}{\partial \beta_j} = \sum_{i=1}^{n} y_i x_{ij} - \sum_{i=1}^{n} \frac{\exp(\mathbf{x}_i^T \boldsymbol{\beta})}{1+\exp(\mathbf{x}_i^T \boldsymbol{\beta})} x_{ij}
$$

Recordando que $\pi_i = \frac{\exp(\mathbf{x}_i^T \boldsymbol{\beta})}{1+\exp(\mathbf{x}_i^T \boldsymbol{\beta})}$, podemos reescribir la derivada como:

$$
\frac{\partial l(\boldsymbol{\beta})}{\partial \beta_j} = \sum_{i=1}^{n} x_{ij}(y_i - \pi_i)
$$

El vector de derivadas parciales, conocido como **función score**, es:

$$
U(\boldsymbol{\beta}) = \begin{pmatrix}
\frac{\partial l(\boldsymbol{\beta})}{\partial \beta_0} \\
\frac{\partial l(\boldsymbol{\beta})}{\partial \beta_1} \\
\vdots \\
\frac{\partial l(\boldsymbol{\beta})}{\partial \beta_k}
\end{pmatrix} = \sum_{i=1}^{n} \mathbf{x}_i (y_i - \pi_i)
$$

donde $\mathbf{x}_i = (1, x_{i1}, x_{i2}, \ldots, x_{ik})^T$ es el vector de covariables para la observación $i$-ésima.

Los estimadores de máxima verosimilitud $\hat{\boldsymbol{\beta}}$ se obtienen resolviendo el sistema de ecuaciones:

$$
\begin{equation}
U(\boldsymbol{\beta}) = \sum_{i=1}^{n} \mathbf{x}_i (y_i - \pi_i) = \mathbf{0}
\end{equation}
$$

Este sistema de ecuaciones no tiene solución analítica cerrada, por lo que se requiere el uso de métodos iterativos como Newton-Raphson o Fisher Scoring para encontrar los estimadores.

### Matriz de Información de Fisher

La matriz de información de Fisher $I(\boldsymbol{\beta})$ juega un papel crucial en la estimación y en las soluciones al problema de separación. Esta matriz se define como el valor esperado del negativo de la matriz Hessiana (segundas derivadas) de la log verosimilitud:

$$
I(\boldsymbol{\beta}) = -E\left[\frac{\partial^2 l(\boldsymbol{\beta})}{\partial \boldsymbol{\beta} \partial \boldsymbol{\beta}^T}\right]
$$

Para la regresión logística, la matriz de información de Fisher tiene la forma:

$$
I(\boldsymbol{\beta}) = \sum_{i=1}^{n} \pi_i(1-\pi_i) \mathbf{x}_i \mathbf{x}_i^T = \mathbf{X}^T \mathbf{W} \mathbf{X}
$$

donde $\mathbf{X}$ es la matriz de diseño con filas $\mathbf{x}_i^T$ y $\mathbf{W}$ es una matriz diagonal con elementos $w_{ii} = \pi_i(1-\pi_i)$.

La matriz de información de Fisher es importante porque:

1. **Inversa de la matriz de covarianza asintótica**: La matriz de covarianza asintótica de los estimadores de máxima verosimilitud es $I(\boldsymbol{\beta})^{-1}$.

2. **Errores estándar**: Los errores estándar de los coeficientes se obtienen de los elementos diagonales de $I(\hat{\boldsymbol{\beta}})^{-1}$.

3. **Penalización de Firth**: La penalización de Firth utiliza esta matriz para modificar la función score y prevenir la separación.

## El Problema de Separación

### Separación Completa

La separación completa ocurre cuando existe un vector $\boldsymbol{\beta}$ tal que:

$$
\begin{cases}
\mathbf{x}_i^T \boldsymbol{\beta} > 0 & \text{si } y_i = 1 \\
\mathbf{x}_i^T \boldsymbol{\beta} < 0 & \text{si } y_i = 0
\end{cases}
$$

En este caso, todos los valores de la variable de respuesta pueden ser perfectamente predichos por el modelo.

### Separación Cuasicompleta

La separación cuasicompleta ocurre cuando existe un vector $\boldsymbol{\beta}$ tal que:

$$
\begin{cases}
\mathbf{x}_i^T \boldsymbol{\beta} \geq 0 & \text{si } y_i = 1 \\
\mathbf{x}_i^T \boldsymbol{\beta} \leq 0 & \text{si } y_i = 0
\end{cases}
$$

Con al menos una igualdad estricta. En este caso, un subconjunto de observaciones puede ser perfectamente predicho.

![Ejemplo de separación completa](~/assets/blog_images/plot_sep.jpg)

**Figura 1:** Ejemplo de separación completa en un conjunto de datos. Los puntos de diferentes clases están perfectamente separados por una línea de decisión.

### Consecuencias

Cuando ocurre separación (completa o cuasicompleta):

1. **Los algoritmos de optimización no convergen**: Los métodos iterativos como Newton-Raphson o Fisher Scoring no logran encontrar una solución finita.

2. **Estimaciones infinitas**: Algunos o todos los coeficientes $\hat{\beta}_j$ tienden a infinito.

3. **Errores estándar muy grandes**: Los errores estándar de los coeficientes se vuelven extremadamente grandes, haciendo que las pruebas de hipótesis sean poco confiables.

4. **Probabilidades predichas extremas**: Las probabilidades predichas $\hat{\pi}_i$ tienden a ser 0 o 1, lo que puede ser problemático para interpretación.

## Soluciones al Problema de Separación

### Penalización de Firth

La penalización de Firth (Firth, 1992) modifica la función score agregando un término de penalización basado en la información de Fisher:

$$
U^*(\boldsymbol{\beta}) = U(\boldsymbol{\beta}) + \frac{1}{2} \text{tr}\left[I(\boldsymbol{\beta})^{-1} \frac{\partial I(\boldsymbol{\beta})}{\partial \boldsymbol{\beta}}\right]
$$

donde $I(\boldsymbol{\beta})$ es la matriz de información de Fisher. Esta modificación reduce el sesgo de las estimaciones y previene la separación.

### Regularización

Otras alternativas incluyen:

- **Ridge Regression (L2)**: Agrega un término de penalización $\lambda \sum_{j=1}^{k} \beta_j^2$ a la función de verosimilitud.

- **Lasso (L1)**: Agrega un término de penalización $\lambda \sum_{j=1}^{k} |\beta_j|$ a la función de verosimilitud.

## Ejemplo Práctico

Consideremos un caso de clasificación de clientes morosos de un banco. Supongamos que tenemos datos con dos covariables $x_1$ y $x_2$, y una variable de respuesta binaria $y$ que indica si el cliente es moroso (1) o no (0).

### Datos con Separación Completa

Cuando intentamos ajustar un modelo de regresión logística estándar, obtenemos:

```
Warning messages:
1: glm.fit: algorithm did not converge
2: glm.fit: fitted probabilities numerically 0 or 1 occurred
```

Esto indica que hay separación en el conjunto de datos. Los coeficientes estimados pueden ser extremadamente grandes y los errores estándar muy grandes, como se muestra en la siguiente tabla:

**Tabla 1:** Resultados de la estimación por máxima verosimilitud con separación

```
                Estimate    Std. Error    z value    Pr(>|z|)
(Intercept)     639.6329    176750.0035     0.00      0.9971
x1              -25.4392      6923.6497    -0.00      0.9971
x2              -25.4442     17615.1711    -0.00      0.9988
```

Como se puede observar, los coeficientes son extremadamente grandes y los errores estándar son desproporcionadamente grandes, lo que hace que los valores z sean prácticamente cero y los p-valores sean cercanos a 1, indicando que el modelo no puede proporcionar inferencias confiables.

**Nota:** La Figura 1 mostrada anteriormente ilustra visualmente cómo se ve la separación completa en los datos, donde las clases están perfectamente separadas.

### Solución con Penalización de Firth

Para solucionar este problema, utilizamos la penalización de Firth en la función score, la cual se realiza en R con el paquete `logistf`:

```r
library(logistf)

# Ajustar modelo con penalización de Firth
lr2 <- logistf(y ~ x1 + x2, data = data)
summary(lr2)
```

Los resultados muestran que:

1. **No hay mensajes de error**: El algoritmo converge correctamente.
2. **Estimaciones finitas**: Todos los coeficientes tienen valores finitos y razonables.
3. **Predicciones perfectas mantenidas**: El modelo mantiene la capacidad de hacer predicciones perfectas, pero con estimaciones estables.

Un ejemplo de salida del modelo con penalización de Firth sería:

```
Model fitted by Penalized ML
Coefficients:
(Intercept) -32.567000  14.3902050  -136.3499317  -11.597743  33.652444
x1           1.265011   0.5566551    0.4609679    5.171369   60.204715
x2           1.858898   1.6290107   -3.1716557   10.285348    1.084355

likelihood ratio test = 62.76578 on 2 df, p = 2.342571e-14
```

Como se puede observar, ahora los coeficientes son finitos y razonables, y el algoritmo converge correctamente sin mensajes de advertencia.

![Gráfica de dispersión con Penalización de Firth](~/assets/blog_images/sep.jpg)

**Figura 2:** Gráfica de dispersión de los datos y banda de decisión estimada con Máxima verosimilitud penalizada. El modelo penalizado también genera estimaciones perfectas ante la separación completa, pero en este caso, sí convergen los estimadores, lo que hace que sea más creíble el modelo. Se observa que la línea de decisión (en azul) separa correctamente las clases mientras mantiene estimaciones finitas y estables.


## Diagnóstico de Separación

Para diagnosticar si hay separación en los datos, podemos:

1. **Revisar los mensajes de advertencia**: Si aparecen mensajes como "algorithm did not converge" o "fitted probabilities numerically 0 or 1", hay indicios de separación.

2. **Revisar los coeficientes**: Si los coeficientes son extremadamente grandes (por ejemplo, > 10 o < -10), puede haber separación.

3. **Revisar los errores estándar**: Si los errores estándar son muy grandes en relación con los coeficientes, es un signo de separación.

4. **Visualizar los datos**: Un gráfico de dispersión puede revelar si las clases son perfectamente separables. El código R proporcionado incluye funciones para generar estos gráficos y visualizar la línea de decisión.

## Conclusiones

La separación es un problema poco frecuente pero hay que tener cuidado con él, ya que se puede tener mucho sesgo en las estimaciones al estar frente a algoritmos que no convergen. Aunque muchos autores hablan de la separación como un problema, hay otros que hablan de ella como un buen resultado, pues lo que se espera son predicciones perfectas. Sin embargo, con los resultados de este trabajo se puede observar que la solución por penalización corrige el problema de convergencia de los algoritmos sin sacrificar las predicciones perfectas.

Con base en lo anterior, se recomienda que cuando se tengan problemas de separación y convergencia en los algoritmos, se utilicen métodos de penalización como el de Firth o la regularización, con el ánimo de corregir estos problemas, pues al momento de presentar tu modelo será más confiable un modelo con estimaciones que convergen, manteniendo las buenas predicciones del mismo.

## Referencias

- Albert, A. & Anderson, J. A. (1984). On the existence of maximum likelihood estimates in logistic regression models. *Biometrika*, 71(1), 1-10.

- Allison, P. D. (2008). Convergence failures in logistic regression. *SAS Global Forum*, 360, 1-11.

- Correa, J. C. & Valencia, M. (2011). La separación en regresión logística, una solución y aplicación. *Revista Facultad Nacional de Salud Pública*, 29(3), 281-288.

- Firth, D. (1992). Bias reduction, the jeffreys prior and glim. In *Advances in GLIM and Statistical Modelling* (pp. 91-100). Springer.

- Heinze, G. & Schemper, M. (2002). A solution to the problem of separation in logistic regression. *Statistics in Medicine*, 21(16), 2409-2419.

- Mansournia, M. A., Geroldinger, A., Greenland, S. & Heinze, G. (2018). Separation in logistic regression: causes, consequences, and control. *American Journal of Epidemiology*, 187(4), 864-870.

- Rindskopf, D. (2002). Infinite parameter estimates in logistic regression: Opportunities, not problems. *Journal of Educational and Behavioral Statistics*, 27(2), 147-161.

