---
title: "bestdist: Encuentra la Mejor Distribución de Probabilidad para tus Datos"
description: "Un tutorial práctico sobre cómo usar el paquete bestdist para ajustar automáticamente distribuciones de probabilidad a tus datos usando pruebas estadísticas y criterios de información."
featured: true
draft: false
pubDate: "Mar 19 2026"
tags:
  - Python
  - Statistics
  - Probability
  - Data Science
  - bestdist
---

## Introducción

Uno de los pasos más importantes —y frecuentemente subestimados— en el análisis de datos es identificar qué distribución de probabilidad describe mejor los datos que tenemos. Conocer la distribución subyacente nos permite hacer inferencias más precisas, simular escenarios, calcular probabilidades exactas y construir mejores modelos.

Sin embargo, el proceso de ajustar distribuciones manualmente es tedioso: hay que probar múltiples distribuciones, ejecutar pruebas de bondad de ajuste, comparar criterios de información y visualizar los resultados. Es aquí donde entra **`bestdist`**, un paquete de Python que automatiza todo este proceso.

En este post exploraremos los fundamentos teóricos detrás del ajuste de distribuciones y cómo usar `bestdist` para encontrar la mejor distribución para tus datos de forma sencilla.

## Fundamento Teórico

### ¿Qué es el ajuste de distribuciones?

Dado un conjunto de observaciones $x_1, x_2, \ldots, x_n$, queremos encontrar la distribución paramétrica $F(x; \boldsymbol{\theta})$ que mejor describe los datos. El proceso tiene dos partes:

1. **Estimación de parámetros**: Encontrar $\hat{\boldsymbol{\theta}}$ que maximice la verosimilitud de los datos.
2. **Evaluación del ajuste**: Determinar qué tan bien se ajusta la distribución a los datos observados.

### Pruebas de Bondad de Ajuste

`bestdist` implementa tres pruebas de bondad de ajuste:

#### Kolmogorov-Smirnov (KS)

La prueba KS mide la distancia máxima entre la función de distribución empírica $\hat{F}_n(x)$ y la distribución teórica $F(x; \hat{\boldsymbol{\theta}})$:

$$
D_n = \sup_x |\hat{F}_n(x) - F(x; \hat{\boldsymbol{\theta}})|
$$

donde $\hat{F}_n(x) = \frac{1}{n} \sum_{i=1}^{n} \mathbf{1}(x_i \leq x)$ es la función de distribución empírica. Un p-valor grande indica un buen ajuste.

#### Anderson-Darling (AD)

La prueba AD es una versión ponderada de la prueba KS que da más importancia a las colas de la distribución:

$$
A^2 = -n - \frac{1}{n} \sum_{i=1}^{n} (2i - 1) \left[\ln F(x_{(i)}) + \ln(1 - F(x_{(n+1-i)}))\right]
$$

donde $x_{(1)} \leq x_{(2)} \leq \cdots \leq x_{(n)}$ son las estadísticas de orden.

#### Chi-cuadrado ($\chi^2$)

Para datos discretos, la prueba Chi-cuadrado compara las frecuencias observadas con las esperadas bajo la distribución hipotética:

$$
\chi^2 = \sum_{k} \frac{(O_k - E_k)^2}{E_k}
$$

donde $O_k$ son las frecuencias observadas y $E_k = n \cdot P(X = k; \hat{\boldsymbol{\theta}})$ son las frecuencias esperadas.

### Criterios de Información: AIC y BIC

Además de las pruebas de hipótesis, `bestdist` usa criterios de información para comparar distribuciones con distinto número de parámetros.

El **Criterio de Información de Akaike (AIC)** penaliza la log-verosimilitud por el número de parámetros $k$:

$$
\text{AIC} = 2k - 2\ell(\hat{\boldsymbol{\theta}})
$$

El **Criterio de Información Bayesiano (BIC)** penaliza más fuertemente el número de parámetros para muestras grandes:

$$
\text{BIC} = k \ln(n) - 2\ell(\hat{\boldsymbol{\theta}})
$$

En ambos casos, **menores valores indican mejor ajuste**. El BIC tiende a seleccionar modelos más parsimoniosos que el AIC.

## Instalación

```bash
pip install bestdist
```

O desde el código fuente:

```bash
git clone https://github.com/Wilmar3752/pdist.git
cd pdist
pip install -e .
```

## Distribuciones Soportadas

`bestdist` cubre las distribuciones más comunes tanto para datos continuos como discretos:

**Continuas (9):** Normal, Gamma, Beta, Weibull, Lognormal, Exponencial, Uniforme, Cauchy, Student-t

**Discretas (4):** Poisson, Binomial, Binomial Negativa, Geométrica

## Uso Básico

### Datos Continuos

Supongamos que tenemos mediciones de tiempos de procesamiento de solicitudes en un servidor, que sospechamos siguen una distribución Gamma.

```python
import numpy as np
from bestdist import DistributionFitter

# Simulamos datos de tiempo de procesamiento (distribución Gamma)
np.random.seed(42)
data = np.random.gamma(shape=2, scale=3, size=1000)

# Crear el fitter (continuo por defecto)
fitter = DistributionFitter(data)

# Ajustar todas las distribuciones
results = fitter.fit()
```

El método `fit()` retorna un resumen con el ranking de todas las distribuciones probadas:

```
  Distribution  Test Statistic        P-Value  ...
0        Gamma        0.013802   9.899e-01
1         Beta        0.017457   9.154e-01
2    Lognormal        0.019547   8.320e-01
3      Weibull        0.023921   6.076e-01
4     StudentT        0.088457   2.942e-07
5       Normal        0.106301   2.707e-10
6       Cauchy        0.141788   5.282e-18
7  Exponential        0.150617   2.862e-20
8      Uniform        0.410416  7.410e-153
```

Para obtener la mejor distribución y visualizarla:

```python
# Obtener la mejor distribución por p-valor (criterio por defecto)
best = fitter.get_best_distribution()
print(f"Mejor distribución: {best['distribution']}")
print(f"Parámetros:         {best['parameters']}")
print(f"P-valor:            {best['p_value']:.4f}")
# Mejor distribución: Gamma
# Parámetros:         {'a': 2.093, 'loc': 0.039, 'scale': 2.934}
# P-valor:            0.9899

# Visualizar el ajuste de la mejor distribución
fitter.plot_best_fit(bins=40)
```

![Ajuste de la distribución Gamma a los datos de tiempos de procesamiento](~/assets/blog_images/bestdist_best_fit_gamma.png)

**Figura 1:** El histograma de los datos superpuesto con la PDF de la distribución Gamma ajustada. La curva se ajusta muy bien a los datos, confirmado por el p-valor de 0.99.

```python
# Comparar todas las distribuciones visualmente
fitter.compare_distributions()
```

![Comparación de todas las distribuciones candidatas](~/assets/blog_images/bestdist_compare_continuous.png)

**Figura 2:** Comparación visual de todas las distribuciones probadas. Las distribuciones Gamma, Beta y Lognormal presentan el mejor ajuste para este conjunto de datos.

### Datos Discretos

Para datos de conteo —por ejemplo, el número de llamadas por hora en un call center— usamos el modo discreto:

```python
from bestdist import DistributionFitter
import numpy as np

# Datos de conteo: número de eventos por unidad de tiempo
np.random.seed(42)
call_counts = np.random.poisson(lam=4.7, size=500)

# Crear fitter para distribuciones discretas
fitter = DistributionFitter(call_counts, dist_type='discrete')
results = fitter.fit()

# Obtener la mejor distribución
best = fitter.get_best_distribution()
print(f"Mejor distribución: {best['distribution']}")
print(f"Parámetros:         {best['parameters']}")
# Mejor distribución: NegativeBinomial
# Parámetros:         {'n': 277.02, 'p': 0.9836}
```

![Ajuste de distribución discreta a datos de llamadas por hora](~/assets/blog_images/bestdist_best_fit_poisson.png)

**Figura 3:** Ajuste de la Binomial Negativa a los datos de conteo. Aunque los datos fueron generados con una Poisson, la Binomial Negativa resulta ganadora por su mayor flexibilidad para capturar la varianza de los datos.

## Criterios de Selección

`bestdist` permite seleccionar la mejor distribución usando tres criterios distintos:

```python
# Por p-valor (mayor p-valor → mejor ajuste estadístico)
best_pvalue = fitter.get_best_distribution(criterion='p_value')

# Por AIC (menor AIC → mejor balance ajuste/complejidad)
best_aic = fitter.get_best_distribution(criterion='aic')

# Por BIC (menor BIC → más parsimonioso)
best_bic = fitter.get_best_distribution(criterion='bic')

print(f"Mejor por p-valor: {best_pvalue['distribution']}")
print(f"Mejor por AIC:     {best_aic['distribution']}")
print(f"Mejor por BIC:     {best_bic['distribution']}")
```

¿Cuándo usar cada criterio?

- **p-valor**: Útil cuando el objetivo es hacer inferencia estadística formal. Sensible al tamaño de muestra.
- **AIC**: Preferible cuando el modelo se usará para predicción y se quiere equilibrar ajuste y complejidad.
- **BIC**: Recomendado cuando la muestra es grande y se quiere el modelo más parsimonioso.

## Uso Avanzado

### Especificar un subconjunto de distribuciones

Cuando ya tienes hipótesis sobre qué familia de distribuciones es más apropiada, puedes limitarte a probar solo esas:

```python
from bestdist import DistributionFitter
from bestdist.distributions.continuous import Normal, Gamma, Lognormal, Exponential

# Probar solo distribuciones de cola derecha para datos de ingresos
income_data = np.random.lognormal(mean=10.5, sigma=0.8, size=500)

fitter = DistributionFitter(
    income_data,
    distributions=[Gamma, Lognormal, Exponential]
)
results = fitter.fit()
best = fitter.get_best_distribution()
print(fitter.summary())
```

### Integración con Pandas

`bestdist` funciona directamente con columnas de DataFrames:

```python
import pandas as pd
from bestdist import DistributionFitter

df = pd.read_csv('ventas.csv')

# Ajustar distribución a una columna
fitter = DistributionFitter(df['monto_venta'])
fitter.fit()

# El resumen retorna un DataFrame listo para análisis
resumen = fitter.summary(top_n=5)
print(resumen)
```

### Distribuciones Personalizadas

Una de las características más poderosas de `bestdist` es su extensibilidad. Puedes implementar cualquier distribución de `scipy.stats` heredando de `BaseDistribution`:

```python
from bestdist.core.base import BaseDistribution
from scipy.stats import pareto, rv_continuous
from typing import Tuple

class Pareto(BaseDistribution):
    """Distribución de Pareto para datos con colas muy pesadas."""

    def _get_scipy_dist(self) -> rv_continuous:
        return pareto

    def _extract_params(self, fit_result: Tuple) -> dict:
        return {
            'b':     float(fit_result[0]),  # forma (índice de Pareto)
            'loc':   float(fit_result[1]),
            'scale': float(fit_result[2])
        }

# Usar la distribución personalizada junto con las nativas
from bestdist.distributions.continuous import Lognormal, Gamma

fitter = DistributionFitter(
    data,
    distributions=[Pareto, Lognormal, Gamma]
)
results = fitter.fit()
```

### Uso de distribuciones individuales

También puedes usar cada distribución por separado para obtener estadísticos y generar muestras:

```python
from bestdist.distributions.continuous import Normal, Lognormal
from bestdist.distributions.discrete import Poisson
import numpy as np

# Distribución Normal
data = np.random.normal(5, 2, 1000)
dist = Normal(data)
dist.fit()

print(f"Media:     {dist.mean:.2f}")
print(f"Desv. Est: {dist.std:.2f}")

# Evaluar PDF y CDF
x = np.linspace(0, 10, 100)
pdf_vals = dist.pdf(x)
cdf_vals = dist.cdf(x)

# Generar muestras sintéticas
samples = dist.rvs(size=200, random_state=42)

# Poisson: probabilidades puntuales
count_data = np.random.poisson(lam=3.5, size=1000)
poisson = Poisson(count_data)
poisson.fit()

print(f"Lambda estimado: {poisson.mu:.4f}")
print(f"P(X = 5) = {poisson.pmf(5):.4f}")
print(f"P(X ≤ 3) = {poisson.cdf(3):.4f}")
```

## Ejemplo Completo: Análisis de Datos de Seguros

Para ilustrar un caso de uso real, supongamos que queremos modelar el monto de siniestros de una aseguradora.

```python
import numpy as np
import pandas as pd
from bestdist import DistributionFitter
from bestdist.distributions.continuous import Gamma, Lognormal, Weibull, Exponential

np.random.seed(0)

# Simular montos de siniestros (distribución Lognormal típica en seguros)
siniestros = np.random.lognormal(mean=8.5, sigma=1.2, size=800)

print(f"Montos de siniestros - Resumen descriptivo:")
print(f"  n        = {len(siniestros)}")
print(f"  Media    = ${siniestros.mean():,.0f}")
print(f"  Mediana  = ${np.median(siniestros):,.0f}")
print(f"  Asimetría = {pd.Series(siniestros).skew():.2f}")

# Probar distribuciones habituales en actuaría
fitter = DistributionFitter(
    siniestros,
    distributions=[Gamma, Lognormal, Weibull, Exponential]
)
fitter.fit()

# Ver ranking completo
print("\nRanking de distribuciones:")
print(fitter.summary())

# Mejor distribución por AIC (recomendado para predicción)
best = fitter.get_best_distribution(criterion='aic')
print(f"\nMejor distribución (AIC): {best['distribution']}")
print(f"Parámetros: {best['parameters']}")

# Visualizar
fitter.plot_best_fit(bins=50)
fitter.compare_distributions()
```

![Mejor ajuste para montos de siniestros](~/assets/blog_images/bestdist_seguros_best_fit.png)

**Figura 4:** Ajuste de la distribución Lognormal a los montos de siniestros. La distribución captura correctamente la asimetría positiva característica de este tipo de datos en seguros.

![Comparación de distribuciones para el ejemplo actuarial](~/assets/blog_images/bestdist_seguros_compare.png)

**Figura 5:** Comparación de las cuatro distribuciones candidatas para los datos de siniestros. La Lognormal y Gamma presentan el mejor ajuste, mientras que la Exponencial y Weibull no logran capturar la forma de la distribución empírica.

## Conclusiones

`bestdist` simplifica considerablemente el proceso de ajuste de distribuciones, que de otra forma requiere escribir código repetitivo para cada distribución candidata. Sus principales ventajas son:

- **Automatización**: prueba múltiples distribuciones en una sola llamada.
- **Flexibilidad**: soporta datos continuos y discretos, y permite distribuciones personalizadas.
- **Múltiples criterios**: p-valor, AIC y BIC para distintos objetivos de análisis.
- **Integración natural** con NumPy y Pandas.

El ajuste de distribuciones es un paso fundamental en la modelación estadística, simulación de Monte Carlo, diseño de experimentos y análisis actuarial. Contar con una herramienta que lo automatice permite al analista enfocarse en interpretar los resultados y tomar decisiones informadas.

El paquete está disponible en [GitHub](https://github.com/Wilmar3752/pdist) y puede instalarse directamente con `pip install bestdist`.

## Referencias

- Akaike, H. (1974). A new look at the statistical model identification. *IEEE Transactions on Automatic Control*, 19(6), 716–723.

- Anderson, T. W. & Darling, D. A. (1952). Asymptotic theory of certain "goodness of fit" criteria based on stochastic processes. *The Annals of Mathematical Statistics*, 23(2), 193–212.

- Kolmogorov, A. N. (1933). Sulla determinazione empirica di una legge di distribuzione. *Giornale dell'Istituto Italiano degli Attuari*, 4, 83–91.

- Schwarz, G. (1978). Estimating the dimension of a model. *The Annals of Statistics*, 6(2), 461–464.

- Sepulveda, W. (2024). bestdist: Find the best probability distribution for your data. GitHub. https://github.com/Wilmar3752/pdist
