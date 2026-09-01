---
title: "El 'Memory Leak' de LightGBM en Producción: Arenas de glibc y Threads de OpenMP"
description: "Cómo un crecimiento indefinido de memoria en Railway resultó ser un problema de fragmentación de glibc combinado con los thread pools de LightGBM y OpenBLAS"
featured: true
pubDate: "30 Aug 2026"
tags:
  - LightGBM
  - Memory Management
  - MLOps
  - Python
  - Railway
  - glibc
---

## El síntoma

Lo primero que noté en el servicio de predicción de [CarPredict](https://carpredict.co) no fue una métrica técnica, sino la factura de Railway subiendo mes a mes. Al principio la fui ignorando:

![Historial de facturación en Railway mostrando el aumento de costos mes a mes](~/assets/blog_images/railway_billing_history.png)

**Figura 1:** Historial de facturación de Railway. El costo pasó de \$5.00 en marzo a \$46.80 en julio — casi \$50 — antes de que investigara en serio qué lo estaba causando. En agosto, tras aplicar el fix, volvió a bajar a \$11.52.

Fue justamente ese salto acercándose a los \$50 el que finalmente me hizo revisar las métricas del servicio en vez de solo pagar la factura. Ahí encontré la causa: el uso de memoria crecía de forma prácticamente indefinida, hasta que el contenedor era reiniciado (probablemente por un OOM kill) y el ciclo volvía a empezar.

![Uso de memoria creciendo indefinidamente hasta el reinicio del contenedor](~/assets/blog_images/railway_memory_growth.png)

**Figura 2:** Métrica de memoria en Railway. El patrón de diente de sierra es característico: la memoria sube de forma sostenida durante horas y cae abruptamente cuando el servicio se reinicia. Railway cobra por uso de recursos (GB-hora), así que ese consumo sostenido y creciente era exactamente lo que estaba inflando la factura mes a mes.

La señal más importante en este gráfico no es el crecimiento en sí, sino su forma: no hay picos aislados ni saltos correlacionados con eventos puntuales, es una **pendiente sostenida** que no baja nunca por sí sola. Eso descarta de entrada la explicación más obvia — "el pipeline de sklearn o el objeto `Booster` de LightGBM se están acumulando en memoria" — porque el modelo se carga una única vez por proceso:

```python
@register
@lru_cache(maxsize=1)
def load_model():
    """Load and cache the sklearn pipeline. Thread-safe via lru_cache."""
    buf = load_model_bytes()
    pipeline = pickle.load(buf)
    _patch_for_serving(pipeline)
    return pipeline
```

Un solo pipeline cacheado con `lru_cache(maxsize=1)` no puede crecer con cada request. El objeto Python vive una sola vez en el proceso. El leak, si existía, no estaba en el nivel de Python.

## El sospechoso real: no es un leak de Python

El endpoint de predicción es una función **sync** (no `async def`) dentro de FastAPI:

```python
@router.post("/", response_model=PredictionResponse, ...)
def predict_price(req: CarFeaturesRequest) -> PredictionResponse:
    model = load_model()
    df = _request_to_dataframe(req)
    prediction = model.predict(df)
    ...
```

Starlette ejecuta los endpoints síncronos en un *thread pool* interno (vía `run_in_threadpool`), así que cada request de predicción corre en un hilo de sistema operativo distinto — no en el event loop de asyncio. Eso significa que, bajo carga concurrente, `model.predict(df)` se está llamando desde múltiples hilos nativos al mismo tiempo.

Ahí es donde entran dos capas nativas que casi nunca pensamos en depurar:

1. **LightGBM usa OpenMP (`libgomp`)** internamente para paralelizar el recorrido de árboles. Por defecto, el número de threads que usa es el número de cores disponibles (o lo que diga `OMP_NUM_THREADS`), y cada llamada a `predict()` puede activar ese pool de threads.
2. **Las transformaciones de scikit-learn/numpy en el pipeline** (imputadores, encoders, etc.) pueden apoyarse en OpenBLAS para operaciones vectorizadas, que tiene su propio pool de threads controlado por `OPENBLAS_NUM_THREADS`.

El problema no es que estos threads "tengan un leak" en el sentido clásico. El problema está una capa más abajo: en cómo **glibc administra la memoria cuando hay múltiples threads asignando y liberando memoria al mismo tiempo**.

## Arenas de glibc: el mecanismo detrás del crecimiento

`malloc` en glibc no usa un único heap global cuando hay multithreading. Para reducir contención sobre un lock compartido, glibc asigna **arenas** (heaps independientes) a los threads que compiten por memoria. Por defecto, el límite es:

$$
\text{MALLOC\_ARENA\_MAX} = 8 \times \text{núcleos disponibles (en sistemas de 64 bits)}
$$

Cada vez que un thread intenta hacer `malloc` y encuentra la arena existente bloqueada por otro thread, glibc puede crear una arena nueva en lugar de esperar. El detalle crítico es que **una arena, una vez creada, casi nunca se destruye**, y la memoria liberada dentro de ella no siempre se devuelve al sistema operativo — sobre todo si quedan objetos "vivos" intercalados con las asignaciones temporales (por ejemplo, un array de numpy de vida corta creado durante la inferencia, seguido de una asignación de más larga duración en la misma región de memoria, lo cual impide que glibc compacte o libere esas páginas).

En un servicio como este, cada request de predicción:

- Crea varios DataFrames y arrays de numpy temporales (uno por columna transformada en el pipeline).
- Dispara el thread pool de OpenMP dentro de LightGBM para el recorrido de árboles.
- Corre en un thread distinto del pool de Starlette.

Con suficiente tráfico y suficiente diversidad de hilos activos, el número de arenas creadas por glibc crece, y cada una retiene páginas de memoria que el proceso nunca libera al sistema operativo. Esto se ve, desde afuera, exactamente como el gráfico de la Figura 2: una fuga de memoria que en realidad es **fragmentación estructural del allocator**, no un objeto Python olvidado en algún lado.

Este es un patrón bien documentado — no soy el primero en toparme con él. Vale la pena leer el post original de [codearcana sobre el "arena leak" en glibc](https://codearcana.com/posts/2016/07/11/arena-leak-in-glibc.html), la guía de [Heroku sobre cómo ajustar el comportamiento de memoria de glibc](https://devcenter.heroku.com/articles/tuning-glibc-memory-behavior), y el hilo de [discuss.python.org sobre memory leaks con `ThreadPoolExecutor` y glibc](https://discuss.python.org/t/memory-leak-when-using-concurrent-futures-threadpoolexecutor-with-glibc-but-possibly-not-with-jemalloc/108311), que describe prácticamente el mismo síntoma que vi en Railway.

## La solución: forzar comportamiento single-threaded en el allocator

La solución fue setear tres variables de entorno en el servicio de Railway:

![Variables de entorno configuradas en Railway](~/assets/blog_images/railway_env_vars_fix.png)

**Figura 3:** `MALLOC_ARENA_MAX`, `OMP_NUM_THREADS` y `OPENBLAS_NUM_THREADS`, todas seteadas a `1`.

```bash
MALLOC_ARENA_MAX=1
OMP_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
```

Cada una ataca una parte distinta del problema:

- **`MALLOC_ARENA_MAX=1`** obliga a glibc a usar una única arena para todo el proceso, sin importar cuántos threads estén asignando memoria simultáneamente. El costo es un poco más de contención sobre el lock del allocator, pero para un servicio de baja/mediana concurrencia como este, ese costo es insignificante comparado con tener memoria acotada y predecible.
- **`OMP_NUM_THREADS=1`** evita que LightGBM dispare su thread pool de OpenMP en cada llamada a `predict()`. Esto es clave porque cada predicción de la API es sobre **una sola fila** (un solo carro a la vez) — paralelizar el recorrido de árboles para una fila no tiene ningún beneficio, solo overhead de crear/sincronizar threads.
- **`OPENBLAS_NUM_THREADS=1`** aplica el mismo criterio a cualquier operación vectorizada de numpy/scipy dentro del pipeline de scikit-learn.

Vale la pena resaltar el hallazgo de un [issue de LightGBM en GitHub](https://github.com/microsoft/LightGBM/issues/4705): llamar a una función multithreaded de LightGBM puede **cambiar la configuración global de threads de OpenMP para todo el proceso**, afectando incluso a otras librerías que dependen de OpenMP en el mismo proceso. Esto refuerza que, en un servidor con múltiples requests concurrentes golpeando el mismo proceso, no controlar explícitamente estas variables deja el comportamiento de threading a merced de quien hizo la última llamada.

Después de aplicar el cambio, el patrón de diente de sierra desapareció: la memoria se estabiliza en un rango bajo y no vuelve a crecer de forma sostenida (línea final de la Figura 2, tras el segundo reinicio), y la factura de agosto lo confirma: volvió a bajar a \$11.52.

## Por qué el trade-off tiene sentido acá

Limitar todo a un solo thread por request suena contraintuitivo — ¿no se supone que el paralelismo hace las cosas más rápido? La respuesta depende del tamaño del batch:

- Para **entrenamiento** (miles o millones de filas), el paralelismo de LightGBM/OpenBLAS es enormemente valioso, y ahí sí conviene dejar que use todos los cores disponibles.
- Para **inferencia sobre una sola fila** en un endpoint de API, el trabajo por request es tan pequeño que el overhead de spinear un thread pool supera cualquier beneficio de paralelizarlo. El paralelismo aquí no compraba velocidad, solo compraba threads adicionales tocando el allocator y, con eso, memoria sin control.

Esto es un caso concreto de una lección más general al llevar modelos de ML a producción: el mismo artefacto que se entrena con toda la CPU disponible se sirve mejor, en el caso de inferencia unitaria de baja latencia, con el threading explícitamente apagado.

## Diagnóstico rápido para este tipo de problema

Si ves un patrón de memoria similar (crecimiento sostenido, sin caídas, hasta un reinicio) en un servicio Python que sirve modelos con librerías nativas (LightGBM, XGBoost, scikit-learn con OpenBLAS, PyTorch/TensorFlow con OpenMP), vale la pena revisar, en este orden:

1. **¿El modelo/objeto se carga una sola vez por proceso?** Si ya está cacheado (como acá con `lru_cache`), descarta un leak a nivel de objetos Python.
2. **¿El framework web ejecuta endpoints sync en un thread pool?** (FastAPI/Starlette lo hace por defecto). Si es así, cada request corre en un thread de sistema operativo distinto.
3. **¿Las librerías nativas usadas (LightGBM, OpenBLAS, MKL) tienen su propio thread pool?** Si no se fija `OMP_NUM_THREADS` / `OPENBLAS_NUM_THREADS` explícitamente, el comportamiento por defecto suele ser "usar todos los cores", multiplicado por cada request concurrente.
4. **¿`MALLOC_ARENA_MAX` está en su valor por defecto** (8 × cores)? Si sí, y hay tráfico concurrente sostenido, es candidato número uno para fragmentación tipo "arena leak".

En este caso, los tres puntos aplicaban a la vez, y arreglar solo uno probablemente no hubiera sido suficiente: `OMP_NUM_THREADS=1` reduce cuántos threads nuevos aparecen, pero `MALLOC_ARENA_MAX=1` es lo que efectivamente acota cuánta memoria puede quedar retenida por thread una vez que ya existen.

## Referencias

- [Arena "leak" in glibc — codearcana](https://codearcana.com/posts/2016/07/11/arena-leak-in-glibc.html)
- [Tuning glibc Memory Behavior — Heroku Dev Center](https://devcenter.heroku.com/articles/tuning-glibc-memory-behavior)
- [Memory leak when using concurrent.futures.ThreadPoolExecutor with glibc — discuss.python.org](https://discuss.python.org/t/memory-leak-when-using-concurrent-futures-threadpoolexecutor-with-glibc-but-possibly-not-with-jemalloc/108311)
- [Calling multithreaded functions sets global number of OMP threads — microsoft/LightGBM#4705](https://github.com/microsoft/LightGBM/issues/4705)
- [LightGBM Parameters Tuning — documentación oficial](https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html)
