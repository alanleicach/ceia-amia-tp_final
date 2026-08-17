# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: ia_env
#     language: python
#     name: python3
# ---

# %% [markdown]
# # TP 1: LDA/QDA y optimización matemática de modelos
#
# Se puede consultar la introducción teórica en el mono-notebook, se prefiere mantener este lo más chico posible.

# %%
# imports
import numpy as np
import numpy.linalg as LA

from base.qda import QDA, TensorizedQDA
from base.cholesky import QDA_Chol1, QDA_Chol2, QDA_Chol3
from utils.bench import Benchmark
from utils.datasets import (get_iris_dataset, get_letters_dataset, 
                            get_penguins_dataset, get_wine_dataset,
                            label_encode)


# %% [markdown]
# ## Ejemplo

# %%
# levantamos el dataset Wine, que tiene 13 features y 178 observaciones en total
X_full, y_full = get_wine_dataset()

X_full.shape, y_full.shape

# %%
# encodeamos a número las clases
y_full_encoded = label_encode(y_full)

y_full[:5], y_full_encoded[:5]

# %%
# generamos el benchmark
# observar que son valores muy bajos de runs para que corra rápido ahora
b = Benchmark(
    X_full, y_full_encoded,
    n_runs = 100,
    warmup = 20,
    mem_runs = 20,
    test_sz = 0.3,
    same_splits = False
)

# %%
# bencheamos un par
to_bench = [QDA]

for model in to_bench:
    b.bench(model)

# %%
# como es una clase, podemos seguir bencheando más después
b.bench(TensorizedQDA)

# %%
# hacemos un summary
b.summary()

# %%
# son muchos datos! nos quedamos con un par nomás
summ = b.summary()

# como es un pandas DataFrame, subseteamos columnas fácil
summ[['train_median_ms', 'test_median_ms','mean_accuracy']]

# %%
# podemos setear un baseline para que fabrique columnas de comparación
summ = b.summary(baseline='QDA')

summ

# %%
# volvemos a subsetear columnas
summ[[
    'train_median_ms', 'test_median_ms','mean_accuracy',
    'train_speedup', 'test_speedup',
    'train_mem_reduction', 'test_mem_reduction'
]]


# %% [markdown]
# ## Tensorización
#
# ### 1) Diferencias entre `QDA` y `TensorizedQDA`

# %% [markdown]
# ### 1.1 ¿Sobre qué paraleliza `TensorizedQDA`? ¿Sobre las $k$ clases, las $n$ observaciones a predecir, o ambas?
#
# `TensorizedQDA` paraleliza las operaciones sobre las $k$ clases mediante tensorización, pero continúa procesando individualmente las $n$ observaciones a predecir.
#
# Durante el entrenamiento, QDA calcula para cada clase su vector de medias $\mu_k$ y su matriz de covarianza $\Sigma_k$. Durante la predicción, cada nueva observación $x$ debe ser evaluada respecto de cada clase utilizando, entre otros términos, la forma cuadrática:
#
# $$
# (x-\mu_k)^T\Sigma_k^{-1}(x-\mu_k)
# $$
#
# En `QDA`, para una misma observación, este cálculo se realiza individualmente para cada clase. En `TensorizedQDA`, en cambio, las medias y las matrices de covarianza inversa correspondientes a las distintas clases se apilan en tensores, permitiendo realizar estas operaciones para todas las clases mediante operaciones matriciales de NumPy.
#
# Por ejemplo, si se tienen $n=4$ observaciones y $k=3$ clases, conceptualmente `QDA` realiza:
#
#     x1 → C1 → C2 → C3
#     x2 → C1 → C2 → C3
#     x3 → C1 → C2 → C3
#     x4 → C1 → C2 → C3
#
# mientras que `TensorizedQDA` realiza:
#
#     x1 → [C1, C2, C3]
#     x2 → [C1, C2, C3]
#     x3 → [C1, C2, C3]
#     x4 → [C1, C2, C3]
#
# Por lo tanto, `TensorizedQDA` elimina el ciclo explícito sobre las $k$ clases, pero mantiene el ciclo sobre las $n$ observaciones. Es decir, la tensorización implementada en esta clase actúa sobre la dimensión correspondiente a las clases, no sobre la dimensión correspondiente a las observaciones.

# %% [markdown]
# ### 1.2 Análisis de los `shapes` de `tensor_inv_cov` y `tensor_means`
#
# > Nota: en la consigna se menciona `tensor_inv_covs`, mientras que en la implementación el atributo se denomina `tensor_inv_cov`.
#
# En `QDA`, para cada una de las $k$ clases se calcula un vector de medias y una matriz de covarianza inversa.
#
# Para una clase $j$:
#
# $$
# \mu_j \in \mathbb{R}^{p \times 1}
# $$
#
# $$
# \Sigma_j^{-1} \in \mathbb{R}^{p \times p}
# $$
#
# donde $p$ es la cantidad de features.
#
# En `TensorizedQDA`, luego de ejecutar el `_fit_params` de `QDA`, estos elementos se apilan mediante:
#
#     self.tensor_inv_cov = np.stack(self.inv_covs)
#     self.tensor_means = np.stack(self.means)
#
# Como existe una media y una matriz de covarianza inversa para cada una de las $k$ clases, los shapes resultantes son:
#
# $$
# \text{tensor\_inv\_cov.shape}=(k,p,p)
# $$
#
# $$
# \text{tensor\_means.shape}=(k,p,1)
# $$
#
# Es decir, la nueva primera dimensión corresponde a las $k$ clases.
#
# Durante la predicción, el método `predict` toma cada observación de forma individual y la transforma en un vector columna mediante:
#
#     X[:, i].reshape(-1, 1)
#
# por lo que, para una observación $x$:
#
# $$
# x.shape=(p,1)
# $$
#
# A continuación, `TensorizedQDA` ejecuta:
#
#     unbiased_x = x - self.tensor_means
#
# Como `x` tiene shape $(p,1)$ y `tensor_means` tiene shape $(k,p,1)$, NumPy utiliza broadcasting para restar la misma observación a la media de cada una de las $k$ clases. El resultado tiene:
#
# $$
# \text{unbiased\_x.shape}=(k,p,1)
# $$
#
# y contiene conceptualmente:
#
# $$
# \begin{bmatrix}
# x-\mu_1 \\
# x-\mu_2 \\
# \vdots \\
# x-\mu_k
# \end{bmatrix}
# $$
#
# Luego se ejecuta:
#
#     unbiased_x.transpose(0,2,1)
#
# El `transpose` mantiene la dimensión correspondiente a las clases e intercambia las dos últimas dimensiones:
#
# $$
# (k,p,1)\rightarrow(k,1,p)
# $$
#
# Esto permite realizar de forma tensorizada la misma forma cuadrática que `QDA` calcula individualmente para cada clase:
#
# $$
# (x-\mu_j)^T\Sigma_j^{-1}(x-\mu_j)
# $$
#
# En el código:
#
#     inner_prod = unbiased_x.transpose(0,2,1) @ self.tensor_inv_cov @ unbiased_x
#
# Los shapes involucrados son:
#
# $$
# (k,1,p)@(k,p,p)@(k,p,1)
# $$
#
# La primera multiplicación produce:
#
# $$
# (k,1,p)@(k,p,p)\rightarrow(k,1,p)
# $$
#
# y la segunda:
#
# $$
# (k,1,p)@(k,p,1)\rightarrow(k,1,1)
# $$
#
# Por lo tanto:
#
# $$
# \text{inner\_prod.shape}=(k,1,1)
# $$
#
# Cada una de las $k$ matrices de dimensión $(1,1)$ contiene la forma cuadrática correspondiente a una clase:
#
# $$
# (x-\mu_j)^T\Sigma_j^{-1}(x-\mu_j)
# $$
#
# Luego el código aplica:
#
#     inner_prod.flatten()
#
# por lo que:
#
# $$
# (k,1,1)\rightarrow(k,)
# $$
#
# obteniéndose un vector con un valor para cada clase.
#
# Por otro lado:
#
#     LA.det(self.tensor_inv_cov)
#
# calcula el determinante de cada una de las $k$ matrices de covarianza inversa. Como `tensor_inv_cov` tiene shape $(k,p,p)$, el resultado tiene shape:
#
# $$
# (k,)
# $$
#
# Así, la expresión:
#
#     0.5*np.log(LA.det(self.tensor_inv_cov)) - 0.5*inner_prod.flatten()
#
# devuelve un vector de shape $(k,)$ que contiene un score log-condicional para cada una de las $k$ clases.
#
# Finalmente, en `_predict_one` se ejecuta:
#
#     np.argmax(self.log_a_priori + self._predict_log_conditionals(x))
#
# `log_a_priori` también tiene un valor por clase, por lo que su shape es $(k,)$. Al sumarlo a los scores log-condicionales se obtiene un score a posteriori para cada clase, y `np.argmax` devuelve el índice de la clase con mayor valor.
#
# La evolución de los shapes para una observación es:
#
# | Elemento | Shape |
# |---|---|
# | `x` | $(p,1)$ |
# | `self.means[j]` | $(p,1)$ |
# | `self.inv_covs[j]` | $(p,p)$ |
# | `tensor_means` | $(k,p,1)$ |
# | `tensor_inv_cov` | $(k,p,p)$ |
# | `unbiased_x` | $(k,p,1)$ |
# | `unbiased_x.transpose(0,2,1)` | $(k,1,p)$ |
# | primera multiplicación matricial | $(k,1,p)$ |
# | `inner_prod` | $(k,1,1)$ |
# | `inner_prod.flatten()` | $(k,)$ |
# | `LA.det(self.tensor_inv_cov)` | $(k,)$ |
# | scores log-condicionales | $(k,)$ |
# | scores a posteriori | $(k,)$ |
# | `np.argmax(...)` | escalar |
#
# Por lo tanto, `TensorizedQDA` llega a la misma predicción que `QDA` porque no modifica el cálculo matemático realizado para cada clase. `QDA` calcula cada forma cuadrática por separado recorriendo las clases, mientras que `TensorizedQDA` apila los parámetros de las $k$ clases y realiza esos mismos cálculos conjuntamente mediante operaciones tensorizadas. En consecuencia, se obtienen los mismos scores por clase y el `argmax` selecciona la misma clase.

# %% [markdown]
# ### 2) Optimización
#
# #### 3. Implementación de `FasterQDA`
#
# Se implementa `FasterQDA` heredando de `TensorizedQDA`, pero redefiniendo el método `predict` para procesar simultáneamente las $n$ observaciones y eliminar el ciclo `for` presente en `BaseBayesianClassifier.predict`.

# %%
class FasterQDA(TensorizedQDA):

    def predict(self, X):

        # X: (p, n)
        # tensor_means: (k, p, 1)
        # broadcasting -> (k, p, n)
        unbiased_X = X - self.tensor_means

        # (k, n, p) @ (k, p, p) @ (k, p, n)
        # -> (k, n, n)
        inner_prod = (
            unbiased_X.transpose(0, 2, 1)
            @ self.tensor_inv_cov
            @ unbiased_X
        )

        # Para cada clase sólo interesan los términos
        # correspondientes a cada observación consigo misma.
        # (k, n, n) -> (k, n)
        quad_terms = np.diagonal(
            inner_prod,
            axis1=1,
            axis2=2
        )

        # determinantes: (k,) -> (k, 1)
        log_conditionals = (
            0.5 * np.log(LA.det(self.tensor_inv_cov))[:, None]
            - 0.5 * quad_terms
        )

        # log_a_priori: (k,) -> (k, 1)
        # resultado: (k, n)
        log_posteriori = (
            self.log_a_priori[:, None]
            + log_conditionals
        )

        # elegimos la clase de mayor score para cada observación
        # (k, n) -> (n,) -> (1, n)
        return np.argmax(log_posteriori, axis=0).reshape(1, -1)


# %%
qda = QDA()
tensorized_qda = TensorizedQDA()
faster_qda = FasterQDA()

qda.fit(X_full.T, y_full_encoded.T)
tensorized_qda.fit(X_full.T, y_full_encoded.T)
faster_qda.fit(X_full.T, y_full_encoded.T)

pred_qda = qda.predict(X_full.T)
pred_tensorized = tensorized_qda.predict(X_full.T)
pred_faster = faster_qda.predict(X_full.T)

print("QDA == TensorizedQDA:",
      np.array_equal(pred_qda, pred_tensorized))

print("QDA == FasterQDA:",
      np.array_equal(pred_qda, pred_faster))

# %% [markdown]
# La implementación de `FasterQDA` fue validada comparando sus predicciones con las de `QDA` y `TensorizedQDA`. En ambos casos se obtuvo igualdad completa de las predicciones:
#
# - `QDA == TensorizedQDA`: `True`
# - `QDA == FasterQDA`: `True`
#
# Esto confirma que la eliminación del ciclo `for` en `predict` no modifica el resultado del clasificador.

# %% [markdown]
# #### 4. ¿Dónde aparece la matriz de $n\times n$?
#
# En `FasterQDA.predict`, al pasar todas las observaciones juntas, el producto
#
#     unbiased_X.transpose(0,2,1) @ tensor_inv_cov @ unbiased_X
#
# no da un escalar por observación, sino un tensor de shape `(k, n, n)`: para cada clase calcula la interacción **entre todo par de observaciones** $(x_i-\mu_k)^T\Sigma_k^{-1}(x_j-\mu_k)$, aunque sólo interesan los términos con $i=j$ (la diagonal). Los términos con $i\neq j$ se calculan y se descartan.

# %%
# X contiene todas las observaciones con la convención del proyecto:
# filas = p features, columnas = n observaciones.
X = X_full.T

# Broadcasting de las medias de las k clases:
# X:                  (p, n)
# tensor_means:       (k, p, 1)
# unbiased_X:         (k, p, n)
unbiased_X = X - tensorized_qda.tensor_means

# Forma cuadrática vectorizada:
# (k, n, p) @ (k, p, p) @ (k, p, n) -> (k, n, n)
# Esta es la matriz n x n mencionada en la consigna.
inner_prod = (
    unbiased_X.transpose(0, 2, 1)
    @ tensorized_qda.tensor_inv_cov
    @ unbiased_X
)

print("X.shape:", X.shape)
print("unbiased_X.shape (k, p, n):", unbiased_X.shape)
print("inner_prod.shape (k, n, n):", inner_prod.shape)

# Sólo se utilizan los n elementos diagonales de cada clase.
print(
    f"valores calculados: {inner_prod.size} | "
    f"valores realmente usados (diagonal): "
    f"{np.diagonal(inner_prod, axis1=1, axis2=2).size}"
)


# %% [markdown]
# #### 5. Demostración de $diag(AB) = \sum_{cols} A \odot B^T$
#
# Por definición del producto matricial, $(AB)_{ii} = \sum_j A_{ij}B_{ji}$.
#
# Como $(B^T)_{ij} = B_{ji}$, resulta $(A \odot B^T)_{ij} = A_{ij}B_{ji}$, y sumando por filas (`axis=1`):
#
# $$
# \sum_j (A\odot B^T)_{ij} = \sum_j A_{ij}B_{ji} = (AB)_{ii}
# $$
#
# Es decir, `np.sum(A * B.T, axis=1)` da la diagonal de `A @ B` sin construir el producto completo. La forma equivalente `np.sum(A.T * B, axis=0).T`, también indicada en la consigna, produce el mismo resultado. Se verifica numéricamente:

# %%
# Verificación numérica de las dos formas equivalentes de la identidad.
rng = np.random.default_rng(0)

# A: (n, p)
# B: (p, n)
A = rng.normal(size=(5, 3))
B = rng.normal(size=(3, 5))

# Método directo: construye A @ B y extrae su diagonal.
diag_matmul = np.diagonal(A @ B)

# Primera forma de la consigna:
# diag(AB) = sum_cols(A * B^T)
diag_trick = np.sum(
    A * B.T,
    axis=1
)

# Segunda forma equivalente:
# diag(AB) = [sum_rows(A^T * B)]^T
diag_trick_2 = np.sum(
    A.T * B,
    axis=0
).T

print("diag(A @ B):            ", diag_matmul)
print("sum(A * B.T, axis=1):   ", diag_trick)
print("sum(A.T * B, axis=0).T: ", diag_trick_2)

print(
    "Primera forma correcta:",
    np.allclose(diag_matmul, diag_trick)
)

print(
    "Segunda forma correcta:",
    np.allclose(diag_matmul, diag_trick_2)
)


# %% [markdown]
# #### 6. Implementación de `EfficientQDA`
#
# Aplicando la identidad anterior a `FasterQDA`: en vez de armar `unbiased_X.transpose(0,2,1) @ tensor_inv_cov @ unbiased_X` (que genera el tensor `(k,n,n)`), calculamos `tensor_inv_cov @ unbiased_X` (shape `(k,p,n)`, sin cambios respecto de `FasterQDA`) y reducimos directamente sobre la dimensión de features (`axis=1`), multiplicando elemento a elemento contra `unbiased_X`. Así se obtiene `quad_terms` de shape `(k,n)` sin pasar nunca por el tensor `(k,n,n)`.

# %%
class EfficientQDA(FasterQDA):

    def predict(self, X):

        # X: (p, n)
        # tensor_means: (k, p, 1)
        # broadcasting -> (k, p, n)
        unbiased_X = X - self.tensor_means

        # (k, p, p) @ (k, p, n) -> (k, p, n)
        adjusted_X = self.tensor_inv_cov @ unbiased_X

        # diag(A @ B) sin construir el (k, n, n): producto elemento a elemento
        # y reducción sobre la dimensión de features (p)
        # (k, p, n) -> (k, n)
        quad_terms = np.sum(unbiased_X * adjusted_X, axis=1)

        # determinantes: (k,) -> (k, 1)
        log_conditionals = (
            0.5 * np.log(LA.det(self.tensor_inv_cov))[:, None]
            - 0.5 * quad_terms
        )

        # log_a_priori: (k,) -> (k, 1)
        # resultado: (k, n)
        log_posteriori = (
            self.log_a_priori[:, None]
            + log_conditionals
        )

        # elegimos la clase de mayor score para cada observación
        # (k, n) -> (n,) -> (1, n)
        return np.argmax(log_posteriori, axis=0).reshape(1, -1)


# %%
# Se instancia y ajusta EfficientQDA con los mismos datos
# utilizados para validar las implementaciones anteriores.
efficient_qda = EfficientQDA()
efficient_qda.fit(
    X_full.T,
    y_full_encoded.T
)

# Se predicen exactamente las mismas observaciones.
pred_efficient = efficient_qda.predict(
    X_full.T
)

# La optimización no debe modificar la clasificación:
# las predicciones deben coincidir con las de QDA.
print(
    "QDA == EfficientQDA:",
    np.array_equal(
        pred_qda,
        pred_efficient
    )
)


# %% [markdown]
# Se valida `EfficientQDA` de la misma forma que `FasterQDA`: sus predicciones coinciden exactamente con las de `QDA` (`True`), confirmando que evitar la matriz `(k, n, n)` no altera el resultado.

# %% [markdown]
# #### 7. Comparación de performance
#
# Se benchean las 4 variantes sobre el mismo `Benchmark` ya creado (`b`), que ya tiene resultados de `QDA` y `TensorizedQDA` de celdas anteriores.

# %%
# El objeto b ya contiene los benchmarks de QDA y TensorizedQDA
# realizados en las celdas anteriores.
# Sólo se agregan las dos variantes nuevas.
b.bench(FasterQDA)
b.bench(EfficientQDA)

# Se genera la misma tabla comparativa usando QDA como baseline.
summ = b.summary(
    baseline='QDA'
)

# Se muestran las métricas solicitadas para comparar
# entrenamiento, predicción, accuracy, speedup y memoria.
summ[[
    'train_median_ms',
    'test_median_ms',
    'mean_accuracy',
    'test_speedup',
    'test_mem_median_mb',
    'test_mem_reduction'
]]


# %%
# Resumen objetivo de esta sección (evita hardcodear valores que cambian entre ejecuciones).
best_train_4 = summ["train_median_ms"].idxmin()
best_test_4 = summ["test_median_ms"].idxmin()
lowest_test_mem_4 = summ["test_mem_median_mb"].idxmin()
highest_test_mem_4 = summ["test_mem_median_mb"].idxmax()

print("Menor tiempo mediano de entrenamiento:", best_train_4)
print("Menor tiempo mediano de predicción:", best_test_4)
print("Menor memoria mediana de predicción:", lowest_test_mem_4)
print("Mayor memoria mediana de predicción:", highest_test_mem_4)
print()
print("Speedup de predicción respecto de QDA:")
print(summ["test_speedup"].sort_values(ascending=False))


# %% [markdown]
# Con las 4 variantes se observa (valores impresos en la celda anterior, ya que dependen de la ejecución concreta):
#
# - **Tiempo de entrenamiento**: del mismo orden en las 4, como se esperaba, ya que todas reusan `_fit_params` de `QDA` sin cambios.
# - **Tiempo de predicción**: `TensorizedQDA` mejora sobre `QDA`, mientras que `FasterQDA` y `EfficientQDA` mejoran mucho más. Esto confirma que el mayor cuello de botella no era el loop sobre clases, sino el loop en Python sobre las $n$ observaciones: vectorizarlo compensa ampliamente el costo de calcular de más en `FasterQDA`.
# - **`FasterQDA` vs `EfficientQDA` en tiempo**: con $n=178$ (Wine), la matriz $(k,n,n)$ es chica (~178x178x3), así que la diferencia entre ambas frente al costo fijo de las llamadas a BLAS suele ser chica.
# - **Memoria**: acá sí se espera una diferencia sistemática. `FasterQDA` debería usar más pico de memoria que `EfficientQDA`, justamente por construir y descartar el tensor $(k,n,n)$; la celda anterior indica cuál tuvo mayor y menor memoria en esta ejecución.
#
# Los resultados se condicen parcialmente con lo esperado: la ganancia de eliminar loops de Python domina por sobre el desperdicio de `FasterQDA` en este dataset (chico), por lo que en tiempo la diferencia con `EfficientQDA` no suele ser grande; pero en memoria sí se ve la penalización teórica de `FasterQDA`. Con un $n$ mucho más grande, se esperaría que `EfficientQDA` también le saque ventaja en tiempo a `FasterQDA`, ya que el tensor $(k,n,n)$ crece cuadráticamente con $n$ mientras que el cálculo de `EfficientQDA` crece linealmente.
#

# %% [markdown]
# #### 8. Si $A=LL^T$, expresar $A^{-1}$ en términos de $L$ y explicar su utilidad en QDA
#
# Sea una matriz definida positiva $A$ con factorización de Cholesky:
#
# $$
# A=LL^T
# $$
#
# donde $L$ es triangular inferior. Invirtiendo ambos miembros:
#
# $$
# A^{-1}=(LL^T)^{-1}
# $$
#
# Utilizando:
#
# $$
# (AB)^{-1}=B^{-1}A^{-1}
# $$
#
# se obtiene:
#
# $$
# A^{-1}=(L^T)^{-1}L^{-1}
# $$
#
# Además:
#
# $$
# (L^T)^{-1}=(L^{-1})^T=L^{-T}
# $$
#
# por lo que:
#
# $$
# \boxed{A^{-1}=L^{-T}L^{-1}}
# $$
#
# Esta identidad es útil para la forma cuadrática de QDA. Si se define:
#
# $$
# u=x-\mu
# $$
#
# entonces:
#
# $$
# u^TA^{-1}u
# =
# u^TL^{-T}L^{-1}u
# $$
#
# Como:
#
# $$
# u^TL^{-T}=(L^{-1}u)^T
# $$
#
# queda:
#
# $$
# \boxed{
# u^TA^{-1}u=(L^{-1}u)^T(L^{-1}u)}
# $$
#
# o, equivalentemente:
#
# $$
# \boxed{
# u^TA^{-1}u=\|L^{-1}u\|_2^2}
# $$
#
# Esto permite evitar la construcción explícita de $A^{-1}$ para evaluar la forma cuadrática: se puede obtener $y=L^{-1}u$ resolviendo el sistema triangular:
#
# $$
# Ly=u
# $$
#
# y luego calcular:
#
# $$
# y^Ty=\|y\|_2^2
# $$
#
# Esto es precisamente la observación de la consigna de que calcular $A^{-1}b$ equivale a resolver el sistema $Ax=b$.
#
# Cholesky también permite tratar eficientemente el término del determinante. Como:
#
# $$
# \det(A)=\det(L)\det(L^T)=\det(L)^2
# $$
#
# y una matriz triangular tiene como determinante el producto de los elementos de su diagonal, el log-determinante puede obtenerse directamente a partir de la diagonal de $L$ o de $L^{-1}$.

# %%
# Verificación numérica de las identidades del punto 8.
#
# Se utilizan las mismas operaciones presentes en base/cholesky.py:
# np.cov(..., bias=True), cholesky(..., lower=True), LA.inv y
# solve_triangular.

from scipy.linalg import cholesky, solve_triangular

class_idx = 0

# Convención del proyecto:
# filas = features (p), columnas = observaciones (n).
X_project = X_full.T
y_project = y_full_encoded.T

# Selección de una clase usando el mismo flatten que base/qda.py.
X_class = X_project[:, y_project.flatten() == class_idx]

# Matriz de covarianza con bias=True, igual que en las clases base.
A = np.cov(
    X_class,
    bias=True
)

# Factorización de Cholesky:
# A = L L^T, con L triangular inferior.
L = cholesky(
    A,
    lower=True
)

# Inversa directa de A, usada sólo como referencia para verificar
# la identidad algebraica.
A_inv_direct = LA.inv(A)

# QDA_Chol1 obtiene L^{-1} mediante una inversión explícita.
L_inv = LA.inv(L)

# Identidad del punto:
# A^{-1} = L^{-T} L^{-1}
A_inv_cholesky = (
    L_inv.T
    @ L_inv
)

print("A.shape:", A.shape)
print("L.shape:", L.shape)

print(
    "A^-1 = L^-T L^-1:",
    np.allclose(
        A_inv_direct,
        A_inv_cholesky
    )
)

# ---------------------------------------------------------------
# Verificación de la forma cuadrática de QDA
# ---------------------------------------------------------------

# Se toma una observación y se centra respecto de la media de su clase.
x = X_class[:, [0]]

mu = X_class.mean(
    axis=1,
    keepdims=True
)

u = x - mu

# Forma cuadrática original:
# u^T A^{-1} u
quad_direct = (
    u.T
    @ A_inv_direct
    @ u
).item()

# En lugar de calcular L^{-1}u explícitamente, QDA_Chol2 resuelve:
# L y = u
#
# Esto produce el mismo y = L^{-1}u.
y_solve = solve_triangular(
    L,
    u,
    lower=True
)

# Por Cholesky:
# u^T A^{-1} u = ||L^{-1}u||^2 = ||y||^2
quad_cholesky = np.sum(
    y_solve**2
)

print(
    "Forma cuadrática directa == forma Cholesky:",
    np.allclose(
        quad_direct,
        quad_cholesky
    )
)

print(
    "Forma cuadrática directa:",
    quad_direct
)

print(
    "Forma cuadrática Cholesky:",
    quad_cholesky
)

# %% [markdown]
# #### 9. Diferencias entre `QDA_Chol1` y `QDA` y obtención paso a paso de las predicciones
#
# `QDA` y `QDA_Chol1` implementan la misma regla de clasificación bayesiana; la diferencia está en la forma de calcular la función log-condicional.
#
# **1. Ajuste de las probabilidades a priori.** Ambas clases heredan de `BaseBayesianClassifier`. Durante `fit`, primero se estima:
#
# $$
# \log P(G=j)
# $$
#
# para cada clase $j$.
#
# **2. Parámetros por clase en `QDA`.** La implementación original calcula explícitamente:
#
# $$
# \Sigma_j^{-1}
# $$
#
# para cada clase, además de su vector de medias $\mu_j$.
#
# La función log-condicional, omitiendo constantes comunes a todas las clases, utiliza:
#
# $$
# \frac{1}{2}\log\det(\Sigma_j^{-1})
# -
# \frac{1}{2}(x-\mu_j)^T\Sigma_j^{-1}(x-\mu_j)
# $$
#
# **3. Parámetros por clase en `QDA_Chol1`.** En lugar de invertir directamente $\Sigma_j$, calcula:
#
# $$
# \Sigma_j=L_jL_j^T
# $$
#
# y almacena:
#
# $$
# L_j^{-1}
# $$
#
# mediante `LA.inv(cholesky(...))`.
#
# **4. Centrado de la observación.** Para una observación $x$:
#
# $$
# u=x-\mu_j
# $$
#
# **5. Transformación mediante Cholesky.** El código calcula:
#
#     y = L_inv @ unbiased_x
#
# es decir:
#
# $$
# y=L_j^{-1}u
# $$
#
# **6. Forma cuadrática.** Por el punto 8:
#
# $$
# u^T\Sigma_j^{-1}u
# =
# \|L_j^{-1}u\|_2^2
# =
# y^Ty
# $$
#
# Por eso el código utiliza:
#
#     (y**2).sum()
#
# **7. Término del determinante.** Como:
#
# $$
# \Sigma_j^{-1}=L_j^{-T}L_j^{-1}
# $$
#
# se cumple:
#
# $$
# \det(\Sigma_j^{-1})=\det(L_j^{-1})^2
# $$
#
# por lo que:
#
# $$
# \frac{1}{2}\log\det(\Sigma_j^{-1})
# =
# \log\det(L_j^{-1})
# $$
#
# Y, dado que $L_j^{-1}$ es triangular:
#
# $$
# \det(L_j^{-1})
# =
# \prod_r(L_j^{-1})_{rr}
# $$
#
# Esto explica:
#
#     np.log(L_inv.diagonal().prod())
#
# **8. Score a posteriori y decisión.** `BaseBayesianClassifier` suma el log-condicional anterior con `log_a_priori` para cada clase y selecciona mediante `argmax` la clase con mayor score.
#
# Por lo tanto, `QDA_Chol1` llega a las mismas predicciones que `QDA`; cambia el procedimiento numérico utilizado para evaluar los mismos términos matemáticos.

# %%
# Comparación paso a paso entre QDA y QDA_Chol1 para una observación.
#
# Se utilizan directamente los atributos generados por las clases de
# base/qda.py y base/cholesky.py para mostrar que ambas formulaciones
# construyen el mismo score.

qda_p9 = QDA()
chol1_p9 = QDA_Chol1()

qda_p9.fit(
    X_full.T,
    y_full_encoded.T
)

chol1_p9.fit(
    X_full.T,
    y_full_encoded.T
)

# Una observación con shape (p, 1), que es el formato utilizado
# internamente por BaseBayesianClassifier._predict_one().
x = X_full.T[:, [0]]

log_cond_qda = []
log_cond_chol1 = []

for class_idx in range(
    len(qda_p9.log_a_priori)
):

    # ============================================================
    # QDA
    # ============================================================

    # QDA almacena Sigma^{-1} y la media de cada clase.
    inv_cov = (
        qda_p9.inv_covs[class_idx]
    )

    mean_qda = (
        qda_p9.means[class_idx]
    )

    unbiased_x_qda = (
        x - mean_qda
    )

    # 1/2 log(det(Sigma^{-1}))
    det_term_qda = (
        0.5
        * np.log(
            LA.det(inv_cov)
        )
    )

    # 1/2 (x-mu)^T Sigma^{-1} (x-mu)
    quad_term_qda = (
        0.5
        * (
            unbiased_x_qda.T
            @ inv_cov
            @ unbiased_x_qda
        ).item()
    )

    score_qda = (
        det_term_qda
        - quad_term_qda
    )

    log_cond_qda.append(
        score_qda
    )

    # ============================================================
    # QDA_Chol1
    # ============================================================

    # QDA_Chol1 almacena L^{-1} y la media de cada clase.
    L_inv = (
        chol1_p9.L_invs[class_idx]
    )

    mean_chol1 = (
        chol1_p9.means[class_idx]
    )

    unbiased_x_chol1 = (
        x - mean_chol1
    )

    # y = L^{-1}(x-mu)
    y = (
        L_inv
        @ unbiased_x_chol1
    )

    # 1/2 log(det(Sigma^{-1}))
    # = log(det(L^{-1}))
    # Como L^{-1} es triangular, el determinante es el producto
    # de los elementos de su diagonal.
    det_term_chol1 = np.log(
        L_inv.diagonal().prod()
    )

    # 1/2 ||L^{-1}(x-mu)||^2
    quad_term_chol1 = (
        0.5
        * np.sum(y**2)
    )

    score_chol1 = (
        det_term_chol1
        - quad_term_chol1
    )

    log_cond_chol1.append(
        score_chol1
    )

log_cond_qda = np.asarray(
    log_cond_qda
)

log_cond_chol1 = np.asarray(
    log_cond_chol1
)

# BaseBayesianClassifier agrega el logaritmo de las probabilidades
# a priori a los scores condicionales.
log_post_qda = (
    qda_p9.log_a_priori
    + log_cond_qda
)

log_post_chol1 = (
    chol1_p9.log_a_priori
    + log_cond_chol1
)

print(
    "Log-condicionales equivalentes:",
    np.allclose(
        log_cond_qda,
        log_cond_chol1
    )
)

print(
    "Scores a posteriori equivalentes:",
    np.allclose(
        log_post_qda,
        log_post_chol1
    )
)

# La decisión final del clasificador es el argmax sobre las clases.
pred_qda_p9 = np.argmax(
    log_post_qda
)

pred_chol1_p9 = np.argmax(
    log_post_chol1
)

print(
    "Predicción QDA:",
    pred_qda_p9
)

print(
    "Predicción QDA_Chol1:",
    pred_chol1_p9
)

print(
    "Misma predicción:",
    pred_qda_p9 == pred_chol1_p9
)

# %% [markdown]
# #### 10. Diferencias entre `QDA_Chol1`, `QDA_Chol2` y `QDA_Chol3`
#
# Las tres variantes parten de la misma factorización:
#
# $$
# \Sigma=LL^T
# $$
#
# pero difieren en qué almacenan durante el entrenamiento y en cómo obtienen $L^{-1}(x-\mu)$ durante la predicción.
#
# ##### `QDA_Chol1`
#
# Durante `_fit_params`:
#
# 1. calcula la covarianza de cada clase;
# 2. obtiene $L$ mediante `cholesky(..., lower=True)`;
# 3. calcula explícitamente $L^{-1}$ utilizando `numpy.linalg.inv`;
# 4. almacena $L^{-1}$.
#
# Durante la predicción realiza una multiplicación:
#
# $$
# y=L^{-1}(x-\mu)
# $$
#
# `numpy.linalg.inv` es una rutina de inversión general, por lo que no expresa de forma específica que $L$ es triangular.
#
# ##### `QDA_Chol2`
#
# Durante `_fit_params` calcula y almacena solamente $L$. No forma $L^{-1}$.
#
# Durante la predicción resuelve:
#
# $$
# Ly=x-\mu
# $$
#
# mediante `solve_triangular`. Matemáticamente:
#
# $$
# y=L^{-1}(x-\mu)
# $$
#
# pero se obtiene por resolución del sistema en vez de multiplicar por una inversa precomputada.
#
# Esta alternativa evita formar explícitamente $L^{-1}$, aunque el sistema triangular debe resolverse para cada clase y cada observación durante `predict`.
#
# ##### `QDA_Chol3`
#
# Al igual que `QDA_Chol1`, almacena $L^{-1}$ y luego predice mediante una multiplicación matricial. La diferencia es que utiliza `dtrtri`, una rutina LAPACK específica para invertir matrices triangulares.
#
# Las diferencias pueden expresarse de la siguiente manera:
#
# | Modelo | Qué almacena | Cómo se obtiene $L^{-1}(x-\mu)$ en `predict` |
# |---|---|---|
# | `QDA_Chol1` | $L^{-1}$ obtenida con inversión general | multiplicación matricial |
# | `QDA_Chol2` | $L$ | `solve_triangular` |
# | `QDA_Chol3` | $L^{-1}$ obtenida con rutina triangular | multiplicación matricial |
#
# Las tres variantes son algebraicamente equivalentes para este problema. Sus diferencias son principalmente computacionales: cantidad y tipo de trabajo realizado durante el ajuste y durante la predicción.

# %%
# Comparación numérica de las tres estrategias implementadas en
# base/cholesky.py.
#
# QDA_Chol1:
#   cholesky(...) -> LA.inv(L)
#
# QDA_Chol2:
#   cholesky(...) -> solve_triangular(L, ...)
#
# QDA_Chol3:
#   cholesky(...) -> dtrtri(L)

from scipy.linalg.lapack import dtrtri

class_idx = 0

X_project = X_full.T
y_project = y_full_encoded.T

X_class = (
    X_project[
        :,
        y_project.flatten() == class_idx
    ]
)

# Misma covarianza que utilizan las tres implementaciones.
cov = np.cov(
    X_class,
    bias=True
)

# Factorización común:
# Sigma = L L^T
L = cholesky(
    cov,
    lower=True
)

# ---------------------------------------------------------------
# QDA_Chol1
# ---------------------------------------------------------------

# Inversión general de L, exactamente como en QDA_Chol1.
L_inv_chol1 = LA.inv(L)

# ---------------------------------------------------------------
# QDA_Chol3
# ---------------------------------------------------------------

# Inversión triangular con la rutina LAPACK usada en QDA_Chol3.
L_inv_chol3 = dtrtri(
    L.copy(),
    lower=1
)[0]

print(
    "L^-1 de Chol1 y Chol3 equivalentes:",
    np.allclose(
        L_inv_chol1,
        L_inv_chol3
    )
)

# ---------------------------------------------------------------
# Comparación de la operación realizada en predict
# ---------------------------------------------------------------

x = X_class[:, [0]]

mu = X_class.mean(
    axis=1,
    keepdims=True
)

unbiased_x = (
    x - mu
)

# Chol1: multiplica por la inversa obtenida con LA.inv.
y_chol1 = (
    L_inv_chol1
    @ unbiased_x
)

# Chol2: no forma L^{-1}; resuelve L y = x-mu.
y_chol2 = solve_triangular(
    L,
    unbiased_x,
    lower=True
)

# Chol3: multiplica por la inversa obtenida con dtrtri.
y_chol3 = (
    L_inv_chol3
    @ unbiased_x
)

print(
    "Chol1 == Chol2 en L^-1(x-mu):",
    np.allclose(
        y_chol1,
        y_chol2
    )
)

print(
    "Chol1 == Chol3 en L^-1(x-mu):",
    np.allclose(
        y_chol1,
        y_chol3
    )
)

# Las tres formulaciones deben producir el mismo término cuadrático.
quad_chol1 = np.sum(
    y_chol1**2
)

quad_chol2 = np.sum(
    y_chol2**2
)

quad_chol3 = np.sum(
    y_chol3**2
)

print(
    "Términos cuadráticos equivalentes:",
    np.allclose(
        [quad_chol1, quad_chol2, quad_chol3],
        quad_chol1
    )
)

# %%
# Verificación de equivalencia entre QDA y las tres variantes Cholesky.
# Se utiliza el mismo conjunto completo para comprobar exclusivamente que
# las distintas formulaciones numéricas producen la misma clasificación.
models_chol_check = [
    QDA,
    QDA_Chol1,
    QDA_Chol2,
    QDA_Chol3
]

predictions_chol_check = {}

for model_class in models_chol_check:
    # Se crea y ajusta una instancia independiente de cada implementación.
    model = model_class()
    model.fit(X_full.T, y_full_encoded.T)

    # Se guardan las predicciones para poder compararlas con QDA.
    predictions_chol_check[model_class.__name__] = model.predict(X_full.T)

# QDA se toma únicamente como referencia de equivalencia funcional.
for name, pred in predictions_chol_check.items():
    print(
        f"QDA == {name}:",
        np.array_equal(predictions_chol_check["QDA"], pred)
    )

# %% [markdown]
# #### 11. Comparación de performance de las siete variantes implementadas hasta este punto
#
# Se comparan:
#
# 1. `QDA`
# 2. `TensorizedQDA`
# 3. `FasterQDA`
# 4. `EfficientQDA`
# 5. `QDA_Chol1`
# 6. `QDA_Chol2`
# 7. `QDA_Chol3`
#
# Además de observar el comportamiento global, interesa específicamente determinar si alguna de las tres variantes Cholesky es claramente mejor o peor que las demás.

# %%
# Benchmark conjunto de las siete variantes disponibles hasta este punto.
# Se crea un Benchmark nuevo para evitar mezclar resultados de secciones
# anteriores y se mantienen las mismas particiones entre modelos.
b_chol = Benchmark(
    X_full,
    y_full_encoded,
    n_runs=100,
    warmup=20,
    mem_runs=20,
    test_sz=0.3,
    same_splits=True
)

models_7 = [
    QDA,
    TensorizedQDA,
    FasterQDA,
    EfficientQDA,
    QDA_Chol1,
    QDA_Chol2,
    QDA_Chol3
]

# Cada modelo se evalúa con exactamente la misma configuración.
for model in models_7:
    b_chol.bench(model)

# Se mantiene QDA como baseline para interpretar speedups y memoria relativa.
summary_chol = b_chol.summary(baseline="QDA")

summary_chol[[
    "train_median_ms",
    "test_median_ms",
    "mean_accuracy",
    "train_speedup",
    "test_speedup",
    "train_mem_reduction",
    "test_mem_reduction"
]]

# %%
# Análisis específico de las tres variantes Cholesky solicitado en el punto 11.
chol_names = [
    "QDA_Chol1",
    "QDA_Chol2",
    "QDA_Chol3"
]

chol_summary = summary_chol.loc[chol_names]

# Menor tiempo significa mejor resultado para estas dos métricas.
best_chol_train = chol_summary["train_median_ms"].idxmin()
best_chol_test = chol_summary["test_median_ms"].idxmin()
worst_chol_test = chol_summary["test_median_ms"].idxmax()

print("Cholesky con menor tiempo mediano de entrenamiento:", best_chol_train)
print("Cholesky con menor tiempo mediano de predicción:", best_chol_test)
print("Cholesky con mayor tiempo mediano de predicción:", worst_chol_test)
print()
print("Detalle de las tres variantes Cholesky:")
print(chol_summary[["train_median_ms", "test_median_ms", "mean_accuracy"]])


# %% [markdown]
# **Interpretación.** Las tres variantes Cholesky calculan la misma regla de decisión, pero distribuyen el costo de manera diferente:
#
# - `QDA_Chol1` calcula $L^{-1}$ con una inversión general y luego utiliza multiplicaciones durante la predicción.
# - `QDA_Chol2` evita construir $L^{-1}$ y resuelve un sistema triangular en cada evaluación clase/observación. Esta estrategia es matemáticamente natural y evita la inversa explícita, pero en esta implementación puede trasladar un costo considerable a `predict` porque `solve_triangular` se invoca repetidamente.
# - `QDA_Chol3` también precomputa $L^{-1}$, pero lo hace con una rutina LAPACK específica para matrices triangulares, y luego predice mediante multiplicaciones.
#
# La celda anterior identifica qué variante resultó mejor y peor en la ejecución concreta. `QDA_Chol3` se utiliza como base de `TensorizedChol` porque conserva una representación mediante $L^{-1}$ que puede apilarse de forma natural y, a diferencia de `QDA_Chol1`, obtiene la inversa triangular mediante una rutina específica para esa estructura.

# %% [markdown]
# ### 4) Optimización
#
# #### 12. Implementación de `TensorizedChol`
#
# Se implementa `TensorizedChol` heredando de `QDA_Chol3`. El objetivo es eliminar el ciclo explícito sobre las $k$ clases, de manera análoga a lo realizado por `TensorizedQDA`.
#
# Luego del ajuste de `QDA_Chol3`, para cada clase $j$ se dispone de:
#
# $$
# L_j^{-1}\in\mathbb{R}^{p\times p}
# $$
#
# $$
# \mu_j\in\mathbb{R}^{p\times1}
# $$
#
# Al apilar estos elementos:
#
# $$
# \text{tensor\_L\_invs.shape}=(k,p,p)
# $$
#
# $$
# \text{tensor\_means.shape}=(k,p,1)
# $$
#
# Para una observación $x\in\mathbb{R}^{p\times1}$, el broadcasting produce:
#
# $$
# (x-\mu).shape=(k,p,1)
# $$
#
# y la multiplicación batch:
#
# $$
# (k,p,p)@(k,p,1)\rightarrow(k,p,1)
# $$
#
# calcula $L_j^{-1}(x-\mu_j)$ para todas las clases simultáneamente.
#
# Por lo tanto, `TensorizedChol` paraleliza las **$k$ clases**, pero todavía procesa individualmente las **$n$ observaciones**, ya que mantiene el método `predict` heredado de `BaseBayesianClassifier`.

# %%
class TensorizedChol(QDA_Chol3):

    def _fit_params(self, X, y):
        # Primero se reutiliza exactamente el ajuste de QDA_Chol3:
        # se calculan L^{-1} y las medias para cada clase.
        super()._fit_params(X, y)

        # Se apilan las k matrices L^{-1}, cada una de shape (p, p).
        # Resultado: (k, p, p)
        self.tensor_L_invs = np.stack(self.L_invs)

        # Se apilan las k medias, cada una de shape (p, 1).
        # Resultado: (k, p, 1)
        self.tensor_means = np.stack(self.means)

        # Para el término del determinante necesitamos:
        #   log(det(L^{-1}))
        #
        # Como L^{-1} es triangular, su determinante es el producto de su
        # diagonal. Numéricamente es preferible sumar los logaritmos:
        #   log(prod(diag)) = sum(log(diag))
        #
        # diagonal: (k, p, p) -> (k, p)
        diag_L_inv = np.diagonal(
            self.tensor_L_invs,
            axis1=1,
            axis2=2
        )

        # Un valor de log-determinante por clase: (k, p) -> (k,)
        self.log_det_L_inv = np.sum(
            np.log(diag_L_inv),
            axis=1
        )

    def _predict_log_conditionals(self, x):
        # x corresponde a UNA observación y tiene shape (p, 1).
        # tensor_means tiene shape (k, p, 1), por lo que el broadcasting
        # genera una versión centrada para cada una de las k clases.
        # Resultado: (k, p, 1)
        unbiased_x = x - self.tensor_means

        # Se calcula L^{-1}(x-mu) simultáneamente para todas las clases.
        # (k, p, p) @ (k, p, 1) -> (k, p, 1)
        transformed_x = self.tensor_L_invs @ unbiased_x

        # Por la identidad del punto 8:
        #   (x-mu)^T Sigma^{-1} (x-mu)
        #   = ||L^{-1}(x-mu)||^2
        #
        # Se suman los cuadrados sobre las dimensiones p y 1,
        # quedando un término cuadrático por clase: (k,)
        quad_terms = np.sum(
            transformed_x**2,
            axis=(1, 2)
        )

        # Se devuelve un score log-condicional por clase: (k,)
        return self.log_det_L_inv - 0.5 * quad_terms

    def _predict_one(self, x):
        # Se suma la priori de cada clase al score log-condicional
        # y se selecciona la clase con mayor valor.
        return np.argmax(
            self.log_a_priori
            + self._predict_log_conditionals(x)
        )


# %%
# Validación de TensorizedChol contra la implementación QDA de referencia.
tensorized_chol = TensorizedChol()
tensorized_chol.fit(X_full.T, y_full_encoded.T)

pred_tensorized_chol = tensorized_chol.predict(X_full.T)

print(
    "QDA == TensorizedChol:",
    np.array_equal(pred_qda, pred_tensorized_chol)
)

# Estos prints permiten comprobar directamente los shapes analizados arriba.
print("tensor_L_invs.shape:", tensorized_chol.tensor_L_invs.shape)
print("tensor_means.shape: ", tensorized_chol.tensor_means.shape)


# %% [markdown]
# #### 13. Implementación de `EfficientChol`
#
# `TensorizedChol` elimina el ciclo sobre las clases, pero continúa procesando una observación por vez. `EfficientChol` combina esa implementación con el insight de `EfficientQDA` para vectorizar también las $n$ observaciones y evitar matrices intermedias de $n\times n$.
#
# Para todas las observaciones:
#
# $$
# X\in\mathbb{R}^{p\times n}
# $$
#
# El broadcasting produce:
#
# $$
# X-\mu\in\mathbb{R}^{k\times p\times n}
# $$
#
# Luego:
#
# $$
# L^{-1}(X-\mu)
# $$
#
# se obtiene mediante:
#
# $$
# (k,p,p)@(k,p,n)\rightarrow(k,p,n)
# $$
#
# Si se define:
#
# $$
# Z=L^{-1}(X-\mu)
# $$
#
# una implementación análoga a `FasterQDA` podría formar:
#
# $$
# Z^TZ\in\mathbb{R}^{k\times n\times n}
# $$
#
# para luego quedarse sólo con su diagonal. Sin embargo, para cada observación $i$:
#
# $$
# z_i^Tz_i=\sum_{r=1}^{p}z_{ri}^2
# $$
#
# Por lo tanto, se puede obtener directamente un tensor $(k,n)$ sumando los cuadrados sobre la dimensión de features. Así se vectorizan simultáneamente clases y observaciones sin introducir la matriz $n\times n$.

# %%
class EfficientChol(TensorizedChol):

    def predict(self, X):
        # X contiene las n observaciones y tiene shape (p, n).
        # tensor_means: (k, p, 1)
        # broadcasting -> (k, p, n)
        unbiased_X = X - self.tensor_means

        # Aplicamos L^{-1} a todas las observaciones y clases de una vez.
        # tensor_L_invs: (k, p, p)
        # unbiased_X:    (k, p, n)
        # Resultado:     (k, p, n)
        transformed_X = self.tensor_L_invs @ unbiased_X

        # Para cada clase y observación necesitamos la norma al cuadrado:
        #   ||L^{-1}(x-mu)||^2
        #
        # Al sumar sobre la dimensión p (axis=1):
        #   (k, p, n) -> (k, n)
        #
        # Esta operación obtiene directamente los términos necesarios y
        # evita construir un tensor (k, n, n).
        quad_terms = np.sum(
            transformed_X**2,
            axis=1
        )

        # log_det_L_inv contiene un valor por clase: (k,).
        # [:, None] lo transforma en (k, 1) para hacer broadcasting
        # sobre las n observaciones. Resultado: (k, n).
        log_conditionals = (
            self.log_det_L_inv[:, None]
            - 0.5 * quad_terms
        )

        # Se agregan las probabilidades a priori para obtener los scores
        # a posteriori de todas las clases y observaciones: (k, n).
        log_posteriori = (
            self.log_a_priori[:, None]
            + log_conditionals
        )

        # Se selecciona la clase con mayor score para cada observación.
        # (k, n) -> (n,) -> (1, n)
        return np.argmax(
            log_posteriori,
            axis=0
        ).reshape(1, -1)


# %%
# Validación funcional de EfficientChol.
efficient_chol = EfficientChol()
efficient_chol.fit(X_full.T, y_full_encoded.T)

pred_efficient_chol = efficient_chol.predict(X_full.T)

# Debe conservar las mismas predicciones que QDA y TensorizedChol.
print(
    "QDA == EfficientChol:",
    np.array_equal(pred_qda, pred_efficient_chol)
)

print(
    "TensorizedChol == EfficientChol:",
    np.array_equal(pred_tensorized_chol, pred_efficient_chol)
)

# %% [markdown]
# La diferencia entre `TensorizedChol` y `EfficientChol` reproduce la lógica desarrollada previamente entre `TensorizedQDA` y `EfficientQDA`:
#
# - `TensorizedChol` vectoriza las clases, pero mantiene el ciclo sobre observaciones.
# - `EfficientChol` vectoriza simultáneamente clases y observaciones.
# - Además, evita formar un tensor $(k,n,n)$ y calcula directamente los $k\times n$ términos cuadráticos necesarios.
#
# Así, `EfficientChol` combina las dos ideas principales del trabajo: la reformulación mediante Cholesky y la eliminación de cálculos innecesarios mediante tensorización.

# %% [markdown]
# #### 14. Comparación de performance de las nueve variantes de QDA
#
# Se comparan las nueve variantes solicitadas:
#
# 1. `QDA`
# 2. `TensorizedQDA`
# 3. `FasterQDA`
# 4. `EfficientQDA`
# 5. `QDA_Chol1`
# 6. `QDA_Chol2`
# 7. `QDA_Chol3`
# 8. `TensorizedChol`
# 9. `EfficientChol`
#
# El benchmark final permite observar conjuntamente el efecto de las optimizaciones de entrenamiento, predicción y uso de memoria.
#

# %%
# Benchmark final de las nueve variantes requeridas por la consigna.
# Se utiliza nuevamente same_splits=True para garantizar comparabilidad.
b_final = Benchmark(
    X_full,
    y_full_encoded,
    n_runs=100,
    warmup=20,
    mem_runs=20,
    test_sz=0.3,
    same_splits=True
)

models_9 = [
    QDA,
    TensorizedQDA,
    FasterQDA,
    EfficientQDA,
    QDA_Chol1,
    QDA_Chol2,
    QDA_Chol3,
    TensorizedChol,
    EfficientChol
]

# Se ejecuta la misma metodología de benchmark para cada modelo.
for model in models_9:
    b_final.bench(model)

# QDA vuelve a utilizarse como baseline común.
summary_final = b_final.summary(baseline="QDA")

summary_final[[
    "train_median_ms",
    "test_median_ms",
    "mean_accuracy",
    "train_speedup",
    "test_speedup",
    "train_mem_reduction",
    "test_mem_reduction"
]]

# %%
# Resumen objetivo de los resultados de la ejecución final.
# Estas líneas permiten contestar "qué se observa" usando los valores
# efectivamente medidos, sin asumir de antemano un ranking fijo.
best_train_final = summary_final["train_median_ms"].idxmin()
best_test_final = summary_final["test_median_ms"].idxmin()
lowest_test_mem_final = summary_final["test_mem_median_mb"].idxmin()
highest_test_mem_final = summary_final["test_mem_median_mb"].idxmax()

print("Menor tiempo mediano de entrenamiento:", best_train_final)
print("Menor tiempo mediano de predicción:", best_test_final)
print("Menor memoria mediana durante predicción:", lowest_test_mem_final)
print("Mayor memoria mediana durante predicción:", highest_test_mem_final)
print()
print("Speedup de predicción respecto de QDA:")
print(summary_final["test_speedup"].sort_values(ascending=False))

# %% [markdown]
# **Interpretación final.** El benchmark permite separar dos tipos de optimización:
#
# 1. **Reformulación mediante Cholesky.** Las variantes `QDA_Chol` cambian la manera de tratar la covarianza y la forma cuadrática. La mejora de entrenamiento puede ser pequeña en un dataset como Wine porque $p$ es reducido y el costo fijo de las rutinas numéricas tiene un peso importante.
#
# 2. **Tensorización de la predicción.** Las variantes `TensorizedQDA` y `TensorizedChol` eliminan el ciclo explícito sobre las clases. `FasterQDA`, `EfficientQDA` y `EfficientChol` eliminan además el ciclo sobre observaciones.
#
# No todas las vectorizaciones tienen el mismo costo de memoria. `FasterQDA` forma un tensor $(k,n,n)$, por lo que el término dominante asociado a las observaciones crece cuadráticamente con $n$:
#
# $$
# O(kn^2)
# $$
#
# En cambio, `EfficientQDA` y `EfficientChol` trabajan principalmente con tensores $(k,p,n)$:
#
# $$
# O(kpn)
# $$
#
# Como la consigna plantea el escenario habitual $p\ll n$, evitar la dependencia cuadrática respecto de $n$ es especialmente relevante cuando crece la cantidad de observaciones a predecir.
#
# `EfficientChol` reúne las optimizaciones desarrolladas durante el trabajo:
#
# $$
# \boxed{
# \text{Cholesky}
# +
# \text{tensorización sobre clases}
# +
# \text{tensorización sobre observaciones}
# +
# \text{eliminación de la matriz }n\times n
# }
# $$
#
# Por lo tanto, es esperable que resulte especialmente competitivo en predicción y que escale mejor que una alternativa que materializa las interacciones $n\times n$. De todos modos, la conclusión concreta sobre qué implementación fue la más rápida o la que utilizó menos memoria debe tomarse de los valores que imprime el benchmark en el entorno de ejecución, ya que los tiempos absolutos dependen de Python, NumPy, SciPy y la biblioteca BLAS utilizada.
#
# Finalmente, todas las optimizaciones se validaron comparando sus predicciones con `QDA`. La equivalencia de las predicciones confirma que las mejoras introducidas modifican la estrategia computacional, pero no la regla matemática del clasificador.

# %% [markdown]
# ## Reproducibilidad del benchmark
#
# Se informan las versiones principales del entorno para contextualizar los tiempos obtenidos. Esto es importante porque las operaciones de álgebra lineal dependen de las implementaciones de NumPy, SciPy y de las bibliotecas BLAS/LAPACK disponibles.

# %%
# Versiones del entorno utilizadas al ejecutar el notebook.
# Se muestran al final para poder asociarlas con los resultados de benchmark.
import sys
import scipy

print("Python:", sys.version)
print("NumPy:", np.__version__)
print("SciPy:", scipy.__version__)
