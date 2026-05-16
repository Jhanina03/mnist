# =============================================================================
#  RED NEURONAL MANUAL - CLASIFICACIÓN MNIST (10 DÍGITOS)
#  Implementación del ciclo de entrenamiento matemáticamente explícito
#  Autor: Paso a paso con GradientTape, Backpropagation y Optimización manual
# =============================================================================

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suprime warnings y logs C++ de TF en consola
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Suprime mensajes de oneDNN
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tensorflow import keras

from sklearn.model_selection import train_test_split

# Fijamos semillas para reproducibilidad total en ambos frameworks
# Sin esto, tf.random.shuffle produce orden diferente en cada ejecución
np.random.seed(42)
tf.random.set_seed(42)

# ===========================================================================
# PASO 1: CARGA Y PREPROCESAMIENTO DE DATOS
# ===========================================================================
# MNIST contiene 70.000 imágenes en escala de grises de dígitos escritos a mano
# Cada imagen es de 28x28 píxeles con valores entre 0 y 255.

print("=" * 60)
print("PASO 1: Cargando y preprocesando el dataset MNIST...")
print("=" * 60)

# Descargamos el dataset usando el módulo oficial de keras
(X_train_raw, y_train_raw), (X_test_raw, y_test_raw) = keras.datasets.mnist.load_data()

# --- Aplanamiento (Flatten) ---
# Las imágenes originales tienen forma (N, 28, 28).
# Una red densa (Dense/Fully Connected) espera vectores 1D.
# Transformamos cada imagen 28x28 en un vector de 784 características.
# Matemáticamente: R^(28×28) → R^784
X_all = X_train_raw.reshape(-1, 784)  # (60000, 784)
y_all = y_train_raw  # (60000,) etiquetas enteras 0-9

# --- Normalización ---
# Dividimos entre 255.0 para que cada píxel quede en el rango [0, 1].
# Esto estabiliza el gradiente y acelera la convergencia:
#   x_normalizado = x / 255.0  ∈ [0, 1]
X_all = X_all.astype("float32") / 255.0

print(f"  Forma del conjunto completo (aplanado): {X_all.shape}")
print(f"  Rango de valores de píxel: [{X_all.min():.2f}, {X_all.max():.2f}]")
print(f"  Clases únicas: {np.unique(y_all)}\n")


# ===========================================================================
# PASO 2: DIVISIÓN ESTRATIFICADA
# ===========================================================================
# Usamos train_test_split con stratify=y para garantizar que la proporción
# de cada clase (0-9) sea idéntica en train y test.
# Esto evita sesgo por desbalance de clases en la partición.

print("=" * 60)
print("PASO 2: División estratificada del dataset...")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X_all,
    y_all,
    test_size=0.2,  # 20% para test, 80% para entrenamiento
    random_state=42,  # Semilla para reproducibilidad
    stratify=y_all,  # ← CLAVE: mantiene proporciones de cada clase
)

# Verificamos la distribución por clase en ambas particiones
print(f"  Tamaño entrenamiento : {X_train.shape[0]} muestras")
print(f"  Tamaño prueba        : {X_test.shape[0]} muestras")
print("\n  Distribución de clases (las 10 clases):")
for clase in range(10):
    train_pct = np.mean(y_train == clase) * 100
    test_pct = np.mean(y_test == clase) * 100
    print(f"    Dígito {clase}: train={train_pct:.2f}%  test={test_pct:.2f}%")
print()


# ===========================================================================
# PASO 3: ARQUITECTURA DE LA RED NEURONAL
# ===========================================================================
# Definimos un modelo Secuencial con capas Dense de Keras.
# Arquitectura:
#   Entrada  : 784 neuronas  (una por píxel)
#   Oculta 1 : 128 neuronas  (activación ReLU)
#   Oculta 2 : 64  neuronas  (activación ReLU)
#   Salida   : 10  neuronas  (activación Softmax → probabilidades por clase)
#
# ReLU(z) = max(0, z)      — introduce no-linealidad, evita el problema de
#                             gradiente desvaneciente para valores positivos.
# Softmax(z_i) = e^z_i / Σ e^z_j  — convierte logits en distribución de prob.

print("=" * 60)
print("PASO 3: Definiendo la arquitectura de la red neuronal...")
print("=" * 60)

model = keras.Sequential(
    [
        # Capa de entrada: declara explícitamente la forma del vector de entrada (784 píxeles)
        # En Keras 3.x, input_shape ya NO se pasa a Dense; se usa keras.Input como primera capa.
        keras.Input(shape=(784,), name="entrada"),
        # Capa oculta 1: 784 → 128, activación ReLU
        keras.layers.Dense(128, activation="relu", name="capa_oculta_1"),
        # Capa oculta 2: 128 → 64, activación ReLU
        keras.layers.Dense(64, activation="relu", name="capa_oculta_2"),
        # Capa de salida: 64 → 10, activación Softmax
        keras.layers.Dense(10, activation="softmax", name="capa_salida"),
    ]
)

# Con keras.Input declarado, el modelo ya está completamente construido
# y los pesos ya están inicializados. No se necesita forward pass adicional.

# Imprimimos una tabla manual de la arquitectura para evitar bugs visuales de model.summary()
print("  Resumen de la Arquitectura de la Red:")
print("  ---------------------------------------------------------")
print("  Capa (Tipo)             Output Shape           Param #")
print("  ---------------------------------------------------------")
for layer in model.layers:
    nombre = layer.name
    try:
        shape = str(layer.output.shape)
    except AttributeError:
        shape = f"(None, {layer.units})" if hasattr(layer, "units") else "N/A"
    params = layer.count_params()
    print(f"  {nombre:22}  {shape:20}   {params:,}")
print("  ---------------------------------------------------------")
print(
    f"  Total de parámetros entrenables: "
    f"{sum(tf.size(w).numpy() for w in model.trainable_variables):,}"
)
print()


# ===========================================================================
# PASO 4: CICLO DE ENTRENAMIENTO PERSONALIZADO (CUSTOM TRAINING LOOP)
# ===========================================================================
# Implementamos el ciclo completo MANUALMENTE sin usar model.compile/fit.
# Esto expone cada etapa matemática del aprendizaje supervisado.

print("=" * 60)
print("PASO 4: Iniciando ciclo de entrenamiento manual...")
print("=" * 60)

# ── Hiperparámetros ──────────────────────────────────────────────────────────
EPOCHS = 15  # Número de épocas (pasadas completas sobre los datos)
BATCH_SIZE = 128  # Tamaño del mini-batch para SGD estocástico
LR = 0.01  # Learning Rate (tasa de aprendizaje) η

# ── Optimizador básico (SGD puro) ────────────────────────────────────────────
# Usamos tf.keras.optimizers.SGD solo para aplicar los gradientes.
# La fórmula de actualización que aplica internamente es:
#   w_nuevo = w_antiguo - η * ∂L/∂w
# Podríamos hacerlo con resta manual de tensores (mostrado en comentarios).
optimizer = keras.optimizers.SGD(learning_rate=LR)

# ── Función de Pérdida: Categorical Cross-Entropy ───────────────────────────
# Para clasificación multiclase usamos:
#   L = -Σ_i  y_i · log(ŷ_i)
# donde y_i es la etiqueta one-hot y ŷ_i es la probabilidad predicha (Softmax).
# Equivalentemente con índices de clase (sparse):
#   L = -log(ŷ_{clase_real})


def calcular_perdida(y_real, y_pred):
    """
    Calcula la Categorical Cross-Entropy de forma explícita.

    Matemáticamente:
        L = (1/N) * Σ_{n=1}^{N} [ -Σ_{k=0}^{9} y_{n,k} · log(ŷ_{n,k}) ]

    Usamos sparse_categorical_crossentropy porque las etiquetas son enteros,
    no vectores one-hot. TF internamente hace la conversión y el cálculo
    numéricamente estable con log-sum-exp.

    Parámetros:
        y_real : tensor de enteros (N,)   — etiquetas verdaderas
        y_pred : tensor de floats (N, 10) — probabilidades predichas (Softmax)

    Retorna:
        Escalar — pérdida promedio del batch
    """
    perdida = tf.keras.losses.sparse_categorical_crossentropy(y_real, y_pred)
    return tf.reduce_mean(perdida)  # Promediamos sobre el batch


def calcular_exactitud(y_real, y_pred):
    """
    Calcula la exactitud de clasificación.
    Toma el argmax de las probabilidades Softmax como clase predicha.
    """
    predicciones = tf.argmax(y_pred, axis=1, output_type=tf.int32)
    y_real_int = tf.cast(y_real, tf.int32)
    correctas = tf.equal(predicciones, y_real_int)
    return tf.reduce_mean(tf.cast(correctas, tf.float32))


# ── Conversión a tensores TF ─────────────────────────────────────────────────
X_train_tf = tf.constant(X_train, dtype=tf.float32)
y_train_tf = tf.constant(y_train, dtype=tf.int32)
X_test_tf = tf.constant(X_test, dtype=tf.float32)
y_test_tf = tf.constant(y_test, dtype=tf.int32)

n_muestras = X_train_tf.shape[0]
n_batches = n_muestras // BATCH_SIZE

historial_perdida = []  # Loss en entrenamiento por época
historial_exactitud = []  # Accuracy en entrenamiento por época
historial_perdida_test = []  # Loss en test por época (para detectar overfitting)
historial_exactitud_test = []  # Accuracy en test por época

# ── Ciclo Principal de Épocas ────────────────────────────────────────────────
for epoca in range(EPOCHS):
    perdidas_epoca = []
    exactitud_epoca = []

    # Permutamos los índices para aleatorizar el orden de los batches
    indices = tf.random.shuffle(tf.range(n_muestras))

    # ── Ciclo sobre Mini-Batches ─────────────────────────────────────────────
    for b in range(n_batches):
        # Seleccionamos los índices del batch actual
        idx_batch = indices[b * BATCH_SIZE : (b + 1) * BATCH_SIZE]

        X_batch = tf.gather(X_train_tf, idx_batch)  # (BATCH_SIZE, 784)
        y_batch = tf.gather(y_train_tf, idx_batch)  # (BATCH_SIZE,)

        # ================================================================
        # FORWARD PASS + CÁLCULO DE GRADIENTES con tf.GradientTape
        # ================================================================
        # tf.GradientTape graba ("tape") todas las operaciones que involucran
        # los `trainable_variables` del modelo dentro del bloque `with`.
        #
        # Matemáticamente implementa la REGLA DE LA CADENA (Chain Rule):
        #
        #   Si la pérdida L depende de los pesos W a través de varias capas:
        #     L = f(ŷ),  ŷ = Softmax(z_L),  z_L = W_L · a_{L-1} + b_L, ...
        #
        #   La derivada total se compone aplicando la regla de la cadena:
        #     ∂L/∂W_k = (∂L/∂ŷ) · (∂ŷ/∂z_L) · (∂z_L/∂a_{L-1}) · ... · (∂a_k/∂W_k)
        #
        #   GradientTape registra cada operación en un grafo computacional.
        #   Al llamar tape.gradient(L, variables), TF recorre el grafo en
        #   sentido inverso (Backpropagation) calculando automáticamente
        #   cada derivada parcial mediante la regla de la cadena.
        #
        #   Esto es equivalente al algoritmo Backprop clásico:
        #     δ_L = ∂L/∂z_L    (error en la capa de salida)
        #     δ_k = (W_{k+1}^T · δ_{k+1}) ⊙ f'(z_k)   (propaga el error)
        #     ∂L/∂W_k = δ_k · a_{k-1}^T                 (gradiente de pesos)

        with tf.GradientTape() as tape:
            # ── FORWARD PASS ──────────────────────────────────────────────
            # Propagamos el batch a través de todas las capas del modelo.
            # Para cada capa Dense: a = f(W·x + b)
            #   Capa 1: a1 = ReLU(W1·X + b1)      → (BATCH, 128)
            #   Capa 2: a2 = ReLU(W2·a1 + b2)     → (BATCH, 64)
            #   Salida: ŷ  = Softmax(W3·a2 + b3)  → (BATCH, 10)
            y_pred = model(X_batch, training=True)  # ŷ: probabilidades Softmax

            # ── CÁLCULO DE PÉRDIDA (Categorical Cross-Entropy) ───────────
            # L = -(1/N) Σ log(ŷ_{n, clase_n})
            perdida = calcular_perdida(y_batch, y_pred)

        # ================================================================
        # BACKPROPAGATION: Cálculo de Gradientes
        # ================================================================
        # tape.gradient(perdida, model.trainable_variables) recorre el grafo
        # computacional registrado por GradientTape en sentido INVERSO
        # (de la pérdida hacia las entradas) y aplica la REGLA DE LA CADENA
        # para obtener:
        #   gradientes = [∂L/∂W1, ∂L/∂b1, ∂L/∂W2, ∂L/∂b2, ∂L/∂W3, ∂L/∂b3]
        gradientes = tape.gradient(perdida, model.trainable_variables)

        # ================================================================
        # OPTIMIZACIÓN: Actualización de Pesos (Descenso de Gradiente)
        # ================================================================
        # Aplicamos la regla de actualización del Descenso de Gradiente Estocástico:
        #   w_nuevo = w_antiguo  -  η · ∂L/∂w
        #
        # Alternativa MANUAL con resta de tensores (comentada):
        # for peso, grad in zip(model.trainable_variables, gradientes):
        #     peso.assign_sub(LR * grad)   # equivalente a: w = w - η·∂L/∂w
        #
        # Usamos el optimizador SGD de TF para aplicar los gradientes:
        optimizer.apply_gradients(zip(gradientes, model.trainable_variables))

        # ── Métricas del batch ────────────────────────────────────────────
        exactitud = calcular_exactitud(y_batch, y_pred)
        perdidas_epoca.append(perdida.numpy())
        exactitud_epoca.append(exactitud.numpy())

    # ── Evaluación al final de cada época ───────────────────────────────────
    perdida_media = np.mean(perdidas_epoca)
    exactitud_media = np.mean(exactitud_epoca)

    # Evaluamos en el conjunto de prueba (sin actualizar gradientes)
    y_pred_test = model(X_test_tf, training=False)
    perdida_test = calcular_perdida(y_test_tf, y_pred_test).numpy()
    exactitud_test = calcular_exactitud(y_test_tf, y_pred_test).numpy()

    historial_perdida.append(perdida_media)
    historial_exactitud.append(exactitud_media)
    historial_perdida_test.append(perdida_test)  # ← datos REALES del test de esta época
    historial_exactitud_test.append(
        exactitud_test
    )  # ← datos REALES del test de esta época

    print(
        f"  Época {epoca + 1:02d}/{EPOCHS} | "
        f"Loss Train: {perdida_media:.4f} | Acc Train: {exactitud_media * 100:.2f}% | "
        f"Loss Test: {perdida_test:.4f}  | Acc Test: {exactitud_test * 100:.2f}%"
    )


# ===========================================================================
# EVALUACIÓN FINAL
# ===========================================================================
print("\n" + "=" * 60)
print("EVALUACIÓN FINAL DEL MODELO")
print("=" * 60)

y_pred_final = model.predict(X_test_tf)
exactitud_final = calcular_exactitud(y_test_tf, y_pred_final).numpy()
perdida_final = calcular_perdida(y_test_tf, y_pred_final).numpy()

print(f"  Loss final en Test  : {perdida_final:.4f}")
print(f"  Exactitud en Test   : {exactitud_final * 100:.2f}%\n")

# Reporte por clase: RECALL por dígito
# Recall_k = TP_k / (TP_k + FN_k)
#          = (ejemplos de clase k clasificados correctamente) / (total ejemplos de clase k)
# Nota: esto NO es Precisión. Precisión = TP_k / (TP_k + FP_k) requeriría
#       contar cuántos de los predichos como k realmente lo eran.
print("  Recall por clase (dígito):")
predicciones_finales = tf.argmax(y_pred_final, axis=1).numpy()
for clase in range(10):
    mask = y_test == clase
    if mask.sum() > 0:
        recall_clase = np.mean(predicciones_finales[mask] == clase)
        print(f"    Dígito {clase}: {recall_clase * 100:.2f}%  ({mask.sum()} muestras)")

print("\n" + "=" * 60)
print("  Entrenamiento manual completado.")
print("  Todos los parámetros fueron actualizados con GradientTape + SGD.")
print("=" * 60)

# ===========================================================================
# VISUALIZACIÓN DEL HISTORIAL DE ENTRENAMIENTO
# ===========================================================================
# Imprimimos el historial completo de Loss y Accuracy por época.
# Un historial en descenso en Loss y ascenso en Accuracy confirma
# que el gradiente descendente está convergiendo correctamente.
print("\n" + "=" * 60)
print("  HISTORIAL DE ENTRENAMIENTO (Loss Train por época):")
print("=" * 60)
for i, (loss, acc) in enumerate(zip(historial_perdida, historial_exactitud), 1):
    barra = "#" * int(acc * 30)  # Barra visual proporcional a la exactitud
    print(f"  Época {i:02d}: Loss={loss:.4f}  Acc={acc * 100:.2f}%  |{barra}")

print(
    f"\n  Mejora total en Loss : {historial_perdida[0]:.4f} -> {historial_perdida[-1]:.4f}"
)
print(
    f"  Mejora total en Acc  : {historial_exactitud[0] * 100:.2f}% -> {historial_exactitud[-1] * 100:.2f}%"
)
print("=" * 60)


# ===========================================================================
# PASO 5: VISUALIZACIÓN EN TERMINAL CON ASCII ART Y EXPORTACIÓN DE IMÁGENES
# ===========================================================================
print("\n" + "=" * 60)
print("PASO 5: Generando visualizaciones...")
print("=" * 60)

# ---------------------------------------------------------------------------
# PARTE A: VISUALIZACIÓN EN TERMINAL (TEXTO / ASCII)
# ---------------------------------------------------------------------------
print("\n--- Matriz de Confusión (Terminal) ---")
num_clases = 10
matriz_conf = np.zeros((num_clases, num_clases), dtype=int)
for real, pred in zip(y_test, predicciones_finales):
    matriz_conf[real][pred] += 1

header = "    " + "".join([f"{i:5d}" for i in range(num_clases)])
print(header)
print("   -" + "-----" * num_clases)

for i in range(num_clases):
    fila_str = "".join([f"{matriz_conf[i, j]:5d}" for j in range(num_clases)])
    print(f"{i} | {fila_str}")

# ---------------------------------------------------------------------------
# PARTE A: MÉTRICAS FINALES Y RESUMEN (LOGS EN TERMINAL)
# ---------------------------------------------------------------------------
# Probabilidad de que salga bien: exactitud_final
# Probabilidad de error testeo vs entrenamiento
error_testeo = (1.0 - exactitud_final) * 100
error_entrenamiento = (1.0 - historial_exactitud[-1]) * 100

print("\n" + "=" * 60)
print("  RESUMEN DE MÉTRICAS FINALES (EVALUACIÓN Y OVERFITTING)")
print("=" * 60)
print(f"  Probabilidad de éxito (Exactitud Testeo): {exactitud_final * 100:.2f}%")
print(f"  Probabilidad de error (Testeo):           {error_testeo:.2f}%")
print(f"  Probabilidad de error (Entrenamiento):    {error_entrenamiento:.2f}%")
print(
    f"  Diferencia de error (Brecha Test-Train):  {abs(error_testeo - error_entrenamiento):.2f}%"
)
print(f"  Pérdida (Loss) Testeo:                    {perdida_final:.4f}")
print(f"  Pérdida (Loss) Entrenamiento:             {historial_perdida[-1]:.4f}")

if abs(error_testeo - error_entrenamiento) < 3.0:
    print(
        "\n  -> Conclusión: La red generaliza bien, NO hay overfitting significativo."
    )
else:
    print(
        "\n  -> Conclusión: Hay una brecha alta entre train y test (Riesgo de Overfitting)."
    )

print("=" * 60)

# ---------------------------------------------------------------------------
# PARTE B: GENERACIÓN Y EXPORTACIÓN DE IMÁGENES (.PNG) EXTERNAS
# ---------------------------------------------------------------------------
print("\n--- Generando y guardando imágenes PNG externamente ---")
epocas_eje = list(range(1, EPOCHS + 1))

# FIGURA 1: CURVAS DE APRENDIZAJE
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
fig1.suptitle("Curvas de Aprendizaje", fontsize=12, fontweight="bold")
ax1.plot(epocas_eje, historial_perdida, "o-", color="#E74C3C", label="Loss Train")
ax1.plot(epocas_eje, historial_perdida_test, "s--", color="#E67E22", label="Loss Test")
ax1.set_title("Pérdida")
ax1.set_xlabel("Época")
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(
    epocas_eje,
    [a * 100 for a in historial_exactitud],
    "o-",
    color="#2ECC71",
    label="Acc Train",
)
ax2.plot(
    epocas_eje,
    [a * 100 for a in historial_exactitud_test],
    "s--",
    color="#27AE60",
    label="Acc Test",
)
ax2.set_title("Exactitud (%)")
ax2.set_xlabel("Época")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
fig1.savefig("curvas_aprendizaje.png", dpi=150, bbox_inches="tight")
print("Figura 1 guardada: curvas_aprendizaje.png")
plt.close(fig1)

# FIGURA 2: MATRIZ DE CONFUSIÓN
fig2, ax3 = plt.subplots(figsize=(7, 6))
im = ax3.imshow(matriz_conf, interpolation="nearest", cmap="Blues")
fig2.colorbar(im, ax=ax3)

ax3.set_xticks(range(num_clases))
ax3.set_yticks(range(num_clases))
ax3.set_xlabel("Dígito PREDICHO", fontsize=10)
ax3.set_ylabel("Dígito REAL", fontsize=10)
ax3.set_title("Matriz de Confusión", fontsize=12, fontweight="bold")

umbrald = matriz_conf.max() / 2.0
for i in range(num_clases):
    for j in range(num_clases):
        color = "white" if matriz_conf[i, j] > umbrald else "black"
        ax3.text(
            j,
            i,
            str(matriz_conf[i, j]),
            ha="center",
            va="center",
            color=color,
            fontsize=8,
        )

plt.tight_layout()
fig2.savefig("matriz_confusion.png", dpi=150, bbox_inches="tight")
print("Figura 2 guardada: matriz_confusion.png")
plt.close(fig2)

# FIGURA 3: PREDICCIONES VISUALES
fig3 = plt.figure(figsize=(12, 4))
fig3.suptitle("Predicciones de la Red", fontsize=12, fontweight="bold")

n_mostrar_imgs = 10
np.random.seed(99)
indices_muestra_imgs = np.random.choice(len(X_test), n_mostrar_imgs, replace=False)

for i, idx in enumerate(indices_muestra_imgs):
    ax = fig3.add_subplot(2, 5, i + 1)
    imagen_plt = X_test[idx].reshape(28, 28)
    ax.imshow(imagen_plt, cmap="gray")
    ax.axis("off")

    etiqueta_real_plt = y_test[idx]
    etiqueta_pred_plt = predicciones_finales[idx]
    es_correcto_plt = etiqueta_real_plt == etiqueta_pred_plt
    color_titulo = "#27AE60" if es_correcto_plt else "#E74C3C"
    simbolo_plt = "✓" if es_correcto_plt else "✗"
    ax.set_title(
        f"R:{etiqueta_real_plt} P:{etiqueta_pred_plt} {simbolo_plt}",
        color=color_titulo,
        fontsize=8,
        fontweight="bold",
    )

plt.tight_layout()
fig3.savefig("predicciones_visuales.png", dpi=150, bbox_inches="tight")
print("Figura 3 guardada: predicciones_visuales.png")
plt.close(fig3)

print("\n" + "=" * 60)
print("  Proceso completado.")
print("=" * 60)
