# Red neuronal manual - clasificación MNIST (10 dígitos)
# Ciclo de entrenamiento con GradientTape, backpropagation y SGD

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tensorflow import keras

from sklearn.model_selection import train_test_split

# Semillas para reproducibilidad
np.random.seed(42)
tf.random.set_seed(42)

# ===========================================================================
# PASO 1: CARGA Y PREPROCESAMIENTO
# ===========================================================================
print("=" * 60)
print("PASO 1: Cargando y preprocesando el dataset MNIST...")
print("=" * 60)

(X_train_raw, y_train_raw), (X_test_raw, y_test_raw) = keras.datasets.mnist.load_data()

# Aplanamos 28x28 → 784 y normalizamos a [0, 1]
X_all = X_train_raw.reshape(-1, 784)
y_all = y_train_raw
X_all = X_all.astype("float32") / 255.0

print(f"  Forma del conjunto completo (aplanado): {X_all.shape}")
print(f"  Rango de valores de píxel: [{X_all.min():.2f}, {X_all.max():.2f}]")
print(f"  Clases únicas: {np.unique(y_all)}\n")


# ===========================================================================
# PASO 2: DIVISIÓN ESTRATIFICADA
# ===========================================================================
print("=" * 60)
print("PASO 2: División estratificada del dataset...")
print("=" * 60)

# 80% train / 20% test, manteniendo proporciones por clase
X_train, X_test, y_train, y_test = train_test_split(
    X_all,
    y_all,
    test_size=0.2,
    random_state=42,
    stratify=y_all,
)

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
# 784 entrada → 128 ReLU → 64 ReLU → 10 Softmax
print("=" * 60)
print("PASO 3: Definiendo la arquitectura de la red neuronal...")
print("=" * 60)

model = keras.Sequential(
    [
        keras.Input(shape=(784,), name="entrada"),
        keras.layers.Dense(128, activation="relu", name="capa_oculta_1"),
        keras.layers.Dense(64, activation="relu", name="capa_oculta_2"),
        keras.layers.Dense(10, activation="softmax", name="capa_salida"),
    ]
)

# Resumen manual de capas (evita bugs visuales de model.summary())
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
# PASO 4: CICLO DE ENTRENAMIENTO MANUAL
# ===========================================================================
print("=" * 60)
print("PASO 4: Iniciando ciclo de entrenamiento manual...")
print("=" * 60)

# Hiperparámetros
EPOCHS = 15
BATCH_SIZE = 128
LR = 0.01

# Optimizador SGD: w = w - lr * grad
optimizer = keras.optimizers.SGD(learning_rate=LR)


def calcular_perdida(y_real, y_pred):
    """Sparse categorical cross-entropy promediada sobre el batch."""
    perdida = tf.keras.losses.sparse_categorical_crossentropy(y_real, y_pred)
    return tf.reduce_mean(perdida)


def calcular_exactitud(y_real, y_pred):
    """Exactitud: fracción de predicciones correctas."""
    predicciones = tf.argmax(y_pred, axis=1, output_type=tf.int32)
    y_real_int = tf.cast(y_real, tf.int32)
    correctas = tf.equal(predicciones, y_real_int)
    return tf.reduce_mean(tf.cast(correctas, tf.float32))


# Conversión a tensores TF
X_train_tf = tf.constant(X_train, dtype=tf.float32)
y_train_tf = tf.constant(y_train, dtype=tf.int32)
X_test_tf = tf.constant(X_test, dtype=tf.float32)
y_test_tf = tf.constant(y_test, dtype=tf.int32)

n_muestras = X_train_tf.shape[0]
n_batches = n_muestras // BATCH_SIZE

historial_perdida = []
historial_exactitud = []
historial_perdida_test = []
historial_exactitud_test = []

# Ciclo principal por épocas
for epoca in range(EPOCHS):
    perdidas_epoca = []
    exactitud_epoca = []

    # Mezcla aleatoria de índices en cada época
    indices = tf.random.shuffle(tf.range(n_muestras))

    for b in range(n_batches):
        idx_batch = indices[b * BATCH_SIZE : (b + 1) * BATCH_SIZE]
        X_batch = tf.gather(X_train_tf, idx_batch)
        y_batch = tf.gather(y_train_tf, idx_batch)

        # Forward pass + cálculo de gradientes
        with tf.GradientTape() as tape:
            y_pred = model(X_batch, training=True)
            perdida = calcular_perdida(y_batch, y_pred)

        # Backpropagation y actualización de pesos
        gradientes = tape.gradient(perdida, model.trainable_variables)
        optimizer.apply_gradients(zip(gradientes, model.trainable_variables))

        exactitud = calcular_exactitud(y_batch, y_pred)
        perdidas_epoca.append(perdida.numpy())
        exactitud_epoca.append(exactitud.numpy())

    # Métricas al cierre de la época
    perdida_media = np.mean(perdidas_epoca)
    exactitud_media = np.mean(exactitud_epoca)

    y_pred_test = model(X_test_tf, training=False)
    perdida_test = calcular_perdida(y_test_tf, y_pred_test).numpy()
    exactitud_test = calcular_exactitud(y_test_tf, y_pred_test).numpy()

    historial_perdida.append(perdida_media)
    historial_exactitud.append(exactitud_media)
    historial_perdida_test.append(perdida_test)
    historial_exactitud_test.append(exactitud_test)

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

# Recall por dígito: ejemplos correctos / total de esa clase
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
# HISTORIAL DE ENTRENAMIENTO
# ===========================================================================
print("\n" + "=" * 60)
print("  HISTORIAL DE ENTRENAMIENTO (Loss Train por época):")
print("=" * 60)
for i, (loss, acc) in enumerate(zip(historial_perdida, historial_exactitud), 1):
    barra = "#" * int(acc * 30)
    print(f"  Época {i:02d}: Loss={loss:.4f}  Acc={acc * 100:.2f}%  |{barra}")

print(
    f"\n  Mejora total en Loss : {historial_perdida[0]:.4f} -> {historial_perdida[-1]:.4f}"
)
print(
    f"  Mejora total en Acc  : {historial_exactitud[0] * 100:.2f}% -> {historial_exactitud[-1] * 100:.2f}%"
)
print("=" * 60)


# ===========================================================================
# PASO 5: VISUALIZACIONES
# ===========================================================================
print("\n" + "=" * 60)
print("PASO 5: Generando visualizaciones...")
print("=" * 60)

# Matriz de confusión en terminal
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

# Resumen de métricas y detección de overfitting
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

# Exportación de figuras PNG
print("\n--- Generando y guardando imágenes PNG externamente ---")
epocas_eje = list(range(1, EPOCHS + 1))

# Figura 1: curvas de aprendizaje (loss y accuracy por época)
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

# Figura 2: matriz de confusión como heatmap
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

# Figura 3: muestra 10 imágenes con etiqueta real vs predicha
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
