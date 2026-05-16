import sys

filepath = r'd:\UTA\7mo Semestre\IA\mnist\red_neuronal_mnist.py'
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

viz_ascii_start = content.find('--- Visualización ASCII')
if viz_ascii_start != -1:
    # go back to the print statement
    viz_ascii_start = content.rfind('print', 0, viz_ascii_start)
    
    content = content[:viz_ascii_start]
    
    new_end = """# ---------------------------------------------------------------------------
# PARTE A: MÉTRICAS FINALES Y RESUMEN (LOGS EN TERMINAL)
# ---------------------------------------------------------------------------
# Probabilidad de que salga bien: exactitud_final
# Probabilidad de error testeo vs entrenamiento
error_testeo = (1.0 - exactitud_final) * 100
error_entrenamiento = (1.0 - historial_exactitud[-1]) * 100

print("\\n" + "=" * 60)
print("  RESUMEN DE MÉTRICAS FINALES (EVALUACIÓN Y OVERFITTING)")
print("=" * 60)
print(f"  Probabilidad de éxito (Exactitud Testeo): {exactitud_final * 100:.2f}%")
print(f"  Probabilidad de error (Testeo):           {error_testeo:.2f}%")
print(f"  Probabilidad de error (Entrenamiento):    {error_entrenamiento:.2f}%")
print(f"  Diferencia de error (Brecha Test-Train):  {abs(error_testeo - error_entrenamiento):.2f}%")
print(f"  Pérdida (Loss) Testeo:                    {perdida_final:.4f}")
print(f"  Pérdida (Loss) Entrenamiento:             {historial_perdida[-1]:.4f}")

if abs(error_testeo - error_entrenamiento) < 3.0:
    print("\\n  -> Conclusión: La red generaliza súper bien, NO hay overfitting significativo.")
else:
    print("\\n  -> Conclusión: Hay una brecha alta entre train y test (Riesgo de Overfitting).")

print("=" * 60)

# ---------------------------------------------------------------------------
# PARTE B: GENERACIÓN Y EXPORTACIÓN DE IMÁGENES (.PNG) EXTERNAS
# ---------------------------------------------------------------------------
print("\\n--- Generando y guardando imágenes PNG externamente ---")
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

ax2.plot(epocas_eje, [a * 100 for a in historial_exactitud], "o-", color="#2ECC71", label="Acc Train")
ax2.plot(epocas_eje, [a * 100 for a in historial_exactitud_test], "s--", color="#27AE60", label="Acc Test")
ax2.set_title("Exactitud (%)")
ax2.set_xlabel("Época")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
fig1.savefig("curvas_aprendizaje.png", dpi=150, bbox_inches="tight")
print("  ✔ Figura 1 guardada: curvas_aprendizaje.png")
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
        ax3.text(j, i, str(matriz_conf[i, j]), ha="center", va="center", color=color, fontsize=8)

plt.tight_layout()
fig2.savefig("matriz_confusion.png", dpi=150, bbox_inches="tight")
print("  ✔ Figura 2 guardada: matriz_confusion.png")
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
    ax.set_title(f"R:{etiqueta_real_plt} P:{etiqueta_pred_plt} {simbolo_plt}", color=color_titulo, fontsize=8, fontweight="bold")

plt.tight_layout()
fig3.savefig("predicciones_visuales.png", dpi=150, bbox_inches="tight")
print("  ✔ Figura 3 guardada: predicciones_visuales.png")
plt.close(fig3)

print("\\n" + "=" * 60)
print("  Proceso completado con éxito.")
print("=" * 60)
"""
    content += new_end
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    print('File modified successfully.')
else:
    print('Could not find visualization block.')
