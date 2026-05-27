# KNN manual para MNIST

Esta versión separa el proyecto en tres archivos simples:

```text
KNN/knn_mnist_academic.py   -> ejecuta la práctica principal
KNN/knn_algorithm.py        -> contiene los cálculos de KNN
KNN/knn_interface.py        -> contiene la ventana minimalista de visualización
```

La parte importante sigue siendo la filosofía de **caja blanca**:

- distancia métrica (Euclidiana, Manhattan o Coseno) calculada "a mano" en los 784 píxeles.
- ordenamiento estricto de distancias para encontrar los `k` vecinos más cercanos.
- voto mayoritario.
- matriz de confusión creada y analizada de forma directa desde los resultados de los tests.
- **Sin usar** librerías ocultas tipo `KNeighborsClassifier`.

## Ejecutar

Para correr la prueba completa (usando por defecto **todo** el dataset de MNIST):

```bash
python KNN/knn_mnist_academic.py
```

Por defecto procesa:
- **50,000** imágenes para entrenamiento (train).
- **10,000** imágenes para validación de K.
- **10,000** imágenes para testeo de exactitud y visualización.

**OJO**: Debido a que se usa el dataset de entrenamiento completo (50k imágenes), cada vez que calcules el KNN de un punto nuevo, internamente medirá las distancias en vivo contra los 50,000 datos, por lo que podría tomar entre 1 y 2 segundos calcular la predicción en tiempo real. 

### Ejecución más rápida (Limitando el tamaño)

Si solo quieres ver cómo funciona la interfaz rápidamente o probar la arquitectura, puedes limitar explícitamente el tamaño de los datos:

```bash
python KNN/knn_mnist_academic.py --train-size 5000 --val-size 1000 --test-size 50
```

*Nota: Un número en `-1` (que es el valor predeterminado) le indica al algoritmo que debe usar todos los datos disponibles de MNIST.*

### Valores de K

El sistema compara múltiples valores de `K` sobre los datos de validación para escoger automáticamente el que tenga mejor rendimiento antes de presentarte la interfaz:

```bash
python KNN/knn_mnist_academic.py --k-values 1,3,5,7,9,11
```

## La Interfaz

La interfaz ha sido reescrita para estar **simplificada al máximo**, sin temas oscuros, ni distracciones:

- **Mapa 2D (Izquierda)**: Muestra una reducción por PCA (en 2D) de los datos de entrenamiento en el fondo. El punto nuevo a testear aparece como una gran estrella roja conectada por líneas a los vecinos exactos que se seleccionaron usando KNN en sus 784 dimensiones reales. Se traza un círculo punteado indicando la zona de influencia.
- **Consultas (Centro)**: Se muestra la imagen real 28x28 del número que está siendo predicho y debajo una gráfica de barras que ilustra cómo se distribuyeron los votos de sus `K` vecinos.
- **Caja Blanca de Distancias (Derecha)**: Un cuadro de texto listando directamente desde la memoria los vecinos encontrados, la clase a la que pertenecen y su distancia métrica para demostrar en tiempo real que el cálculo no es mágico.

### Controles Interactivos
Puedes utilizar los botones de **◀ Anterior** y **Siguiente ▶** para procesar la siguiente imagen en tiempo real.
Adicionalmente, las barras desplegables de `Métrica` y `K` recalcularán **en ese mismo momento** los vecinos usando la nueva información.

## Matriz de Confusión

Dado que usamos las 10,000 imágenes de test, ahora la matriz arrojará estadísticas sobre un número colosal de combinaciones. Haciendo clic en el botón de "Matriz de Confusión" en la interfaz verás un cuadro de calor mostrando si, por ejemplo, el número `7` tiende a confundirse más con el `1`.
