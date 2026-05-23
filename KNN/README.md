# KNN manual para MNIST

Esta version separa el proyecto en tres archivos simples:

```text
KNN/knn_mnist_academic.py   -> ejecuta la practica
KNN/knn_algorithm.py        -> contiene los calculos de KNN
KNN/knn_interface.py        -> contiene la ventana de Tkinter + Matplotlib
```

La parte importante sigue siendo caja blanca:

- distancia euclidiana calculada a mano en los 784 pixeles;
- ordenamiento de distancias;
- seleccion de los `k` vecinos mas cercanos;
- voto mayoritario;
- probabilidad como `votos / k`;
- desempate por menor distancia promedio;
- matriz de confusion hecha sin metricas externas.

Se usa Keras para cargar MNIST, NumPy para arreglos numericos, Tkinter para la
ventana y Matplotlib para los graficos. No se usa `KNeighborsClassifier`.

## Ejecutar

```bash
python KNN/knn_mnist_academic.py
```

Por defecto usa:

- 5.000 imagenes para train;
- 1.000 imagenes para validacion;
- 1.000 imagenes para test.

MNIST completo tiene 70.000 imagenes, pero KNN manual compara cada imagen nueva
contra todos los ejemplos guardados. Usar todo el dataset puede ser muy pesado
porque no hay una libreria optimizada escondiendo el costo.

La seleccion de `k` esta optimizada para no recalcular distancias por cada valor:
primero busca vecinos con el `k` mas grande y luego reutiliza esos mismos vecinos
para evaluar `k=1`, `k=3`, `k=5`, etc. Esto permite comparar valores de k sin repetir
el calculo completo de distancias.

La ventana muestra caso por caso:

- una muestra de puntos de entrenamiento proyectados a 3D con PCA para mantener la ventana fluida;
- el nuevo dato como estrella negra;
- los vecinos usados por KNN con circulos;
- lineas desde el nuevo dato hacia sus vecinos;
- radio visual hasta el vecino mas lejano de los k usados;
- la imagen 28x28 consultada;
- probabilidades por voto;
- texto con distancias ordenadas;
- botones `Anterior`, `Siguiente`, `Auto` y `Matriz de confusion`.

Las distancias completas hacia todos los puntos de entrenamiento se guardan en:

```text
KNN/output/distancias_consulta_000.txt
KNN/output/distancias_consulta_001.txt
...
```

Cada vez que se abre la ventana, se borran primero los
`distancias_consulta_*.txt` anteriores para que la corrida nueva empiece limpia.

## Controles

- boton `Siguiente` o flecha derecha: avanzar.
- boton `Anterior` o flecha izquierda: retroceder.
- boton `Auto` o espacio: activar/desactivar avance automatico.
- boton `Matriz de confusion`: abrir la matriz en otra ventana.
- `Esc`: cerrar.

Por defecto no avanza solo. Si quieres que avance automaticamente:

```bash
python KNN/knn_mnist_academic.py --auto
```

## Matriz de confusion

La matriz de confusion no explica una prediccion individual. Sirve para evaluar
el comportamiento global del KNN: por ejemplo, si el modelo confunde muchos `7`
como `1`, eso aparece en la fila del `7` y la columna del `1`.

Como no es parte directa del calculo de vecinos, queda opcional:

```bash
python KNN/knn_mnist_academic.py --show-confusion
```

## Que hace paso a paso

1. Carga MNIST desde Keras.
2. Normaliza los pixeles a valores entre 0 y 1.
3. Aplana cada imagen 28x28 a un vector de 784 valores.
4. Guarda los ejemplos de entrenamiento, que en KNN son la "memoria" del modelo.
5. Prueba varios valores de `k` con validacion.
6. Para cada imagen nueva calcula su distancia hacia todos los ejemplos guardados.
7. Ordena esas distancias y toma los `k` vecinos mas cercanos.
8. Vota por clase y calcula probabilidad como `votos / k`.
9. Evalua en test con matriz de confusion.
10. Abre la ventana para ver el proceso caso por caso.

## Ejemplo mas pequeno

```bash
python KNN/knn_mnist_academic.py --train-per-class 30 --validation-per-class 10 --test-per-class 5 --cases 8
```

## Ejemplo mas grande

```bash
python KNN/knn_mnist_academic.py --train-per-class 3000 --validation-per-class 500 --test-per-class 500
```

## Solo consola, sin ventana

```bash
python KNN/knn_mnist_academic.py --no-window
```
