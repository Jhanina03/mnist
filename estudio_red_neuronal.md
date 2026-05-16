# 🧠 Guía de Estudio: Red Neuronal MNIST desde Cero

¡Hola! Soy tu profesor de Inteligencia Artificial. He revisado exhaustivamente el código `red_neuronal_mnist.py`. Te confirmo que **técnicamente está impecable**. La lógica matemática, el uso de tensores, la propagación hacia atrás (Backpropagation) y el control de *overfitting* están implementados con excelentes prácticas pedagógicas.

Como me mencionaste que necesitas entender todo esto a profundidad, he preparado esta guía de estudio. Léela con calma.

---

## 1. El Problema a Resolver (¿Qué hace este programa?)
Tenemos un conjunto de datos muy famoso llamado **MNIST**. Contiene miles de imágenes en blanco y negro de números escritos a mano (del 0 al 9).
El objetivo es **enseñarle a la computadora a ver una imagen y decirnos qué número es**. Esto se llama **Clasificación Multiclase** (porque hay 10 clases posibles, los números del 0 al 9).

Para lograrlo, programamos una **Red Neuronal Artificial**, que es un modelo matemático inspirado en el cerebro humano.

---

## 2. Explicación Paso a Paso del Código

### PASO 1: Preparación de los Datos (Preprocesamiento)
La computadora no "ve" imágenes, ve matrices de números. Cada imagen de MNIST mide 28 píxeles de ancho por 28 píxeles de alto.
* **Aplanamiento (Flatten):** La red no acepta la imagen como un cuadrito de 28x28. Así que tomamos todas las filas de la imagen y las ponemos en una sola línea larga. Matemáticamente: `28 * 28 = 784`. Cada imagen se convierte en un vector de 784 numeritos.
* **Normalización:** Los colores de los píxeles van del 0 (negro) al 255 (blanco). Para que a la red no le cueste procesar números tan grandes, dividimos todo para `255.0`. Así, todos los valores quedan entre `0` y `1`. ¡A las redes neuronales les encantan los números pequeños!

### PASO 2: División de los Datos (Train y Test)
Tú no puedes enseñarle a la red con todos los datos y luego hacerle el examen con los mismos datos que ya memorizó. Eso sería trampa.
* **80% Entrenamiento (Train):** Usamos 48,000 imágenes para que la red "estudie" y ajuste sus parámetros.
* **20% Testeo (Test):** Guardamos 12,000 imágenes bajo llave. Solo las usaremos al final para "tomarle el examen" y ver si realmente aprendió a generalizar o si solo se memorizó las respuestas.

### PASO 3: La Arquitectura de la Red Neuronal
Nuestra red es tipo "Secuencial", las capas van una tras otra como una línea de ensamblaje.
1. **Capa de Entrada:** 784 neuronas. Entra un píxel por neurona.
2. **Capa Oculta 1:** 128 neuronas. Aquí la red extrae patrones básicos (ej. líneas curvas, rectas). Usa una función de activación llamada **ReLU** (Rectified Linear Unit). *ReLU simplemente convierte cualquier número negativo en 0. Esto ayuda a que la red aprenda más rápido.*
3. **Capa Oculta 2:** 64 neuronas. Extrae patrones más complejos (ej. círculos, formas de números). También usa **ReLU**.
4. **Capa de Salida:** 10 neuronas (una para cada número del 0 al 9). Usa la función **Softmax**. *Softmax convierte los números locos que saca la red en **probabilidades porcentuales**. Por ejemplo: "Estoy 90% segura de que es un 8 y 10% segura de que es un 3".*

### PASO 4: El Corazón del Aprendizaje (GradientTape y Backpropagation)
Este es el proceso iterativo. Hacemos 15 pasadas completas por todos los datos de estudio (15 Épocas).
Por cada lote de imágenes:
1. **Forward Pass (Hacia adelante):** La red lanza una predicción ciega.
2. **Cálculo de Pérdida (Loss):** Se compara lo que dijo la red con la respuesta correcta usando una fórmula llamada *Categorical Cross-Entropy*. Si la red dijo "Es un 2" y realmente era un "7", el "Loss" será muy alto.
3. **GradientTape y Gradientes:** Aquí ocurre la magia del cálculo diferencial. TensorFlow calcula las "Derivadas Parciales" (Gradientes) para descubrir *qué pesos matemáticos fallaron y causaron el error*.
4. **SGD (Descenso de Gradiente):** El optimizador ajusta ligeramente las conexiones de la red (los pesos) en la dirección correcta para que la próxima vez se equivoque menos.

### PASO 5: Resultados y Overfitting
Al final, le pasamos las 12,000 imágenes de prueba que la red nunca había visto y medimos dos cosas:
* **Exactitud:** Qué porcentaje adivinó correctamente. ¡Logramos más de un 93%!
* **Overfitting (Sobreajuste):** Ocurre cuando un estudiante se memoriza el libro pero se aplaza en el examen. Si la Exactitud de Entrenamiento fuera 99% pero la de Testeo fuera 60%, habría Overfitting. Como en nuestro caso ambas rondan el 93%, decimos que **el modelo generalizó perfectamente**.

---

## 3. Glosario Rápido para Exponer
Si el profesor te pregunta en tu sustentación:
* **¿Qué función de activación usaron?** *"Usamos ReLU para las capas ocultas por su eficiencia, y Softmax en la salida para obtener probabilidades."*
* **¿Cómo actualizaron los pesos?** *"Con Descenso de Gradiente Estocástico (SGD) guiado por Backpropagation usando tf.GradientTape."*
* **¿Cómo validaron que no se memorizó los datos?** *"Separamos el dataset con train_test_split. Al final comparamos el error de entrenamiento vs. testeo y confirmamos que la brecha es mínima (menos del 1%), descartando un overfitting."*
