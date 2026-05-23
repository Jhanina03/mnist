"""
Logica del KNN para MNIST.

Aqui va lo matematico: carga de datos, distancias, vecinos, votos,
probabilidades, seleccion de k y matriz de confusion.
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
from tensorflow import keras


class MNISTData:
    """Carga MNIST y prepara muestras balanceadas para KNN."""

    def __init__(self, seed=42):
        self.rng = np.random.default_rng(seed)

    def load(self, train_per_class, validation_per_class, test_per_class):
        """Devuelve datos planos para calcular y datos 28x28 para mostrar."""
        print_header("PASO 1: cargar MNIST")
        (x_train_raw, y_train_raw), (x_test_raw, y_test_raw) = keras.datasets.mnist.load_data()

        x_train_flat = self._normalize_and_flatten(x_train_raw)
        x_test_flat = self._normalize_and_flatten(x_test_raw)

        train_idx, validation_idx = self._train_validation_indices(
            y_train_raw,
            train_per_class,
            validation_per_class,
        )
        test_idx = self._balanced_indices(y_test_raw, test_per_class)

        data = {
            "x_train": x_train_flat[train_idx],
            "y_train": y_train_raw[train_idx].astype(int),
            "x_validation": x_train_flat[validation_idx],
            "y_validation": y_train_raw[validation_idx].astype(int),
            "x_test": x_test_flat[test_idx],
            "y_test": y_test_raw[test_idx].astype(int),
            "raw_test": x_test_raw[test_idx],
        }

        print(f"  Train      : {data['x_train'].shape[0]} imagenes")
        print(f"  Validacion : {data['x_validation'].shape[0]} imagenes")
        print(f"  Test       : {data['x_test'].shape[0]} imagenes")
        print(f"  Cada imagen queda como vector de {data['x_train'].shape[1]} pixeles")
        return data

    def _normalize_and_flatten(self, images):
        """Pasa pixeles de 0-255 a 0-1 y aplana cada imagen 28x28."""
        return images.reshape(images.shape[0], -1).astype(np.float32) / 255.0

    def _balanced_indices(self, labels, per_class):
        """Toma la misma cantidad de ejemplos por digito."""
        selected = []
        for digit in range(10):
            digit_indices = np.where(labels == digit)[0]
            self.rng.shuffle(digit_indices)
            selected.extend(digit_indices[:per_class])

        selected = np.array(selected)
        self.rng.shuffle(selected)
        return selected

    def _train_validation_indices(self, labels, train_per_class, validation_per_class):
        """Separa train y validacion sin repetir imagenes."""
        train_indices = []
        validation_indices = []

        for digit in range(10):
            digit_indices = np.where(labels == digit)[0]
            self.rng.shuffle(digit_indices)
            cut = train_per_class
            end = train_per_class + validation_per_class
            train_indices.extend(digit_indices[:cut])
            validation_indices.extend(digit_indices[cut:end])

        train_indices = np.array(train_indices)
        validation_indices = np.array(validation_indices)
        self.rng.shuffle(train_indices)
        self.rng.shuffle(validation_indices)
        return train_indices, validation_indices


class WhiteBoxKNN:
    """KNN hecho a mano: distancia, vecinos, votos y probabilidad."""

    def __init__(self, k=5):
        self.k = k
        self.x_train = None
        self.y_train = None
        self.train_squared_norms = None

    def fit(self, x_train, y_train):
        """KNN no ajusta pesos; solo guarda los ejemplos conocidos."""
        self.x_train = np.asarray(x_train, dtype=np.float32)
        self.y_train = np.asarray(y_train, dtype=int)
        self.train_squared_norms = np.sum(self.x_train * self.x_train, axis=1)
        print_header("PASO 2: guardar ejemplos para KNN")
        print(f"  Ejemplos guardados: {len(self.x_train)}")
        print(f"  k inicial         : {self.k}")

    def distances_to(self, query):
        """Calcula distancia euclidiana real en las 784 dimensiones."""
        differences = self.x_train - query
        squared_distances = np.sum(differences * differences, axis=1)
        return np.sqrt(squared_distances)

    def neighbors_for(self, query, k=None):
        """Ordena las distancias y toma los k ejemplos mas cercanos."""
        if k is None:
            k = self.k

        distances = self.distances_to(query)
        ordered_indices = np.argsort(distances)
        neighbor_indices = ordered_indices[:k]
        return neighbor_indices, distances

    def predict_one(self, query, k=None):
        """Predice un digito y devuelve datos para explicar la decision."""
        if k is None:
            k = self.k

        neighbor_indices, distances = self.neighbors_for(query, k)
        neighbor_labels = self.y_train[neighbor_indices]
        votes = np.bincount(neighbor_labels, minlength=10)
        probabilities = votes / k
        tied_digits = np.where(votes == votes.max())[0]

        if len(tied_digits) == 1:
            prediction = int(tied_digits[0])
        else:
            prediction = self._break_tie(neighbor_labels, distances[neighbor_indices], tied_digits)

        return {
            "prediction": prediction,
            "probabilities": probabilities,
            "confidence": float(probabilities[prediction]),
            "neighbor_indices": neighbor_indices,
            "neighbor_labels": neighbor_labels,
            "distances": distances,
        }

    def predict_many(self, x_data, k=None):
        """Predice varias imagenes reutilizando vecinos calculados por lotes."""
        if k is None:
            k = self.k

        neighbor_indices, neighbor_distances = self.nearest_neighbors_many(x_data, k)
        return self.predict_from_neighbors(neighbor_indices, neighbor_distances, k)

    def nearest_neighbors_many(self, x_data, k, batch_size=256):
        """Calcula vecinos para muchas imagenes sin repetir trabajo innecesario."""
        x_data = np.asarray(x_data, dtype=np.float32)
        all_indices = []
        all_distances = []

        for start in range(0, len(x_data), batch_size):
            end = min(start + batch_size, len(x_data))
            batch = x_data[start:end]
            squared_distances = self._squared_distances_batch(batch)

            # Solo nos interesan los k mas cercanos, no ordenar todos los ejemplos.
            neighbor_indices = np.argpartition(squared_distances, kth=k - 1, axis=1)[:, :k]
            neighbor_squared = np.take_along_axis(squared_distances, neighbor_indices, axis=1)
            order = np.argsort(neighbor_squared, axis=1)

            sorted_indices = np.take_along_axis(neighbor_indices, order, axis=1)
            sorted_squared = np.take_along_axis(neighbor_squared, order, axis=1)
            sorted_distances = np.sqrt(np.maximum(sorted_squared, 0.0))

            all_indices.append(sorted_indices)
            all_distances.append(sorted_distances)

        return np.vstack(all_indices), np.vstack(all_distances)

    def predict_from_neighbors(self, neighbor_indices, neighbor_distances, k):
        """Predice usando vecinos ya calculados, util para probar varios k."""
        predictions = np.zeros(neighbor_indices.shape[0], dtype=int)
        confidences = np.zeros(neighbor_indices.shape[0], dtype=np.float32)

        labels_by_query = self.y_train[neighbor_indices[:, :k]]
        distances_by_query = neighbor_distances[:, :k]

        for row in range(labels_by_query.shape[0]):
            labels = labels_by_query[row]
            distances = distances_by_query[row]
            votes = np.bincount(labels, minlength=10)
            tied_digits = np.where(votes == votes.max())[0]

            if len(tied_digits) == 1:
                prediction = int(tied_digits[0])
            else:
                prediction = self._break_tie(labels, distances, tied_digits)

            predictions[row] = prediction
            confidences[row] = votes[prediction] / k

        return predictions, confidences

    def _squared_distances_batch(self, batch):
        """Calcula ||x-y||^2 para un lote contra todo train."""
        batch_squared_norms = np.sum(batch * batch, axis=1, keepdims=True)
        dot_products = batch @ self.x_train.T
        squared_distances = batch_squared_norms + self.train_squared_norms[None, :] - 2.0 * dot_products
        return np.maximum(squared_distances, 0.0)

    def _break_tie(self, neighbor_labels, neighbor_distances, tied_digits):
        """Si hay empate de votos, gana la clase con vecinos mas cercanos."""
        best_digit = int(tied_digits[0])
        best_average_distance = float("inf")

        for digit in tied_digits:
            average_distance = np.mean(neighbor_distances[neighbor_labels == digit])
            if average_distance < best_average_distance:
                best_digit = int(digit)
                best_average_distance = float(average_distance)

        return best_digit


def choose_best_k(model, x_validation, y_validation, k_values):
    """Prueba varios k reutilizando el mismo calculo de vecinos."""
    print_header("PASO 3: escoger k con validacion")
    best_k = None
    best_accuracy = -1
    max_k = max(k_values)

    print("  Calculando vecinos de validacion una sola vez...")
    neighbor_indices, neighbor_distances = model.nearest_neighbors_many(x_validation, max_k)

    for k in k_values:
        predictions, _ = model.predict_from_neighbors(neighbor_indices, neighbor_distances, k)
        accuracy = np.mean(predictions == y_validation)
        print(f"  k={k:<2d} -> exactitud validacion: {accuracy * 100:6.2f}%")

        if accuracy > best_accuracy:
            best_k = k
            best_accuracy = accuracy

    model.k = best_k
    print(f"  k elegido: {best_k}")
    return best_k


def evaluate_model(model, x_test, y_test):
    """Calcula exactitud y matriz de confusion sin usar metricas externas."""
    print_header("PASO 4: evaluar en test")
    print("  Calculando vecinos de prueba por lotes...")
    predictions, confidences = model.predict_many(x_test, model.k)
    matrix = confusion_matrix(y_test, predictions)
    accuracy = np.trace(matrix) / np.sum(matrix)

    print(f"  Exactitud test: {accuracy * 100:.2f}%")
    print(f"  Porcentaje de error: {(1 - accuracy) * 100:.2f}%")
    print_confusion_matrix(matrix)
    return predictions, confidences, matrix, accuracy


def confusion_matrix(y_true, y_predicted):
    """Cuenta cuantos ejemplos de cada digito caen en cada prediccion."""
    matrix = np.zeros((10, 10), dtype=int)
    for real, predicted in zip(y_true, y_predicted):
        matrix[real, predicted] += 1
    return matrix


def print_confusion_matrix(matrix):
    """Imprime la matriz como tabla para revisar errores por clase."""
    print("  Matriz de confusion:")
    print("      " + "".join(f"{digit:5d}" for digit in range(10)))
    print("     " + "-" * 50)
    for digit, row in enumerate(matrix):
        print(f"  {digit} | " + "".join(f"{value:5d}" for value in row))


def print_header(text):
    """Encabezado sencillo para separar pasos en consola."""
    print("\n" + "=" * 72)
    print(text)
    print("=" * 72)
