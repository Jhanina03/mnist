import argparse

from knn_algorithm import (
    MNISTData,
    WhiteBoxKNN,
    choose_best_k,
    evaluate_model,
    compare_metrics,
)
from knn_interface import LiveKNNViewer

DEFAULT_K_VALUES = [1, 3, 5, 7, 9, 12]


def parse_k_values(text):
    """Convierte '1,3,5' en una lista de enteros."""
    values = [int(value.strip()) for value in text.split(",") if value.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Debe existir al menos un valor de k.")
    if min(values) <= 0:
        raise argparse.ArgumentTypeError("Los valores de k deben ser positivos.")
    return values


def build_parser():
    """Argumentos utiles para cambiar tamano y velocidad sin tocar el codigo."""
    parser = argparse.ArgumentParser(description="KNN manual e interactivo para MNIST.")
    parser.add_argument(
        "--train-size",
        type=int,
        default=-1,
        help="Total de imágenes de entrenamiento (-1 para usar todas).",
    )
    parser.add_argument(
        "--val-size",
        type=int,
        default=-1,
        help="Total de imágenes de validación (-1 para usar todas las restantes).",
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=-1,
        help="Total de imágenes de test (-1 para usar todas).",
    )
    parser.add_argument(
        "--k-values",
        type=parse_k_values,
        default=DEFAULT_K_VALUES,
        help="Valores de k a comparar en validacion (default: 1,3,5,7,9,12).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--metric",
        type=str,
        default="euclidean",
        choices=["euclidean", "manhattan", "cosine"],
        help="Metrica de distancia a usar (default: euclidean).",
    )
    parser.add_argument(
        "--compare-metrics",
        action="store_true",
        help="Compara euclidean, manhattan y cosine en validacion antes de entrenar.",
    )
    return parser


def validate_arguments(args):
    """Revisa lo basico para que el experimento no arranque con datos absurdos."""
    for size_name in ["train_size", "val_size", "test_size"]:
        val = getattr(args, size_name)
        if val != -1 and val <= 0:
            raise ValueError(f"El argumento {size_name} debe ser -1 o un número mayor que 0. Valor actual: {val}")


def main():
    """Ejecuta toda la practica."""
    args = build_parser().parse_args()

    if args.metric == "manhattan":
        if args.train_size == -1:
            args.train_size = 2000
        if args.val_size == -1:
            args.val_size = 500
        if args.test_size == -1:
            args.test_size = 500
        print("\n" + "=" * 80)
        print(
            "NOTA: Para evitar que el cálculo de Manhattan se demore horas en Python puro,"
        )
        print("se redujo el tamaño del dataset automáticamente.")
        print("=" * 80 + "\n")

    validate_arguments(args)
    k_candidates = sorted(set(args.k_values))

    data_loader = MNISTData(seed=args.seed)
    data = data_loader.load(args.train_size, args.val_size, args.test_size)

    # Comparar metricas si el usuario lo pide; si no, usa la elegida por --metric.
    metric = args.metric
    if args.compare_metrics:
        _, metric = compare_metrics(
            data["x_train"],
            data["y_train"],
            data["x_validation"],
            data["y_validation"],
            k=max(k_candidates),
        )

    model = WhiteBoxKNN(k=max(k_candidates), metric=metric)
    model.fit(data["x_train"], data["y_train"])

    best_k = choose_best_k(
        model, data["x_validation"], data["y_validation"], k_candidates
    )
    _, _, matrix, accuracy = evaluate_model(model, data["x_test"], data["y_test"])

    viewer = LiveKNNViewer(
        model=model,
        data=data,
        k=best_k,
        k_values=k_candidates,
        matrix=matrix,
        accuracy=accuracy,
    )
    # Pasar todos los tests que haya en data["x_test"]
    viewer.run(len(data["x_test"]))


if __name__ == "__main__":
    main()
