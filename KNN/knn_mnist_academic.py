"""Punto de entrada para ejecutar el KNN manual sobre MNIST."""

import argparse
from pathlib import Path

from knn_algorithm import MNISTData, WhiteBoxKNN, choose_best_k, evaluate_model, compare_metrics
from knn_interface import LiveKNNViewer, clean_distance_reports


OUTPUT_DIR = Path(__file__).resolve().parent / "output"
DEFAULT_K_VALUES = [1, 3, 5, 7, 9]


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
    parser.add_argument("--train-per-class", type=int, default=500)
    parser.add_argument("--validation-per-class", type=int, default=100)
    parser.add_argument("--test-per-class", type=int, default=100)
    parser.add_argument(
        "--k-values",
        type=parse_k_values,
        default=DEFAULT_K_VALUES,
        help="Valores de k a comparar en validacion (default: 1,3,5,7,9).",
    )
    parser.add_argument("--delay", type=float, default=1.8)
    parser.add_argument("--cases", type=int, default=20)
    parser.add_argument("--distance-lines", type=int, default=28)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--auto", action="store_true")
    parser.add_argument("--show-confusion", action="store_true")
    parser.add_argument("--no-window", action="store_true")
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
    sizes = [args.train_per_class, args.validation_per_class, args.test_per_class]
    if min(sizes) <= 0:
        raise ValueError("Las cantidades por clase deben ser mayores que cero.")
    if args.delay <= 0:
        raise ValueError("El delay debe ser mayor que cero.")
    if args.cases <= 0:
        raise ValueError("La cantidad de casos a mostrar debe ser mayor que cero.")
    if args.distance_lines <= 0:
        raise ValueError("distance-lines debe ser mayor que cero.")
    if max(args.k_values) > args.train_per_class * 10:
        raise ValueError("k no puede ser mayor que la cantidad total de ejemplos de train.")


def main():
    """Ejecuta toda la practica."""
    args = build_parser().parse_args()
    validate_arguments(args)
    k_candidates = sorted(set(args.k_values))
    clean_distance_reports(OUTPUT_DIR)

    data_loader = MNISTData(seed=args.seed)
    data = data_loader.load(args.train_per_class, args.validation_per_class, args.test_per_class)

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

    best_k = choose_best_k(model, data["x_validation"], data["y_validation"], k_candidates)
    _, _, matrix, accuracy = evaluate_model(model, data["x_test"], data["y_test"])

    if not args.no_window:
        viewer = LiveKNNViewer(
            model=model,
            data=data,
            k=best_k,
            k_values=k_candidates,
            delay=args.delay,
            distance_lines=args.distance_lines,
            matrix=matrix,
            accuracy=accuracy,
            output_dir=OUTPUT_DIR,
            auto_play=args.auto,
            show_confusion=args.show_confusion,
        )
        viewer.run(args.cases)


if __name__ == "__main__":
    main()
