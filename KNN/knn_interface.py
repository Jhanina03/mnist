"""
Interfaz visual para observar KNN en accion.

Esta version usa Tkinter para la ventana y Matplotlib embebido para los
graficos. Asi se mantiene como aplicacion de escritorio, sin HTML.
"""

from pathlib import Path
import tkinter as tk
from tkinter import ttk

import matplotlib

matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from knn_algorithm import print_header


class LiveKNNViewer:
    """Ventana de Tkinter que muestra el proceso de KNN muestra por muestra."""

    def __init__(
        self,
        model,
        data,
        k,
        delay,
        distance_lines,
        matrix,
        accuracy,
        output_dir,
        auto_play=False,
        show_confusion=False,
    ):
        self.model = model
        self.data = data
        self.k = k
        self.delay_ms = int(delay * 1000)
        self.distance_lines = distance_lines
        self.matrix = matrix
        self.accuracy = accuracy
        self.auto_play = auto_play
        self.show_confusion_at_start = show_confusion
        self.current_index = 0
        self.auto_job = None
        self.confusion_window = None
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print_header("PASO 5: preparar ventana Tkinter + Matplotlib")
        self._clear_old_distance_reports()
        self.train_3d, self.test_3d, self.train_subset_indices = self._manual_pca_3d(
            data["x_train"], data["x_test"]
        )
        print("  La grafica usa PCA 3D solo para visualizar.")
        print("  Se dibuja una muestra de ejemplos para no perder velocidad.")
        print("  Las distancias del KNN se calculan con todos los ejemplos en 784 dimensiones.")

    def run(self, total_cases):
        """Abre la ventana principal y deja el control en botones/teclas."""
        self.total_cases = min(total_cases, len(self.data["x_test"]))
        self.root = tk.Tk()
        self.root.title("KNN manual sobre MNIST")
        self._maximize_window()
        self._build_window()
        self._bind_keys()

        self.draw_case(0)
        if self.show_confusion_at_start:
            self.show_confusion_matrix()
        if self.auto_play:
            self._schedule_auto()

        self.root.mainloop()

    def _build_window(self):
        """Organiza la ventana para que cada grafico tenga su propio espacio."""
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(2, weight=1)

        title = ttk.Label(
            self.root,
            text="KNN manual sobre MNIST: vecinos, distancias y votos",
            font=("Segoe UI", 15, "bold"),
        )
        title.grid(row=0, column=0, sticky="ew", padx=12, pady=(8, 2))

        info = ttk.Label(
            self.root,
            text=(
                "Nota: la gráfica muestra una proyección PCA 3D de una muestra de puntos "
                "para mantener la ventana fluida. "
                "El KNN usa todos los ejemplos de entrenamiento en las 784 dimensiones reales."
            ),
            font=("Segoe UI", 10),
            wraplength=980,
            justify="center",
        )
        info.grid(row=1, column=0, sticky="ew", padx=12, pady=(0, 4))

        main = ttk.Frame(self.root)
        main.grid(row=2, column=0, sticky="nsew", padx=10, pady=8)
        main.columnconfigure(0, weight=5)
        main.columnconfigure(1, weight=2)
        main.columnconfigure(2, weight=3)
        main.rowconfigure(0, weight=1)

        left = ttk.LabelFrame(main, text="Mapa PCA 3D: dato nuevo y vecinos")
        center = ttk.Frame(main)
        right = ttk.LabelFrame(main, text="Distancias hacia ejemplos guardados")
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        center.grid(row=0, column=1, sticky="nsew", padx=4)
        right.grid(row=0, column=2, sticky="nsew", padx=(8, 0))

        left.rowconfigure(0, weight=1)
        left.columnconfigure(0, weight=1)
        center.rowconfigure(0, weight=1)
        center.rowconfigure(1, weight=1)
        center.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)

        self.map_figure = plt.figure(figsize=(7.6, 6.2), dpi=100)
        self.map_axis = self.map_figure.add_subplot(111, projection="3d")
        self.map_canvas = FigureCanvasTkAgg(self.map_figure, master=left)
        self.map_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew", padx=6, pady=6)

        image_frame = ttk.LabelFrame(center, text="Imagen consultada")
        prob_frame = ttk.LabelFrame(center, text="Probabilidad por votos")
        image_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 6))
        prob_frame.grid(row=1, column=0, sticky="nsew", pady=(6, 0))
        image_frame.rowconfigure(0, weight=1)
        image_frame.columnconfigure(0, weight=1)
        prob_frame.rowconfigure(0, weight=1)
        prob_frame.columnconfigure(0, weight=1)

        self.image_figure, self.image_axis = plt.subplots(figsize=(3.2, 3.0), dpi=100)
        self.image_canvas = FigureCanvasTkAgg(self.image_figure, master=image_frame)
        self.image_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew", padx=6, pady=6)

        self.prob_figure, self.prob_axis = plt.subplots(figsize=(3.2, 3.0), dpi=100)
        self.prob_canvas = FigureCanvasTkAgg(self.prob_figure, master=prob_frame)
        self.prob_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew", padx=6, pady=6)

        self.distance_text = tk.Text(right, wrap="none", font=("Consolas", 9), height=30)
        y_scroll = ttk.Scrollbar(right, orient="vertical", command=self.distance_text.yview)
        x_scroll = ttk.Scrollbar(right, orient="horizontal", command=self.distance_text.xview)
        self.distance_text.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
        self.distance_text.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")

        controls = ttk.Frame(self.root)
        controls.grid(row=3, column=0, sticky="ew", padx=10, pady=(0, 10))
        controls.columnconfigure(7, weight=1)

        self.prev_button = ttk.Button(controls, text="Anterior", command=self.previous_case)
        self.next_button = ttk.Button(controls, text="Siguiente", command=self.next_case)
        self.auto_button = ttk.Button(controls, text="Auto: OFF", command=self.toggle_auto)
        self.confusion_button = ttk.Button(controls, text="Matriz de confusion", command=self.show_confusion_matrix)
        self.exit_button = ttk.Button(controls, text="Salir", command=self.root.destroy)
        self.status_label = ttk.Label(controls, text="", font=("Segoe UI", 10, "bold"))

        self.prev_button.grid(row=0, column=0, padx=4)
        self.next_button.grid(row=0, column=1, padx=4)
        self.auto_button.grid(row=0, column=2, padx=4)
        self.confusion_button.grid(row=0, column=3, padx=4)
        self.exit_button.grid(row=0, column=4, padx=4)
        self.status_label.grid(row=0, column=7, sticky="e", padx=8)

    def draw_case(self, case_index):
        """Dibuja una consulta completa sin reconstruir toda la ventana."""
        case_index = max(0, min(case_index, self.total_cases - 1))
        self.current_index = case_index

        query = self.data["x_test"][case_index]
        real_label = self.data["y_test"][case_index]
        result = self.model.predict_one(query, self.k)
        prediction = result["prediction"]
        probabilities = result["probabilities"]
        distances = result["distances"]
        neighbor_indices = result["neighbor_indices"]

        self._write_full_distance_report(case_index, distances)
        self._draw_map(case_index, neighbor_indices, prediction, real_label)
        self._draw_digit(case_index, prediction, real_label, result["confidence"])
        self._draw_probabilities(probabilities)
        self._write_distance_panel(case_index, result, distances)
        self._update_status(real_label, prediction, result["confidence"])

    def next_case(self):
        """Avanza al siguiente numero."""
        if self.current_index < self.total_cases - 1:
            self.draw_case(self.current_index + 1)
        else:
            self._stop_auto()

    def previous_case(self):
        """Regresa al numero anterior."""
        if self.current_index > 0:
            self.draw_case(self.current_index - 1)

    def toggle_auto(self):
        """Activa o desactiva el avance automatico."""
        self.auto_play = not self.auto_play
        self.auto_button.configure(text="Auto: ON" if self.auto_play else "Auto: OFF")

        if self.auto_play:
            self._schedule_auto()
        else:
            self._stop_auto()

    def show_confusion_matrix(self):
        """Abre la matriz en una ventana aparte para que no aplaste la vista principal."""
        if self.confusion_window is not None and self.confusion_window.winfo_exists():
            self.confusion_window.lift()
            return

        self.confusion_window = tk.Toplevel(self.root)
        self.confusion_window.title("Matriz de confusion global")
        self.confusion_window.geometry("860x650")

        figure, axis = plt.subplots(figsize=(8.2, 5.8), dpi=100)
        image = axis.imshow(self.matrix, cmap="Blues", interpolation="nearest")
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        axis.set_title(f"Matriz de confusion global | Exactitud test: {self.accuracy * 100:.2f}% | Error global: {(1 - self.accuracy) * 100:.2f}%")
        axis.set_xlabel("Digito predicho")
        axis.set_ylabel("Digito real")
        axis.set_xticks(range(10))
        axis.set_yticks(range(10))

        threshold = self.matrix.max() / 2 if self.matrix.max() else 0
        for row in range(10):
            for col in range(10):
                color = "white" if self.matrix[row, col] > threshold else "black"
                axis.text(col, row, str(self.matrix[row, col]), ha="center", va="center", color=color)

        figure.tight_layout()
        canvas = FigureCanvasTkAgg(figure, master=self.confusion_window)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        canvas.draw_idle()

        help_text = (
            "Lectura: filas = digito real, columnas = digito predicho. "
            "Sirve para ver en que clases se equivoca KNN globalmente."
        )
        ttk.Label(self.confusion_window, text=help_text).pack(fill="x", padx=10, pady=8)

    def _draw_map(self, case_index, neighbor_indices, prediction, real_label):
        """Muestra la nube de puntos, los vecinos y las conexiones en 3D."""
        self.map_axis.clear()
        colors = plt.cm.tab10(np.arange(10))

        for digit in range(10):
            mask = self.data["y_train"] == digit
            class_indices = np.where(mask)[0]
            plot_indices = np.intersect1d(self.train_subset_indices, class_indices)
            self.map_axis.scatter(
                self.train_3d[plot_indices, 0],
                self.train_3d[plot_indices, 1],
                self.train_3d[plot_indices, 2],
                s=14,
                color=colors[digit],
                alpha=0.28,
                label=str(digit),
            )

        query_point = self.test_3d[case_index]
        neighbor_points = self.train_3d[neighbor_indices]

        for point in neighbor_points:
            self.map_axis.plot(
                [query_point[0], point[0]],
                [query_point[1], point[1]],
                [query_point[2], point[2]],
                color="black",
                alpha=0.28,
                linewidth=1,
            )

        self.map_axis.scatter(
            neighbor_points[:, 0],
            neighbor_points[:, 1],
            neighbor_points[:, 2],
            s=115,
            facecolors="none",
            edgecolors="black",
            linewidths=1.8,
            label="vecinos",
        )
        self.map_axis.scatter(
            query_point[0],
            query_point[1],
            query_point[2],
            marker="*",
            s=260,
            color="black",
            edgecolors="white",
            linewidths=1,
            label="nuevo dato",
        )

        status = "correcto" if prediction == real_label else "error"
        self.map_axis.set_title(f"PCA 3D | real={real_label} pred={prediction} ({status})")
        self.map_axis.set_xlabel("Componente principal 1")
        self.map_axis.set_ylabel("Componente principal 2")
        self.map_axis.set_zlabel("Componente principal 3")
        self.map_axis.grid(alpha=0.22)
        self.map_axis.legend(loc="upper right", ncol=2, fontsize=8)
        self.map_figure.tight_layout()
        self.map_canvas.draw_idle()

    def _draw_digit(self, case_index, prediction, real_label, confidence):
        """Muestra el numero que esta entrando al KNN."""
        self.image_axis.clear()
        self.image_axis.imshow(self.data["raw_test"][case_index], cmap="gray")
        self.image_axis.set_title(f"Real {real_label} | Pred {prediction} | Prob {confidence:.2f}")
        self.image_axis.axis("off")
        self.image_figure.tight_layout()
        self.image_canvas.draw_idle()

    def _draw_probabilities(self, probabilities):
        """Muestra los votos normalizados como probabilidad."""
        self.prob_axis.clear()
        digits = np.arange(10)
        self.prob_axis.bar(digits, probabilities, color=plt.cm.tab10(digits))
        self.prob_axis.set_xlabel("Digito")
        self.prob_axis.set_ylabel("votos / k")
        self.prob_axis.set_xticks(digits)
        self.prob_axis.set_ylim(0, 1)
        self.prob_axis.grid(axis="y", alpha=0.25)
        self.prob_figure.tight_layout()
        self.prob_canvas.draw_idle()

    def _write_distance_panel(self, case_index, result, distances):
        """Escribe las distancias mas importantes en el panel derecho."""
        ordered = np.argsort(distances)
        self.distance_text.configure(state="normal")
        self.distance_text.delete("1.0", "end")

        lines = [
            f"Consulta {case_index + 1}/{self.total_cases}",
            f"k = {self.k}",
            "",
            "Proceso KNN:",
            "1. Distancia euclidiana en 784 pixeles.",
            "2. Ordenar distancias de menor a mayor.",
            "3. Tomar los k vecinos mas cercanos.",
            "4. Votar por clase.",
            "",
            "Vecinos usados:",
        ]

        for rank, train_index in enumerate(result["neighbor_indices"], start=1):
            label = self.data["y_train"][train_index]
            distance = distances[train_index]
            lines.append(f"  {rank:02d}. train[{train_index:04d}] y={label} d={distance:.4f}")

        lines.extend(["", "Distancias ordenadas:"])

        neighbor_set = set(result["neighbor_indices"])
        for rank, train_index in enumerate(ordered[: self.distance_lines], start=1):
            label = self.data["y_train"][train_index]
            distance = distances[train_index]
            mark = "*" if train_index in neighbor_set else " "
            lines.append(f"{mark} {rank:03d}. train[{train_index:04d}] y={label} d={distance:.4f}")

        if self.distance_lines < len(ordered):
            report_name = f"distancias_consulta_{case_index:03d}.txt"
            lines.extend(["", "Lista completa:", f"KNN/output/{report_name}"])

        self.distance_text.insert("1.0", "\n".join(lines))
        self.distance_text.configure(state="disabled")

    def _update_status(self, real_label, prediction, confidence):
        """Actualiza el resumen inferior."""
        status = "correcto" if real_label == prediction else "error"
        text = (
            f"Consulta {self.current_index + 1}/{self.total_cases} | "
            f"Real: {real_label} | Predicho: {prediction} | "
            f"Probabilidad: {confidence:.2f} | {status} | "
            f"Exactitud test: {self.accuracy * 100:.2f}% | "
            f"Error global: {(1 - self.accuracy) * 100:.2f}%"
        )
        self.status_label.configure(text=text)

    def _manual_pca_3d(self, x_train, x_test, max_plot=1500):
        """Reduce los 784 pixeles a 3 coordenadas para la visualizacion."""
        mean_image = np.mean(x_train, axis=0)
        centered_train = x_train - mean_image

        # SVD da tres direcciones utiles para ver la nube, no para clasificar.
        _, _, vt = np.linalg.svd(centered_train, full_matrices=False)
        components = vt[:3]
        train_3d = centered_train @ components.T
        test_3d = (x_test - mean_image) @ components.T

        if len(x_train) > max_plot:
            rng = np.random.default_rng(42)
            train_subset_indices = rng.choice(len(x_train), size=max_plot, replace=False)
        else:
            train_subset_indices = np.arange(len(x_train))

        return train_3d, test_3d, train_subset_indices

    def _write_full_distance_report(self, case_index, distances):
        """Guarda todas las distancias para no saturar la ventana."""
        ordered = np.argsort(distances)
        path = self.output_dir / f"distancias_consulta_{case_index:03d}.txt"

        with path.open("w", encoding="utf-8") as file:
            file.write(f"Distancias completas para consulta {case_index}\n")
            file.write("rank,train_index,label,distance\n")
            for rank, train_index in enumerate(ordered, start=1):
                label = self.data["y_train"][train_index]
                distance = distances[train_index]
                file.write(f"{rank},{train_index},{label},{distance:.8f}\n")

    def _schedule_auto(self):
        """Programa el siguiente avance automatico."""
        self._stop_auto()
        self.auto_button.configure(text="Auto: ON")
        self.auto_job = self.root.after(self.delay_ms, self._auto_step)

    def _auto_step(self):
        """Avanza mientras queden consultas."""
        if not self.auto_play:
            return

        if self.current_index >= self.total_cases - 1:
            self._stop_auto()
            self.auto_play = False
            self.auto_button.configure(text="Auto: OFF")
            return

        self.next_case()
        self.auto_job = self.root.after(self.delay_ms, self._auto_step)

    def _stop_auto(self):
        """Cancela el avance automatico pendiente."""
        if self.auto_job is not None:
            self.root.after_cancel(self.auto_job)
            self.auto_job = None

    def _clear_old_distance_reports(self):
        """Borra reportes de distancias de corridas anteriores."""
        clean_distance_reports(self.output_dir)

    def _maximize_window(self):
        """Intenta abrir maximizado, con fallback para otros sistemas."""
        try:
            self.root.state("zoomed")
        except tk.TclError:
            self.root.geometry("1280x760")

    def _bind_keys(self):
        """Atajos comodos para revisar los casos."""
        self.root.bind("<Right>", lambda event: self.next_case())
        self.root.bind("<Left>", lambda event: self.previous_case())
        self.root.bind("<space>", lambda event: self.toggle_auto())
        self.root.bind("<Escape>", lambda event: self.root.destroy())


def clean_distance_reports(output_dir):
    """Borra solo los txt de distancias generados por esta practica."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    deleted = 0

    for path in output_dir.glob("distancias_consulta_*.txt"):
        path.unlink()
        deleted += 1

    if deleted:
        print(f"  Se limpiaron {deleted} archivos de distancias anteriores.")
