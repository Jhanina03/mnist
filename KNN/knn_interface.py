import tkinter as tk
from tkinter import ttk
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from knn_algorithm import evaluate_model

class LiveKNNViewer:
    """Ventana de Tkinter ultra simplificada para mostrar el proceso KNN."""

    def __init__(self, model, data, k, matrix, accuracy, k_values=None):
        self.model = model
        self.data = data
        self.k = k
        self.k_values = k_values if k_values is not None else [k]
        self.matrix = matrix
        self.accuracy = accuracy
        self.current_index = 0
        self.confusion_window = None

        self.train_2d, self.test_2d, self.train_subset_indices = self._manual_pca_2d(
            data["x_train"], data["x_test"]
        )

    def _manual_pca_2d(self, x_train, x_test, max_plot=800):
        """Reduce los 784 pixeles a 2 coordenadas para la visualizacion."""
        mean_image = np.mean(x_train, axis=0)
        centered_train = x_train - mean_image
        _, _, vt = np.linalg.svd(centered_train, full_matrices=False)
        components = vt[:2]
        train_2d = centered_train @ components.T
        test_2d = (x_test - mean_image) @ components.T

        if len(x_train) > max_plot:
            rng = np.random.default_rng(42)
            train_subset_indices = rng.choice(len(x_train), size=max_plot, replace=False)
        else:
            train_subset_indices = np.arange(len(x_train))

        return train_2d, test_2d, train_subset_indices

    def run(self, total_cases):
        """Inicia la ventana principal."""
        self.total_cases = total_cases
        self.root = tk.Tk()
        self.root.title("KNN MNIST - Visualización 2D")
        
        try:
            self.root.state("zoomed")
        except tk.TclError:
            self.root.geometry("1200x800")
            
        self._build_window()
        self.draw_case(0)
        self.root.mainloop()

    def _build_window(self):
        """Crea la interfaz limpia y minimalista."""
        # Top Label
        tk.Label(
            self.root, 
            text="NOTA: El gráfico es una representación 2D para visualización, pero en realidad en la lógica del sistema los cálculos y distancias KNN se hacen con las 784 dimensiones de cada imagen.",
            fg="blue", font=("Arial", 10, "bold"), wraplength=800, justify="left"
        ).pack(pady=5, padx=10, fill="x")

        # Main frame
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill="both", expand=True, padx=10, pady=5)

        # Left: Map 2D
        left_frame = tk.Frame(main_frame)
        left_frame.pack(side="left", fill="both", expand=True)
        
        self.map_fig, self.map_ax = plt.subplots(figsize=(5.5, 5))
        self.map_canvas = FigureCanvasTkAgg(self.map_fig, master=left_frame)
        self.map_canvas.get_tk_widget().pack(fill="both", expand=True)

        # Center: Image + Probabilities
        center_frame = tk.Frame(main_frame)
        center_frame.pack(side="left", fill="y", expand=False, padx=5)
        
        self.img_fig, self.img_ax = plt.subplots(figsize=(2.5, 2.5))
        self.img_canvas = FigureCanvasTkAgg(self.img_fig, master=center_frame)
        self.img_canvas.get_tk_widget().pack(fill="both", expand=True)
        
        self.prob_fig, self.prob_ax = plt.subplots(figsize=(2.5, 2.5))
        self.prob_canvas = FigureCanvasTkAgg(self.prob_fig, master=center_frame)
        self.prob_canvas.get_tk_widget().pack(fill="both", expand=True)

        # Right: Distances List
        right_frame = tk.Frame(main_frame)
        right_frame.pack(side="left", fill="y", expand=False, padx=5)
        
        tk.Label(right_frame, text="Reporte de Cálculos (Caja Blanca)", font=("Arial", 10, "bold")).pack(pady=5)
        
        scroll = tk.Scrollbar(right_frame)
        scroll.pack(side="right", fill="y")
        self.dist_text = tk.Text(right_frame, width=46, height=28, state="disabled", font=("Courier", 9), yscrollcommand=scroll.set)
        self.dist_text.pack(side="left", fill="y")
        scroll.config(command=self.dist_text.yview)

        # Bottom controls
        controls = tk.Frame(self.root)
        controls.pack(fill="x", pady=10, padx=10)

        tk.Button(controls, text="◀ Anterior", command=self.previous_case).pack(side="left", padx=5)
        tk.Button(controls, text="Siguiente ▶", command=self.next_case).pack(side="left", padx=5)
        
        tk.Label(controls, text="Métrica:").pack(side="left", padx=(15,2))
        self.metric_combo = ttk.Combobox(controls, values=["euclidean", "manhattan", "cosine"], state="readonly", width=10)
        self.metric_combo.set(getattr(self.model, "metric", "euclidean"))
        self.metric_combo.bind("<<ComboboxSelected>>", self.change_metric)
        self.metric_combo.pack(side="left")

        tk.Label(controls, text="K:").pack(side="left", padx=(15,2))
        self.k_combo = ttk.Combobox(controls, values=[str(kv) for kv in self.k_values], state="readonly", width=4)
        self.k_combo.set(str(self.k))
        self.k_combo.bind("<<ComboboxSelected>>", self.change_k)
        self.k_combo.pack(side="left")

        tk.Button(controls, text="Matriz de Confusión", command=self.show_confusion_matrix).pack(side="left", padx=15)
        
        self.status_label = tk.Label(controls, text="", font=("Arial", 10, "bold"))
        self.status_label.pack(side="right", padx=10)

    def draw_case(self, case_index):
        """Calcula el KNN de un caso de prueba y lo dibuja."""
        self.current_index = max(0, min(case_index, self.total_cases - 1))
        
        query = self.data["x_test"][self.current_index]
        real_label = self.data["y_test"][self.current_index]
        result = self.model.predict_one(query, self.k)
        
        self._draw_map(self.current_index, result["neighbor_indices"], result["prediction"], real_label)
        self._draw_digit(self.current_index, result["prediction"], real_label, result["confidence"])
        self._draw_probabilities(result["probabilities"])
        self._update_distances_text(query, result["distances"])
        
        correct = (result["prediction"] == real_label)
        status_text = f"Caso {self.current_index + 1}/{self.total_cases} | Real: {real_label} Pred: {result['prediction']} | {'✓' if correct else '✗'}"
        self.status_label.config(text=status_text, fg="green" if correct else "red")

    def _update_distances_text(self, query, distances):
        self.dist_text.config(state="normal")
        self.dist_text.delete("1.0", tk.END)
        
        if hasattr(self.model, "k_validation_scores") and self.model.k_validation_scores:
            self.dist_text.insert(tk.END, "HISTORIAL DE ENTRENAMIENTO (Selección de K):\n")
            for k_cand, score in self.model.k_validation_scores.items():
                marker = " (K Elegido)" if k_cand == self.model.k else ""
                self.dist_text.insert(tk.END, f" K={k_cand:<2} : {score*100:.2f}%{marker}\n")
            self.dist_text.insert(tk.END, "-" * 32 + "\n\n")

        try:
            proto_dist = self.model.class_distances(query)
            self.dist_text.insert(tk.END, "DISTANCIA DEL DATO ACTUAL A PROTOTIPOS:\n")
            for c in range(10):
                self.dist_text.insert(tk.END, f" Clase {c}: {proto_dist[c]:.2f}\n")
            self.dist_text.insert(tk.END, "-" * 32 + "\n\n")
        except Exception:
            pass

        self.dist_text.insert(tk.END, "TOP VECINOS DEL DATO NUEVO (TEST):\n")
        self.dist_text.insert(tk.END, f"{'Top':<4} | {'Dígito':<6} | {'Distancia':<9}\n")
        self.dist_text.insert(tk.END, "-" * 32 + "\n")
        
        ordered = np.argsort(distances)
        for rank in range(30):
            idx = ordered[rank]
            dist = distances[idx]
            lbl = self.data["y_train"][idx]
            marker = " <-" if rank < self.k else ""
            self.dist_text.insert(tk.END, f"{rank+1:<4} | {lbl:<6} | {dist:.2f}{marker}\n")
            
        self.dist_text.config(state="disabled")

    def _draw_map(self, case_index, neighbor_indices, prediction, real_label):
        self.map_ax.clear()
        colors = plt.cm.tab10(np.arange(10))

        # Puntos de entrenamiento (fondo)
        for digit in range(10):
            mask = self.data["y_train"] == digit
            class_indices = np.where(mask)[0]
            plot_indices = np.intersect1d(self.train_subset_indices, class_indices)
            self.map_ax.scatter(
                self.train_2d[plot_indices, 0], self.train_2d[plot_indices, 1],
                s=10, color=colors[digit], alpha=0.3, label=str(digit)
            )

        query_point = self.test_2d[case_index]
        neighbor_points = self.train_2d[neighbor_indices]
        neighbor_labels = self.data["y_train"][neighbor_indices]

        # Puntos de los vecinos
        for pt, lbl in zip(neighbor_points, neighbor_labels):
            self.map_ax.scatter(pt[0], pt[1], s=50, color=colors[lbl], edgecolors="black", zorder=5)

        # Lineas a los vecinos
        for point in neighbor_points:
            self.map_ax.plot([query_point[0], point[0]], [query_point[1], point[1]], color="black", alpha=0.5, linewidth=1)

        # Circulo de radio
        if len(neighbor_points) > 0:
            diffs = neighbor_points - query_point
            radius = float(np.max(np.sqrt(np.sum(diffs * diffs, axis=1))))
            theta = np.linspace(0, 2 * np.pi, 100)
            self.map_ax.plot(query_point[0] + radius * np.cos(theta), query_point[1] + radius * np.sin(theta), "k--", alpha=0.5, label=f"radio k={self.k}")

        # Punto de test
        self.map_ax.scatter(query_point[0], query_point[1], marker="*", s=200, color="red", edgecolors="black", zorder=6, label="nuevo dato")
        
        self.map_ax.set_title(f"Mapa 2D (Test vs Train)")
        self.map_ax.legend(loc="upper right", ncol=2, fontsize=8)
        self.map_fig.tight_layout()
        self.map_canvas.draw_idle()

    def _draw_digit(self, case_index, prediction, real_label, confidence):
        self.img_ax.clear()
        self.img_ax.imshow(self.data["raw_test"][case_index], cmap="gray")
        self.img_ax.set_title(f"Predicho: {prediction} ({confidence:.0%})", color="green" if prediction==real_label else "red")
        self.img_ax.axis("off")
        self.img_fig.tight_layout()
        self.img_canvas.draw_idle()

    def _draw_probabilities(self, probabilities):
        self.prob_ax.clear()
        digits = np.arange(10)
        self.prob_ax.bar(digits, probabilities, color=["green" if p == probabilities.max() else "gray" for p in probabilities])
        self.prob_ax.set_xticks(digits)
        self.prob_ax.set_ylim(0, 1.1)
        self.prob_ax.set_title("Votos por clase")
        self.prob_fig.tight_layout()
        self.prob_canvas.draw_idle()

    def next_case(self):
        if self.current_index < self.total_cases - 1:
            self.draw_case(self.current_index + 1)

    def previous_case(self):
        if self.current_index > 0:
            self.draw_case(self.current_index - 1)

    def change_metric(self, event=None):
        new_metric = self.metric_combo.get()
        if new_metric != self.model.metric:
            self.model.metric = new_metric
            self.matrix = None
            self.accuracy = None
            self.root.config(cursor="watch")
            self.root.update()
            self.draw_case(self.current_index)
            self.root.config(cursor="")

    def change_k(self, event=None):
        new_k = int(self.k_combo.get())
        if new_k != self.k:
            self.k = new_k
            self.model.k = new_k
            self.matrix = None
            self.accuracy = None
            self.root.config(cursor="watch")
            self.root.update()
            self.draw_case(self.current_index)
            self.root.config(cursor="")

    def show_confusion_matrix(self):
        if self.confusion_window is not None and self.confusion_window.winfo_exists():
            self.confusion_window.lift()
            return

        if self.matrix is None:
            self.root.config(cursor="watch")
            self.root.update()
            print("Calculando matriz de confusión completa (10,000 tests)... esto tomará un momento.")
            _, _, self.matrix, self.accuracy = evaluate_model(
                self.model, self.data["x_test"], self.data["y_test"],
                progress_callback=self.root.update
            )
            self.root.config(cursor="")

        self.confusion_window = tk.Toplevel(self.root)
        self.confusion_window.title("Matriz de confusión (Datos Test)")
        
        fig, ax = plt.subplots(figsize=(6, 5))
        cax = ax.imshow(self.matrix, cmap="Blues")
        fig.colorbar(cax)
        
        for i in range(10):
            for j in range(10):
                ax.text(j, i, str(self.matrix[i, j]), ha="center", va="center", color="black" if self.matrix[i,j] < self.matrix.max()/2 else "white")
                
        ax.set_xticks(np.arange(10))
        ax.set_yticks(np.arange(10))
        ax.set_xlabel("Predicho")
        ax.set_ylabel("Real")
        ax.set_title(f"Exactitud global: {self.accuracy:.2%}")
        
        canvas = FigureCanvasTkAgg(fig, master=self.confusion_window)
        canvas.get_tk_widget().pack(fill="both", expand=True)

def clean_distance_reports(*args, **kwargs):
    pass
