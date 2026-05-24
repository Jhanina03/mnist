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

from knn_algorithm import print_header, evaluate_model


# ── Paleta de colores del tema oscuro ──────────────────────────────────────────
BG_DARK    = "#0d1117"   # fondo principal (negro azulado)
BG_PANEL   = "#161b22"   # fondo de paneles
BG_CARD    = "#1c2230"   # fondo de tarjetas
ACCENT     = "#58a6ff"   # azul celeste (acento principal)
ACCENT2    = "#bc8cff"   # violeta (acento secundario)
SUCCESS    = "#3fb950"   # verde correcto
ERROR_COL  = "#f85149"   # rojo error
TEXT_MAIN  = "#e6edf3"   # texto principal
TEXT_DIM   = "#8b949e"   # texto secundario
BORDER     = "#30363d"   # bordes sutiles
BTN_BG     = "#21262d"   # fondo de botones
BTN_HOVER  = "#30363d"   # botón hover

# Colores matplotlib para fondo oscuro
MPL_BG     = "#161b22"
MPL_AX_BG  = "#0d1117"
MPL_TEXT   = "#e6edf3"
MPL_GRID   = "#21262d"
# ──────────────────────────────────────────────────────────────────────────────


def _apply_dark_style(root):
    """Configura ttk.Style con el tema oscuro personalizado."""
    style = ttk.Style(root)
    style.theme_use("clam")

    # Fondo general
    style.configure(".", background=BG_DARK, foreground=TEXT_MAIN,
                    font=("Segoe UI", 10), bordercolor=BORDER,
                    troughcolor=BG_PANEL, selectbackground=ACCENT,
                    selectforeground=BG_DARK)

    # Frame y LabelFrame
    style.configure("TFrame", background=BG_DARK)
    style.configure("TLabelframe", background=BG_PANEL, foreground=ACCENT,
                    bordercolor=BORDER, relief="flat", padding=6)
    style.configure("TLabelframe.Label", background=BG_PANEL,
                    foreground=ACCENT, font=("Segoe UI", 10, "bold"))

    # Labels
    style.configure("TLabel", background=BG_DARK, foreground=TEXT_MAIN)
    style.configure("Title.TLabel", background=BG_DARK, foreground=ACCENT,
                    font=("Segoe UI", 16, "bold"))
    style.configure("Sub.TLabel", background=BG_DARK, foreground=TEXT_DIM,
                    font=("Segoe UI", 9))
    style.configure("Status.TLabel", background=BG_DARK, foreground=ACCENT2,
                    font=("Segoe UI", 10, "bold"))
    style.configure("MetricLabel.TLabel", background=BG_DARK,
                    foreground=TEXT_DIM, font=("Segoe UI", 10))

    # Botones
    style.configure("TButton", background=BTN_BG, foreground=TEXT_MAIN,
                    bordercolor=BORDER, relief="flat", padding=(10, 5),
                    font=("Segoe UI", 10))
    style.map("TButton",
              background=[("active", ACCENT), ("pressed", "#1158c7")],
              foreground=[("active", BG_DARK)])

    # Botón de navegación (más prominente)
    style.configure("Nav.TButton", background=ACCENT, foreground=BG_DARK,
                    bordercolor=ACCENT, relief="flat", padding=(12, 6),
                    font=("Segoe UI", 10, "bold"))
    style.map("Nav.TButton",
              background=[("active", "#79b8ff"), ("pressed", "#1158c7")],
              foreground=[("active", BG_DARK)])

    # Botón Salir
    style.configure("Exit.TButton", background="#2d1b1b", foreground=ERROR_COL,
                    bordercolor=ERROR_COL, relief="flat", padding=(10, 5),
                    font=("Segoe UI", 10))
    style.map("Exit.TButton",
              background=[("active", ERROR_COL)],
              foreground=[("active", BG_DARK)])

    # Combobox
    style.configure("TCombobox", fieldbackground=BG_CARD, background=BTN_BG,
                    foreground=TEXT_MAIN, arrowcolor=ACCENT,
                    selectbackground=ACCENT, selectforeground=BG_DARK,
                    bordercolor=BORDER, relief="flat")
    style.map("TCombobox",
              fieldbackground=[("readonly", BG_CARD)],
              foreground=[("readonly", TEXT_MAIN)])

    # Scrollbar
    style.configure("TScrollbar", background=BG_PANEL, troughcolor=BG_DARK,
                    arrowcolor=TEXT_DIM, bordercolor=BORDER)

    # Separador
    style.configure("TSeparator", background=BORDER)


def _setup_dark_matplotlib():
    """Aplica estilo oscuro a todos los graficos de matplotlib."""
    plt.rcParams.update({
        "figure.facecolor":   MPL_BG,
        "axes.facecolor":     MPL_AX_BG,
        "axes.edgecolor":     BORDER,
        "axes.labelcolor":    TEXT_DIM,
        "axes.titlecolor":    TEXT_MAIN,
        "text.color":         TEXT_MAIN,
        "xtick.color":        TEXT_DIM,
        "ytick.color":        TEXT_DIM,
        "grid.color":         MPL_GRID,
        "legend.facecolor":   BG_PANEL,
        "legend.edgecolor":   BORDER,
        "legend.labelcolor":  TEXT_MAIN,
        "savefig.facecolor":  MPL_BG,
        "figure.edgecolor":   BG_DARK,
    })


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
        k_values=None,
    ):
        self.model = model
        self.data = data
        self.k = k
        self.k_values = k_values if k_values is not None else [k]
        self.delay_ms = int(delay * 1000)
        self.distance_lines = distance_lines
        self.matrix = matrix
        self.accuracy = accuracy
        self.auto_play = auto_play
        self.show_confusion_at_start = show_confusion
        self.current_index = 0
        self.auto_job = None
        self.zoom_mode = False  # False = vista completa, True = zoom a vecinos
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
        metric_name = getattr(self.model, "metric", "euclidean")
        self.root.title(f"KNN manual sobre MNIST  |  métrica: {metric_name}")

        # Aplicar tema oscuro antes de construir la ventana
        self.root.configure(bg=BG_DARK)
        _apply_dark_style(self.root)
        _setup_dark_matplotlib()

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

        # ── Cabecera ──────────────────────────────────────────────────────────
        header_frame = tk.Frame(self.root, bg=BG_PANEL, pady=0)
        header_frame.grid(row=0, column=0, sticky="ew")
        header_frame.columnconfigure(0, weight=1)

        # Línea de acento en la parte superior
        accent_bar = tk.Frame(header_frame, bg=ACCENT, height=3)
        accent_bar.grid(row=0, column=0, sticky="ew")

        title_inner = tk.Frame(header_frame, bg=BG_PANEL)
        title_inner.grid(row=1, column=0, sticky="ew", padx=18, pady=(8, 4))
        title_inner.columnconfigure(1, weight=1)

        # Ícono de punto de acento
        tk.Label(title_inner, text="◈", bg=BG_PANEL, fg=ACCENT,
                 font=("Segoe UI", 20)).grid(row=0, column=0, padx=(0, 10))

        tk.Label(
            title_inner,
            text="KNN Manual sobre MNIST",
            bg=BG_PANEL, fg=TEXT_MAIN,
            font=("Segoe UI", 16, "bold"),
        ).grid(row=0, column=1, sticky="w")

        tk.Label(
            title_inner,
            text="Visualización interactiva de vecinos, distancias y votos",
            bg=BG_PANEL, fg=TEXT_DIM,
            font=("Segoe UI", 10),
        ).grid(row=1, column=1, sticky="w")

        # Nota informativa
        note_frame = tk.Frame(self.root, bg=BG_DARK)
        note_frame.grid(row=1, column=0, sticky="ew", padx=14, pady=(6, 2))
        tk.Label(
            note_frame,
            text=(
                "ℹ  La gráfica 3D usa PCA para visualizar. "
                "El KNN clasifica en las 784 dimensiones reales del píxel."
            ),
            bg=BG_DARK, fg=TEXT_DIM,
            font=("Segoe UI", 9),
            anchor="w",
        ).pack(fill="x")

        # ── Área principal ────────────────────────────────────────────────────
        main = tk.Frame(self.root, bg=BG_DARK)
        main.grid(row=2, column=0, sticky="nsew", padx=10, pady=6)
        main.columnconfigure(0, weight=5)
        main.columnconfigure(1, weight=2)
        main.columnconfigure(2, weight=3)
        main.rowconfigure(0, weight=1)

        # Panel izquierdo (mapa 3D)
        left = self._make_panel(main, "  🗺  Mapa PCA 3D — dato nuevo y vecinos")
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 6))

        # Panel central (imagen + probabilidades)
        center = tk.Frame(main, bg=BG_DARK)
        center.grid(row=0, column=1, sticky="nsew", padx=4)
        center.rowconfigure(0, weight=1)
        center.rowconfigure(1, weight=1)
        center.columnconfigure(0, weight=1)

        # Panel derecho (distancias)
        right = self._make_panel(main, "  📊  Distancias hacia ejemplos guardados")
        right.grid(row=0, column=2, sticky="nsew", padx=(6, 0))

        # Configurar paneles internos
        left.rowconfigure(0, weight=1)
        left.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)

        # Mapa 3D
        self.map_figure = plt.figure(figsize=(7.6, 6.2), dpi=100)
        self.map_figure.patch.set_facecolor(MPL_BG)
        self.map_axis = self.map_figure.add_subplot(111, projection="3d")
        self.map_axis.set_facecolor(MPL_AX_BG)
        self.map_canvas = FigureCanvasTkAgg(self.map_figure, master=left)
        self.map_canvas.get_tk_widget().configure(bg=BG_PANEL, highlightthickness=0)
        self.map_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew", padx=6, pady=6)

        # Imagen del dígito
        image_frame = self._make_panel(center, "  🔢  Imagen consultada")
        image_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 4))
        image_frame.rowconfigure(0, weight=1)
        image_frame.columnconfigure(0, weight=1)

        self.image_figure, self.image_axis = plt.subplots(figsize=(3.2, 3.0), dpi=100)
        self.image_figure.patch.set_facecolor(MPL_BG)
        self.image_axis.set_facecolor(MPL_AX_BG)
        self.image_canvas = FigureCanvasTkAgg(self.image_figure, master=image_frame)
        self.image_canvas.get_tk_widget().configure(bg=BG_PANEL, highlightthickness=0)
        self.image_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew", padx=6, pady=6)

        # Probabilidades
        prob_frame = self._make_panel(center, "  📈  Probabilidad por votos")
        prob_frame.grid(row=1, column=0, sticky="nsew", pady=(4, 0))
        prob_frame.rowconfigure(0, weight=1)
        prob_frame.columnconfigure(0, weight=1)

        self.prob_figure, self.prob_axis = plt.subplots(figsize=(3.2, 3.0), dpi=100)
        self.prob_figure.patch.set_facecolor(MPL_BG)
        self.prob_axis.set_facecolor(MPL_AX_BG)
        self.prob_canvas = FigureCanvasTkAgg(self.prob_figure, master=prob_frame)
        self.prob_canvas.get_tk_widget().configure(bg=BG_PANEL, highlightthickness=0)
        self.prob_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew", padx=6, pady=6)

        # Panel de texto de distancias (tema oscuro)
        self.distance_text = tk.Text(
            right,
            wrap="none",
            font=("Cascadia Code", 9) if self._font_exists("Cascadia Code") else ("Consolas", 9),
            height=30,
            bg=MPL_AX_BG,
            fg=TEXT_MAIN,
            insertbackground=ACCENT,
            selectbackground=ACCENT,
            selectforeground=BG_DARK,
            borderwidth=0,
            relief="flat",
            padx=8,
            pady=6,
        )
        y_scroll = ttk.Scrollbar(right, orient="vertical", command=self.distance_text.yview)
        x_scroll = ttk.Scrollbar(right, orient="horizontal", command=self.distance_text.xview)
        self.distance_text.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
        self.distance_text.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")

        # Configurar tags de color para el texto de distancias
        self.distance_text.tag_configure("header",  foreground=ACCENT,  font=("Consolas", 9, "bold"))
        self.distance_text.tag_configure("section", foreground=ACCENT2, font=("Consolas", 9, "bold"))
        self.distance_text.tag_configure("hit",     foreground=SUCCESS,  font=("Consolas", 9, "bold"))
        self.distance_text.tag_configure("dim",     foreground=TEXT_DIM)

        # ── Barra de controles ────────────────────────────────────────────────
        controls_outer = tk.Frame(self.root, bg=BG_PANEL)
        controls_outer.grid(row=3, column=0, sticky="ew")
        controls_outer.columnconfigure(0, weight=1)

        # Línea divisoria
        tk.Frame(controls_outer, bg=BORDER, height=1).grid(row=0, column=0, sticky="ew")

        controls = tk.Frame(controls_outer, bg=BG_PANEL)
        controls.grid(row=1, column=0, sticky="ew", padx=12, pady=8)
        controls.columnconfigure(9, weight=1)

        # Botones de navegación
        self.prev_button = ttk.Button(
            controls, text="◀  Anterior", command=self.previous_case, style="Nav.TButton"
        )
        self.next_button = ttk.Button(
            controls, text="Siguiente  ▶", command=self.next_case, style="Nav.TButton"
        )
        self.auto_button = ttk.Button(
            controls, text="▷  Auto: OFF", command=self.toggle_auto
        )
        self.confusion_button = ttk.Button(
            controls, text="⊞  Matriz de confusión", command=self.show_confusion_matrix
        )
        self.exit_button = ttk.Button(
            controls, text="✕  Salir", command=self.root.destroy, style="Exit.TButton"
        )

        self.prev_button.grid(row=0, column=0, padx=(0, 4))
        self.next_button.grid(row=0, column=1, padx=4)
        self.auto_button.grid(row=0, column=2, padx=4)

        # Separador visual
        tk.Frame(controls, bg=BORDER, width=1, height=28).grid(
            row=0, column=3, padx=8, pady=2
        )

        # Selector de métrica con etiqueta
        tk.Label(controls, text="Métrica:", bg=BG_PANEL, fg=TEXT_DIM,
                 font=("Segoe UI", 10)).grid(row=0, column=4, padx=(4, 2))

        metric_name = getattr(self.model, "metric", "euclidean")
        self.metric_combo = ttk.Combobox(
            controls,
            values=["euclidean", "manhattan", "cosine"],
            state="readonly",
            width=11,
        )
        self.metric_combo.set(metric_name)
        self.metric_combo.bind("<<ComboboxSelected>>", self.change_metric)
        self.metric_combo.grid(row=0, column=5, padx=4)

        # Selector de K con etiqueta
        tk.Label(controls, text="k:", bg=BG_PANEL, fg=TEXT_DIM,
                 font=("Segoe UI", 10)).grid(row=0, column=6, padx=(8, 2))

        self.k_combo = ttk.Combobox(
            controls,
            values=[str(kv) for kv in self.k_values],
            state="readonly",
            width=3,
        )
        self.k_combo.set(str(self.k))
        self.k_combo.bind("<<ComboboxSelected>>", self.change_k)
        self.k_combo.grid(row=0, column=7, padx=4)

        # Separador visual
        tk.Frame(controls, bg=BORDER, width=1, height=28).grid(
            row=0, column=8, padx=8, pady=2
        )

        # Botón de zoom a vecinos
        self.zoom_button = ttk.Button(
            controls, text="🔍 Zoom: OFF", command=self.toggle_zoom
        )
        self.zoom_button.grid(row=0, column=9, padx=4)

        self.confusion_button.grid(row=0, column=10, padx=4)
        self.exit_button.grid(row=0, column=11, padx=(4, 0))

        # Label de estado (alineado a la derecha)
        self.status_label = tk.Label(
            controls, text="",
            bg=BG_PANEL, fg=ACCENT2,
            font=("Segoe UI", 10, "bold"),
            anchor="e",
        )
        self.status_label.grid(row=0, column=12, sticky="e", padx=(12, 0))

    # ── Lógica de dibujo (SIN CAMBIOS) ────────────────────────────────────────

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
        if self.auto_play:
            self.auto_button.configure(text="⏸  Auto: ON")
        else:
            self.auto_button.configure(text="▷  Auto: OFF")

        if self.auto_play:
            self._schedule_auto()
        else:
            self._stop_auto()

    def toggle_zoom(self):
        """Activa o desactiva el zoom centrado en el vecindario."""
        self.zoom_mode = not self.zoom_mode
        if self.zoom_mode:
            self.zoom_button.configure(text="🔍 Zoom: ON")
        else:
            self.zoom_button.configure(text="🔍 Zoom: OFF")
        self.draw_case(self.current_index)

    def show_confusion_matrix(self):
        """Abre la matriz en una ventana aparte para que no aplaste la vista principal."""
        if self.confusion_window is not None and self.confusion_window.winfo_exists():
            self.confusion_window.lift()
            return

        self.confusion_window = tk.Toplevel(self.root)
        self.confusion_window.title("Matriz de confusión global")
        self.confusion_window.geometry("860x650")
        self.confusion_window.configure(bg=BG_DARK)

        figure, axis = plt.subplots(figsize=(8.2, 5.8), dpi=100)
        figure.patch.set_facecolor(MPL_BG)
        axis.set_facecolor(MPL_AX_BG)
        image = axis.imshow(self.matrix, cmap="Blues", interpolation="nearest")
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        axis.set_title(
            f"Exactitud test: {self.accuracy * 100:.2f}%  |  "
            f"Error global: {(1 - self.accuracy) * 100:.2f}%",
            color=TEXT_MAIN,
        )
        axis.set_xlabel("Dígito predicho", color=TEXT_DIM)
        axis.set_ylabel("Dígito real", color=TEXT_DIM)
        axis.set_xticks(range(10))
        axis.set_yticks(range(10))
        axis.tick_params(colors=TEXT_DIM)
        for spine in axis.spines.values():
            spine.set_edgecolor(BORDER)

        threshold = self.matrix.max() / 2 if self.matrix.max() else 0
        for row in range(10):
            for col in range(10):
                color = "white" if self.matrix[row, col] > threshold else "black"
                axis.text(col, row, str(self.matrix[row, col]),
                          ha="center", va="center", color=color)

        figure.tight_layout()
        canvas = FigureCanvasTkAgg(figure, master=self.confusion_window)
        canvas.get_tk_widget().configure(bg=BG_PANEL, highlightthickness=0)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        canvas.draw_idle()

        tk.Label(
            self.confusion_window,
            text=(
                "Filas = dígito real  |  Columnas = dígito predicho  |  "
                "La diagonal principal son los aciertos."
            ),
            bg=BG_DARK, fg=TEXT_DIM,
            font=("Segoe UI", 9),
        ).pack(fill="x", padx=12, pady=8)

    def _draw_map(self, case_index, neighbor_indices, prediction, real_label):
        """Muestra la nube de puntos, los vecinos y las conexiones en 3D."""
        self.map_axis.clear()
        self.map_axis.set_facecolor(MPL_AX_BG)
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
        neighbor_labels = self.data["y_train"][neighbor_indices]

        # Dibujar los puntos reales de los vecinos con color sólido y más grandes
        for pt, lbl in zip(neighbor_points, neighbor_labels):
            self.map_axis.scatter(
                pt[0], pt[1], pt[2],
                s=60,                       # Más grande que el s=14 de los puntos de fondo
                color=colors[lbl],          # Su color original de clase
                alpha=1.0,                  # Totalmente opaco (sin transparencia)
                edgecolors="white",         # Borde blanco
                linewidths=0.8,
                zorder=5
            )

        for point in neighbor_points:
            self.map_axis.plot(
                [query_point[0], point[0]],
                [query_point[1], point[1]],
                [query_point[2], point[2]],
                color=ACCENT,
                alpha=0.35,
                linewidth=1,
            )

        self.map_axis.scatter(
            neighbor_points[:, 0],
            neighbor_points[:, 1],
            neighbor_points[:, 2],
            s=115,
            facecolors="none",
            edgecolors=ACCENT,
            linewidths=1.8,
            label="vecinos",
        )
        self.map_axis.scatter(
            query_point[0],
            query_point[1],
            query_point[2],
            marker="*",
            s=280,
            color=ACCENT2,
            edgecolors="white",
            linewidths=0.8,
            label="nuevo dato",
        )

        # Radio de búsqueda: círculo punteado hasta el vecino más lejano
        if len(neighbor_points) > 0:
            diffs = neighbor_points - query_point
            radius = float(np.max(np.sqrt(np.sum(diffs * diffs, axis=1))))
            theta = np.linspace(0, 2 * np.pi, 120)
            cx, cy, cz = query_point
            circle_x = cx + radius * np.cos(theta)
            circle_y = cy + radius * np.sin(theta)
            circle_z = np.full_like(theta, cz)
            self.map_axis.plot(
                circle_x, circle_y, circle_z,
                linestyle="--",
                color=TEXT_DIM,
                alpha=0.45,
                linewidth=1.1,
                label=f"radio k={self.k}",
            )

        # Si está activo el zoom, recortar los ejes alrededor del vecindario
        if self.zoom_mode and len(neighbor_points) > 0:
            cx, cy, cz = query_point
            margin = radius * 1.1
            self.map_axis.set_xlim(cx - margin, cx + margin)
            self.map_axis.set_ylim(cy - margin, cy + margin)
            self.map_axis.set_zlim(cz - margin, cz + margin)

        status = "correcto ✓" if prediction == real_label else "error ✗"
        color_status = SUCCESS if prediction == real_label else ERROR_COL
        zoom_tag = "  [ZOOM ON]" if self.zoom_mode else ""
        self.map_axis.set_title(
            f"PCA 3D  |  real={real_label}  pred={prediction}  ({status}){zoom_tag}",
            color=color_status,
            fontsize=10,
        )
        self.map_axis.set_xlabel("PC 1", color=TEXT_DIM, fontsize=8)
        self.map_axis.set_ylabel("PC 2", color=TEXT_DIM, fontsize=8)
        self.map_axis.set_zlabel("PC 3", color=TEXT_DIM, fontsize=8)
        self.map_axis.tick_params(colors=TEXT_DIM, labelsize=7)
        self.map_axis.xaxis.pane.fill = False
        self.map_axis.yaxis.pane.fill = False
        self.map_axis.zaxis.pane.fill = False
        self.map_axis.xaxis.pane.set_edgecolor(BORDER)
        self.map_axis.yaxis.pane.set_edgecolor(BORDER)
        self.map_axis.zaxis.pane.set_edgecolor(BORDER)
        self.map_axis.grid(alpha=0.15, color=MPL_GRID)
        self.map_axis.legend(loc="upper right", ncol=2, fontsize=7,
                             facecolor=BG_PANEL, edgecolor=BORDER,
                             labelcolor=TEXT_DIM)
        self.map_figure.tight_layout()
        self.map_canvas.draw_idle()

    def _draw_digit(self, case_index, prediction, real_label, confidence):
        """Muestra el numero que esta entrando al KNN."""
        self.image_axis.clear()
        self.image_figure.patch.set_facecolor(MPL_BG)
        self.image_axis.set_facecolor(MPL_AX_BG)
        self.image_axis.imshow(self.data["raw_test"][case_index], cmap="gray")
        correct = prediction == real_label
        title_color = SUCCESS if correct else ERROR_COL
        self.image_axis.set_title(
            f"Real: {real_label}  →  Pred: {prediction}  ({confidence:.0%})",
            color=title_color, fontsize=9, fontweight="bold",
        )
        self.image_axis.axis("off")
        self.image_figure.tight_layout()
        self.image_canvas.draw_idle()

    def _draw_probabilities(self, probabilities):
        """Muestra los votos normalizados como probabilidad."""
        self.prob_axis.clear()
        self.prob_figure.patch.set_facecolor(MPL_BG)
        self.prob_axis.set_facecolor(MPL_AX_BG)
        digits = np.arange(10)
        bar_colors = [SUCCESS if p == probabilities.max() else "#2d4a6b"
                      for p in probabilities]
        bars = self.prob_axis.bar(digits, probabilities, color=bar_colors,
                                  edgecolor=BORDER, linewidth=0.5)
        # Etiqueta de valor en barras con valor > 0
        for bar, prob in zip(bars, probabilities):
            if prob > 0:
                self.prob_axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    f"{prob:.0%}",
                    ha="center", va="bottom",
                    color=TEXT_DIM, fontsize=7,
                )
        self.prob_axis.set_xlabel("Dígito", color=TEXT_DIM, fontsize=8)
        self.prob_axis.set_ylabel("votos / k", color=TEXT_DIM, fontsize=8)
        self.prob_axis.set_xticks(digits)
        self.prob_axis.tick_params(colors=TEXT_DIM, labelsize=8)
        self.prob_axis.set_ylim(0, 1.15)
        self.prob_axis.grid(axis="y", alpha=0.15, color=MPL_GRID)
        for spine in self.prob_axis.spines.values():
            spine.set_edgecolor(BORDER)
        self.prob_figure.tight_layout()
        self.prob_canvas.draw_idle()

    def _write_distance_panel(self, case_index, result, distances):
        """Escribe las distancias mas importantes en el panel derecho."""
        ordered = np.argsort(distances)
        query = self.data["x_test"][case_index]
        class_dists = self.model.class_distances(query)
        closest_class = min(class_dists, key=class_dists.get)
        metric_name = getattr(self.model, "metric", "euclidean")
        k_candidates = getattr(self.model, "k_candidates", None) or self.k_values or [self.k]
        k_scores = getattr(self.model, "k_validation_scores", {})

        self.distance_text.configure(state="normal")
        self.distance_text.delete("1.0", "end")

        def ins(text, tag=None):
            if tag:
                self.distance_text.insert("end", text, tag)
            else:
                self.distance_text.insert("end", text)

        ins(
            f"Consulta {case_index + 1}/{self.total_cases}  |  "
            f"k elegido={self.k}  |  {metric_name}\n",
            "header",
        )
        ins("-" * 48 + "\n", "dim")

        ins("\nSeleccion de k por validacion:\n", "section")
        for candidate_k in k_candidates:
            score = k_scores.get(candidate_k)
            tag = "hit" if candidate_k == self.k else "dim"
            prefix = ">>" if candidate_k == self.k else "  "
            if score is None:
                ins(f"{prefix} k={candidate_k:<2d}: sin dato\n", tag)
            else:
                ins(f"{prefix} k={candidate_k:<2d}: {score * 100:5.2f}%\n", tag)

        ins("\nProceso KNN:\n", "section")
        ins(f"  1. Distancia {metric_name} en 784 píxeles.\n", "dim")
        ins("  2. Ordenar distancias de menor a mayor.\n", "dim")
        ins("  3. Tomar los k vecinos más cercanos.\n", "dim")
        ins("  4. Votar por clase.\n", "dim")

        ins("\nDistancia al prototipo de cada clase:\n", "section")
        ins(f"  (más cercano por prototipo: clase {closest_class})\n", "dim")

        for clase, dist in class_dists.items():
            if clase == closest_class:
                ins(f"  ▶▶ clase {clase}: {dist:.4f}\n", "hit")
            else:
                ins(f"     clase {clase}: {dist:.4f}\n", "dim")

        ins("\nVecinos usados:\n", "section")
        for rank, train_index in enumerate(result["neighbor_indices"], start=1):
            label = self.data["y_train"][train_index]
            distance = distances[train_index]
            ins(f"  {rank:02d}. train[{train_index:04d}] y={label} d={distance:.4f}\n")

        ins("\nDistancias ordenadas:\n", "section")
        neighbor_set = set(result["neighbor_indices"])
        for rank, train_index in enumerate(ordered[: self.distance_lines], start=1):
            label = self.data["y_train"][train_index]
            distance = distances[train_index]
            if train_index in neighbor_set:
                ins(f"* {rank:03d}. train[{train_index:04d}] y={label} d={distance:.4f}\n", "hit")
            else:
                ins(f"  {rank:03d}. train[{train_index:04d}] y={label} d={distance:.4f}\n", "dim")

        if self.distance_lines < len(ordered):
            report_name = f"distancias_consulta_{case_index:03d}.txt"
            ins("\n", "dim")
            ins("Lista completa en:\n", "section")
            ins(f"KNN/output/{report_name}\n", "dim")

        self.distance_text.configure(state="disabled")

    def _update_status(self, real_label, prediction, confidence):
        """Actualiza el resumen inferior."""
        correct = real_label == prediction
        status = "✓ correcto" if correct else "✗ error"
        color = SUCCESS if correct else ERROR_COL
        text = (
            f"Consulta {self.current_index + 1}/{self.total_cases}   "
            f"Real: {real_label}   Pred: {prediction}   "
            f"Prob: {confidence:.0%}   {status}   "
            f"Exactitud: {self.accuracy * 100:.2f}%"
        )
        self.status_label.configure(text=text, fg=color)

    # ── Utilidades internas (SIN CAMBIOS en lógica) ───────────────────────────

    def _manual_pca_3d(self, x_train, x_test, max_plot=800):
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
        self.auto_button.configure(text="⏸  Auto: ON")
        self.auto_job = self.root.after(self.delay_ms, self._auto_step)

    def _auto_step(self):
        """Avanza mientras queden consultas."""
        if not self.auto_play:
            return

        if self.current_index >= self.total_cases - 1:
            self._stop_auto()
            self.auto_play = False
            self.auto_button.configure(text="▷  Auto: OFF")
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

    def change_metric(self, event=None):
        """Cambia la metrica de distancia de forma interactiva y recalcula."""
        new_metric = self.metric_combo.get()
        if new_metric == getattr(self.model, "metric", ""):
            return

        self.model.metric = new_metric
        self.root.title(f"KNN manual sobre MNIST  |  métrica: {new_metric}")

        # Desactivar temporalmente la ventana durante el calculo
        self.metric_combo.configure(state="disabled")
        self.root.configure(cursor="watch")
        self.root.update()

        try:
            # Recalcular matriz de confusion y exactitud para test con la nueva metrica
            _, _, self.matrix, self.accuracy = evaluate_model(
                self.model, self.data["x_test"], self.data["y_test"]
            )
            # Si la ventana de la matriz de confusion esta abierta, la cerramos para que se recree limpia
            if self.confusion_window is not None and self.confusion_window.winfo_exists():
                self.confusion_window.destroy()
                self.confusion_window = None
        finally:
            self.metric_combo.configure(state="readonly")
            self.root.configure(cursor="")

        # Redibujar la consulta actual con la nueva metrica
        self.draw_case(self.current_index)

    def change_k(self, event=None):
        """Cambia el valor de K de forma interactiva y recalcula."""
        try:
            new_k = int(self.k_combo.get())
        except ValueError:
            return
        if new_k == self.k:
            return

        self.k = new_k
        self.model.k = new_k

        # Desactivar temporalmente la ventana durante el calculo
        self.k_combo.configure(state="disabled")
        self.root.configure(cursor="watch")
        self.root.update()

        try:
            # Recalcular matriz de confusion y exactitud para test con el nuevo K
            _, _, self.matrix, self.accuracy = evaluate_model(
                self.model, self.data["x_test"], self.data["y_test"]
            )
            # Si la ventana de la matriz de confusion esta abierta, la cerramos para que se recree limpia
            if self.confusion_window is not None and self.confusion_window.winfo_exists():
                self.confusion_window.destroy()
                self.confusion_window = None
        finally:
            self.k_combo.configure(state="readonly")
            self.root.configure(cursor="")

        # Redibujar la consulta actual con el nuevo K
        self.draw_case(self.current_index)

    @staticmethod
    def _font_exists(font_name):
        """Comprueba si una fuente esta disponible en el sistema."""
        import tkinter.font as tkfont
        try:
            return font_name in tkfont.families()
        except Exception:
            return False

    @staticmethod
    def _make_panel(parent, title):
        """Crea un LabelFrame estilizado con fondo oscuro."""
        frame = tk.LabelFrame(
            parent,
            text=title,
            bg=BG_PANEL,
            fg=ACCENT,
            font=("Segoe UI", 10, "bold"),
            relief="flat",
            bd=1,
            highlightbackground=BORDER,
            highlightthickness=1,
            padx=4,
            pady=4,
        )
        return frame


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
