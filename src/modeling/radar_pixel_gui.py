import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
import time

# Radarbild auf 3000x1000 vergrößern
radar_image = np.random.randn(1000, 3000)  # Zufällig generiertes Radarbild (1000x3000)

# Vorberechnung der Verteilungen für jeden Pixel
gradient_image = np.gradient(radar_image)
absolute_gradient = np.sqrt(gradient_image[0] ** 2 + gradient_image[1] ** 2)
variation_image = np.var(radar_image, axis=0)
mean_image = np.mean(radar_image, axis=0)


class RadarGUI(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Radargramm Amplitudenanalyse")
        self.geometry("1000x800")

        # Protokoll für das Schließen des Fensters
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        # Erstellen des Canvas, auf dem das Radargramm angezeigt wird
        self.fig, self.ax = plt.subplots(figsize=(10, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, self)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Add this line:
        self.fig.canvas.mpl_connect("resize_event", self.on_resize)

        # Initialize plot elements and variables
        self.rect_artist = None
        self.background = None
        self._last_hist_update = 0

        # Frame für die Kontrollelemente
        self.control_frame = ttk.Frame(self)
        self.control_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        # Parameter Dropdown-Menü
        self.parameter_var = tk.StringVar(value="Amplitudenverteilung")

        self.ax.imshow(radar_image, cmap="gray", origin="lower")
        self.ax.set_title("Radargramm")
        self.canvas.draw()

        # Cache the background
        self.background = self.fig.canvas.copy_from_bbox(self.ax.bbox)

        # Fenster für Verteilungen
        self.dist_fig, self.dist_ax = plt.subplots(figsize=(10, 3))
        self.dist_canvas = FigureCanvasTkAgg(self.dist_fig, self)
        self.dist_canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        self.dist_ax.set_title("Verteilung")

        # Dropdown-Menü für Parameterwahl (verwendet die gleiche Variable)
        self.parameter_menu = ttk.Combobox(
            self.control_frame, textvariable=self.parameter_var, state="readonly"
        )
        self.parameter_menu["values"] = [
            "Amplitudenverteilung",
            "Absoluter Gradient",
            "Variation",
            "Mittelwert",
        ]
        self.parameter_menu.pack(side=tk.LEFT, padx=10, pady=5)

        # Dropdown-Menü für Fenstergrößen
        self.field_size_var = tk.StringVar(
            value="14"
        )  # Voreinstellung: 29x29 (Abstand 14)
        self.field_size_menu = ttk.Combobox(
            self.control_frame, textvariable=self.field_size_var, state="readonly"
        )
        # Generiere Liste von Abständen von 1 bis 100
        self.field_size_menu["values"] = list(range(2, 101))
        self.field_size_menu.pack(side=tk.RIGHT, padx=10, pady=5)

        # Label für die Fenstergröße
        self.field_size_label = ttk.Label(
            self.control_frame, text="Fensterabstand: 14 (29x29)"
        )
        self.field_size_label.pack(side=tk.RIGHT, padx=10, pady=5)

        # Binde Event an Größenänderung
        self.field_size_menu.bind("<<ComboboxSelected>>", self.update_field_size_label)

        # Event-Handler für Mausklick und Bewegung
        self.cid_press = self.fig.canvas.mpl_connect(
            "button_press_event", self.on_press
        )
        self.cid_release = self.fig.canvas.mpl_connect(
            "button_release_event", self.on_release
        )
        self.cid_motion = self.fig.canvas.mpl_connect(
            "motion_notify_event", self.on_motion
        )

        self.is_dragging = False  # Zustand der Maus (gedrückt oder nicht)

    def update_field_size_label(self, event=None):
        size = int(self.field_size_var.get())
        window_size = 2 * size + 1
        self.field_size_label.config(
            text=f"Fensterabstand: {size} ({window_size}x{window_size})"
        )

    def on_press(self, event):
        if event.xdata is not None and event.ydata is not None:
            self.is_dragging = True
            self.update_window(event)

    def on_release(self, event):
        self.is_dragging = False

    def on_motion(self, event):
        if self.is_dragging:
            self.update_window(event)

    def on_field_size_change(self):
        """Handler für Änderungen der Fenstergröße"""
        # Update das Label
        self.update_field_size_label()

        # Aktualisiere die Verteilung mit der aktuellen Mausposition
        if hasattr(self, "last_event") and self.last_event is not None:
            self.update_window(self.last_event)

    def update_window(self, event):
        if event.xdata is None or event.ydata is None:
            return

        # Store current event
        self.last_event = event

        x, y = int(event.xdata), int(event.ydata)
        size = int(self.field_size_var.get())

        if (
            x - size >= 0
            and x + size < radar_image.shape[1]
            and y - size >= 0
            and y + size < radar_image.shape[0]
        ):
            current_time = time.time()
            if not hasattr(self, "_last_hist_update"):
                self._last_hist_update = 0

            # Update histogram less frequently
            if current_time - self._last_hist_update > 0.1:
                self._last_hist_update = current_time
                self.update_histogram(x, y, size)

            # Clear previous rectangle
            if self.rect_artist is not None:
                self.rect_artist.remove()
                self.rect_artist = None

            # Force redraw periodically
            if not hasattr(self, "_last_full_redraw"):
                self._last_full_redraw = 0

            if current_time - self._last_full_redraw > 1.0:
                self._last_full_redraw = current_time
                self.ax.draw(self.canvas.renderer)
                self.background = self.fig.canvas.copy_from_bbox(self.ax.bbox)

            # Create new rectangle
            self.rect_artist = plt.Rectangle(
                (x - size, y - size),
                2 * size + 1,
                2 * size + 1,
                linewidth=2,
                edgecolor="r",
                facecolor="none",
            )
            self.ax.add_patch(self.rect_artist)

            # Update display
            self.canvas.draw_idle()
            self.canvas.flush_events()

    def update_histogram(self, x, y, size):
        self.dist_ax.clear()
        parameter = self.parameter_var.get()

        variations, means = self.calculate_window_statistics(x, y, size)

        if parameter == "Variation":
            self.dist_ax.hist(
                variations,
                bins=20,
                color="r",
                density=True,
                alpha=0.6,
            )
            self.dist_ax.set_title(f"Variationen der Pixel im Fenster um ({x}, {y})")
        elif parameter == "Mittelwert":
            self.dist_ax.hist(
                means,
                bins=20,
                color="y",
                density=True,
                alpha=0.6,
            )
            self.dist_ax.set_title(f"Mittelwerte der Pixel im Fenster um ({x}, {y})")
        elif parameter == "Amplitudenverteilung":
            window = radar_image[y - size : y + size + 1, x - size : x + size + 1]
            self.dist_ax.hist(
                window.flatten(), bins=20, color="g", density=True, alpha=0.6
            )
            self.dist_ax.set_title(f"Amplitudenverteilung um Pixel ({x}, {y})")
        elif parameter == "Absoluter Gradient":
            grad_window = absolute_gradient[
                y - size : y + size + 1, x - size : x + size + 1
            ]
            self.dist_ax.hist(
                grad_window.flatten(),
                bins=20,
                color="b",
                density=True,
                alpha=0.6,
            )
            self.dist_ax.set_title(f"Absoluter Gradient um Pixel ({x}, {y})")

        self.dist_canvas.draw()

    def update_distribution_view(self, event=None):
        parameter = self.parameter_var.get()  # Nutze die gemeinsame Variable
        self.dist_ax.clear()

        if parameter == "Amplitudenverteilung":
            self.dist_ax.hist(
                radar_image.flatten(), bins=20, color="g", density=True, alpha=0.6
            )
            self.dist_ax.set_title("Amplitudenverteilung")
        elif parameter == "Absoluter Gradient":
            self.dist_ax.hist(
                absolute_gradient.flatten(), bins=20, color="b", density=True, alpha=0.6
            )
            self.dist_ax.set_title("Absoluter Gradient")
        elif parameter == "Variation":
            self.dist_ax.hist(
                variation_image.flatten(), bins=20, color="r", density=True, alpha=0.6
            )
            self.dist_ax.set_title("Variation")
        elif parameter == "Mittelwert":
            self.dist_ax.hist(
                mean_image.flatten(), bins=20, color="y", density=True, alpha=0.6
            )
            self.dist_ax.set_title("Mittelwert")

        self.dist_canvas.draw()

    def on_close(self):
        # Trenne alle matplotlib-Event-Handler
        self.fig.canvas.mpl_disconnect(self.cid_press)
        self.fig.canvas.mpl_disconnect(self.cid_release)
        self.fig.canvas.mpl_disconnect(self.cid_motion)

        # Schließe die matplotlib-Figuren
        plt.close(self.fig)
        plt.close(self.dist_fig)

        # Beende das Tkinter-Fenster
        self.destroy()

        # Beende den Python-Prozess, falls noch aktiv
        self.quit()

    def calculate_window_statistics(self, x, y, size):
        """Berechnet Variation und Mittelwerte für alle Pixel im Fenster"""
        variations = []
        means = []

        # Für jeden Pixel im Fenster
        for i in range(2 * size + 1):
            for j in range(2 * size + 1):
                pixel_y = y - size + i
                pixel_x = x - size + j

                # Überprüfe Randbedinungen
                if (
                    pixel_x - size >= 0
                    and pixel_x + size < radar_image.shape[1]
                    and pixel_y - size >= 0
                    and pixel_y + size < radar_image.shape[0]
                ):
                    # Berechne Fenster um aktuellen Pixel
                    pixel_window = radar_image[
                        pixel_y - size : pixel_y + size + 1,
                        pixel_x - size : pixel_x + size + 1,
                    ]
                    variations.append(np.var(pixel_window))
                    means.append(np.mean(pixel_window))

        return np.array(variations), np.array(means)

    def on_resize(self, event):
        # Update background cache on window resize
        self.ax.draw(self.canvas.renderer)
        self.background = self.fig.canvas.copy_from_bbox(self.ax.bbox)


# Hauptprogramm starten
if __name__ == "__main__":
    app = RadarGUI()
    app.mainloop()
