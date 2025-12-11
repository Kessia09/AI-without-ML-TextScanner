"""
Tkinter-based OCR app (same behavior/deps as text-reader.py) with:
- Image load and camera capture
- ROI selection (click + drag)
- Run OCR with overlay showing detected text
- Extracted text display panel
"""
import cv2
import pytesseract
from pytesseract import Output
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk
class OCRApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Kessi OCR Panel (Split UI)")
        self.canvas_width = 900
        self.canvas_height = 600
        self.current_image = None
        self.video_capture = None
        self.running_camera = False
        self.pause_live_preview = False
        self.display_image_meta = None
        self.roi_start = None
        self.roi_box = None  # (x1, y1, x2, y2) in image coords
        self.photo_image = None
        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
    # ---------------- UI ----------------
    def _build_ui(self) -> None:
        main = tk.Frame(self.root, bg="#1e1f22")
        main.pack(fill=tk.BOTH, expand=True)
        # Left: image canvas
        canvas_frame = tk.Frame(main, bg="#1e1f22", padx=8, pady=8)
        canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(
            canvas_frame,
            width=self.canvas_width,
            height=self.canvas_height,
            bg="#0f0f10",
            highlightthickness=1,
            highlightbackground="#444",
            cursor="crosshair",
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_press)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_release)
        # Right: controls + text output stacked vertically
        side_panel = tk.Frame(main, width=340, bg="#25262b", padx=8, pady=8)
        side_panel.pack(side=tk.LEFT, fill=tk.Y)
        side_panel.pack_propagate(False)
        tk.Label(
            side_panel, text="Controls", fg="white", bg="#25262b", anchor="w", font=("Arial", 11, "bold")
        ).pack(fill=tk.X, pady=(0, 4))
        btn_opts = {"fill": tk.X, "padx": 0, "pady": 3}
        tk.Button(side_panel, text="Load Image", command=self.load_image).pack(**btn_opts)
        tk.Button(side_panel, text="Start/Resume Camera", command=self.start_camera).pack(**btn_opts)
        tk.Button(side_panel, text="Stop Camera", command=self.stop_camera).pack(**btn_opts)
        tk.Button(side_panel, text="Run OCR", command=self.run_ocr).pack(**btn_opts)
        tk.Button(side_panel, text="Clear ROI", command=self.clear_roi).pack(**btn_opts)
        tk.Label(
            side_panel,
            text="Tip: drag on the image to set ROI.\nOverlay appears after OCR.",
            fg="#d0d0d0",
            bg="#25262b",
            justify=tk.LEFT,
            wraplength=300,
        ).pack(fill=tk.X, pady=(6, 6))
        tk.Label(
            side_panel, text="Extracted Text", fg="white", bg="#25262b", anchor="w", font=("Arial", 11, "bold")
        ).pack(fill=tk.X, pady=(4, 2))
        self.text_output = scrolledtext.ScrolledText(side_panel, height=12, wrap=tk.WORD)
        self.text_output.pack(fill=tk.BOTH, expand=True, pady=(0, 6))
        self.status_var = tk.StringVar(value="Load an image or start the camera.")
        tk.Label(side_panel, textvariable=self.status_var, fg="#7cd4ff", bg="#25262b", anchor="w").pack(
            fill=tk.X, pady=(4, 0)
        )
    # -------------- Utility --------------
    def _cv_to_tk_image(self, cv_img: np.ndarray):
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        img_h, img_w = pil_img.height, pil_img.width
        scale = min(self.canvas_width / img_w, self.canvas_height / img_h)
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)
        resized = pil_img.resize((new_w, new_h), Image.LANCZOS)
        offset_x = (self.canvas_width - new_w) // 2
        offset_y = (self.canvas_height - new_h) // 2
        self.display_image_meta = {
            "scale": scale,
            "offset_x": offset_x,
            "offset_y": offset_y,
            "img_w": img_w,
            "img_h": img_h,
        }
        self.photo_image = ImageTk.PhotoImage(resized)
        return self.photo_image, offset_x, offset_y
    def _canvas_to_image_coords(self, x: float, y: float):
        if not self.display_image_meta:
            return None
        m = self.display_image_meta
        img_x = int((x - m["offset_x"]) / m["scale"])
        img_y = int((y - m["offset_y"]) / m["scale"])
        if 0 <= img_x < m["img_w"] and 0 <= img_y < m["img_h"]:
            return img_x, img_y
        return None
    def _image_to_canvas_coords(self, x: float, y: float):
        m = self.display_image_meta
        canvas_x = x * m["scale"] + m["offset_x"]
        canvas_y = y * m["scale"] + m["offset_y"]
        return canvas_x, canvas_y
    def display_image(self, cv_img: np.ndarray) -> None:
        if cv_img is None:
            return
        tk_img, offset_x, offset_y = self._cv_to_tk_image(cv_img)
        self.canvas.delete("all")
        self.canvas.create_image(offset_x, offset_y, image=tk_img, anchor=tk.NW, tags="img")
        self.draw_roi_box()
    def draw_roi_box(self) -> None:
        self.canvas.delete("roi")
        if not self.roi_box or not self.display_image_meta:
            return
        x1, y1, x2, y2 = self.roi_box
        c1 = self._image_to_canvas_coords(x1, y1)
        c2 = self._image_to_canvas_coords(x2, y2)
        self.canvas.create_rectangle(*c1, *c2, outline="yellow", width=2, tags="roi")
    # -------------- Mouse events --------------
    def on_mouse_press(self, event) -> None:
        coords = self._canvas_to_image_coords(event.x, event.y)
        if coords:
            self.roi_start = coords
            self.roi_box = None
            self.draw_roi_box()
    def on_mouse_drag(self, event) -> None:
        if not self.roi_start:
            return
        coords = self._canvas_to_image_coords(event.x, event.y)
        if coords:
            x0, y0 = self.roi_start
            x1, y1 = coords
            self.roi_box = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
            self.draw_roi_box()
    def on_mouse_release(self, event) -> None:
        if not self.roi_start:
            return
        coords = self._canvas_to_image_coords(event.x, event.y)
        if coords:
            x0, y0 = self.roi_start
            x1, y1 = coords
            self.roi_box = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
        self.roi_start = None
        self.draw_roi_box()
    # -------------- Camera --------------
    def start_camera(self) -> None:
        if self.video_capture is None:
            self.video_capture = cv2.VideoCapture(0)
            if not self.video_capture.isOpened():
                self.video_capture.release()
                self.video_capture = None
                messagebox.showerror("Camera Error", "Cannot open camera.")
                return
        self.running_camera = True
        self.pause_live_preview = False
        self.status_var.set("Camera running. Drag to set ROI or click Run OCR.")
        self.update_camera_frame()
    def stop_camera(self) -> None:
        self.running_camera = False
        self.pause_live_preview = False
        if self.video_capture is not None:
            self.video_capture.release()
            self.video_capture = None
        self.status_var.set("Camera stopped.")
    def update_camera_frame(self) -> None:
        if not self.running_camera or self.pause_live_preview:
            return
        if self.video_capture is None:
            return
        ret, frame = self.video_capture.read()
        if not ret:
            self.status_var.set("Failed to read from camera.")
            return
        self.current_image = frame
        self.display_image(frame)
        self.root.after(30, self.update_camera_frame)
    # -------------- OCR --------------
    @staticmethod
    def preprocess_for_ocr(img: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, d=5, sigmaColor=50, sigmaSpace=50)
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            10,
        )
        return binary
    def _prepare_roi(self):
        if self.current_image is None:
            return None, None
        h, w = self.current_image.shape[:2]
        if not self.roi_box:
            return self.current_image.copy(), (0, 0)
        x1, y1, x2, y2 = self.roi_box
        x1 = max(0, min(w - 1, int(x1)))
        x2 = max(0, min(w, int(x2)))
        y1 = max(0, min(h - 1, int(y1)))
        y2 = max(0, min(h, int(y2)))
        if x2 <= x1 or y2 <= y1:
            return self.current_image.copy(), (0, 0)
        roi_img = self.current_image[y1:y2, x1:x2].copy()
        return roi_img, (x1, y1)
    def run_ocr(self) -> None:
        if self.current_image is None:
            messagebox.showwarning("No Image", "Load an image or start the camera first.")
            return
        if self.running_camera:
            self.pause_live_preview = True
        roi_img, (offset_x, offset_y) = self._prepare_roi()
        if roi_img is None:
            messagebox.showwarning("No Image", "Load an image or start the camera first.")
            return
        processed = self.preprocess_for_ocr(roi_img)
        ocr_config = "--psm 6"
        try:
            ocr_text = pytesseract.image_to_string(processed, config=ocr_config)
            data = pytesseract.image_to_data(processed, output_type=Output.DICT, config=ocr_config)
        except Exception as e:
            messagebox.showerror("OCR Error", f"OCR failed: {e}")
            return
        annotated = self.current_image.copy()
        n = len(data["text"])
        for i in range(n):
            text = data["text"][i].strip()
            conf = float(data["conf"][i]) if data["conf"][i] != "-1" else -1.0
            if not text or conf < 10:
                continue
            x = data["left"][i] + offset_x
            y = data["top"][i] + offset_y
            w = data["width"][i]
            h = data["height"][i]
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                annotated,
                text,
                (x, max(0, y - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )
        if self.roi_box:
            x1, y1, x2, y2 = self.roi_box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 1)
        self.text_output.delete("1.0", tk.END)
        stripped = ocr_text.strip()
        if stripped:
            self.text_output.insert(tk.END, stripped)
            self.status_var.set("OCR complete. Overlay shows detected text. Resume camera if needed.")
        else:
            self.text_output.insert(tk.END, "[No text detected]")
            self.status_var.set("No text detected. Try a larger ROI, better lighting, or move closer.")
        self.display_image(annotated)
    # -------------- Misc --------------
    def clear_roi(self) -> None:
        self.roi_box = None
        self.draw_roi_box()
        self.status_var.set("ROI cleared.")
    def load_image(self) -> None:
        file_path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
                ("All files", "*.*"),
            ],
        )
        if not file_path:
            return
        img = cv2.imread(file_path)
        if img is None:
            messagebox.showerror("Load Error", "Could not load the selected image.")
            return
        self.current_image = img
        self.pause_live_preview = False
        self.running_camera = False
        self.display_image(img)
        self.status_var.set("Image loaded. Select ROI or run OCR.")
    def on_close(self) -> None:
        self.stop_camera()
        self.root.destroy()
def main() -> None:
    root = tk.Tk()
    app = OCRApp(root)
    root.mainloop()
if __name__ == "__main__":
    main()
