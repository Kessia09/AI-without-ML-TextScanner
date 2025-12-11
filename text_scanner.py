import sys
import cv2
import numpy as np
from PIL import Image
import pytesseract
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QRect, QPoint
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QPen
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QTextEdit, QVBoxLayout, QHBoxLayout, QFileDialog, QMessageBox, QCheckBox

class CameraThread(QThread):
    frame_ready = pyqtSignal(np.ndarray)

    def __init__(self, camera_index=0):
        super().__init__()
        self.camera_index = camera_index
        self.running = False
        self.cap = None

    def run(self):
        self.cap = cv2.VideoCapture(self.camera_index)
        time.sleep(0.2)
        self.running = True
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            self.frame_ready.emit(frame)
            time.sleep(0.02)
        if self.cap is not None:
            self.cap.release()

    def stop(self):
        self.running = False
        self.wait()

class ImageLabel(QLabel):
    roi_changed = pyqtSignal(QRect)

    def __init__(self):
        super().__init__()
        self.setMouseTracking(True)
        self.dragging = False
        self.start_point = QPoint()
        self.end_point = QPoint()
        self.current_pixmap = None
        self.selection_enabled = False

    def enable_selection(self, enabled: bool):
        self.selection_enabled = enabled
        self.dragging = False
        self.start_point = QPoint()
        self.end_point = QPoint()
        self.update()

    def setPixmap(self, pm: QPixmap):
        super().setPixmap(pm)
        self.current_pixmap = pm

    def mousePressEvent(self, event):
        if not self.selection_enabled:
            return
        if event.button() == Qt.LeftButton:
            self.dragging = True
            self.start_point = event.pos()
            self.end_point = event.pos()
            self.update()

    def mouseMoveEvent(self, event):
        if not self.selection_enabled:
            return
        if self.dragging:
            self.end_point = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if not self.selection_enabled:
            return
        if event.button() == Qt.LeftButton and self.dragging:
            self.dragging = False
            self.end_point = event.pos()
            rect = QRect(self.start_point, self.end_point).normalized()
            self.roi_changed.emit(rect)
            self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.selection_enabled and not self.start_point.isNull() and not self.end_point.isNull():
            painter = QPainter(self)
            pen = QPen(QColor(0, 255, 0), 2, Qt.DashLine)
            painter.setPen(pen)
            rect = QRect(self.start_point, self.end_point).normalized()
            painter.drawRect(rect)
            painter.end()

class OCRApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('PyTesseract Printed Text Scanner')
        self.setGeometry(100, 100, 1200, 700)

        self.current_frame = None
        self.display_frame = None
        self.roi_rect = None

        self.image_label = ImageLabel()
        self.image_label.setStyleSheet('background-color: #222;')
        self.image_label.setMinimumSize(640, 480)
        self.image_label.roi_changed.connect(self.on_roi_changed)

        self.load_btn = QPushButton('Load Image')
        self.load_btn.clicked.connect(self.load_image)
        self.start_cam_btn = QPushButton('Start Camera')
        self.start_cam_btn.clicked.connect(self.start_camera)
        self.stop_cam_btn = QPushButton('Stop Camera')
        self.stop_cam_btn.clicked.connect(self.stop_camera)
        self.stop_cam_btn.setEnabled(False)
        self.select_roi_chk = QCheckBox('Select ROI')
        self.select_roi_chk.stateChanged.connect(self.toggle_roi_selection)
        self.ocr_btn = QPushButton('Run OCR')
        self.ocr_btn.clicked.connect(self.run_ocr)
        self.clear_btn = QPushButton('Clear OCR')
        self.clear_btn.clicked.connect(self.clear_results)

        self.text_output = QTextEdit()
        self.text_output.setReadOnly(True)
        self.text_output.setMinimumWidth(400)

        left_v = QVBoxLayout()
        left_v.addWidget(self.image_label)
        controls_h = QHBoxLayout()
        controls_h.addWidget(self.load_btn)
        controls_h.addWidget(self.start_cam_btn)
        controls_h.addWidget(self.stop_cam_btn)
        controls_h.addWidget(self.select_roi_chk)
        controls_h.addWidget(self.ocr_btn)
        controls_h.addWidget(self.clear_btn)
        left_v.addLayout(controls_h)

        main_h = QHBoxLayout()
        main_h.addLayout(left_v)
        main_h.addWidget(self.text_output)
        self.setLayout(main_h)

        self.cam_thread = None

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Open Image', '', 'Images (*.png *.jpg *.jpeg *.bmp)')
        if not path:
            return
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            QMessageBox.warning(self, 'Error', 'Could not read image')
            return
        self.current_frame = img
        self.display_frame = img.copy()
        self.roi_rect = None
        self.update_image_display(self.display_frame)
        self.text_output.clear()

    def start_camera(self):
        if self.cam_thread is not None and self.cam_thread.isRunning():
            return
        self.cam_thread = CameraThread(0)
        self.cam_thread.frame_ready.connect(self.on_frame_ready)
        self.cam_thread.start()
        self.start_cam_btn.setEnabled(False)
        self.stop_cam_btn.setEnabled(True)
        self.load_btn.setEnabled(False)

    def stop_camera(self):
        if self.cam_thread is not None:
            self.cam_thread.stop()
            self.cam_thread = None
        self.start_cam_btn.setEnabled(True)
        self.stop_cam_btn.setEnabled(False)
        self.load_btn.setEnabled(True)

    def toggle_roi_selection(self, state):
        enabled = (state == Qt.Checked)
        self.image_label.enable_selection(enabled)
        if not enabled:
            self.roi_rect = None

    def on_frame_ready(self, frame: np.ndarray):
        self.current_frame = frame
        self.display_frame = frame.copy()
        if self.roi_rect is not None:
            x, y, w, h = self.roi_rect
            cv2.rectangle(self.display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        self.update_image_display(self.display_frame)

    def on_roi_changed(self, rect: QRect):
        if self.current_frame is None:
            return
        label_w, label_h = self.image_label.width(), self.image_label.height()
        img_h, img_w = self.current_frame.shape[:2]
        x1 = int(rect.left() * img_w / label_w)
        y1 = int(rect.top() * img_h / label_h)
        x2 = int(rect.right() * img_w / label_w)
        y2 = int(rect.bottom() * img_h / label_h)
        x1, x2 = max(0, min(img_w-1, x1)), max(0, min(img_w-1, x2))
        y1, y2 = max(0, min(img_h-1, y1)), max(0, min(img_h-1, y2))
        if x2 - x1 > 5 and y2 - y1 > 5:
            self.roi_rect = (x1, y1, x2-x1, y2-y1)
        else:
            self.roi_rect = None
        if self.current_frame is not None:
            disp = self.current_frame.copy()
            if self.roi_rect:
                x, y, w, h = self.roi_rect
                cv2.rectangle(disp, (x, y), (x + w, y + h), (0, 255, 0), 2)
            self.update_image_display(disp)

    def update_image_display(self, frame_bgr: np.ndarray):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qt_image).scaled(self.image_label.width(), self.image_label.height(), Qt.KeepAspectRatio)
        self.image_label.setPixmap(pix)

    def run_ocr(self):
        if self.current_frame is None:
            QMessageBox.information(self, 'No image', 'Load an image or start the camera first')
            return
        img = self.current_frame.copy()
        if self.roi_rect is not None:
            x, y, w, h = self.roi_rect
            img = img[y:y+h, x:x+w]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 3)
        proc = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        pil_img = Image.fromarray(proc)
        try:
            data = pytesseract.image_to_data(pil_img, output_type=pytesseract.Output.DICT)
        except Exception as e:
            QMessageBox.warning(self, 'Tesseract error', f'OCR failed: {e}')
            return
        n_boxes = len(data['level'])
        overlay = img.copy()
        extracted_lines = []
        for i in range(n_boxes):
            text = data['text'][i].strip()
            conf = int(data['conf'][i]) if data['conf'][i].isdigit() else -1
            if text != "" and conf > 30:
                (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
                cv2.rectangle(overlay, (x, y), (x+w, y+h), (0, 255, 0), 2)
                org = (x, y-10 if y-10>10 else y+h+20)
                cv2.putText(overlay, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                extracted_lines.append(text)
        if self.roi_rect is not None:
            preview = self.current_frame.copy()
            x0, y0, w0, h0 = self.roi_rect
            if overlay.shape[0] != h0 or overlay.shape[1] != w0:
                overlay_resized = cv2.resize(overlay, (w0, h0))
            else:
                overlay_resized = overlay
            preview[y0:y0+h0, x0:x0+w0] = overlay_resized
            self.display_frame = preview
        else:
            self.display_frame = overlay
        self.update_image_display(self.display_frame)
        final_text = '\n'.join(extracted_lines)
        if final_text.strip() == '':
            try:
                final_text = pytesseract.image_to_string(pil_img).strip()
            except Exception:
                final_text = ''
        self.text_output.setPlainText(final_text)

    def clear_results(self):
        self.text_output.clear()
        self.roi_rect = None
        if self.current_frame is not None:
            self.update_image_display(self.current_frame)

    def closeEvent(self, event):
        if self.cam_thread is not None:
            self.cam_thread.stop()
            self.cam_thread = None
        event.accept()

def main():
    app = QApplication(sys.argv)
    window = OCRApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
