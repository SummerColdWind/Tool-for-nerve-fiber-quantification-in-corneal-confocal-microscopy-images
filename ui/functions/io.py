
from PyQt6.QtWidgets import QFileDialog


from utils.common import load_image

def open_image(window, file_path=None):
    if file_path is None:
        file_path, _ = QFileDialog.getOpenFileName(
            None, "Select a image", "", "Image file (*.png *.jpg *.jpeg *.bmp *.gif)"
        )
    if file_path:
        try:
            window.raw_image = load_image(file_path)
            window.cur_image = None
            window.show_image()
            print(f"Image loaded successfully")
        except Exception as e:
            print(f"Image loading failed: {e}")


