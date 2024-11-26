from PyQt6.QtGui import QDragEnterEvent, QDropEvent

from ui.functions.io import open_image



def label_drag_enter(qlabel, event: QDragEnterEvent):
    if event.mimeData().hasUrls():  # 如果拖动内容包含文件
        event.accept()
    else:
        event.ignore()


# QLabel 的 dropEvent
def label_drop(window, event: QDropEvent):
    if event.mimeData().hasUrls():  # 如果拖放的是文件
        files = [url.toLocalFile() for url in event.mimeData().urls()]
        open_image(window, "\n".join(files))
    else:
        event.ignore()
