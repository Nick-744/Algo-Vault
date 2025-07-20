# PyQt5-based class for a simple drag-and-drop
# interface for image files. Enables users to
# dynamically input images by dragging them into the application!

from PyQt5.QtGui import (
    QPixmap, QDragEnterEvent, QDragMoveEvent, QDropEvent
)
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout
)
from PyQt5.QtCore import (Qt, pyqtSignal)
import sys

class _ImageLabel(QLabel):
    def __init__(self) -> None:
        super().__init__();
        
        self.setAlignment(Qt.AlignCenter)
        self.setText('\n\n Drop an image here... \n\n')
        self.setStyleSheet('''
            QLabel {
                border: 4px dashed #aaa;
                font-size: 24px;
                color: #666;
            }
        ''')

        return;
    
    def set_pixmap(self, image: QPixmap) -> None:
        # Scale the image to fit the label 
        # while maintaining aspect ratio!
        scaled_pixmap = image.scaled(
            self.size(), 
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        )
        super().setPixmap(scaled_pixmap)

        return;

class DragAndDropImage(QWidget):
    image_dropped = pyqtSignal(str) # Signal [image path]

    def __init__(self) -> None:
        super().__init__();

        self.setWindowTitle('Drag and Drop Image')
        self.resize(500, 500)
        self.setAcceptDrops(True)

        main_layout = QVBoxLayout()

        self.photo_viewer = _ImageLabel()
        main_layout.addWidget(self.photo_viewer)

        self.setLayout(main_layout)

        return;

    # --- Methods from QWidget parent class ---
    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        self.__handle_drag_event(event)
        
        return;

    def dragMoveEvent(self, event: QDragMoveEvent) -> None:
        self.__handle_drag_event(event)

        return;
    
    def dropEvent(self, event: QDropEvent) -> None:
        if event.mimeData().hasImage:
            event.setDropAction(Qt.CopyAction)

            file_path = event.mimeData().urls()[0].toLocalFile()
            self.__set_image(file_path)
            self.image_dropped.emit(file_path) # Emit the signal!

            event.accept()
        else:
            event.ignore()

        return;

    # --- Private methods ---
    def __handle_drag_event(self, event: QDragMoveEvent) -> None:
        if event.mimeData().hasImage:
            event.accept()
        else:
            event.ignore()

        return;

    def __set_image(self, file_path: str) -> str:
        self.photo_viewer.set_pixmap(QPixmap(file_path))
        return file_path;

# --- Demo code ---
if __name__ == '__main__':
    app = QApplication(sys.argv)

    main_window = DragAndDropImage()
    
    def on_drop(path: str) -> None:
        print(f'Image dropped: {path}')
        app.quit()

        return;
    
    main_window.image_dropped.connect(on_drop)
    main_window.show()

    sys.exit(app.exec_())
