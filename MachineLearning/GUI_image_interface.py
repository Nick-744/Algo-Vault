from PyQt5.QtGui import (
    QPixmap, QDragEnterEvent, QDragMoveEvent, QDropEvent
)
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout
)
from PyQt5.QtCore import (Qt, pyqtSignal, QTimer)
import sys

class _ImageLabel(QLabel):
    def __init__(self) -> None:
        super().__init__();
        
        self.setAlignment(Qt.AlignCenter)
        self.setText('\n\n Drop an image here... \n\n')
        self.setStyleSheet('''
            QLabel {
                border: 4px dashed #aaa;
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
    # Signal carrying the image path:
    image_dropped = pyqtSignal(str)

    def __init__(self) -> None:
        super().__init__();

        self.setWindowTitle('Drag and Drop Image')
        self.resize(500, 500)
        self.setAcceptDrops(True)

        main_layout = QVBoxLayout()

        self.photo_viewer = _ImageLabel()
        main_layout.addWidget(self.photo_viewer)

        self.setLayout(main_layout)

        # Timer for closing the window after showing the image
        self.close_timer = QTimer()
        self.close_timer.timeout.connect(self.close_and_process)
        self.close_timer.setSingleShot(True) # Only fire once
        
        self.pending_file_path = None # Image file path

        return;

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        self.__enter_move_event_handler(event)
        
        return;

    def dragMoveEvent(self, event: QDragMoveEvent) -> None:
        self.__enter_move_event_handler(event)

        return;
    
    def dropEvent(self, event: QDropEvent) -> None:
        if event.mimeData().hasImage:
            event.setDropAction(Qt.CopyAction)
            file_path = event.mimeData().urls()[0].toLocalFile()

            # Store the file path for later processing
            self.pending_file_path = file_path
            
            # Display the image immediately
            self.__set_image(file_path)
            
            # Update the window title to show processing status
            self.setWindowTitle('Processing...')
            
            # Start timer to close window and
            # emit signal after 2 seconds...
            self.close_timer.start(2000)

            event.accept()
        else:
            event.ignore()

        return;

    def close_and_process(self) -> None:
        ''' Called by timer to close window and emit signal '''
        if self.pending_file_path:
            # Emit the signal with the image path
            self.image_dropped.emit(self.pending_file_path)
            
        # Close the window
        self.close()
        
        return;

    def __enter_move_event_handler(self,
                                   event: QDragMoveEvent) -> None:
        if event.mimeData().hasImage:
            event.accept()
        else:
            event.ignore()

        return;

    def __set_image(self, file_path: str) -> str:
        self.photo_viewer.set_pixmap(QPixmap(file_path))

        return file_path;

if __name__ == '__main__':
    app = QApplication(sys.argv)

    main_window = DragAndDropImage()
    main_window.show()

    sys.exit(app.exec_())
