import sys
import os

# Add the bundled src directory to Python path
if getattr(sys, 'frozen', False):
    # Running as compiled executable
    app_dir = sys._MEIPASS
    src_path = os.path.join(app_dir, 'src')
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    # Also add root path for absolute imports
    root_path = app_dir
    if root_path not in sys.path:
        sys.path.insert(0, root_path)
