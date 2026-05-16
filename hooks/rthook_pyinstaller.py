import sys
import os

# Debug: print paths at startup
if getattr(sys, 'frozen', False):
    app_dir = sys._MEIPASS
    print(f"[RTDEBUG] app_dir: {app_dir}", file=sys.stderr)
    print(f"[RTDEBUG] sys.path: {sys.path[:5]}", file=sys.stderr)

    # List contents of app directory
    if os.path.exists(app_dir):
        contents = os.listdir(app_dir)
        print(f"[RTDEBUG] app_dir contents (first 20): {contents[:20]}", file=sys.stderr)

    # Check if yaml exists
    yaml_path = os.path.join(app_dir, 'yaml')
    if os.path.exists(yaml_path):
        print(f"[RTDEBUG] yaml directory found: {yaml_path}", file=sys.stderr)
        print(f"[RTDEBUG] yaml contents: {os.listdir(yaml_path)[:10]}", file=sys.stderr)
    else:
        print(f"[RTDEBUG] yaml directory NOT found at {yaml_path}", file=sys.stderr)

    # Add the bundled root to Python path for absolute imports
    if app_dir not in sys.path:
        sys.path.insert(0, app_dir)

    # Add src directory
    src_path = os.path.join(app_dir, 'src')
    if src_path not in sys.path:
        sys.path.insert(0, src_path)