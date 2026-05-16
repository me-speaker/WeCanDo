import sys
import os

# Set up Python path for bundled app
if getattr(sys, 'frozen', False):
    app_dir = sys._MEIPASS

    # Debug output
    print(f"[RTDEBUG] app_dir: {app_dir}", file=sys.stderr)
    print(f"[RTDEBUG] sys.path[:5]: {sys.path[:5]}", file=sys.stderr)

    # List app directory
    if os.path.exists(app_dir):
        contents = sorted(os.listdir(app_dir))
        print(f"[RTDEBUG] app_dir contents: {contents}", file=sys.stderr)

    # Check for yaml
    yaml_path = os.path.join(app_dir, 'yaml')
    if os.path.exists(yaml_path):
        print(f"[RTDEBUG] yaml found: {yaml_path}", file=sys.stderr)
    else:
        print(f"[RTDEBUG] yaml NOT found at {yaml_path}", file=sys.stderr)

    # Add app_dir to sys.path for imports
    if app_dir not in sys.path:
        sys.path.insert(0, app_dir)

    # Add src directory
    src_path = os.path.join(app_dir, 'src')
    if src_path not in sys.path:
        sys.path.insert(0, src_path)