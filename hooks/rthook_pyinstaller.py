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

    # CRITICAL: Add torch lib directory to DLL search path on Windows
    if sys.platform == 'win32':
        base_dir = app_dir

        # Add multiple DLL directories
        dll_dirs = [
            os.path.join(base_dir, 'torch', 'lib'),
            os.path.join(base_dir, 'numpy.libs'),
            os.path.join(base_dir, 'scipy.libs'),
            base_dir,
        ]

        for d in dll_dirs:
            if os.path.exists(d):
                os.add_dll_directory(d)
                print(f"[RTDEBUG] Added DLL directory: {d}", file=sys.stderr)

        # Prepend to PATH for subprocesses
        path_additions = os.pathsep.join([d for d in dll_dirs if os.path.exists(d)])
        os.environ['PATH'] = path_additions + os.pathsep + os.environ.get('PATH', '')

        # List torch DLLs
        torch_lib_path = os.path.join(base_dir, 'torch', 'lib')
        if os.path.exists(torch_lib_path):
            torch_dlls = [f for f in os.listdir(torch_lib_path) if f.endswith('.dll')]
            print(f"[RTDEBUG] torch DLLs found: {torch_dlls}", file=sys.stderr)

        # Disable CUDA for torch
        os.environ['TORCH_CUDA_ARCHIST'] = 'disabled'
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        os.environ['USE_CUDA'] = '0'

        # List torch directory contents for debugging
        torch_base = os.path.join(app_dir, 'torch')
        if os.path.exists(torch_base):
            print(f"[RTDEBUG] torch base contents: {sorted(os.listdir(torch_base))}", file=sys.stderr)