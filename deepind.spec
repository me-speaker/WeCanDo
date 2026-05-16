# -*- mode: python ; coding: utf-8 -*-
"""
GUI version spec for DAE-ELM PyQt5 graphical interface.
"""

from PyInstaller.utils.hooks import collect_all, collect_submodules

# 收集 xgboost、matplotlib、sklearn 和 yaml 的所有依赖
xgboost_datas, xgboost_binaries, xgboost_hiddenimports = collect_all('xgboost')
matplotlib_datas, matplotlib_binaries, matplotlib_hiddenimports = collect_all('matplotlib')
sklearn_datas, sklearn_binaries, sklearn_hiddenimports = collect_all('sklearn')
# 必须同时使用 'PyYAML' 和 'yaml'
yaml_datas, yaml_binaries, yaml_hiddenimports = collect_all('PyYAML')
yaml2_datas, yaml2_binaries, yaml2_hiddenimports = collect_all('yaml')

# 合并 yaml 数据和隐式导入
yaml_datas += yaml2_datas
yaml_binaries += yaml2_binaries
yaml_hiddenimports += yaml2_hiddenimports

# 收集 src 的所有子模块
src_hiddenimports = collect_submodules('src')

a = Analysis(
    ['gui_main.py'],
    pathex=[".", "src"],
    binaries=xgboost_binaries + matplotlib_binaries + sklearn_binaries + yaml_binaries + yaml2_binaries,
    datas=[
        ('src', 'src'),
        ('daeelm_agents', 'daeelm_agents'),
        ('bridge', 'bridge'),
        ('skill_hub', 'skill_hub'),
        ('deployment', 'deployment'),
        ('gui', 'gui'),
        ('docs', 'docs'),
    ] + xgboost_datas + matplotlib_datas + sklearn_datas + yaml_datas + yaml2_datas,
    hiddenimports=[
        'torch',
        'sklearn',
        'sklearn.ensemble',
        'sklearn.feature_selection',
        'xgboost',
        'matplotlib',
        'matplotlib.backends.backend_qt5agg',
        'numpy',
        'pandas',
        'scipy',
        'yaml',
        'yaml.cyaml',
        'PyYAML',
        'scikit-learn',
        'PyQt5',
        'PyQt5.QtCore',
        'PyQt5.QtGui',
        'PyQt5.QtWidgets',
    ] + xgboost_hiddenimports + matplotlib_hiddenimports + sklearn_hiddenimports + yaml_hiddenimports + src_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['hooks/rthook_pyinstaller.py'],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='deepind',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='deepind',
)