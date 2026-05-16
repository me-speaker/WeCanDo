# -*- mode: python ; coding: utf-8 -*-
"""
CLI version spec for DAE-ELM command line interface.
"""

from PyInstaller.utils.hooks import collect_all

xgboost_datas, xgboost_binaries, xgboost_hiddenimports = collect_all('xgboost')
matplotlib_datas, matplotlib_binaries, matplotlib_hiddenimports = collect_all('matplotlib')
sklearn_datas, sklearn_binaries, sklearn_hiddenimports = collect_all('sklearn')
torch_datas, torch_binaries, torch_hiddenimports = collect_all('torch')

a = Analysis(
    ['src/__main__.py'],
    pathex=[],
    binaries=xgboost_binaries + matplotlib_binaries + sklearn_binaries + torch_binaries,
    datas=[
        ('src', 'src'),
        ('daeelm_agents', 'daeelm_agents'),
        ('bridge', 'bridge'),
        ('skill_hub', 'skill_hub'),
        ('deployment', 'deployment'),
    ] + xgboost_datas + matplotlib_datas + sklearn_datas + torch_datas,
    hiddenimports=[
        'torch',
        'sklearn',
        'xgboost',
        'matplotlib',
        'numpy',
        'pandas',
        'scipy',
        'yaml',
        'PyYAML',
        'scikit-learn',
    ] + xgboost_hiddenimports + matplotlib_hiddenimports + sklearn_hiddenimports + torch_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
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
    name='deepind_cli',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='deepind_cli',
)