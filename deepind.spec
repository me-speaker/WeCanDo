# -*- mode: python ; coding: utf-8 -*-
"""
GUI version spec for DAE-ELM PyQt5 graphical interface.
"""

from PyInstaller.utils.hooks import collect_all

# 收集 xgboost、matplotlib 和 sklearn 的所有依赖（包括原生库）
xgboost_datas, xgboost_binaries, xgboost_hiddenimports = collect_all('xgboost')
matplotlib_datas, matplotlib_binaries, matplotlib_hiddenimports = collect_all('matplotlib')
sklearn_datas, sklearn_binaries, sklearn_hiddenimports = collect_all('sklearn')

a = Analysis(
    ['gui_main.py'],
    pathex=[".", "src"],
    binaries=xgboost_binaries + matplotlib_binaries + sklearn_binaries,
    datas=[
        ('src', 'src'),
        ('daeelm_agents', 'daeelm_agents'),
        ('bridge', 'bridge'),
        ('skill_hub', 'skill_hub'),
        ('deployment', 'deployment'),
        ('gui', 'gui'),
        ('docs', 'docs'),
    ] + xgboost_datas + matplotlib_datas + sklearn_datas,
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
        'PyYAML',
        'scikit-learn',
        'PyQt5',
        'PyQt5.QtCore',
        'PyQt5.QtGui',
        'PyQt5.QtWidgets',
        # src modules
        'src.core.config_manager',
        'src.core.registry',
        'src.core.base',
        'src.core.factory',
        'src.core.runner',
        'src.core.interfaces',
        'src.core.interfaces.base',
        'src.core.interfaces.config_parser',
        'src.core.interfaces.data_loader',
        'src.core.interfaces.model_updater',
        'src.optimization.algorithms.auto_optimizer',
        'src.optimization.algorithms.lbfgs',
        'src.optimization.algorithms.cg',
        'src.optimization.algorithms.bayesian',
        'src.optimization.algorithms.genetic',
        'src.optimization.algorithms.nga',
        'src.optimization.algorithms.nsga2',
        'src.optimization.algorithms.funcs_CG',
        'src.optimization.algorithms.get_opt_by_CG-FR_foil',
        'src.optimization.base',
        'src.optimization.config_loader',
        'src.optimization.constraints',
        'src.models.surrogates.elm',
        'src.models.surrogates.gaussian_process',
        'src.models.surrogates.neural_network',
        'src.models.surrogates.xgboost_model',
        'src.models.surrogates.get_fit_para',
        'src.models.surrogates.funcs_fit',
        'src.models.postprocessors',
        'src.features.feature_selector',
        'src.data.preprocessing.base',
        'src.data.preprocessing.missing_value_imputation',
        'src.data.preprocessing.noise_reduction',
        'src.data.preprocessing.outlier_detection',
        'src.data.preprocessing.signal_smoothing',
    ] + xgboost_hiddenimports + matplotlib_hiddenimports + sklearn_hiddenimports,
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