# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_submodules

# Automatically collect all PySide6 submodules
pyside6_hiddenimports = collect_submodules('PySide6')
pyside6_hiddenimports += collect_submodules('imgaug')
pyside6_hiddenimports += collect_submodules('cv2')
pyside6_hiddenimports += collect_submodules('onnxruntime')
pyside6_hiddenimports += collect_submodules('pymatting')

a = Analysis(
    ['..//multiprocess_augmentation//multiprocess_augmentation.py'],
    pathex=[],
    binaries=[],
    datas=[('..//core', 'core'), ('C:/Program Files/Tesseract-OCR/', 'tesseract')],
    hiddenimports=pyside6_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['..//core//utilities//u2net'],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='Multiprocess Augmentation',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['..//core//gui//images//camera.png'],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='Multiprocess Augmentation',
    icon=['..//core//gui//images//camera.png'],
    distpath='.//build//dist'
)
