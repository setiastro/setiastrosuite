# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_data_files, collect_submodules
from PyInstaller.utils.hooks import (
    collect_submodules,
    collect_data_files,
    collect_dynamic_libs
)

from PyInstaller.utils.hooks import collect_data_files
from PyInstaller.utils.hooks import get_package_paths

# 1) Collect photutils data (including CITATION.rst).
photutils_data = collect_data_files('photutils')

# 2) Collect all photutils submodules (pure Python).
photutils_submodules = collect_submodules('photutils')

# 3) Collect any compiled libraries from photutils (the .pyd / .dll / .so).
photutils_binaries = collect_dynamic_libs('photutils')


# 1. Collect Dask data (templates, etc.)
dask_data = collect_data_files('dask', include_py_files=False)
from PyInstaller.utils.hooks import get_package_paths

photutils_path = get_package_paths('photutils')[0]

directory = './.venv/lib/python3.*/site-packages'

a = Analysis(
    ['setiastrosuitemacQT6.py'],
    pathex=[],
    binaries=[],
    datas=[
        (directory + '/astroquery/CITATION', 'astroquery'),
        (directory + '/photutils/CITATION.rst', 'photutils'),
        ('celestial_catalog.csv', '.'), 
        ('astrosuite.png', '.'), 
        ('wimilogo.png', '.'), 
        ('wrench_icon.png', '.'), 
        ('platesolve.png', '.'),
        ('psf.png', '.'),        
        ('eye.png', '.'), 
        ('disk.png', '.'), 
        ('nuke.png', '.'), 
        ('rescale.png', '.'),  
        ('staralign.png', '.'),       
        ('LExtract.png', '.'),
        ('LInsert.png', '.'),
        ('slot1.png', '.'),
        ('slot0.png', '.'),
        ('slot2.png', '.'),
        ('hdr.png', '.'),
        ('slot3.png', '.'),
        ('slot4.png', '.'),
        ('slot5.png', '.'),
        ('slot6.png', '.'),
        ('slot7.png', '.'),
        ('slot8.png', '.'),
        ('slot9.png', '.'),   
        ('rescale.png', '.'),  
        ('staralign.png', '.'),
        ('pixelmath.png', '.'),
        ('histogram.png', '.'),   
        ('invert.png', '.'),
        ('fliphorizontal.png', '.'),
        ('flipvertical.png', '.'),
        ('rotateclockwise.png', '.'),
        ('rotatecounterclockwise.png', '.'),
        ('maskcreate.png', '.'),
        ('maskapply.png', '.'),
        ('maskremove.png', '.'),
        ('mosaic.png', '.'),
        ('openfile.png', '.'),
        ('graxpert.png', '.'),
        ('abeicon.png', '.'),
        ('undoicon.png', '.'),
        ('blaster.png', '.'),
        ('redoicon.png', '.'),
        ('cropicon.png', '.'),
        ('rgbcombo.png', '.'),
        ('copyslot.png', '.'),
        ('rgbextract.png', '.'),        
        ('hubble.png', '.'), ('staradd.png', '.'),('starnet.png', '.'),('clahe.png', '.'),('morpho.png', '.'),('whitebalance.png', '.'),('neutral.png', '.'),('green.png', '.'),
        ('imgs', 'imgs'),
        ('collage.png', '.'), 
        ('annotated.png', '.'), 
        ('colorwheel.png', '.'), 
        ('font.png', '.'), 
        ('spinner.gif', '.'), 
        ('cvs.png', '.'), 
        (directory + '/astroquery/simbad/data', 'astroquery/simbad/data'), 
        (directory + '/astropy/CITATION', 'astropy')
    ] + dask_data+ photutils_data,
    hiddenimports=[] + photutils_submodules,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['torch', 'torchvision'],
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
    name='setiastrosuitemac',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # Enable terminal console
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=[directory + 'astrosuite.icns'],
    onefile=True  # Enable single-file mode
)
