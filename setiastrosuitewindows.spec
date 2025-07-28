# -*- mode: python ; coding: utf-8 -*-
import sys ; sys.setrecursionlimit(sys.getrecursionlimit() * 10)

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_submodules,
    collect_dynamic_libs,
    get_package_paths
)
from PyInstaller.utils.hooks import collect_all
sklearn_api_submods = collect_submodules('sklearn.externals.array_api_compat.numpy')

#############################################
# Collect everything we need
#############################################

# 0) Kaleido (Plotly static-image engine)
kaleido_datas, kaleido_binaries, kaleido_hiddenimports = collect_all('kaleido')

# 1) photutils data (CITATION.rst) and submodules
photutils_data       = collect_data_files('photutils')
photutils_submodules = collect_submodules('photutils')
photutils_binaries   = collect_dynamic_libs('photutils')

# 2) Dask data (templates, etc.)
dask_data = collect_data_files('dask', include_py_files=False)

# 3) astroalign, astropy, scikit-image, OpenCV submodules
astroalign_submodules = collect_submodules('astroalign')
astropy_submodules    = collect_submodules('astropy')
skimage_submodules    = collect_submodules('skimage')
cv2_submodules        = collect_submodules('cv2')
cv2_binaries          = collect_dynamic_libs('cv2')

# 4) sep_pjw submodules & binaries (the library astroalign references as `import sep_pjw as sep`)
sep_pjw_submodules = collect_submodules('sep_pjw')
sep_pjw_binaries   = collect_dynamic_libs('sep_pjw')

#############################################
# Build up hiddenimports and binaries
#############################################
hiddenimports = [
    'lz4.block',
    'zstandard',
    'base64',
    'ast',
    'cv2',
    'astropy.io.fits',
    'astropy.wcs',
    'skimage.transform',
    'skimage.feature',
    'scipy.spatial',
    'astroalign',
    'sep',
    'sep_pjw',
    'sep_pjw._version',
    '_version',
    *photutils_submodules,
    *astroalign_submodules,
    *astropy_submodules,
    *skimage_submodules,
    *cv2_submodules,
    *sep_pjw_submodules,
    *kaleido_hiddenimports,    # include Kaleido
    *sklearn_api_submods
]

binaries = []
binaries += photutils_binaries
binaries += cv2_binaries
binaries += sep_pjw_binaries
binaries += kaleido_binaries      # include Kaleido

#############################################
# The Analysis block
#############################################
a = Analysis(
    ['setiastrosuiteQT6.py'],
    pathex=[],
    binaries=binaries,
    datas=[
        ('imgs', 'imgs'),
        ('C:\\Users\\Gaming\\Desktop\\Python Code\\venv\\Lib\\site-packages\\astroquery\\CITATION', 'astroquery'),
        ('C:\\Users\\Gaming\\Desktop\\Python Code\\venv\\Lib\\site-packages\\_version.py', '.'),
        ('wimilogo.png', '.'),
        ('wrench_icon.png', '.'),
        ('astrosuite.ico', '.'),
        ('staradd.png', '.'),
        ('starnet.png', '.'),
        ('clahe.png', '.'),
        ('morpho.png', '.'),
        ('whitebalance.png', '.'),
        ('neutral.png', '.'),
        ('green.png', '.'),
        ('eye.png', '.'),
        ('disk.png', '.'),
        ('nuke.png', '.'),
        ('astrosuite.png', '.'),
        ('hubble.png', '.'),
        ('collage.png', '.'),
        ('annotated.png', '.'),
        ('colorwheel.png', '.'),
        ('font.png', '.'),
        ('spinner.gif', '.'),
        ('cvs.png', '.'),
        ('C:\\Users\\Gaming\\Desktop\\Python Code\\venv\\Lib\\site-packages\\astroquery\\simbad\\data', 'astroquery\\simbad\\data'),
        ('C:\\Users\\Gaming\\Desktop\\Python Code\\venv\\Lib\\site-packages\\photutils\\CITATION.rst', 'photutils'),
        ('LExtract.png', '.'),
        ('LInsert.png', '.'),
        ('slot1.png', '.'),
        ('numba_utils.py', '.'),
        ('slot0.png', '.'),
        ('slot2.png', '.'),
        ('slot3.png', '.'),
        ('slot4.png', '.'),
        ('slot5.png', '.'),
        ('slot6.png', '.'),
        ('slot7.png', '.'),
        ('slot8.png', '.'),
        ('slot9.png', '.'),
        ('supernova.png', '.'),
        ('platesolve.png', '.'),
        ('psf.png', '.'),
        ('pixelmath.png', '.'),
        ('histogram.png', '.'),
        ('mosaic.png', '.'),
        ('rgbcombo.png', '.'),
        ('rgbextract.png', '.'),
        ('hdr.png', '.'),
        ('blaster.png', '.'),
        ('cropicon.png', '.'),
        ('openfile.png', '.'),
        ('abeicon.png', '.'),
        ('invert.png', '.'),
        ('fliphorizontal.png', '.'),
        ('flipvertical.png', '.'),
        ('rotateclockwise.png', '.'),
        ('spcc.png', '.'),
        ('SASP_data.fits', '.'),
        ('rescale.png', '.'),
        ('rotatecounterclockwise.png', '.'),
        ('staralign.png', '.'),
        ('maskcreate.png', '.'),
        ('maskapply.png', '.'),
        ('livestacking.png', '.'),
        ('maskremove.png', '.'),
        ('stacking.png', '.'),
        ('undoicon.png', '.'),
        ('exoicon.png', '.'),
        ('convo.png', '.'),
        ('HRDiagram.png', '.'),
        ('pedestal.png', '.'),
        ('pen.png', '.'),
        ('redoicon.png', '.'),
        ('gridicon.png', '.'),
        ('graxpert.png', '.'),
        ('dse.png', '.'),
        ('starspike.png', '.'),
        ('jwstpupil.png', '.'),
        ('copyslot.png', '.'),
        ('aperture.png', '.'),
        ('starregistration.png', '.'),
        ('C:\\Users\\Gaming\\Desktop\\Python Code\\venv\\Lib\\site-packages\\astropy\\CITATION', 'astropy'),
        ('celestial_catalog.csv', '.'),
        ('C:\\Users\\Gaming\\Desktop\\Python Code\\imgs', 'imgs'),
        ('xisf.py', '.'),
        *kaleido_datas                # include Kaleido data
    ] + dask_data + photutils_data + collect_data_files('lightkurve', includes=['data/lightkurve.mplstyle']),
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['torch', 'torchvision', 'PyQt5', 'PySide6', 'shiboken6'],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

# Modify EXE to exclude binaries so that COLLECT can handle them.
exe = EXE(
    pyz,
    a.scripts,
    exclude_binaries=True,
    runtime_hooks=['minimize_console.py'],
    name='setiastrosuiteQT6',
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
    icon='astrosuite.ico',
)

# COLLECT creates the one-folder bundle containing the executable and an _internal folder.
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='setiastrosuiteQT6'
)
