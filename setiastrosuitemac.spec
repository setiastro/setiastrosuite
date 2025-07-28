# -*- mode: python ; coding: utf-8 -*-
import sys, os
from PyInstaller.utils.hooks import (
    collect_submodules,
    collect_dynamic_libs,
    collect_data_files,
    collect_all
)


sklearn_api_submods = collect_submodules('sklearn.externals.array_api_compat.numpy')
#############################################
# Collect everything we need
#############################################

# 0) Kaleido (Plotly static-image engine)
kaleido_datas, kaleido_binaries, kaleido_hiddenimports = collect_all('kaleido')

# 1) Photutils submodules
photutils_submodules = collect_submodules('photutils')

# 2) sep_pjw submodules & binaries
sep_pjw_submodules = collect_submodules('sep_pjw')
sep_pjw_binaries   = collect_dynamic_libs('sep_pjw')

# 3) typing_extensions code files
typingext_datas = collect_data_files('typing_extensions')

# 4) importlib_metadata back-port
importlib_metadata_datas = collect_data_files('importlib_metadata')

# 5) NUMCODECS (for zfpy extension)
numcodecs_submodules = collect_submodules('numcodecs')
numcodecs_binaries   = collect_dynamic_libs('numcodecs')
numcodecs_datas      = collect_data_files('numcodecs')

# 6) PyTorch dynamic libs & data
torch_binaries = collect_dynamic_libs('torch')
torch_datas    = collect_data_files('torch')
#############################################
# Build up hiddenimports and binaries
#############################################
binaries = []
binaries += sep_pjw_binaries
binaries += kaleido_binaries
binaries += numcodecs_binaries
binaries += torch_binaries     # <-- include all torch .so/.dylib

hiddenimports = []
hiddenimports += photutils_submodules
hiddenimports += ['sep_pjw', '_version']
hiddenimports += sep_pjw_submodules
hiddenimports += kaleido_hiddenimports
hiddenimports += [
    'typing_extensions',
    'importlib_metadata',
    'numcodecs',           # ensure the package is there
    'numcodecs.zfpy',      # explicit for the zfpy extension
    'torch._C',            # ensure the torch C-extension is loaded
]
hiddenimports += sklearn_api_submods
directory = '/Users/franklinmarek/cosmicclarity/setiastrosuite'


#############################################
# Data collection
#############################################
datas=[
    (directory + '/venv/lib/python3.12/site-packages/astroquery/CITATION', 'astroquery'),
    (directory + '/venv/lib/python3.12/site-packages/photutils/CITATION.rst', 'photutils'),
    ('celestial_catalog.csv', '.'),
    ('astrosuite.png', '.'),
    ('stacking.png', '.'),
    ('wimilogo.png', '.'),
    ('wrench_icon.png', '.'),
    ('numba_utils.py', '.'),
    ('mosaic.png', '.'),
    ('platesolve.png', '.'),
    ('starregistration.png', '.'),
    ('supernova.png', '.'),
    ('dse.png', '.'),
    ('psf.png', '.'),
    ('gridicon.png', '.'),
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
    ('spcc.png', '.'),
    ('SASP_data.fits', '.'),
    ('slot3.png', '.'),
    ('slot4.png', '.'),
    ('slot5.png', '.'),
    ('convo.png', '.'),
    ('slot6.png', '.'),
    ('slot7.png', '.'),
    ('slot8.png', '.'),
    ('exoicon.png', '.'),
    ('slot9.png', '.'),
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
    ('pen.png', '.'),
    ('openfile.png', '.'),
    ('aperture.png', '.'),
    ('starspike.png', '.'),
    ('graxpert.png', '.'),
    ('jwstpupil.png', '.'),
    ('pedestal.png', '.'),
    ('abeicon.png', '.'),
    ('undoicon.png', '.'),
    ('blaster.png', '.'),
    ('redoicon.png', '.'),
    ('cropicon.png', '.'),
    ('rgbcombo.png', '.'),
    ('copyslot.png', '.'),
    ('rgbextract.png', '.'),
    ('hubble.png', '.'),
    ('livestacking.png', '.'),
    ('staradd.png', '.'),
    ('starnet.png', '.'),
    ('HRDiagram.png', '.'),
    ('clahe.png', '.'),
    ('morpho.png', '.'),
    ('whitebalance.png', '.'),
    ('neutral.png', '.'),
    ('green.png', '.'),
    ('imgs', 'imgs'),
    ('collage.png', '.'),
    ('annotated.png', '.'),
    ('colorwheel.png', '.'),
    ('font.png', '.'),
    ('spinner.gif', '.'),
    ('cvs.png', '.'),
    (directory + '/venv/lib/python3.12/site-packages/astroquery/simbad/data', 'astroquery/simbad/data'),
    (directory + '/venv/lib/python3.12/site-packages/astropy/CITATION', 'astropy'),
    (directory + '/venv/lib/python3.12/site-packages/_version.py', '.')
]

# Append other package data
from PyInstaller.utils.hooks import collect_data_files as _cdf

datas += _cdf('dask', include_py_files=False)
datas += _cdf('photutils')
datas += typingext_datas
datas += importlib_metadata_datas
datas += kaleido_datas
datas += numcodecs_datas    # <-- ensure .so and support files for numcodecs
datas += torch_datas       # <-- include torch’s data files
datas += collect_data_files('lightkurve', includes=['data/lightkurve.mplstyle'])

#############################################
# Build the spec
#############################################
a = Analysis(
    ['setiastrosuitemacQT6.py'],  # your entrypoint
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['fix_importlib_metadata.py'],  # keep your metadata shim
    excludes=['PyQt5'],  # drop torch & torchvision from excludes!
    noarchive=True,
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
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=directory + '/astrosuite.icns',
    onefile=True
)
