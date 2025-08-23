# -*- mode: python ; coding: utf-8 -*-

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

#############################################
# Build up hiddenimports and binaries
#############################################
binaries = []
binaries += sep_pjw_binaries
binaries += kaleido_binaries
binaries += numcodecs_binaries

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
]
hiddenimports += sklearn_api_submods

#############################################
# Data collection
#############################################
datas=[
    # astroquery citation
    ('/home/seti-astro/Desktop/setiastrosuite/venv/lib/python3.12/site-packages/astroquery/CITATION',
        'astroquery'),
    # astroquery simbad data
    ('/home/seti-astro/Desktop/setiastrosuite/venv/lib/python3.12/site-packages/astroquery/simbad/data',
        'astroquery/simbad/data'),
    # photutils citation
    ('/home/seti-astro/Desktop/setiastrosuite/venv/lib/python3.12/site-packages/photutils/CITATION.rst',
        'photutils'),
    # astropy citation
    ('/home/seti-astro/Desktop/setiastrosuite/venv/lib/python3.12/site-packages/astropy/CITATION',
        'astropy'),
    # local resources
    ('wimilogo.png', '.'),
    ('wrench_icon.png', '.'),
    ('astrosuite.png', '.'),
    ('exoicon.png', '.'),
    ('LExtract.png', '.'),
    ('LInsert.png', '.'),
    ('slot1.png', '.'),
    ('slot0.png', '.'),
    ('slot2.png', '.'),
    ('slot3.png', '.'),
    ('slot4.png', '.'),
    ('slot5.png', '.'),
    ('slot6.png', '.'),
    ('slot7.png', '.'),
    ('slot8.png', '.'),
    ('HRDiagram.png', '.'),
    ('gridicon.png', '.'),
    ('spcc.png', '.'),
    ('SASP_data.fits', '.'),
    ('convo.png', '.'),
    ('astrobin_filters.csv', '.'),
    ('slot9.png', '.'),
    ('numba_utils.py', '.'),
    ('livestacking.png', '.'),
    ('stacking.png', '.'),
    ('dse.png', '.'),
    ('starregistration.png', '.'),
    ('supernova.png', '.'),
    ('pen.png', '.'),
    ('pixelmath.png', '.'),
    ('histogram.png', '.'),
    ('rgbcombo.png', '.'),
    ('jwstpupil.png', '.'),
    ('pedestal.png', '.'),
    ('cropicon.png', '.'),
    ('openfile.png', '.'),
    ('blaster.png', '.'),
    ('mosaic.png', '.'),
    ('graxpert.png', '.'),
    ('starspike.png', '.'),
    ('aperture.png', '.'),
    ('hdr.png', '.'),
    ('invert.png', '.'),
    ('fliphorizontal.png', '.'),
    ('flipvertical.png', '.'),
    ('rotateclockwise.png', '.'),
    ('rotatecounterclockwise.png', '.'),
    ('platesolve.png', '.'),
    ('psf.png', '.'),
    ('rescale.png', '.'),
    ('staralign.png', '.'),
    ('maskcreate.png', '.'),
    ('maskapply.png', '.'),
    ('maskremove.png', '.'),
    ('abeicon.png', '.'),
    ('undoicon.png', '.'),
    ('redoicon.png', '.'),
    ('rgbextract.png', '.'),
    ('copyslot.png', '.'),
    ('staradd.png', '.'),
    ('gridicon.png', ','),
    ('starnet.png', '.'),
    ('clahe.png', '.'),
    ('morpho.png', '.'),
    ('whitebalance.png', '.'),
    ('neutral.png', '.'),
    ('green.png', '.'),
    ('celestial_catalog.csv', '.'),
    ('eye.png', '.'),
    ('disk.png', '.'),
    ('nuke.png', '.'),
    ('hubble.png', '.'),
    ('collage.png', '.'),
    ('annotated.png', '.'),
    ('colorwheel.png', '.'),
    ('font.png', '.'),
    ('spinner.gif', '.'),
    ('cvs.png', '.'),
    ('imgs', 'imgs'),
]

# Append other package data
from PyInstaller.utils.hooks import collect_data_files as _cdf

datas += _cdf('dask', include_py_files=False)
datas += _cdf('photutils')
datas += typingext_datas
datas += importlib_metadata_datas
datas += kaleido_datas
datas += numcodecs_datas    # <-- ensure .so and support files for numcodecs
datas += collect_data_files('lightkurve', includes=['data/lightkurve.mplstyle'])

#############################################
# Build the spec
#############################################
a = Analysis(
    ['setiastrosuitelinuxQT6.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['fix_importlib_metadata.py'],  # leave your metadata‐shim hook
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
    name='setiastrosuite_linux_ubuntu24.04',
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
    icon=['astrosuite.png'],
)
