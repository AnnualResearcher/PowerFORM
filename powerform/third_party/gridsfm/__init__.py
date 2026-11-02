"""GridSFM (Microsoft), vendored: the modules the Table 1 GridSFM rows need.

Upstream: https://github.com/microsoft/GridSFM, commit
1ca775fd436d7ce013a1c0ab946e61ac7ef59ad6 (package `gridsfm` 1.1.0, the
`model/gridsfm/` directory). MIT License, see LICENSE in this directory.

Every module file here is byte-identical to that commit. This `__init__` is not
upstream's, which imports every submodule at package load, including the OPFData
downloader adapter that is not vendored. Import
the submodules directly, e.g. `from powerform.third_party.gridsfm.model import
GridTransformerBackbone`; the names upstream re-exports from the package root are
re-exported here for the modules present.

See NOTICE for what is included and why.
"""
from .data import (  # noqa: F401
    batch_data_list,
    load_opfdata,
    load_pyg_json,
    prepare_for_inference,
)
from .model import GridTransformerBackbone  # noqa: F401
from .schema import AC_LINE_KEY, TRANSFORMER_KEY  # noqa: F401

__version__ = "1.1.0"
UPSTREAM = "https://github.com/microsoft/GridSFM"
UPSTREAM_COMMIT = "1ca775fd436d7ce013a1c0ab946e61ac7ef59ad6"
