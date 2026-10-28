# FSA-ZoomNeXt model registry.
# Original ZoomNeXt classes are kept for baseline/FLOPs comparison.

from .zoomnext.zoomnext import (
    EffB1_ZoomNeXt,
    EffB4_ZoomNeXt,
    PvtV2B2_ZoomNeXt,
    PvtV2B3_ZoomNeXt,
    PvtV2B4_ZoomNeXt,
    PvtV2B5_ZoomNeXt,
    RN50_ZoomNeXt,
    videoPvtV2B5_ZoomNeXt,
)
from .zoomnext.fsa_zoomnext import (
    PvtV2B2_FSA_ZoomNeXt,
    PvtV2B3_FSA_ZoomNeXt,
    PvtV2B4_FSA_ZoomNeXt,
    PvtV2B5_FSA_ZoomNeXt,
)
