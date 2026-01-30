"""Source reconstruction module.

This module contains functions for MEG source reconstruction:
- Coregistration (MEG to MRI coordinate system)
- Source space setup
- BEM model creation
- Forward solution computation
- Inverse solution application
- Morphing to fsaverage
- Atlas/parcellation application
"""

from code.source_reconstruction import utils

__all__ = ["utils"]
