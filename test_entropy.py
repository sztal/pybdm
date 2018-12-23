from bdm.base import BDMIgnore as BDM
from bdm.utils import get_ctm_dataset, get_reduced_idx, get_reduced_shape
import numpy as np
from collections import Counter

bdm1 = BDM(1)
bdm2 = BDM(2)
ctm1 = get_ctm_dataset('CTM-B2-D12B')
ctm2 = get_ctm_dataset('CTM-B2-D4x4')

print(len(ctm1))
