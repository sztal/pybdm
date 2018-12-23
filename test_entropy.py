from bdm.base import BDMIgnore as BDM
from bdm.utils import get_ctm_dataset, get_reduced_idx, get_reduced_shape
import numpy as np
from collections import Counter

bdm1 = BDM(1)
bdm2 = BDM(2)
ctm = get_ctm_dataset('CTM-B2-D12')
c1 = Counter([('111111111111', 1.95207842085224e-08)])
c2 = Counter([('000000000000', 1.95207842085224e-08)])
c3 = Counter([('101010101010', 7.49862566298854e-09)])

x=np.random.randint(0,2,48)

