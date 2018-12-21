from bdm.base import BDMIgnore as BDM
import numpy as np

bdm = BDM(2)
d1 = np.ones((10,10),dtype=int)
d2 = np.random.randint(2,size=(10,10))

c1 = bdm.count_and_lookup(d1)
bdm1 = bdm.compute_bdm(c1)
c2 = bdm.count_and_lookup(d2)
bdm2 = bdm.compute_bdm(c2)

print(bdm1,bdm2)



