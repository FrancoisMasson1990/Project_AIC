import numpy as np
from scipy.ndimage import measurements

a = np.zeros((10,10))
a[3:-3,3:-3] = 10
prediction = np.ma.masked_where(a == 0, a)
lw, num = measurements.label(prediction)
area = measurements.sum(a, lw, index=np.arange(lw.max() + 1))
print(lw)
print(prediction)