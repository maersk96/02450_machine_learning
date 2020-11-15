import numpy as np
import dtc
x = np.array([0, 2, 1, 2, 1, 1, 1]).reshape(1,-1)

# Evaluate the classification tree for the new data object
x_class = dtc.predict(x)[0]