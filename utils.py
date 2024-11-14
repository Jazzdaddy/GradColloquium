import numpy as np
from scipy.integrate import trapezoid

def error(X, Xrec, dx, dt):

   e1 = dx * trapezoid((X - Xrec)**2)
   error = np.sqrt(dt * np.trapezoid(e1))

   return error