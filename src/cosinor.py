import numpy as np

def cosinor_fit(t_hours, y, period=24.0):
    t = np.asarray(t_hours)
    y = np.asarray(y)
    omega = 2*np.pi/period
    X = np.column_stack([np.ones_like(t),
                         np.cos(omega*t),
                         np.sin(omega*t)])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    M, A, B = beta
    amp = float(np.sqrt(A*A + B*B))
    phi = float(np.arctan2(B, A))
    acrophase_hours = (phi / omega) % period
    yhat = X @ beta
    resid = y - yhat
    return dict(mesor=float(M), amplitude=amp,
                acrophase_hours=float(acrophase_hours),
                yhat=yhat, resid=resid)
