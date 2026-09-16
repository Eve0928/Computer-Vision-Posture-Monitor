# tracker.py
import numpy as np
from config import L_EAR, R_EAR, L_SHOULDER, R_SHOULDER

def calculate_posture_metric(keypoints):
    """
    Computes a scale-invariant posture ratio:
    Vertical distance between ears and shoulders, normalized by shoulder width.
    """
    l_ear, r_ear = keypoints[L_EAR], keypoints[R_EAR]
    l_sh, r_sh = keypoints[L_SHOULDER], keypoints[R_SHOULDER]

    # Require minimum confidence score from detection
    min_conf = 0.5
    if (l_ear[2] < min_conf and r_ear[2] < min_conf) or \
       (l_sh[2] < min_conf or r_sh[2] < min_conf):
        return None

    # Midpoint of ears and shoulders
    ear_y = (l_ear[1] + r_ear[1]) / 2.0 if (l_ear[2] >= min_conf and r_ear[2] >= min_conf) else (l_ear[1] if l_ear[2] >= min_conf else r_ear[1])
    sh_y = (l_sh[1] + r_sh[1]) / 2.0

    # Vertical distance
    vert_dist = sh_y - ear_y

    # Shoulder width for scale normalization
    shoulder_width = np.linalg.norm(np.array([l_sh[0], l_sh[1]]) - np.array([r_sh[0], r_sh[1]]))
    if shoulder_width == 0:
        return None

    return vert_dist / shoulder_width