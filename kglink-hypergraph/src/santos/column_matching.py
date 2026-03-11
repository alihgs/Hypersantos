import numpy as np

def cosine(a, b):

    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))