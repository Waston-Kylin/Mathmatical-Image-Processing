from FraDec3D import FraDec3D


def FraDecMultiLevel3D(A, D, L):
    """
    Apply L decomposition levels.

    At each subsequent level, only the low-pass (0,0,0) component from
    the preceding level is decomposed further.
    """
    if L < 1:
        raise ValueError("The decomposition level L must be >= 1.")

    decomposition = []
    current = A

    for level in range(L):
        coeff_level = FraDec3D(current, D, level + 1)
        decomposition.append(coeff_level)
        current = coeff_level[0][0][0]

    return decomposition
