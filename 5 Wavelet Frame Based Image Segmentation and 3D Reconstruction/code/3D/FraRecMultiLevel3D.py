import copy
from FraRec3D import FraRec3D


def FraRecMultiLevel3D(C, R, L):
    """
    Reconstruct a signal from L levels of nested 3-D frame coefficients.

    The coefficient structure is deep-copied because reconstruction
    replaces low-pass entries while traversing from coarse to fine.
    """
    if L < 1:
        raise ValueError("The reconstruction level L must be >= 1.")
    if len(C) != L:
        raise ValueError(
            f"Expected {L} coefficient levels, received {len(C)}."
        )

    coeff_list = copy.deepcopy(C)

    for level in range(L - 1, 0, -1):
        coeff_list[level - 1][0][0][0] = FraRec3D(
            coeff_list[level], R, level + 1
        )

    return FraRec3D(coeff_list[0], R, 1)
