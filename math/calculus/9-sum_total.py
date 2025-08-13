"""Module pour calculer la somme des carrés des entiers de 1 à n."""


def summation_i_squared(n):
    """
    Calcule la somme des carrés des entiers de 1 à n.

    Args:
        n (int): Limite supérieure de la somme.

    Returns:
        int: Somme des carrés de 1 à n, ou None si n n'est pas un entier
        positif.
    """
    if not isinstance(n, int) or n < 1:
        return None

    return n * (n + 1) * (2 * n + 1) // 6
