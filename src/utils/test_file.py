def add_numbers(a: int, b: int) -> int:
    """
    2つの数値を加算する関数

    Args:
        a (int): 1つ目の数値
        b (int): 2つ目の数値

    Returns:
        int: 加算結果
    """
    return a + b


def test_add_numbers():
    assert add_numbers(1, 2) == 3
    assert add_numbers(-1, 1) == 0
