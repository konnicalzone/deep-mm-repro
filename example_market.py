import numpy as np

from baselines import numba_BSC, numba_DA, numba_RSD, numba_SD, numba_TTC


def example_market():
    P = np.array([[
        [1/3, 1, 2/3],
        [2/3, 1, 1/3],
        [1, 1/3, 2/3],
    ]])

    Q = np.array([[
        [1, 1/3, 2/3],
        [2/3, 1, 1/3],
        [1/3, 2/3, 1],
    ]])

    menPreferences = np.argsort(-P, axis=2)
    womenPreferences = np.argsort(-np.swapaxes(Q, 1, 2), axis=2)
    order = [1, 0, 2, 3, 5, 4]
    orders = np.array([[1, 0, 2, 3, 5, 4], [0, 5, 4, 3, 1, 2]])

    return P, Q, menPreferences, womenPreferences, order, orders


def main():
    P, Q, menPreferences, womenPreferences, order, orders = example_market()
    R = numba_DA(P, Q, menPreferences, womenPreferences)
    R2 = numba_SD(P, Q, menPreferences, womenPreferences, order)
    R3 = numba_TTC(P, Q, menPreferences, womenPreferences)
    R4 = numba_BSC(P, Q, menPreferences, womenPreferences)
    R5 = numba_RSD(P, Q, menPreferences, womenPreferences, orders)
    expected_R = np.array([[
        [0, 0, 1],
        [0, 1, 0],
        [1, 0, 0],
    ]])

    print("menPreferences:")
    print(menPreferences)
    print("womenPreferences:")
    print(womenPreferences)
    print("R(DA):")
    print(R.astype(int))
    print("R2(SD):")
    print(R2.astype(int))
    print("R3(TTC):")
    print(R3.astype(int))
    print(f"R4(BSC):\n{R4.astype(int)}")
    print(f"R5(RSD):\n{R5}")
    print(f"R5 averaged over all orders: \n{R5[0]}, order shape: {orders.shape[0]}")
    print("expected R:")
    print(expected_R)
    print("matches expected:", np.array_equal(R, expected_R))


if __name__ == "__main__":
    main()
