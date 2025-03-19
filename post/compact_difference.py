import numpy as np


def mesh_tanh(N, alpha):
    eta = np.linspace(-1, 1, N)
    deta = 2.0 / (N - 1.0)

    ap = alpha
    aeta = ap * eta
    ap2 = np.power(ap, 2)
    ap3 = np.power(ap, 3)
    ap4 = np.power(ap, 4)
    tap = np.tanh(ap)

    # y = np.tanh(aeta) / tap
    sech2 = np.power(np.cosh(aeta), -2)
    tanh2 = np.power(np.tanh(aeta), 2)
    taeta = np.tanh(aeta)

    dyde1 = ap / np.tanh(ap) * np.power(np.cosh(aeta), -2)
    dyde2 = -2 * ap2 / np.tanh(ap) * np.power(np.cosh(aeta),
                                              -2) * np.tanh(aeta)
    dyde3 = (-2 * ap3 * sech2**2 + 4 * ap3 * sech2 * tanh2) / tap
    dyde4 = -2 * ap4 * np.cosh(ap) / np.sinh(ap) * np.power(
        np.cosh(aeta), -5) * (-11 * np.sinh(aeta) + np.sinh(3 * aeta))

    c11 = 1 / dyde1

    c21 = -dyde2 * np.power(dyde1, -2)
    c22 = np.power(dyde1, -2)

    c31 = -dyde3 / np.power(dyde1, 4)
    c31 += 3 * np.power(dyde2, 2) / np.power(dyde1, 5)
    c32 = -3 * dyde2 / np.power(dyde1, 4)
    c33 = np.power(dyde1, -3)

    c41 = -dyde4 * np.power(dyde1, -5)
    c41 += 10 * dyde3 * dyde2 * np.power(dyde1, -6)
    c41 -= 15 * np.power(dyde2, 3) * np.power(dyde1, -7)
    c42 = -4 * dyde3 * np.power(dyde1, -5)
    c42 += 15 * np.power(dyde2, 2) * np.power(dyde1, -6)
    c43 = -6 * dyde2 * np.power(dyde1, -5)
    c44 = np.power(dyde1, -4)

    mesh_dict = {
        'eta': eta,
        'deta': deta,
        'c11': c11,
        'c21': c21,
        'c22': c22,
        'c31': c31,
        'c32': c32,
        'c33': c33,
        'c41': c41,
        'c42': c42,
        'c43': c43,
        'c44': c44
    }
    return mesh_dict


def operator_D1(N, h):
    # A1@u'=B1@u
    # computation grid interval: h

    # init
    A1 = np.zeros((N, N))
    B1 = np.zeros((N, N))

    # interior 8th order
    alpha = 4 / 9
    beta = 1 / 36
    a = 40 / 27
    b = 25 / 54
    # alpha = 1.0 / 3.0
    # beta = 0.0
    # a = 14.0 / 9.0
    # b = 1.0 / 9.0
    for i in range(2, N - 2):
        A1[i, i - 2] = beta
        A1[i, i - 1] = alpha
        A1[i, i] = 1.0
        A1[i, i + 1] = alpha
        A1[i, i + 2] = beta

        B1[i, i - 2] = -b / (4 * h)
        B1[i, i - 1] = -a / (2 * h)
        B1[i, i + 1] = a / (2 * h)
        B1[i, i + 2] = b / (4 * h)

    # boundary 2nd grid 4th order
    alpha = 1 / 4
    a = 3 / 2
    A1[1, 0] = alpha
    A1[1, 1] = 1
    A1[1, 2] = alpha

    A1[-2, -3] = alpha
    A1[-2, -2] = 1
    A1[-2, -1] = alpha

    B1[1, 0] = -a / (2 * h)
    B1[1, 2] = a / (2 * h)

    B1[-2, -3] = -a / (2 * h)
    B1[-2, -1] = a / (2 * h)

    # boundary 1st grid
    A1[0, 0] = 1
    A1[0, 1] = 3

    A1[-1, -1] = 1
    A1[-1, -2] = 3

    B1[0, 0] = -17 / 6 / h
    B1[0, 1] = 1.5 / h
    B1[0, 2] = 1.5 / h
    B1[0, 3] = -1 / 6 / h

    B1[-1, -1] = 17 / 6 / h
    B1[-1, -2] = -1.5 / h
    B1[-1, -3] = -1.5 / h
    B1[-1, -4] = 1 / 6 / h

    return np.linalg.solve(A1, B1)


def operator_D2(N, h):
    # A2@u''=B2@u
    # computation grid interval: h

    A2 = np.zeros((N, N))
    B2 = np.zeros((N, N))

    # params interior 8th order
    alpha = 344 / 1179
    beta = 23 / 2358
    a = 320 / 393
    b = 310 / 393
    for i in range(2, N - 2):
        A2[i, i - 2] = beta
        A2[i, i - 1] = alpha
        A2[i, i] = 1.0
        A2[i, i + 1] = alpha
        A2[i, i + 2] = beta

        B2[i, i - 2] = b / (4 * h**2)
        B2[i, i - 1] = a / (h**2)
        B2[i, i] = -2 * b / (4 * h**2) - 2 * a / (h**2)
        B2[i, i + 1] = a / (h**2)
        B2[i, i + 2] = b / (4 * h**2)
    # boundary 2nd grid (4th order most compact)
    alpha = 1 / 10
    a = 6 / 5
    A2[1, 0] = alpha
    A2[1, 1] = 1
    A2[1, 2] = alpha

    A2[N - 2, N - 3] = alpha
    A2[N - 2, N - 2] = 1
    A2[N - 2, N - 1] = alpha

    B2[1, 0] = a / (h**2)
    B2[1, 1] = -2 * a / (h**2)
    B2[1, 2] = a / (h**2)

    B2[N - 2, N - 3] = a / (h**2)
    B2[N - 2, N - 2] = -2 * a / (h**2)
    B2[N - 2, N - 1] = a / (h**2)

    # boundary 1st grid
    A2[0, 0] = 1
    A2[0, 1] = 10  # 11  #

    A2[-1, -1] = 1
    A2[-1, -2] = 10  # 11  #

    B2[0, 0] = 145 / 12 / (h**2)  # 13 / (h**2)  #
    B2[0, 1] = -76 / 3 / (h**2)  #-27 / (h**2)  #
    B2[0, 2] = 29 / 2 / (h**2)  #15 / (h**2)  #
    B2[0, 3] = -4 / 3 / (h**2)  # -1 / (h**2)  #
    B2[0, 4] = 1 / 12 / (h**2)

    B2[-1, -1] = 145 / 12 / (h**2)  # 13 / (h**2)  #
    B2[-1, -2] = -76 / 3 / (h**2)  #-27 / (h**2)  #
    B2[-1, -3] = 29 / 2 / (h**2)  #15 / (h**2)  #
    B2[-1, -4] = -4 / 3 / (h**2)  # -1 / (h**2)  #
    B2[-1, -5] = 1 / 12 / (h**2)

    return np.linalg.solve(A2, B2)


def operator_D3(N, h):
    # A3@u'''=B3@u
    # computation grid interval: h

    h3 = np.power(h, 3)

    # init
    A3 = np.zeros((N, N))
    B3 = np.zeros((N, N))

    # interior (8th order)
    alpha = 147 / 332
    beta = 1 / 166
    a = 160 / 83
    b = -5 / 166
    # alpha = 7 / 16
    # beta = 0
    # a = 2
    # b = -1 / 8
    for i in range(3, N - 3):
        A3[i, i - 2] = beta
        A3[i, i - 1] = alpha
        A3[i, i] = 1.0
        A3[i, i + 1] = alpha
        A3[i, i + 2] = beta

        B3[i, i - 3] = -b / (8 * h3)
        B3[i, i - 2] = -a / (2 * h3)
        B3[i, i - 1] = a / h3 + 3 * b / (8 * h3)
        B3[i, i + 1] = -a / h3 - 3 * b / (8 * h3)
        B3[i, i + 2] = a / (2 * h3)
        B3[i, i + 3] = b / (8 * h3)

    # boundary 3rd grid (6th order)
    alpha = 4 / 9
    beta = 1 / 126
    a = 40 / 21
    for i in [2, N - 3]:
        A3[i, i - 2] = beta
        A3[i, i - 1] = alpha
        A3[i, i] = 1.0
        A3[i, i + 1] = alpha
        A3[i, i + 2] = beta

        B3[i, i - 2] = -a / (2 * h3)
        B3[i, i - 1] = a / h3
        B3[i, i + 1] = -a / h3
        B3[i, i + 2] = a / (2 * h3)

    # # boundary 2nd grid (4th order, 4-stencil)
    # A3[1, 1] = 0.5
    # A3[1, 2] = 0.5

    # A3[-2, -2] = 0.5
    # A3[-2, -3] = 0.5

    # B3[1, 0] = -1 / h3
    # B3[1, 1] = 3 / h3
    # B3[1, 2] = -3 / h3
    # B3[1, 3] = 1 / h3

    # B3[-2, -1] = 1 / h3
    # B3[-2, -2] = -3 / h3
    # B3[-2, -3] = 3 / h3
    # B3[-2, -4] = -1 / h3

    # boundary 2nd grid (4th order, 5-stencil)
    alpha = 1 / 2
    a = 2
    A3[1, 1] = alpha
    A3[1, 2] = 1
    A3[1, 3] = alpha

    A3[N - 2, N - 4] = alpha
    A3[N - 2, N - 3] = 1
    A3[N - 2, N - 2] = alpha

    B3[1, 0] = -a / (2 * h3)
    B3[1, 1] = a / h3
    B3[1, 3] = -a / h3
    B3[1, 4] = a / (2 * h3)

    B3[N - 2, N - 5] = -a / (2 * h3)
    B3[N - 2, N - 4] = a / h3
    B3[N - 2, N - 2] = -a / h3
    B3[N - 2, N - 1] = a / (2 * h3)

    # boundary 1st grid (5th bc scheme, beta=0)
    alpha = -7
    a = 8
    b = -26
    c = 30
    d = -14
    e = 2

    A3[0, 0] = 1.0
    A3[0, 1] = alpha

    B3[0, 0] = a / h3
    B3[0, 1] = b / h3
    B3[0, 2] = c / h3
    B3[0, 3] = d / h3
    B3[0, 4] = e / h3

    A3[-1, -1] = 1.0
    A3[-1, -2] = alpha

    B3[-1, -1] = -a / h3
    B3[-1, -2] = -b / h3
    B3[-1, -3] = -c / h3
    B3[-1, -4] = -d / h3
    B3[-1, -5] = -e / h3

    return np.linalg.solve(A3, B3)


def operator_D4(N, h):
    # A4@u''''=B4@u
    # computation grid interval: h
    h4 = np.power(h, 4)
    # init
    A4 = np.zeros((N, N))
    B4 = np.zeros((N, N))

    # interior (8th order)
    alpha = 7 / 26
    a = 19 / 13
    b = 1 / 13
    for i in range(3, N - 3):
        A4[i, i - 1] = alpha
        A4[i, i] = 1.0
        A4[i, i + 1] = alpha

        B4[i, i - 3] = b / (6 * h4)
        B4[i, i - 2] = a / h4
        B4[i, i - 1] = -4 * a / h4 - 9 * b / (6 * h4)
        B4[i, i] = 6 * a / h4 + 16 * b / (6 * h4)
        B4[i, i + 1] = -4 * a / h4 - 9 * b / (6 * h4)
        B4[i, i + 2] = a / h4
        B4[i, i + 3] = b / (6 * h4)

    # boundary 3rd grid (4th order, the most compact)
    alpha = 1 / 4  # 124 / 474  #
    beta = 0  #-1 / 474  #
    a = 3 / 2  #120 / 79  #
    for i in [2, N - 3]:
        A4[i, i - 2] = beta
        A4[i, i - 1] = alpha
        A4[i, i] = 1.0
        A4[i, i + 1] = alpha
        A4[i, i + 2] = beta

        B4[i, i - 2] = a / h4
        B4[i, i - 1] = -4 * a / h4
        B4[i, i] = 6 * a / h4
        B4[i, i + 1] = -4 * a / h4
        B4[i, i + 2] = a / h4

    # boundary 2nd grid (6th bc scheme)
    alpha = -2
    beta = 7
    a = 6
    b = -24
    c = 36
    d = -24
    e = 6

    A4[1, 0] = 1.0
    A4[1, 1] = alpha
    A4[1, 2] = beta

    B4[1, 0] = a / h4
    B4[1, 1] = b / h4
    B4[1, 2] = c / h4
    B4[1, 3] = d / h4
    B4[1, 4] = e / h4

    A4[-2, -1] = 1.0
    A4[-2, -2] = alpha
    A4[-2, -3] = beta

    B4[-2, -1] = a / h4
    B4[-2, -2] = b / h4
    B4[-2, -3] = c / h4
    B4[-2, -4] = d / h4
    B4[-2, -5] = e / h4

    # boundary 1st grid (5th bc scheme, beta=0)
    alpha = -2
    beta = 0
    a = -1 + beta
    b = 4 * (1 - beta)
    c = 6 * (-1 + beta)
    d = 4 * (1 - beta)
    e = -1 + beta

    A4[0, 0] = 1.0
    A4[0, 1] = alpha
    A4[0, 2] = beta

    B4[0, 0] = a / h4
    B4[0, 1] = b / h4
    B4[0, 2] = c / h4
    B4[0, 3] = d / h4
    B4[0, 4] = e / h4

    A4[-1, -1] = 1.0
    A4[-1, -2] = alpha
    A4[-1, -3] = beta

    B4[-1, -1] = a / h4
    B4[-1, -2] = b / h4
    B4[-1, -3] = c / h4
    B4[-1, -4] = d / h4
    B4[-1, -5] = e / h4

    # # boundary 1st grid
    # A4[0, 0] = 1

    # B4[0, 0] = 1 / h4
    # B4[0, 1] = -4 / h4
    # B4[0, 2] = 6 / h4
    # B4[0, 3] = -4 / h4
    # B4[0, 4] = 1 / h4

    # A4[-1, -1] = 1

    # B4[-1, -1] = 1 / h4
    # B4[-1, -2] = -4 / h4
    # B4[-1, -3] = 6 / h4
    # B4[-1, -4] = -4 / h4
    # B4[-1, -5] = 1 / h4

    return np.linalg.solve(A4, B4)
