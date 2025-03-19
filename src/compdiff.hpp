#ifndef COMPDIFF_HPP
#define COMPDIFF_HPP

#include <vector>
#include <iostream>
#include <stdexcept>
#include "linalg.hpp"

using namespace std;

void op_A1B1(const size_t N, const double h, Mat &A1, Mat &B1)
{
    if (N < 5)
        throw invalid_argument("N must be at least 5 for 1st derivative scheme");

    // 8th order for inner grids
    constexpr double alpha_2 = 4.0 / 9.0;
    constexpr double beta_2 = 1.0 / 36.0;
    constexpr double a_2 = 40.0 / 27.0;
    constexpr double b_2 = 25.0 / 54.0;
    for (size_t i = 2; i < N - 2; i++)
    {
        A1(i, i - 2) = beta_2;
        A1(i, i - 1) = alpha_2;
        A1(i, i) = 1.0;
        A1(i, i + 1) = alpha_2;
        A1(i, i + 2) = beta_2;

        B1(i, i - 2) = -b_2 / (4 * h);
        B1(i, i - 1) = -a_2 / (2 * h);
        B1(i, i + 1) = a_2 / (2 * h);
        B1(i, i + 2) = b_2 / (4 * h);
    }

    // 4th order for 2nd boundary grids
    constexpr double alpha_1 = 0.25;
    constexpr double a_1 = 1.5;
    for (int i : {1, int(N - 2)})
    {
        A1(i, i - 1) = alpha_1;
        A1(i, i) = 1.0;
        A1(i, i + 1) = alpha_1;

        B1(i, i - 1) = -a_1 / (2 * h);
        B1(i, i + 1) = a_1 / (2 * h);
    }

    // 4th order for 1st boundary grids
    constexpr double alpha_0 = 3.0;
    constexpr double a_0 = -17.0 / 6.0;
    constexpr double b_0 = 1.5;
    constexpr double c_0 = 1.5;
    constexpr double d_0 = -1.0 / 6.0;
    A1(0, 0) = 1.0;
    A1(0, 1) = alpha_0;

    A1(N - 1, N - 1) = 1.0;
    A1(N - 1, N - 2) = alpha_0;

    B1(0, 0) = a_0 / h;
    B1(0, 1) = b_0 / h;
    B1(0, 2) = c_0 / h;
    B1(0, 3) = d_0 / h;

    B1(N - 1, N - 1) = -a_0 / h;
    B1(N - 1, N - 2) = -b_0 / h;
    B1(N - 1, N - 3) = -c_0 / h;
    B1(N - 1, N - 4) = -d_0 / h;
}

void op_A2B2(const size_t N, const double h, Mat &A2, Mat &B2)
{
    if (N < 5)
        throw invalid_argument("N must be at least 5 for 2nd derivative scheme");

    double h2 = pow(h, 2);

    // 8th order for inner grids
    constexpr double alpha_2 = 344.0 / 1179.0;
    constexpr double beta_2 = 23.0 / 2358.0;
    constexpr double a_2 = 320.0 / 393.0;
    constexpr double b_2 = 310.0 / 393.0;
    for (size_t i = 2; i < N - 2; i++)
    {
        A2(i, i - 2) = beta_2;
        A2(i, i - 1) = alpha_2;
        A2(i, i) = 1.0;
        A2(i, i + 1) = alpha_2;
        A2(i, i + 2) = beta_2;

        B2(i, i - 2) = b_2 / (4 * h2);
        B2(i, i - 1) = a_2 / h2;
        B2(i, i) = -2 * b_2 / (4 * h2) - 2 * a_2 / h2;
        B2(i, i + 1) = a_2 / h2;
        B2(i, i + 2) = b_2 / (4 * h2);
    }

    // // 4th order for 2nd boundary grids
    // constexpr double alpha_1 = 0.1;
    // constexpr double a_1 = 1.2;
    // for (int i : {1, int(N - 2)})
    // {
    //     A2(i, i - 1) = alpha_1;
    //     A2(i, i) = 1.0;
    //     A2(i, i + 1) = alpha_1;

    //     B2(i, i - 1) = a_1 / h2;
    //     B2(i, i) = -2 * a_1 / h2;
    //     B2(i, i + 1) = a_1 / h2;
    // }

    // 6th order for 2nd boundary grids
    constexpr double alpha_1 = 2.0 / 11.0;
    constexpr double a_1 = 12.0 / 11.0;
    constexpr double b_1 = 3.0 / 11.0;
    for (size_t i : {size_t(1), size_t(N - 2)})
    {
        int i_shift = 0;
        if (i == 1)
        {
            i_shift = i + 1;
        }
        if (i == static_cast<size_t>(N - 2))
        {
            i_shift = i - 1;
        }

        A2(i, i_shift - 1) = alpha_1;
        A2(i, i_shift) = 1.0;
        A2(i, i_shift + 1) = alpha_1;

        B2(i, i_shift - 2) = b_1 / (4 * h2);
        B2(i, i_shift - 1) = a_1 / h2;
        B2(i, i_shift) = -2 * b_1 / (4 * h2) - 2 * a_1 / h2;
        B2(i, i_shift + 1) = a_1 / h2;
        B2(i, i_shift + 2) = b_1 / (4 * h2);
    }

    // 4th order for 1st boundary grids
    constexpr double alpha_0 = 10.0; // 10.0;
    constexpr double beta_0 = 1.0;   // 0.0;
    constexpr double a_0 = 12.0;     // 145.0 / 12.0;
    constexpr double b_0 = -24.0;    // -76.0 / 3.0;
    constexpr double c_0 = 12.0;     // 29.0 / 2.0;
    constexpr double d_0 = 0.0;      //-4.0 / 3.0;
    constexpr double e_0 = 0.0;      // 1.0 / 12.0;
    A2(0, 0) = 1.0;
    A2(0, 1) = alpha_0;
    A2(0, 2) = beta_0;

    A2(N - 1, N - 1) = 1.0;
    A2(N - 1, N - 2) = alpha_0;
    A2(N - 1, N - 3) = beta_0;

    B2(0, 0) = a_0 / h2;
    B2(0, 1) = b_0 / h2;
    B2(0, 2) = c_0 / h2;
    B2(0, 3) = d_0 / h2;
    B2(0, 4) = e_0 / h2;

    B2(N - 1, N - 1) = a_0 / h2;
    B2(N - 1, N - 2) = b_0 / h2;
    B2(N - 1, N - 3) = c_0 / h2;
    B2(N - 1, N - 4) = d_0 / h2;
    B2(N - 1, N - 4) = e_0 / h2;
}

void op_A3B3(const size_t N, const double h, Mat &A3, Mat &B3)
{
    if (N < 7)
        throw invalid_argument("N must be at least 7 for 3rd derivative scheme");

    double h3 = pow(h, 3);

    // 8th order for inner grids
    constexpr double alpha_3 = 147.0 / 332.0;
    constexpr double beta_3 = 1.0 / 166.0;
    constexpr double a_3 = 160.0 / 83.0;
    constexpr double b_3 = -5.0 / 166.0;
    for (size_t i = 3; i < N - 3; i++)
    {
        A3(i, i - 2) = beta_3;
        A3(i, i - 1) = alpha_3;
        A3(i, i) = 1.0;
        A3(i, i + 1) = alpha_3;
        A3(i, i + 2) = beta_3;

        B3(i, i - 3) = -b_3 / (8 * h3);
        B3(i, i - 2) = -a_3 / (2 * h3);
        B3(i, i - 1) = a_3 / h3 + 3 * b_3 / (8 * h3);
        B3(i, i + 1) = -a_3 / h3 - 3 * b_3 / (8 * h3);
        B3(i, i + 2) = a_3 / (2 * h3);
        B3(i, i + 3) = b_3 / (8 * h3);
    }

    // 6th order for 3rd boundary grids
    constexpr double alpha_2 = 4.0 / 9.0;
    constexpr double beta_2 = 1.0 / 126.0;
    constexpr double a_2 = 40.0 / 21.0;
    for (int i : {2, int(N - 3)})
    {
        A3(i, i - 2) = beta_2;
        A3(i, i - 1) = alpha_2;
        A3(i, i) = 1.0;
        A3(i, i + 1) = alpha_2;
        A3(i, i + 2) = beta_2;

        B3(i, i - 2) = -a_2 / (2 * h3);
        B3(i, i - 1) = a_2 / h3;
        B3(i, i + 1) = -a_2 / h3;
        B3(i, i + 2) = a_2 / (2 * h3);
    }

    // 5th order for 2nd boundary grids
    constexpr double alpha_1 = 0.5;
    constexpr double a_1 = 2.0;
    A3(1, 1) = alpha_1;
    A3(1, 2) = 1.0;
    A3(1, 3) = alpha_1;

    A3(N - 2, N - 4) = alpha_1;
    A3(N - 2, N - 3) = 1.0;
    A3(N - 2, N - 2) = alpha_1;

    B3(1, 0) = -a_1 / (2 * h3);
    B3(1, 1) = a_1 / h3;
    B3(1, 3) = -a_1 / h3;
    B3(1, 4) = a_1 / (2 * h3);

    B3(N - 2, N - 5) = -a_1 / (2 * h3);
    B3(N - 2, N - 4) = a_1 / h3;
    B3(N - 2, N - 2) = -a_1 / h3;
    B3(N - 2, N - 1) = a_1 / (2 * h3);

    // 5th order for 1st boundary grids
    constexpr double alpha_0 = -7.0;
    constexpr double a_0 = 8.0;
    constexpr double b_0 = -26.0;
    constexpr double c_0 = 30.0;
    constexpr double d_0 = -14.0;
    constexpr double e_0 = 2.0;
    A3(0, 0) = 1.0;
    A3(0, 1) = alpha_0;

    A3(N - 1, N - 1) = 1.0;
    A3(N - 1, N - 2) = alpha_0;

    B3(0, 0) = a_0 / h3;
    B3(0, 1) = b_0 / h3;
    B3(0, 2) = c_0 / h3;
    B3(0, 3) = d_0 / h3;
    B3(0, 4) = e_0 / h3;

    B3(N - 1, N - 1) = -a_0 / h3;
    B3(N - 1, N - 2) = -b_0 / h3;
    B3(N - 1, N - 3) = -c_0 / h3;
    B3(N - 1, N - 4) = -d_0 / h3;
    B3(N - 1, N - 4) = -e_0 / h3;
}

void op_A4B4(const size_t N, const double h, Mat &A4, Mat &B4)
{
    if (N < 7)
        throw invalid_argument("N must be at least 7 for 4th derivative scheme");

    double h4 = pow(h, 4);

    // 8th order for inner grids
    constexpr double alpha_3 = 7.0 / 26.0;
    constexpr double a_3 = 19.0 / 13.0;
    constexpr double b_3 = 1.0 / 13.0;
    for (size_t i = 3; i < N - 3; i++)
    {
        A4(i, i - 1) = alpha_3;
        A4(i, i) = 1.0;
        A4(i, i + 1) = alpha_3;

        B4(i, i - 3) = b_3 / (6 * h4);
        B4(i, i - 2) = a_3 / h4;
        B4(i, i - 1) = -4 * a_3 / h4 - 9 * b_3 / (6 * h4);
        B4(i, i) = 6 * a_3 / h4 + 16 * b_3 / (6 * h4);
        B4(i, i + 1) = -4 * a_3 / h4 - 9 * b_3 / (6 * h4);
        B4(i, i + 2) = a_3 / h4;
        B4(i, i + 3) = b_3 / (6 * h4);
    }

    // 4th order for 3rd boundary grids (the most compact)
    constexpr double alpha_2 = 0.25;
    constexpr double a_2 = 1.5;
    for (int i : {2, int(N - 3)})
    {
        A4(i, i - 1) = alpha_2;
        A4(i, i) = 1.0;
        A4(i, i + 1) = alpha_2;

        B4(i, i - 2) = a_2 / h4;
        B4(i, i - 1) = -4 * a_2 / h4;
        B4(i, i) = 6 * a_2 / h4;
        B4(i, i + 1) = -4 * a_2 / h4;
        B4(i, i + 2) = a_2 / h4;
    }

    // 6th order for 2nd boundary grids
    constexpr double alpha_1 = -2.0;
    constexpr double beta_1 = 7.0;
    constexpr double a_1 = 6.0;
    constexpr double b_1 = -24.0;
    constexpr double c_1 = 36.0;
    constexpr double d_1 = -24.0;
    constexpr double e_1 = 6.0;

    A4(1, 0) = 1.0;
    A4(1, 1) = alpha_1;
    A4(1, 2) = beta_1;

    B4(1, 0) = a_1 / h4;
    B4(1, 1) = b_1 / h4;
    B4(1, 2) = c_1 / h4;
    B4(1, 3) = d_1 / h4;
    B4(1, 4) = e_1 / h4;

    A4(N - 2, N - 1) = 1.0;
    A4(N - 2, N - 2) = alpha_1;
    A4(N - 2, N - 3) = beta_1;

    B4(N - 2, N - 1) = a_1 / h4;
    B4(N - 2, N - 2) = b_1 / h4;
    B4(N - 2, N - 3) = c_1 / h4;
    B4(N - 2, N - 4) = d_1 / h4;
    B4(N - 2, N - 5) = e_1 / h4;

    // 5th order for 1st boundary grids
    constexpr double alpha_0 = -2.0;
    constexpr double a_0 = -1.0;
    constexpr double b_0 = 4.0;
    constexpr double c_0 = -6.0;
    constexpr double d_0 = 4.0;
    constexpr double e_0 = -1.0;

    A4(0, 0) = 1.0;
    A4(0, 1) = alpha_0;

    B4(0, 0) = a_0 / h4;
    B4(0, 1) = b_0 / h4;
    B4(0, 2) = c_0 / h4;
    B4(0, 3) = d_0 / h4;
    B4(0, 4) = e_0 / h4;

    A4(N - 1, N - 1) = 1.0;
    A4(N - 1, N - 2) = alpha_0;

    B4(N - 1, N - 1) = a_0 / h4;
    B4(N - 1, N - 2) = b_0 / h4;
    B4(N - 1, N - 3) = c_0 / h4;
    B4(N - 1, N - 4) = d_0 / h4;
    B4(N - 1, N - 5) = e_0 / h4;
}

#endif
