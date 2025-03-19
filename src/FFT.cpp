#include <cassert>
#include <cstring>
#include "setup.hpp"

using namespace std;

/*
In this programm, all the matrics are about discretisize in wall-normal direction, thus are all of double
*/

void init_fft()
{
    fft_temp_x1 = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * NX2);
    fft_temp_x2 = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * NX2);
    fft_temp_z1 = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * NZ2);
    fft_temp_z2 = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * NZ2);

    if (!fft_temp_x1 || !fft_temp_x2 || !fft_temp_z1 || !fft_temp_z2)
    {
        std::cerr << "Error: FFTW memory allocation failed!\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    plan_xf = fftw_plan_dft_1d(NX2, fft_temp_x1, fft_temp_x2, FFTW_FORWARD, FFTW_ESTIMATE);
    plan_zf = fftw_plan_dft_1d(NZ2, fft_temp_z1, fft_temp_z2, FFTW_FORWARD, FFTW_ESTIMATE);
    plan_xb = fftw_plan_dft_1d(NX2, fft_temp_x1, fft_temp_x2, FFTW_BACKWARD, FFTW_ESTIMATE);
    plan_zb = fftw_plan_dft_1d(NZ2, fft_temp_z1, fft_temp_z2, FFTW_BACKWARD, FFTW_ESTIMATE);

    if (!plan_xf || !plan_xb || !plan_zf || !plan_zb)
    {
        std::cerr << "Error: FFTW plan creation failed!\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

void cleanup_fft()
{
    fftw_destroy_plan(plan_xf);
    fftw_destroy_plan(plan_xb);
    fftw_destroy_plan(plan_zf);
    fftw_destroy_plan(plan_zb);

    fftw_free(fft_temp_x1);
    fftw_free(fft_temp_x2);
    fftw_free(fft_temp_z1);
    fftw_free(fft_temp_z2);
}

void Fourier2Physical_dealiased(const Field<Complex> &u_, Field<double> &up_)
{
    // u_.shape()=[NXHPX][NY][NZPZ]
    // up_.shape()=[NX2][NYPZ][NZ2PX]

    // check if input shape is right
    assert(u_.size_z() == size_t(NZPZ));
    assert(u_.size_x() == size_t(NXHPX));
    assert(u_.size_y() == size_t(NY));

    // check if output shape is right
    assert(up_.size_z() == size_t(NZ2PX));
    assert(up_.size_x() == size_t(NX2));
    assert(up_.size_y() == size_t(NYPZ));

    // change parallel direction from y to z
    Field<Complex> var_x1z1_pz(NZ, NXHPX, NYPZ);
    EXCHANGE_Y2Z(u_, var_x1z1_pz);

    // IFFT along z direction
    Field<Complex> var_x1z2_pz(NZ2, NXHPX, NYPZ);

    for (size_t i = 0; i < NXHPX; i++)
    {
        for (size_t j = 0; j < NYPZ; j++)
        {
            // zero-pad
            std::memset(fft_temp_z1, 0, sizeof(fftw_complex) * NZ2);

            for (size_t k = 0; k < NZH; k++)
            {
                fft_temp_z1[k][0] = var_x1z1_pz(k, i, j).real();
                fft_temp_z1[k][1] = var_x1z1_pz(k, i, j).imag();

                fft_temp_z1[k + NZ][0] = var_x1z1_pz(k + NZH, i, j).real();
                fft_temp_z1[k + NZ][1] = var_x1z1_pz(k + NZH, i, j).imag();
            }
            fft_temp_z1[NZ][0] = 0.0; // set Nyquist wave zero
            fft_temp_z1[NZ][1] = 0.0; // set Nyquist wave zero

            fftw_execute(plan_zb); // IFFT in z direction
            for (size_t k = 0; k < NZ2; k++)
            {
                var_x1z2_pz(k, i, j) = Complex(fft_temp_z2[k][0], fft_temp_z2[k][1]);
            }
        }
    }

    // change parallel direction from z to x
    Field<Complex> var_x1z2_px(NZ2PX, NXH, NYPZ);
    EXCHANGE_Z2X(var_x1z2_pz, var_x1z2_px);

    // IFFT along x direction
    for (size_t j = 0; j < NYPZ; j++)
    {
        for (size_t k = 0; k < NZ2PX; k++)
        {
            // zero-pad
            std::memset(fft_temp_x1, 0, sizeof(fftw_complex) * NX2);

            for (size_t i = 0; i < NXH; i++)
            {
                fft_temp_x1[i][0] = var_x1z2_px(k, i, j).real();
                fft_temp_x1[i][1] = var_x1z2_px(k, i, j).imag();

                if (i > 0) // Nyquist wave zero for i==0
                {
                    fft_temp_x1[i + NX][0] = var_x1z2_px(k, NXH - i, j).real();
                    fft_temp_x1[i + NX][1] = -var_x1z2_px(k, NXH - i, j).imag();
                }
            }

            fftw_execute(plan_xb); // IFFT in x direction
            for (size_t i = 0; i < NX2; i++)
            {
                up_(k, i, j) = fft_temp_x2[i][0];
            }
        }
    }
}

void Physical2Fourier_dealiased(const Field<double> &up_, Field<Complex> &u_)
{
    // up_.shape()=[NX2][NYPZ][NZ2PX]
    // u_.shape()=[NXHPX][NY][NZPZ]

    // check if input shape is right
    assert(up_.size_z() == size_t(NZ2PX));
    assert(up_.size_x() == size_t(NX2));
    assert(up_.size_y() == size_t(NYPZ));

    // check if output shape is right
    assert(u_.size_z() == size_t(NZPZ));
    assert(u_.size_x() == size_t(NXHPX));
    assert(u_.size_y() == size_t(NY));

    // FFT in x direction
    Field<Complex> var_x1z2_px(NZ2PX, NXH, NYPZ);
    for (size_t k = 0; k < NZ2PX; k++)
    {
        for (size_t j = 0; j < NYPZ; j++)
        {
            std::memset(fft_temp_x1, 0, sizeof(fftw_complex) * NX2);
            // FFT in x direction
            for (size_t i = 0; i < NX2; i++)
            {
                fft_temp_x1[i][0] = up_(k, i, j);
            }
            fftw_execute(plan_xf);
            // zero-pad
            for (size_t i = 0; i < NXH; i++)
            {
                var_x1z2_px(k, i, j) = Complex(fft_temp_x2[i][0], fft_temp_x2[i][1]) / (double)NX2;
            }
        }
    }

    // change parallel direction
    Field<Complex> var_x1z2_pz(NZ2, NXHPX, NYPZ);
    EXCHANGE_X2Z(var_x1z2_px, var_x1z2_pz);

    // FFT along z direction
    Field<Complex> var_x1z1_pz(NZ, NXHPX, NYPZ);
    for (size_t i = 0; i < NXHPX; i++)
    {
        for (size_t j = 0; j < NYPZ; j++)
        {
            // FFT in z direction
            for (size_t k = 0; k < NZ2; k++)
            {
                fft_temp_z1[k][0] = var_x1z2_pz(k, i, j).real();
                fft_temp_z1[k][1] = var_x1z2_pz(k, i, j).imag();
            }
            fftw_execute(plan_zf);
            // zero-pad
            for (size_t k = 0; k < NZH; k++)
            {
                var_x1z1_pz(k, i, j) = Complex(fft_temp_z2[k][0], fft_temp_z2[k][1]) / (double)NZ2;
                var_x1z1_pz(k + NZH, i, j) = Complex(fft_temp_z2[k + NZ][0], fft_temp_z2[k + NZ][1]) / (double)NZ2;
            }
        }
    }

    // change paralle direction from z to y
    EXCHANGE_Z2Y(var_x1z1_pz, u_);
}