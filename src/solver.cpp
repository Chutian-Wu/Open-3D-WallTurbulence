#include "setup.hpp"
#include "linalg.hpp"
#include "compdiff.hpp"

double coth(double x)
{
    return 1.0 / std::tanh(x);
}

double sech(double x)
{
    return 1.0 / std::cosh(x);
}

void allocate_var()
{
    /**
     * @brief wavenumbers
     * kx: positive half
     * kz: all
     */
    kx_local = Vec<double>(NXHPX);
    kz_local = Vec<double>(NZPZ);

    for (size_t i = 0; i < NXHPX; ++i)
    {
        kx_local[i] = (double)(PXID * NXHPX + i) * dkx;
    }
    for (size_t k = 0; k < NZPZ; ++k)
    {
        double kz_temp = (double)(NZPZ * PZID + k);
        if (kz_temp >= NZH)
        {
            kz_temp -= NZ;
        }
        kz_local[k] = kz_temp * dkz;
    }

    // if (my_rank == 0)
    // {
    //     std::cout << "rank: " << my_rank << " kx_local: ";
    //     kx_local.print();
    //     std::cout << "rank: " << my_rank << " kz_local: ";
    //     kz_local.print();
    // }

    /**
     * @brief y_mesh stuffs
     * s: computational grid [-1,1]
     * y: physical grid y=tanh(alpha s)/tanh(alpha)
     * alpha_mesh: alpha in upper eqn, the larger the refined near walls
     * dyds1: 1st derivative of y on s
     * dyds2: 2nd derivative of y on s
     * dyds3: 3rd derivative of y on s
     * dyds4: 4th derivative of y on s
     */
    s = Vec<double>(NY);
    y = Vec<double>(NY);
    dyds1 = Vec<double>(NY);
    dyds2 = Vec<double>(NY);
    dyds3 = Vec<double>(NY);
    dyds4 = Vec<double>(NY);

    ds = 2.0 / (NY - 1.0);

    double ap = alpha_mesh;
    double ap2 = pow(alpha_mesh, 2);
    double ap3 = pow(alpha_mesh, 3);
    double ap4 = pow(alpha_mesh, 4);

    for (size_t i = 0; i < NY; i++)
    {
        s[i] = 2 * i / (NY - 1.0) - 1.0; // s in [-1,1]
        double as = ap * s[i];

        y[i] = tanh(as) / tanh(ap);                                                          // y in [-1,1]
        dyds1[i] = ap * coth(ap) * pow(sech(as), 2);                                         // dyds1
        dyds2[i] = -2 * ap2 * coth(ap) * pow(sech(as), 2) * tanh(as);                        // dyds2
        dyds3[i] = 2 * ap3 * (-2 + cosh(2 * as)) * coth(ap) * pow(sech(as), 4);              // dyds3
        dyds4[i] = -2 * ap4 * coth(ap) * pow(sech(as), 5) * (-11 * sinh(as) + sinh(3 * as)); // dyds4
    }

    c11 = Vec<double>(NY);
    c21 = Vec<double>(NY);
    c22 = Vec<double>(NY);
    c31 = Vec<double>(NY);
    c32 = Vec<double>(NY);
    c33 = Vec<double>(NY);
    c41 = Vec<double>(NY);
    c42 = Vec<double>(NY);
    c43 = Vec<double>(NY);
    c44 = Vec<double>(NY);

    c11 = dyds1.pow(-1);

    c21 = -dyds2 * dyds1.pow(-3);
    c22 = dyds1.pow(-2);

    c31 = -dyds3 * dyds1.pow(-4) + 3.0 * dyds2.pow(2) * dyds1.pow(-5);
    c32 = -3.0 * dyds2 * dyds1.pow(-4);
    c33 = dyds1.pow(-3);

    c41 = -dyds4 * dyds1.pow(-5) + 10.0 * dyds3 * dyds2 * dyds1.pow(-6) - 15.0 * dyds2.pow(3) * dyds1.pow(-7);
    c42 = -4.0 * dyds3 * dyds1.pow(-5) + 15.0 * dyds2.pow(2) * dyds1.pow(-6);
    c43 = -6.0 * dyds2 * dyds1.pow(-5);
    c44 = dyds1.pow(-4);

    /**
     * @brief allocate fields vars
     * vars in wave space [NXHPX][NY][NZPZ]
     * vars in physical space [NX2][NYPZ][NZ2PX]
     */
    // velocity, vortocity, Lamb vector in physical space
    velx = Field<Complex>(NZPZ, NXHPX, NY);
    vely = Field<Complex>(NZPZ, NXHPX, NY);
    velz = Field<Complex>(NZPZ, NXHPX, NY);
    vorx = Field<Complex>(NZPZ, NXHPX, NY);
    vory = Field<Complex>(NZPZ, NXHPX, NY);
    vorz = Field<Complex>(NZPZ, NXHPX, NY);
    lmbx = Field<Complex>(NZPZ, NXHPX, NY);
    lmby = Field<Complex>(NZPZ, NXHPX, NY);
    lmbz = Field<Complex>(NZPZ, NXHPX, NY);

    Hv = Field<Complex>(NZPZ, NXHPX, NY);
    Hg = Field<Complex>(NZPZ, NXHPX, NY);

    vely_0 = Field<Complex>(NZPZ, NXHPX, NY);
    vely_1 = Field<Complex>(NZPZ, NXHPX, NY);
    vely_2 = Field<Complex>(NZPZ, NXHPX, NY);

    vory_0 = Field<Complex>(NZPZ, NXHPX, NY);
    vory_1 = Field<Complex>(NZPZ, NXHPX, NY);
    vory_2 = Field<Complex>(NZPZ, NXHPX, NY);

    Hv_0 = Field<Complex>(NZPZ, NXHPX, NY);
    Hv_1 = Field<Complex>(NZPZ, NXHPX, NY);
    Hv_2 = Field<Complex>(NZPZ, NXHPX, NY);

    Hg_0 = Field<Complex>(NZPZ, NXHPX, NY);
    Hg_1 = Field<Complex>(NZPZ, NXHPX, NY);
    Hg_2 = Field<Complex>(NZPZ, NXHPX, NY);

    lmbx_mean_0 = Vec<Complex>(NY);
    lmbx_mean_1 = Vec<Complex>(NY);
    lmbx_mean_2 = Vec<Complex>(NY);

    lmbz_mean_0 = Vec<Complex>(NY);
    lmbz_mean_1 = Vec<Complex>(NY);
    lmbz_mean_2 = Vec<Complex>(NY);

    velx_mean_0 = Vec<Complex>(NY);
    velx_mean_1 = Vec<Complex>(NY);
    velx_mean_2 = Vec<Complex>(NY);

    velz_mean_0 = Vec<Complex>(NY);
    velz_mean_1 = Vec<Complex>(NY);
    velz_mean_2 = Vec<Complex>(NY);

    // velocity, vortocity, Lamb vector in wave space
    velx_p = Field<double>(NZ2PX, NX2, NYPZ);
    vely_p = Field<double>(NZ2PX, NX2, NYPZ);
    velz_p = Field<double>(NZ2PX, NX2, NYPZ);
    vorx_p = Field<double>(NZ2PX, NX2, NYPZ);
    vory_p = Field<double>(NZ2PX, NX2, NYPZ);
    vorz_p = Field<double>(NZ2PX, NX2, NYPZ);
    lmbx_p = Field<double>(NZ2PX, NX2, NYPZ);
    lmby_p = Field<double>(NZ2PX, NX2, NYPZ);
    lmbz_p = Field<double>(NZ2PX, NX2, NYPZ);
}

void init_mat()
{
    /**
     * @brief 1st derivative operator
     * D1: derivative on computational grid
     * DY1: derivative on physical grid
     */
    Mat A1(NY, NY, 0.0);
    Mat B1(NY, NY, 0.0);
    op_A1B1(NY, ds, A1, B1);
    Mat D1 = A1.solve(B1);
    DY1 = c11 * D1;

    /**
     * @brief 2nd derivative operator
     * D2: derivative on computational grid
     * DY2: derivative on physical grid
     */
    Mat A2(NY, NY, 0.0);
    Mat B2(NY, NY, 0.0);
    op_A2B2(NY, ds, A2, B2);
    Mat D2 = A2.solve(B2);
    DY2 = c21 * D1 + c22 * D2;

    /**
     * @brief 3rd derivative operator
     * D3: derivative on computational grid
     * DY3: derivative on physical grid
     */
    Mat A3(NY, NY, 0.0);
    Mat B3(NY, NY, 0.0);
    op_A3B3(NY, ds, A3, B3);
    Mat D3 = A3.solve(B3);
    DY3 = c31 * D1 + c32 * D2 + c33 * D3;

    /**
     * @brief 4th derivative operator
     * D4: derivative on computational grid
     * DY4: derivative on physical grid
     */
    Mat A4(NY, NY, 0.0);
    Mat B4(NY, NY, 0.0);
    op_A4B4(NY, ds, A4, B4);
    Mat D4 = A4.solve(B4);
    DY4 = c41 * D1 + c42 * D2 + c43 * D3 + c44 * D4;
}

void get_vor_from_vel()
{
    /**
     * @brief solve vorticity from velocity
     * vorx = ∂y w - ∂z v
     * vory = ∂z u - ∂x w
     * vorz = ∂x v - ∂y u
     */

    for (size_t k = 0; k < NZPZ; k++)
    {
        for (size_t i = 0; i < NXHPX; i++)
        {
            double kx = kx_local[i];
            double kz = kz_local[k];
            vorx(k, i) = DY1 * velz(k, i) - IUNIT * kz * vely(k, i);
            vory(k, i) = IUNIT * kz * velx(k, i) - IUNIT * kx * velz(k, i);
            vorz(k, i) = IUNIT * kx * vely(k, i) - DY1 * velx(k, i);
        }
    }
}

void nonlinear_source()
{
    /**
     * @brief the source terms of equations for wall-normal componnets
     * Kim, P. Moin, R. Moser, Turbulence statistics in fully developed channel flow at low Reynolds number,
     * Journal of Fluid Mechanics 177 (1987) 133–166
     */

    for (size_t k = 0; k < NZPZ; k++)
    {
        for (size_t i = 0; i < NXHPX; i++)
        {
            double kz = kz_local[k];
            double kx = kx_local[i];
            double k2 = pow(kx, 2) + pow(kz, 2);

            Hv(k, i) = -k2 * lmby(k, i) - DY1 * (IUNIT * kx * lmbx(k, i) + IUNIT * kz * lmbz(k, i));
            Hg(k, i) = IUNIT * kz * lmbx(k, i) - IUNIT * kx * lmbz(k, i);
        }
    }
}

void Lamb_vec()
{
    /**
     * @brief Lamb vector
     * lmbx = v vorz - w vory
     * lmby = w vorx - u vorz
     * lmbz = u vory - v vorx
     * (1) IFFT (3/2 rule) to physical space (3/2 refined)
     * (2) multiplication in physical space
     * (3) FFT back
     */

    // Transform the velocity field
    Fourier2Physical_dealiased(velx, velx_p);
    Fourier2Physical_dealiased(vely, vely_p);
    Fourier2Physical_dealiased(velz, velz_p);

    // Transform the vorticity field
    Fourier2Physical_dealiased(vorx, vorx_p);
    Fourier2Physical_dealiased(vory, vory_p);
    Fourier2Physical_dealiased(vorz, vorz_p);

    lmbx_p = vely_p * vorz_p - velz_p * vory_p;
    lmby_p = velz_p * vorx_p - velx_p * vorz_p;
    lmbz_p = velx_p * vory_p - vely_p * vorx_p;

    // Transform the result back to Fourier space using 3/2 rule
    Physical2Fourier_dealiased(lmbx_p, lmbx);
    Physical2Fourier_dealiased(lmby_p, lmby);
    Physical2Fourier_dealiased(lmbz_p, lmbz);

    if (my_rank == 0)
    {

        lmbx_mean_2 = lmbx_mean_1;
        lmbx_mean_1 = lmbx_mean_0;
        lmbx_mean_0 = lmbx(0, 0);

        lmbz_mean_2 = lmbz_mean_1;
        lmbz_mean_1 = lmbz_mean_0;
        lmbz_mean_0 = lmbz(0, 0);
    }
}

void solve_friction_force()
{
    Vec<Complex> dudy(NY), dwdy(NY);
    dudy = DY1 * velx(0, 0);
    dwdy = DY1 * velz(0, 0);
    double Ly = y[NY - 1] - y[0];
    fx_fric = -nu / Ly * (dudy[NY - 1].real() - dudy[0].real());
    fz_fric = -nu / Ly * (dwdy[NY - 1].real() - dwdy[0].real());
    std::cout << "fx_fric, " << fx_fric << ", bottom, " << 2 * dudy[0].real() * nu / Ly << ", upper, " << 2 * dudy[NY - 1].real() * nu / Ly << "\n";
    // std::cout << "fz_fric " << fz_fric << std::endl;

    /**
     * @brief displace
     */
    fx_fric_2 = fx_fric_1;
    fx_fric_1 = fx_fric_0;
    fx_fric_0 = fx_fric;

    fz_fric_2 = fz_fric_1;
    fz_fric_1 = fz_fric_0;
    fz_fric_0 = fz_fric;
}

void init_fields()
{
    // initialize velocity field ///////////////////////////////////////////////
    /**
     * @brief laminar flow
     * y=1.5(1-y^2)
     */
    if (my_rank == 0)
    {
        for (size_t j = 0; j < NY; j++)
        {
            velx(0, 0, j) = Complex(1.5 * (1 - pow(y[j], 2)), 0.0);
        }
    }

    /**
     * @brief TS wave as perturbation
     * u'=(alpha/k1) (y-y^3) sin[k1 x]Cos[k2 z]
     * v'=alpha (y^2/2-y^4/4-1/4) cos[k1 x]cos[k2 z]
     * w'=(-2 alpha/k2) (y-y^3) cos[k1 x]sin[k2 z]
     */

    double k1 = 2.0;
    double k2 = 2.0;
    double alpha = 1.0;

    for (size_t k = 0; k < NZPZ; k++)
    {
        for (size_t i = 0; i < NXHPX; i++)
        {
            for (size_t j = 0; j < NY; j++)
            {
                double y_ = y[j];
                if (kz_local[k] == k2 && kx_local[i] == k1)
                {
                    velx(k, i, j) += alpha / k1 * (y_ - pow(y_, 3)) * (-0.25 * IUNIT);
                    vely(k, i, j) += alpha * (0.5 * pow(y_, 2) - 0.25 * pow(y_, 4) - 0.25) * (0.25);
                    velz(k, i, j) += -2 * alpha / k2 * (y_ - pow(y_, 3)) * (-0.25 * IUNIT);
                }

                if (kz_local[k] == -k2 && kx_local[i] == k1)
                {
                    velx(k, i, j) += alpha / k1 * (y_ - pow(y_, 3)) * (-0.25 * IUNIT);
                    vely(k, i, j) += alpha * (0.5 * pow(y_, 2) - 0.25 * pow(y_, 4) - 0.25) * (0.25);
                    velz(k, i, j) += -2 * alpha / k2 * (y_ - pow(y_, 3)) * (0.25 * IUNIT);
                }
            }
        }
    }

    get_vor_from_vel();
    Lamb_vec();
    nonlinear_source();

    vely_0 = vely;
    vely_1 = vely;
    vely_2 = vely;

    vory_0 = vory;
    vory_1 = vory;
    vory_2 = vory;

    Hv_0 = Hv;
    Hv_1 = Hv;
    Hv_2 = Hv;

    Hg_0 = Hg;
    Hg_1 = Hg;
    Hg_2 = Hg;

    if (my_rank == 0)
    {
        solve_friction_force();
        fx_fric_0 = fx_fric;
        fx_fric_1 = fx_fric;
        fx_fric_2 = fx_fric;

        fz_fric_0 = fz_fric;
        fz_fric_1 = fz_fric;
        fz_fric_2 = fz_fric;

        velx_mean_0 = velx(0, 0);
        velx_mean_1 = velx(0, 0);
        velx_mean_2 = velx(0, 0);

        velz_mean_0 = velz(0, 0);
        velz_mean_1 = velz(0, 0);
        velz_mean_2 = velz(0, 0);

        lmbx_mean_0 = lmbx(0, 0);
        lmbx_mean_1 = lmbx(0, 0);
        lmbx_mean_2 = lmbx(0, 0);

        lmbz_mean_0 = lmbz(0, 0);
        lmbz_mean_1 = lmbz(0, 0);
        lmbz_mean_2 = lmbz(0, 0);
    }
}

void solve_mean()
{

    if (my_rank > 0)
    {
        return;
    }

    // time integration scheme coeffcients
    double gamma_0 = 11.0 / 6.0; // weight for linear term
    double alpha_0 = 3.0;        // weight for linear term
    double alpha_1 = -1.5;       // weight for linear term
    double alpha_2 = 1.0 / 3.0;  // weight for linear term
    double beta_0 = 3.0;         // weight for nonlinear term
    double beta_1 = -3.0;        // weight for nonlinear term
    double beta_2 = 1.0;         // weight for nonlinear term

    /**
     * @brief mean streamwise and spanwise velocity
     */
    Mat A(NY, NY);
    A = -nu * DY2 + gamma_0 / dt * Mat::eye(NY);

    Vec<Complex> source_u(NY), source_w(NY);
    source_u = beta_0 * (lmbx_mean_0 + fx_fric_0) + beta_1 * (lmbx_mean_1 + fx_fric_1) + beta_2 * (lmbx_mean_2 + fx_fric_2);
    source_u = source_u + (alpha_0 * velx_mean_0 + alpha_1 * velx_mean_1 + alpha_2 * velx_mean_2) / dt;

    source_w = beta_0 * (lmbz_mean_0 + fz_fric_0) + beta_1 * (lmbz_mean_1 + fz_fric_1) + beta_2 * (lmbz_mean_2 + fz_fric_2);
    // source_w = source_w + (beta_0 * fz_fric_0 + beta_1 * fz_fric_1 + beta_2 * fz_fric_2);
    source_w = source_w + (alpha_0 * velz_mean_0 + alpha_1 * velz_mean_1 + alpha_2 * velz_mean_2) / dt;

    // std::cout << "time step " << id_step << "\n";
    // std::cout << "lmbx_mean_0\n";
    // lmbx_mean_0.print();
    // std::cout << "lmbx_mean_1\n";
    // lmbx_mean_1.print();
    // std::cout << "lmbx_mean_2\n";
    // lmbx_mean_2.print();
    // std::cout << "fx_fric_0 " << fx_fric_0 << " fx_fric_1 " << fx_fric_1 << " fx_fric_2 " << fx_fric_2 << "\n";
    // std::cout << "source_w\n";
    // source_w.print();

    /**
     * @brief Boundary conditions
     */
    A.set_row(0, 0.0);
    A(0, 0) = 1.0;
    A.set_row(NY - 1, 0.0);
    A(NY - 1, NY - 1) = 1.0;

    source_u[0] = 0.0;
    source_u[NY - 1] = 0.0;

    source_w[0] = 0.0;
    source_w[NY - 1] = 0.0;

    velx(0, 0) = A.solve(source_u);
    velz(0, 0) = A.solve(source_w);

    // std::cout << "matrix A\n";
    // A.print();

    // std::cout << "solved u_mean \n";
    // velx(0, 0).print();

    /**
     * @brief displace
     */
    velx_mean_2 = velx_mean_1;
    velx_mean_1 = velx_mean_0;
    velx_mean_0 = velx(0, 0);

    velz_mean_2 = velz_mean_1;
    velz_mean_1 = velz_mean_0;
    velz_mean_0 = velz(0, 0);

    /**
     * @brief friction force
     */
    solve_friction_force();

    /**
     * @brief mean vorticity
     */
    vorx(0, 0) = DY1 * velz(0, 0);
    vorz(0, 0) = -DY1 * velx(0, 0);
}

void solve_horizontal()
{

    for (size_t k = 0; k < NZPZ; k++)
    {
        for (size_t i = 0; i < NXHPX; i++)
        {
            double kx = kx_local[i];
            double kz = kz_local[k];
            double k2 = pow(kx, 2) + pow(kz, 2);
            if (k2 < 1.e-8)
            {
                solve_mean();
                continue;
            }

            velx(k, i) = IUNIT * kx / k2 * (DY1 * vely(k, i)) - IUNIT * kz / k2 * vory(k, i);
            velz(k, i) = IUNIT * kx / k2 * vory(k, i) + IUNIT * kz / k2 * (DY1 * vely(k, i));

            vorx(k, i) = IUNIT / k2 * (kx * (DY1 * vory(k, i)) + kz * (DY2 * vely(k, i)) - kz * k2 * vely(k, i));
            vorz(k, i) = IUNIT / k2 * (kz * (DY1 * vory(k, i)) - kx * (DY2 * vely(k, i)) + kx * k2 * vely(k, i));
        }
    }
}

void time_Integration()
{
    /**
     * @brief time integration using high-order stiff scheme
     * Karniadakis G E, Israeli M, Orszag S A. High-order splitting methods for the incompressible Navier-Stokes equations[J].
     * Journal of computational physics, 1991, 97(2): 414-443.
     */

    // time integration scheme coeffcients
    double gamma_0 = 11.0 / 6.0;
    double alpha_0 = 3.0;
    double alpha_1 = -1.5;
    double alpha_2 = 1.0 / 3.0;
    double beta_0 = 3.0;
    double beta_1 = -3.0;
    double beta_2 = 1.0;

    // solve ODE
    Vec<Complex> source_v(NY), source_g(NY);
    Mat Av(NY, NY), Ag(NY, NY);
    for (size_t k = 0; k < NZPZ; k++)
    {
        for (size_t i = 0; i < NXHPX; i++)
        {
            double k2 = pow(kx_local[i], 2) + pow(kz_local[k], 2);
            double k4 = pow(k2, 2);

            // right-hand terms
            source_v = Hv_0(k, i) * beta_0 + Hv_1(k, i) * beta_1 + Hv_2(k, i) * beta_2;
            source_v = source_v - (k2 / dt) * (vely_0(k, i) * alpha_0 + vely_1(k, i) * alpha_1 + vely_2(k, i) * alpha_2);
            source_v = source_v + (DY2 * (vely_0(k, i) * alpha_0 + vely_1(k, i) * alpha_1 + vely_2(k, i) * alpha_2)) / dt;

            source_g = Hg_0(k, i) * beta_0 + Hg_1(k, i) * beta_1 + Hg_2(k, i) * beta_2;
            source_g = source_g + (vory_0(k, i) * alpha_0 + vory_1(k, i) * alpha_1 + vory_2(k, i) * alpha_2) / dt;

            // linear systems
            Av = -nu * DY4 + (gamma_0 / dt + 2 * nu * k2) * DY2 - (nu * k4 + gamma_0 * k2 / dt) * Mat::eye(NY);
            Ag = -nu * DY2 + (nu * k2 + gamma_0 / dt) * Mat::eye(NY);

            // Boundary conditions
            Av.set_row(0, 0.0);
            Av(0, 0) = 1.0;
            Av.set_row(1, DY1, 0);
            Av.set_row(NY - 2, DY1, NY - 1);
            Av.set_row(NY - 1, 0.0);
            Av(NY - 1, NY - 1) = 1.0;

            Ag.set_row(0, 0.0);
            Ag(0, 0) = 1.0;
            Ag.set_row(NY - 1, 0.0);
            Ag(NY - 1, NY - 1) = 1.0;

            source_v[0] = Complex(0.0, 0.0);
            source_v[1] = Complex(0.0, 0.0);
            source_v[NY - 2] = Complex(0.0, 0.0);
            source_v[NY - 1] = Complex(0.0, 0.0);

            source_g[0] = Complex(0.0, 0.0);
            source_g[NY - 1] = Complex(0.0, 0.0);

            // solve linear system and update
            vely(k, i) = Av.solve(source_v);
            vory(k, i) = Ag.solve(source_g);
        }
    }

    // solve velocity and displace
    solve_horizontal();
    Lamb_vec();
    nonlinear_source();

    vely_2 = vely_1;
    vely_1 = vely_0;
    vely_0 = vely;

    vory_2 = vory_1;
    vory_1 = vory_0;
    vory_0 = vory;

    Hv_2 = Hv_1;
    Hv_1 = Hv_0;
    Hv_0 = Hv;

    Hg_2 = Hg_1;
    Hg_1 = Hg_0;
    Hg_0 = Hg;
}
