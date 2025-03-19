#ifndef LINALG_HPP
#define LINALG_HPP
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <complex>
#include <iomanip>
#include <lapacke.h>
#include <cblas.h>

/**
 * @brief Vector class
 * basic constructor
 * Addition:
 * Subtraction:
 * Multiplication:
 * Division:
 * Others: [vec.dot(vec)]
 */

template <typename T>
class Vec
{
private:
    std::vector<T> vec;

public:
    // Constructors
    Vec() = default;
    explicit Vec(size_t size) : vec(size) {}
    Vec(size_t size, const T &value) : vec(size, value) {}
    explicit Vec(const std::vector<T> &v) : vec(v) {} // Initialize from std::vector<T>

    // Accessors
    size_t size() const noexcept { return vec.size(); }
    T &operator[](size_t i) noexcept { return vec[i]; }
    const T &operator[](size_t i) const noexcept { return vec[i]; }
    T *data() { return vec.data(); }
    const T *data() const { return vec.data(); }

    /*Addition*/

    // [vec + vec]
    Vec operator+(const Vec &other) const
    {
        if (size() != other.size())
        {
            throw std::runtime_error("Vector sizes do not match for addition");
        }

        Vec result(size());
        std::transform(vec.begin(), vec.end(), other.vec.begin(), result.vec.begin(), std::plus<T>());
        return result;
    }

    // [vec + s]
    Vec operator+(const T &s) const
    {
        Vec result(size());
        std::transform(vec.begin(), vec.end(), result.vec.begin(), [&s](const T &val)
                       { return val + s; });
        return result;
    }
    // [s + vec]
    friend Vec operator+(const T &s, const Vec &v) { return v + s; }

    /*Subtraction*/

    // [-v]
    Vec operator-() const
    {
        Vec result(size());
        std::transform(vec.begin(), vec.end(), result.vec.begin(), std::negate<T>());
        return result;
    }

    // [vec - vec]
    Vec operator-(const Vec &other) const
    {
        if (size() != other.size())
        {
            throw std::runtime_error("Vector sizes do not match for subtraction.");
        }

        Vec result(size());
        std::transform(vec.begin(), vec.end(), other.vec.begin(), result.vec.begin(), std::minus<T>());
        return result;
    }

    // [vec - s]
    Vec operator-(const T &s) const
    {
        Vec result(size());
        std::transform(vec.begin(), vec.end(), result.vec.begin(), [&s](const T &val)
                       { return val - s; });
        return result;
    }

    // [s - vec]
    friend Vec operator-(const T &s, const Vec &v) { return -(v - s); }

    /* Multiplication */

    // [vec * vec] (element-wise multiplication)
    Vec operator*(const Vec &other) const
    {
        if (size() != other.size())
        {
            throw std::runtime_error("Vector sizes do not match multiply.");
        }
        Vec result(size());
        std::transform(vec.begin(), vec.end(), other.vec.begin(), result.vec.begin(), std::multiplies<T>());
        return result;
    }

    // [vec * s]
    Vec operator*(const T &s) const
    {
        Vec result(size());
        std::transform(vec.begin(), vec.end(), result.vec.begin(), [&s](const T &val)
                       { return val * s; });
        return result;
    }

    // [s * vec]
    friend Vec operator*(const T &s, const Vec &v) { return v * s; }

    /*Division*/

    // [vec / vec] (element-wise divide)
    Vec operator/(const Vec &other) const
    {
        if (size() != other.size())
        {
            throw std::runtime_error("Vector sizes do not match for element-wise division.");
        }

        Vec result(size());
        std::transform(vec.begin(), vec.end(), other.vec.begin(), result.vec.begin(),
                       [](const T &a, const T &b) -> T
                       {
                           if (abs(b) == 0)
                           {
                               throw std::runtime_error("Division by zero in element-wise division");
                           }
                           return a / b;
                       });
        return result;
    }

    // [vec / s]
    Vec operator/(const T &s) const
    {
        if (std::abs(s) == 0)
        {
            throw std::runtime_error("Division by zero is not allowed.");
        }

        Vec result(size());
        std::transform(vec.begin(), vec.end(), result.vec.begin(), [&s](const T &val)
                       { return val / s; });
        return result;
    }

    // [s / vec]
    friend Vec operator/(const T &s, const Vec &v)
    {
        Vec result(v.size());
        std::transform(v.vec.begin(), v.vec.end(), result.vec.begin(),
                       [s](const T &val)
                       {
                           if (std::abs(val) == 0)
                               throw std::runtime_error("Division by zero in vector.");
                           return s / val;
                       });
        return result;
    }

    /*other*/
    // power (element-wise pow)
    Vec pow(const int &exponent) const
    {
        Vec result(size());
        std::transform(vec.begin(), vec.end(), result.vec.begin(), [exponent](const T &val)
                       { return std::pow(val, static_cast<T>(exponent)); });
        return result;
    }

    T dot(const Vec &other) const
    {
        if (size() != other.size())
        {
            throw std::runtime_error("Vector sizes do not match for dot product.");
        }

        return std::inner_product(vec.begin(), vec.end(), other.vec.begin(), T(0.0));
    }

    // Print vector
    void print() const
    {
        for (const auto &val : vec)
            std::cout << val << " ";
        std::cout << std::endl;
    }
};

/**
 * @brief Matrix class
 * Initialization
 * Diagonal: D()
 * transpose: T()
 * Addition: mat + mat, mat + scalar, scalar + mat
 * Subtraction: -mat, mat - mat, mat - calar, scalar - mat
 * Multiplication: mat.element_wise_product(mat), mat.mat_mul(mat), mat * mat, mat * vec, vec * mat, vec * scalar, scalar * mat
 * Solver: inv_Gauss(),
 */

class Mat
{
    /**
     * @brief col-major
     * data stored in a 1d vector to match LAPACK conventions
     */
private:
    size_t rows, cols;
    std::vector<double> mat;
    inline size_t index(size_t i, size_t j) const noexcept // internal index calculation
    {
        assert(i < rows && j < cols);
        return i + j * rows;
    }

public:
    // default constructor
    Mat() : rows(0), cols(0) {}
    // constructir with size and optional initial value
    Mat(size_t rows, size_t cols, double value = 0.0) : rows(rows), cols(cols), mat(rows * cols, value) {}
    // copy constructor
    Mat(const Mat &other) : rows(other.rows), cols(other.cols), mat(other.mat) {}
    // move constructor
    Mat(Mat &&other) noexcept : rows(other.rows), cols(other.cols), mat(std::move(other.mat))
    {
        other.rows = 0;
        other.cols = 0;
    }
    // assignment operators
    Mat &operator=(const Mat &other)
    {
        if (this != &other)
        {
            rows = other.rows;
            cols = other.cols;
            mat = other.mat;
        }
        return *this;
    }
    // move assignment operator
    Mat &operator=(Mat &&other) noexcept
    {
        if (this != &other)
        {
            rows = other.rows;
            cols = other.cols;
            mat = std::move(other.mat);
            other.rows = 0;
            other.cols = 0;
        }
        return *this;
    }

    // get matrix dimensions
    size_t row_count() const { return rows; }
    size_t col_count() const { return cols; }

    double &operator()(size_t i, size_t j) { return mat[index(i, j)]; }             // access elements (read/write))
    const double &operator()(size_t i, size_t j) const { return mat[index(i, j)]; } // access elements (read only)
    double *data() { return mat.data(); }                                           // get pointer to raw data (for LAPACK)
    const double *data() const { return mat.data(); }                               // get pointer to raw data (for LAPACK)

    // static function to return an identity matrix
    static Mat eye(const size_t n)
    {
        Mat identity(n, n, 0.0);
        for (size_t i = 0; i < n; i++)
        {
            identity(i, i) = 1.0;
        }
        return identity;
    }

    // print matrix
    void print() const
    {
        for (size_t i = 0; i < rows; ++i)
        {
            for (size_t j = 0; j < cols; ++j)
            {
                std::cout << (*this)(i, j) << " ";
            }
            std::cout << "\n";
        }
    }

    // set row a constant value
    void set_row(size_t target_row, const double s)
    {
        for (size_t j = 0; j < cols; j++)
        {
            (*this)(target_row, j) = s;
        }
    }

    // Set row using a row from another matrix
    void set_row(size_t target_row, const Mat &source_mat, size_t source_row)
    {
        if (target_row >= rows)
        {
            throw std::out_of_range("Target row index out of bounds");
        }
        if (source_row >= source_mat.row_count())
        {
            throw std::out_of_range("Source row index out of bounds");
        }
        if (source_mat.col_count() != cols)
        {
            throw std::invalid_argument("Source matrix column count does not match target's column count");
        }
        for (size_t j = 0; j < cols; ++j)
        {
            (*this)(target_row, j) = source_mat(source_row, j);
        }
    }

    /*Addition*/
    Mat operator+(const Mat &other) const // [mat + mat]
    {
        if (rows != other.rows || cols != other.cols)
        {
            throw std::runtime_error("Matrix dimensions do not match for addition");
        }
        Mat result(rows, cols);
        std::transform(mat.begin(), mat.end(), other.mat.begin(), result.mat.begin(), std::plus<double>());
        return result;
    }

    Mat operator+(const double &s) const // [mat + s]
    {
        Mat result(rows, cols);
        std::transform(mat.begin(), mat.end(), result.mat.begin(), [s](double val)
                       { return val + s; });
        return result;
    }

    friend Mat operator+(const double &s, const Mat &m) { return m + s; } // [s + mat]

    /*Subtraction*/
    Mat operator-() const // [-mat]
    {
        Mat result(rows, cols);
        std::transform(mat.begin(), mat.end(), result.mat.begin(), [](double val)
                       { return -val; });
        return result;
    }

    Mat operator-(const Mat &other) const // [mat - mat]
    {
        if (rows != other.rows || cols != other.cols)
        {
            throw std::runtime_error("Matrix dimensions do not match for subtraction");
        }
        Mat result(rows, cols);
        std::transform(mat.begin(), mat.end(), other.mat.begin(), result.mat.begin(),
                       [](double a, double b)
                       { return a - b; });
        return result;
    }

    Mat operator-(const double &s) const // [mat - s]
    {
        Mat result(rows, cols);
        std::transform(mat.begin(), mat.end(), result.mat.begin(), [s](double val)
                       { return val - s; });
        return result;
    }

    friend Mat operator-(const double &s, const Mat &m) { return -(m - s); } // [s - mat]

    /*Multiplication*/
    Mat element_wise_product(const Mat &other) const // [mat .* mat] element-wise multiplication
    {
        if (rows != other.rows || cols != other.cols)
        {
            throw std::runtime_error("Matrix dimensions do not match for elementw-wise multiplication");
        }
        Mat result(rows, cols);
        std::transform(mat.begin(), mat.end(), other.mat.begin(), result.mat.begin(), [](double a, double b)
                       { return a * b; });
        return result;
    }

    Mat operator*(const Mat &other) const // [mat * mat] standard matrices multiplication
    {
        if (cols != other.rows)
        {
            throw std::runtime_error("Matrix dimensions do not match for multiplication");
        }
        Mat result(rows, other.cols, 0.0);

        cblas_dgemm(
            CblasColMajor,     // Matrix storage order (column-major)
            CblasNoTrans,      // No transpose for A
            CblasNoTrans,      // No transpose for B
            rows,              // Number of rows in A and C
            other.cols,        // Number of columns in B and C
            cols,              // Number of columns in A and rows in B
            1.0,               // Alpha (scalar multiplier for A * B)
            mat.data(),        // Pointer to matrix A
            rows,              // Leading dimension of A (number of rows)
            other.mat.data(),  // Pointer to matrix B
            other.rows,        // Leading dimension of B (number of rows)
            0.0,               // Beta (scalar multiplier for C)
            result.mat.data(), // Pointer to matrix C
            result.rows        // Leading dimension of C (number of rows)
        );

        return result;
    }

    template <typename T>
    Vec<T> operator*(const Vec<T> &v) const // [mat * vec]
    {
        if (cols != v.size())
        {
            throw std::runtime_error("Matrix dimensions do not match for vector multiplication");
        }
        Vec<T> result(rows, static_cast<T>(0.0));

        // Perform matrix-vector multiplication
        for (size_t j = 0; j < cols; ++j)
        {
            T vj = v[j];
            for (size_t i = 0; i < rows; ++i)
            {
                result[i] += (*this)(i, j) * vj;
            }
        }
        return result;
    }

    friend Mat operator*(const Vec<double> &v, const Mat &m) // [vec * mat] essentially diagonal(vec)@mat, mat[i,:]*vec[i]
    {
        if (m.row_count() != v.size())
        {
            throw std::runtime_error("Matrix dimensions do not match for vector row-multiplication");
        }
        Mat result(m.row_count(), m.col_count());
        for (size_t j = 0; j < m.col_count(); ++j)
        {
            for (size_t i = 0; i < m.row_count(); ++i)
            {
                result(i, j) = v[i] * m(i, j);
            }
        }
        return result;
    }

    Mat operator*(const double &s) const // [mat * scalar]
    {
        Mat result(rows, cols);

        std::transform(mat.begin(), mat.end(), result.mat.begin(), [s](const double &val)
                       { return val * s; });
        return result;
    }

    friend Mat operator*(const double &s, const Mat &m) // [scalar * mat]
    {
        Mat result(m.row_count(), m.col_count());

        std::transform(m.mat.begin(), m.mat.end(), result.mat.begin(), [s](const double &val)
                       { return val * s; });
        return result;
    }

    /**
     * solve linear system
     */
    Mat solve(const Mat &B) const
    {
        if (rows != cols)
        {
            throw std::invalid_argument("Matrix A must be square to compute its inverse.");
        }
        if (rows != B.rows)
        {
            throw std::invalid_argument("Matrix dimensions do not match for solving A^{-1}B.");
        }

        Mat A_copy = *this; // Create a copy of A for LU factorization (A will be overwritten)
        Mat X = B;          // Create a copy of B for the solution (B will be overwritten)

        std::vector<int> ipiv(rows); // Pivot indices
        int info = LAPACKE_dgetrf(
            LAPACK_COL_MAJOR, // Matrix storage order (column-major)
            rows,             // Number of rows in A
            cols,             // Number of columns in A
            A_copy.data(),    // Pointer to matrix A
            rows,             // Leading dimension of A
            ipiv.data()       // Pivot indices
        );                    // Perform LU factorization of A_copy

        if (info != 0)
        {
            throw std::runtime_error("LU factorization failed. Matrix A may be singular.");
        }

        info = LAPACKE_dgetrs(
            LAPACK_COL_MAJOR, // Matrix storage order (column-major)
            'N',              // No transpose
            rows,             // Number of rows in A
            X.cols,           // Number of columns in B
            A_copy.data(),    // Pointer to LU-factored matrix A
            rows,             // Leading dimension of A
            ipiv.data(),      // Pivot indices
            X.data(),         // Pointer to matrix B (will be overwritten with the solution)
            X.rows            // Leading dimension of B
        );                    // Solve the system A_copy * X = B

        if (info != 0)
        {
            throw std::runtime_error("Solving the system failed.");
        }

        return X; // X now contains A^{-1}B
    }

    template <typename T>
    Vec<T> solve_posdef(const Vec<T> &b) const
    {
        if (rows != cols)
        {
            throw std::invalid_argument("Matrix A must be square to solve linear systems.");
        }
        if (b.size() % rows != 0)
        {
            throw std::invalid_argument("Matrix dimensions do not match for solving A^{-1}b.");
        }

        size_t n = b.size() / rows; // Number of right-hand sides
        Mat A_copy = *this;         // Create a copy of the matrix A for Cholesky factorization

        int info = 0; // Perform Cholesky factorization of A_copy

        info = LAPACKE_dpotrf(
            LAPACK_COL_MAJOR, // Matrix storage order (column-major)
            'U',              // Upper triangular part of A is stored
            rows,             // Number of rows in A
            A_copy.data(),    // Pointer to matrix A
            rows              // Leading dimension of A
        );                    // Use LAPACK's dpotrf for double-precision

        if (info != 0)
        {
            throw std::runtime_error("Cholesky factorization failed. Matrix A may not be positive definite.");
        }

        Vec<T> X = b; // Create a copy of the right-hand side vector b

        // Solve the system A_copy * X = B
        if constexpr (std::is_same_v<T, double>)
        {

            info = LAPACKE_dpotrs(
                LAPACK_COL_MAJOR, // Matrix storage order (column-major)
                'U',              // Upper triangular part of A is stored
                rows,             // Number of rows in A
                n,                // Number of right-hand sides
                A_copy.data(),    // Pointer to Cholesky-factored matrix A
                rows,             // Leading dimension of A
                X.data(),         // Pointer to matrix B (will be overwritten with the solution)
                rows              // Leading dimension of B
            );                    // Use LAPACK's dpotrs for double-precision
        }
        else if constexpr (std::is_same_v<T, std::complex<double>>)
        {
            // Use LAPACK's zpotrs for complex double-precision
            // Create a complex copy of A_copy for LAPACKE_zpotrs
            std::vector<std::complex<double>> A_complex(A_copy.data(), A_copy.data() + rows * cols);

            info = LAPACKE_zpotrs(
                LAPACK_COL_MAJOR,                                            // Matrix storage order (column-major)
                'U',                                                         // Upper triangular part of A is stored
                rows,                                                        // Number of rows in A
                n,                                                           // Number of right-hand sides
                reinterpret_cast<lapack_complex_double *>(A_complex.data()), // Pointer to Cholesky-factored matrix A
                rows,                                                        // Leading dimension of A
                reinterpret_cast<lapack_complex_double *>(X.data()),         // Pointer to matrix B (will be overwritten with the solution)
                rows                                                         // Leading dimension of B
            );                                                               // Use LAPACK's zpotrs for complex double-precision
        }

        if (info != 0)
        {
            throw std::runtime_error("Solving the system failed.");
        }

        return X; // X now contains the solution
    }

    template <typename T>
    Vec<T> solve(const Vec<T> &b) const
    {
        if (rows != cols)
        {
            throw std::invalid_argument("Matrix A must be square to solve linear systems.");
        }
        if (b.size() % rows != 0)
        {
            throw std::invalid_argument("Matrix dimensions do not match for solving A^{-1}b.");
        }

        size_t n = b.size() / rows; // Number of right-hand sides
        Mat A_copy = *this;         // Create a copy of the matrix A for Cholesky factorization

        // Perform LU factorization of A_copy
        std::vector<int> ipiv(rows); // Pivot indices
        int info = LAPACKE_dgetrf(
            LAPACK_COL_MAJOR, // Matrix storage order (column-major)
            rows,             // Number of rows in A
            cols,             // Number of columns in A
            A_copy.data(),    // Pointer to matrix A
            rows,             // Leading dimension of A
            ipiv.data()       // Pivot indices
        );

        if (info != 0)
        {
            throw std::runtime_error("LU factorization failed. Matrix A may be singular.");
        }

        Vec<T> X = b; // Create a copy of the right-hand side vector b

        // Solve the system A_copy * X = B
        if constexpr (std::is_same_v<T, double>)
        {

            info = LAPACKE_dgetrs(
                LAPACK_COL_MAJOR, // Matrix storage order (column-major)
                'N',              // No transpose
                rows,             // Number of rows in A
                n,                // Number of right-hand sides
                A_copy.data(),    // Pointer to LU-factored matrix A
                rows,             // Leading dimension of A
                ipiv.data(),      // Pivot indices
                X.data(),         // Pointer to matrix B (will be overwritten with the solution)
                rows              // Leading dimension of B
            );                    // Use LAPACK's dgetrs for double-precision
        }
        else if constexpr (std::is_same_v<T, std::complex<double>>)
        {
            // Use LAPACK's zgetrs for complex double-precision
            // Create a complex copy of A_copy for LAPACKE_zgetrs
            std::vector<std::complex<double>> A_complex(A_copy.data(), A_copy.data() + rows * cols);

            info = LAPACKE_zgetrs(
                LAPACK_COL_MAJOR,                                            // Matrix storage order (column-major)
                'N',                                                         // No transpose
                rows,                                                        // Number of rows in A
                n,                                                           // Number of right-hand sides
                reinterpret_cast<lapack_complex_double *>(A_complex.data()), // Pointer to LU-factored matrix A
                rows,                                                        // Leading dimension of A
                ipiv.data(),                                                 // Pivot indices
                reinterpret_cast<lapack_complex_double *>(X.data()),         // Pointer to matrix B (will be overwritten with the solution)
                rows                                                         // Leading dimension of B
            );                                                               // Use LAPACK's zpotrs for complex double-precision
        }

        if (info != 0)
        {
            throw std::runtime_error("Solving the system failed.");
        }

        return X; // X now contains the solution
    }
};

/**
 * @brief flow field var
 */
template <typename T>
class Field
{
private:
    size_t nz, nx, ny;
    std::vector<std::vector<Vec<T>>> var;

public:
    // default constructor
    Field() : nz(0), nx(0), ny(0) {}
    // constructor with size and optional initial value
    Field(size_t nz, size_t nx, size_t ny, const T &value = T()) : nz(nz), nx(nx), ny(ny), var(nz, std::vector<Vec<T>>(nx, Vec<T>(ny, value))) {}

    // access elements with bounds checking
    T &operator()(size_t k, size_t i, size_t j)
    {
        if (k >= nz || i >= nx || j >= ny)
        {
            throw std::out_of_range("Field indices out of range");
        }
        return var[k][i][j];
    }

    // const version of the access operator
    const T &operator()(size_t k, size_t i, size_t j) const
    {
        if (k >= nz || i >= nx || j >= ny)
        {
            throw std::out_of_range("Field indices out of range");
        }
        return var[k][i][j];
    }

    // acess the Vec
    Vec<T> &operator()(size_t k, size_t i)
    {
        if (k >= nz || i >= nx)
        {
            throw std::out_of_range("Field indices out of range");
        }
        return var[k][i];
    }
    // const version of the acess operator
    const Vec<T> &operator()(size_t k, size_t i) const
    {
        if (k >= nz || i >= nx)
        {
            throw std::out_of_range("Field indices out of range");
        }
        return var[k][i];
    }

    // get the dimensions of the field
    size_t size_z() const { return nz; }
    size_t size_x() const { return nx; }
    size_t size_y() const { return ny; }

    /**
     * @brief element-wise addition
     * [Field + Field]
     */
    Field<T> operator+(const Field<T> &other) const
    {
        if (nz != other.nz || nx != other.nx || ny != other.ny)
        {
            throw std::runtime_error("Field dimensions do not match for element-wise addition");
        }

        Field<T> result(nz, nx, ny);
        for (size_t k = 0; k < nz; k++)
        {
            for (size_t i = 0; i < nx; i++)
            {
                result(k, i) = var[k][i] + other(k, i);
            }
        }
        return result;
    }

    /**
     * @brief element-wise subtraction
     * [Field - Field]
     */
    Field<T> operator-(const Field<T> &other) const
    {
        if (nz != other.nz || nx != other.nx || ny != other.ny)
        {
            throw std::runtime_error("Field dimensions do not match for element-wise subtraction");
        }

        Field<T> result(nz, nx, ny);
        for (size_t k = 0; k < nz; k++)
        {
            for (size_t i = 0; i < nx; i++)
            {
                result(k, i) = var[k][i] - other(k, i);
            }
        }
        return result;
    }

    /**
     * @brief element-wise multiplication
     * [Field - Field]
     */
    Field<T> operator*(const Field<T> &other) const
    {
        if (nz != other.nz || nx != other.nx || ny != other.ny)
        {
            throw std::runtime_error("Field dimensions do not match for element-wise multiplication");
        }

        Field<T> result(nz, nx, ny);
        for (size_t k = 0; k < nz; k++)
        {
            for (size_t i = 0; i < nx; i++)
            {
                result(k, i) = var[k][i] * other(k, i);
            }
        }
        return result;
    }
};

#endif