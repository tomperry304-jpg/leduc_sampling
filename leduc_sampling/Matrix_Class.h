#pragma once

#include <vector>
#include <cstddef>
#include <string>
#include <iostream>
#include <stdexcept>
#include <omp.h>  
#include <random>
#include <algorithm>
#include <cmath>

class Matrix {
private:
    std::vector<std::vector<float>> matrix_;
    std::size_t n_;
    std::size_t m_;
public:
    Matrix(std::size_t n, std::size_t m, float fill = 0)
        :matrix_(n, std::vector<float>(m, fill)), n_(n), m_(m) {}

    ~Matrix() {

    }
    //Copy Constructor
    Matrix(const Matrix& other) {
        matrix_ = other.matrix_;
        n_ = other.n_;
        m_ = other.m_;
    }
    //Copy Assinghment
    Matrix& operator=(const Matrix& other) {
        if (this == &other) {
            return *this;
        }
        matrix_ = other.matrix_;
        n_ = other.n_;
        m_ = other.m_;
        return *this;
    }
    //Move Constructor
    Matrix(Matrix&& other) noexcept
        : matrix_(std::move(other.matrix_)), n_(other.n_), m_(other.m_) {}


    //Move Assinghment
    Matrix& operator=(Matrix&& other) noexcept {
        if (this == &other) {
            return *this;
        }
        matrix_ = std::move(other.matrix_);
        n_ = other.n_;
        m_ = other.m_;
        return *this;
    }

    //Matrix Assinghment
    void Matrix_Assinghment(std::size_t n, std::size_t m, float value) {
        matrix_[n][m] = value;
    }
    //Extract Value reference
    float& Extract_Value(std::size_t n, std::size_t m) {
        return matrix_[n][m];
    }

    //Extract Value const
    float Extract_Value(std::size_t n, std::size_t m) const {
        return matrix_[n][m];
    }

    //Add
    Matrix operator+(const Matrix& other) const {
        if (n_ == other.n_ && m_ == other.m_) {
            Matrix temp = Matrix(n_, m_);
            temp.matrix_ = matrix_;
            for (std::size_t i = 0; i < n_; i++) {
                for (std::size_t j = 0; j < m_; j++) {
                    temp.matrix_[i][j] += other.matrix_[i][j];
                }
            }
            return temp;
        }
        throw std::invalid_argument("dimension mismatch");
    }


    //Add (Scalar)
    Matrix operator+(const float num) const {
        Matrix temp = Matrix(n_, m_);
        temp.matrix_ = matrix_;
        for (std::size_t i = 0; i < n_; i++) {
            for (std::size_t j = 0; j < m_; j++) {
                temp.matrix_[i][j] += num;
            }
        }
        return temp;
    }

    //Subtraction (non destructive)
    Matrix operator-(const Matrix& other) const {
        if (n_ == other.n_ && m_ == other.m_) {
            Matrix temp = Matrix(n_, m_);
            temp.matrix_ = matrix_;
            for (std::size_t i = 0; i < n_; i++) {
                for (std::size_t j = 0; j < m_; j++) {
                    temp.matrix_[i][j] -= other.matrix_[i][j];
                }
            }
            return temp;
        }
        throw std::invalid_argument("dimension mismatch");
    }


    void subtracing_scaled_matrix(const Matrix& other, float n) {
        if (n_ == other.n_ && m_ == other.m_) {
            for (std::size_t i = 0; i < n_; i++) {
                for (std::size_t j = 0; j < m_; j++) {
                    matrix_[i][j] -= other.matrix_[i][j]*n;
                }
            }
        }
        else {
            throw std::invalid_argument("dimension mismatch");
        }
    }



    //Scalar Multiplication (non Destructive)
    Matrix operator*(float value) const {
        Matrix temp = Matrix(n_, m_);
        temp.matrix_ = matrix_;
        for (std::size_t i = 0; i < n_; i++) {
            for (std::size_t j = 0; j < m_; j++) {
                temp.matrix_[i][j] *= value;
            }
        }
        return temp;
    }

    friend Matrix operator*(const std::vector<float>& a, const std::vector<float>& b);

    //Matrix Multiplication
    Matrix operator*(const Matrix& other) const {

        if (m_ == other.n_) {
            std::size_t block_size = 50;
            Matrix temp = Matrix(n_, other.m_);
            Matrix other_matrix = other;
            other_matrix.Transpose();
#pragma omp parallel for schedule(dynamic)
            for (long long ii = 0; ii < (long long)n_; ii += block_size) {
                for (std::size_t jj = 0; jj < other_matrix.n_; jj += block_size) {

                    std::size_t i_end = std::min((std::size_t)(ii + block_size), n_);
                    std::size_t j_end = std::min((std::size_t)(jj + block_size), other_matrix.n_);

                    std::size_t i_even = ii + ((i_end - ii) / 2) * 2;
                    std::size_t j_even = jj + ((j_end - jj) / 2) * 2;

                    for (std::size_t i = ii; i < i_even; i += 2) {
                        for (std::size_t j = jj; j < j_even; j += 2) {
                            float c0 = 0, c1 = 0, c2 = 0, c3 = 0;
                            for (std::size_t k = 0; k < m_; k++) {
                                float a0 = matrix_[i][k], a1 = matrix_[i + 1][k];
                                float b0 = other_matrix.matrix_[j][k], b1 = other_matrix.matrix_[j + 1][k];
                                c0 += a0 * b0;
                                c1 += a0 * b1;
                                c2 += a1 * b0;
                                c3 += a1 * b1;
                            }
                            temp.matrix_[i][j] += c0;
                            temp.matrix_[i][j + 1] += c1;
                            temp.matrix_[i + 1][j] += c2;
                            temp.matrix_[i + 1][j + 1] += c3;
                        }
                    }
                    for (std::size_t i = i_even; i < i_end; i++) {
                        for (std::size_t j = jj; j < j_even; j++) {
                            float c0 = 0;
                            for (std::size_t k = 0; k < m_; k++) {
                                c0 += matrix_[i][k] * other_matrix.matrix_[j][k];
                            }
                            temp.matrix_[i][j] += c0;
                        }
                    }
                    for (std::size_t i = ii; i < i_even; i++) {
                        for (std::size_t j = j_even; j < j_end; j++) {
                            float c0 = 0;
                            for (std::size_t k = 0; k < m_; k++) {
                                c0 += matrix_[i][k] * other_matrix.matrix_[j][k];
                            }
                            temp.matrix_[i][j] += c0;
                        }
                    }
                    for (std::size_t i = i_even; i < i_end; i++) {
                        for (std::size_t j = j_even; j < j_end; j++) {
                            float c0 = 0;
                            for (std::size_t k = 0; k < m_; k++) {
                                c0 += matrix_[i][k] * other_matrix.matrix_[j][k];
                            }
                            temp.matrix_[i][j] += c0;
                        }
                    }
                }
            }
            return temp;
        }
        throw std::invalid_argument("dimension mismatch");
    }


    //Matrix * vec Multiplication
    friend std::vector<float> operator*(const std::vector<float>& vector, const Matrix& matrix) {

        if (matrix.n_ == vector.size()) {
            int n_ = matrix.n_;
            std::vector<float> temp(matrix.m_, 0);

            std::size_t ii = matrix.m_ - matrix.m_ % 2;
#pragma omp parallel for schedule(dynamic)
            for (std::size_t i = 0; i < ii; i += 2) {
                    float c0 = 0, c1 = 0, c2 = 0, c3 = 0;
                    for (std::size_t k = 0; k < n_; k++) {
                        float a0 = matrix.matrix_[k][i], a1 = matrix.matrix_[k][i+1];
                        float b0 = vector[k];
                        c0 += a0 * b0;
                        c1 += a1 * b0;
                    }
                    temp[i] += c0;
                    temp[i+1] += c1;
            }
            for (std::size_t i = ii; i < matrix.m_; i ++) {
                float c0 = 0;
                for (std::size_t k = 0; k < n_; k++) {
                    c0 += matrix.matrix_[k][i] * vector[k];
                }
                temp[i] += c0;
            }

            return temp;
        }
        throw std::invalid_argument("dimension mismatch");
    }

    //Matrix * vec Multiplication (transposed)
    friend std::vector<float> matrix_vec_multiply_transposed(const std::vector<float>& vector, const Matrix& matrix) {

        if (matrix.m_ == vector.size()) {
            std::vector<float> temp(matrix.n_, 0);

#pragma omp parallel for schedule(dynamic)
            for (std::size_t i = 0; i < matrix.n_; i++) {
                float c0 = 0;
                for (std::size_t k = 0; k < matrix.m_; k++) {
                    c0 += matrix.matrix_[i][k] * vector[k];
                }
                temp[i] = c0;
            }

            return temp;
        }
        throw std::invalid_argument("dimension mismatch");
    }




    //Transpose
    void Transpose() {
        Matrix temp = Matrix(m_, n_);
        for (std::size_t i = 0; i < n_; i++) {
            for (std::size_t j = 0; j < m_; j++) {
                temp.matrix_[j][i] = matrix_[i][j];
            }
        }
        matrix_ = temp.matrix_;
        n_ = temp.n_;
        m_ = temp.m_;
    }

    //Determinat
    float Determinat() {
        if (n_ == m_) {
            float sighn = 1;
            Matrix temp = Matrix(n_, m_);
            temp.matrix_ = matrix_;
            for (std::size_t i = 0; i < n_ - 1; i++) {
                if (matrix_[i][i] == 0) {
                    for (std::size_t l = i + 1; l < n_; l++) {
                        if (matrix_[l][i] != 0) {
                            std::vector<float> temp_row = matrix_[l];
                            matrix_[l] = matrix_[i];
                            matrix_[i] = temp_row;
                            sighn *= -1;
                            break;
                        }
                        if (l == n_ - 1) {
                            matrix_ = temp.matrix_;
                            return 0;
                        }
                    }
                }
                for (std::size_t j = i + 1; j < m_; j++) {
                    if (matrix_[j][i] != 0) {
                        float ratio = matrix_[j][i] / matrix_[i][i];
                        for (std::size_t k = 0; k < m_; k++) {
                            matrix_[j][k] -= matrix_[i][k] * ratio;
                        }
                    }
                }
            }
            float determinat = 1;
            for (std::size_t i = 0; i < n_; i++) {
                determinat *= matrix_[i][i];
            }
            matrix_ = temp.matrix_;
            determinat *= sighn;
            return determinat;
        }
        throw std::invalid_argument("dimension mismatch");
    }


    //Element-wise function apply destructive
    template<typename func, typename... Args>
    void Apply_in_place(func f, Args... arguments) {
        for (std::size_t i = 0; i < n_; i++) {
            for (std::size_t j = 0; j < m_; j++) {
                matrix_[i][j] = f(matrix_[i][j], arguments...);
            }
        }
    }


    //Element-wise function apply const
    Matrix Apply(float (*f)(float)) const {
        Matrix temp(n_, m_);
        for (std::size_t i = 0; i < n_; i++) {
            for (std::size_t j = 0; j < m_; j++) {
                temp.matrix_[i][j] = f(matrix_[i][j]);
            }
        }
        return temp;
    }


    //Hadamard (element-wise) product
    Matrix Element_Wise_Product(const Matrix& other) const {
        if (n_ == other.n_ && m_ == other.m_) {
            Matrix temp = Matrix(n_, m_);
#pragma omp parallel for schedule(static)
            for (long long i = 0; i < (long long)n_; i++) {
                for (std::size_t j = 0; j < m_; j++) {
                    temp.matrix_[i][j] = matrix_[i][j] * other.matrix_[i][j];
                }
            }
            return temp;
        }
        throw std::invalid_argument("dimension mismatch");
    }

    //Hadamard (element-wise) product
    void Element_Wise_Product_Destructive(const Matrix& other) {
        if (n_ == other.n_ && m_ == other.m_) {
#pragma omp parallel for schedule(static)
            for (long long i = 0; i < (long long)n_; i++) {
                for (std::size_t j = 0; j < m_; j++) {
                    matrix_[i][j] *= other.matrix_[i][j];
                }
            }
        }
        else {
            throw std::invalid_argument("dimension mismatch");
        }

    }


    //Non-destructive transpose
    Matrix Non_Destructive_Transpose() const {
        Matrix temp = Matrix(m_, n_);
        for (std::size_t i = 0; i < n_; i++) {
            for (std::size_t j = 0; j < m_; j++) {
                temp.matrix_[j][i] = matrix_[i][j];
            }
        }
        return temp;
    }
    //Dimension getters. Trivial rows() and cols() const accessors
    std::size_t rows() const {
        return n_;
    }

    std::size_t cols() const {
        return m_;
    }
    //Random init helper
    static Matrix Random_Intaliser(std::size_t n, std::size_t m) {
        static std::mt19937 gen(42);
        Matrix Random(n, m);
        float scale = std::sqrt(1.0f / n);
        std::normal_distribution<float> dist(0.0f, scale);
        for (std::size_t i = 0; i < n; i++) {
            for (std::size_t j = 0; j < m; j++) {
                Random.matrix_[i][j] = dist(gen);
            }
        }
        return Random;
    }


    //Row-extraction helper. Pull row i out of your 60000×784 image matrix as a standalone 1×784 Matrix
    Matrix Row_Extraction(std::size_t i) const {
        Matrix Row(1, m_);
        Row.matrix_[0] = matrix_[i];
        return Row;
    }

    //Matrix Total
    float Total() {
        float total = 0;
        for (std::size_t i = 0; i < n_; i++) {
            for (std::size_t j = 0; j < m_; j++) {
                total += matrix_[i][j];
            }
        }
        return total;
    }

    //Mas J location;
    std::size_t Max_j() {
        float max = matrix_[0][0];
        float max_j = 0;
        for (std::size_t i = 0; i < n_; i++) {
            for (std::size_t j = 0; j < m_; j++) {
                if (matrix_[i][j] > max) {
                    max = matrix_[i][j];
                    max_j = j;
                }
            }
        }
        return max_j;
    }

};


inline Matrix operator*(const std::vector<float>& a, const std::vector<float>& b) {
    int n_ = a.size();
    int m_ = b.size();
    Matrix other(n_, m_);
    for (int i = 0; i < n_; i++)
        for (int j = 0; j < m_; j++)
            other.matrix_[i][j] = a[i] * b[j];
    return other;
}