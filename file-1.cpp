#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <Eigen/Dense>
#include <fstream>

// Morse potential
double V(double r) {
    double a = 1.0;
    double De = 2.0;
    double re = 2.0;
    return De * (exp(-2 * a * (r - re)) - 2 * exp(-a * (r - re)));
}

// Collective Bohr Hamiltonian
Eigen::MatrixXd H(int N, double a) {
    Eigen::MatrixXd Hmat = Eigen::MatrixXd::Zero(N, N);
    for (int i = 0; i < N; ++i) {
        Hmat(i, i) = -2 * (i + 1) / a;
        if (i > 0) {
            Hmat(i, i - 1) = sqrt(i) / a;
        }
        if (i < N - 1) {
            Hmat(i, i + 1) = sqrt(i + 1) / a;
        }
    }
    return Hmat;
}

// Lanczos algorithm
std::pair<double, Eigen::VectorXd> lanczos(Eigen::MatrixXd& A, int m) {
    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(A.rows(), m);
    Eigen::VectorXd alpha = Eigen::VectorXd::Zero(m);
    Eigen::VectorXd beta = Eigen::VectorXd::Zero(m - 1);

    Q.col(0) = A.col(0);
    alpha(0) = Q.col(0).dot(A * Q.col(0));
    for (int i = 1; i < m; ++i) {
        Q.col(i) = A * Q.col(i - 1) - beta(i - 1) * Q.col(i - 1);
        alpha(i) = Q.col(i).dot(A * Q.col(i));
        if (i < m - 1) {
            beta(i) = sqrt(Q.col(i).dot(Q.col(i)));
            Q.col(i) /= beta(i);
        }
    }

    Eigen::MatrixXd T = Eigen::MatrixXd::Zero(m, m);
    for (int i = 0; i < m - 1; ++i) {
        T(i, i + 1) = beta(i);
        T(i + 1, i) = beta(i);
    }
    T(m - 1, m - 1) = alpha(m - 1);

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(T);
    double E0 = eigensolver.eigenvalues()(0);
    Eigen::VectorXd c = eigensolver.eigenvectors().col(0);

    return {E0, Q * c};
}

int main() {
    int N = 10;
    double a = 1.0;
    Eigen::MatrixXd A = H(N, a);
    for (int i = 0; i < N; ++i) {
        A(i, i) += V(i + 1);
    }
    double E0;
    Eigen::VectorXd psi;
    std::tie(E0, psi) = lanczos(A, N);
    std::cout << "E0: " << E0 << std::endl;
    std::cout << "psi: " << psi.transpose() << std::endl;

    std::ofstream outfile("results.txt");
    outfile << "E0: " << E0 << std::endl;
    outfile << "psi: " << psi.transpose() << std::endl;
    outfile.close();

    return 0;
}