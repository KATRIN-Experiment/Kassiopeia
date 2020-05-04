#include "KBiconjugateGradientStabilized.hh"
#include "KGaussSeidel.hh"
#include "KGaussianElimination.hh"
#include "KJacobiPreconditioner.hh"
#include "KMultiElementRobinHood.hh"
#include "KPreconditionedBiconjugateGradientStabilized.hh"
#include "KPreconditionedIterativeKrylovSolver.hh"
#include "KRobinHood.hh"
#include "KSimpleIterativeKrylovSolver.hh"
#include "KSimpleVector.hh"
#include "KSquareMatrix.hh"
#include "KSuccessiveSubspaceCorrection.hh"

#include <cstdlib>
#include <fstream>
#include <getopt.h>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#ifdef KEMFIELD_USE_VTK
#include "KVTKResidualGraph.hh"
#endif

using namespace KEMField;

clock_t start;

void StartTimer()
{
    start = clock();
}

double Time()
{
    {
        double end = clock();
        return ((double) (end - start)) / CLOCKS_PER_SEC;  // time in seconds
    }

    return 0.;
}

class KDiscreteDirichlet2ndDerivative : public KSquareMatrix<double>
{
  public:
    KDiscreteDirichlet2ndDerivative(unsigned int dimension, double delta) :
        KSquareMatrix<double>(),
        fDimension(dimension),
        fOffDiagonal(1. / (delta * delta)),
        fDiagonal(-2. / (delta * delta)),
        fFirstLastDiagonal(-3. / (delta * delta)),
        fZero(0.)
    {}

    unsigned int Dimension() const
    {
        return fDimension;
    }

    const double& operator()(unsigned int i, unsigned int j) const;

  private:
    unsigned int fDimension;

    const double fOffDiagonal;
    const double fDiagonal;
    const double fFirstLastDiagonal;
    const double fZero;
};

const double& KDiscreteDirichlet2ndDerivative::operator()(unsigned int i, unsigned int j) const
{
    if (std::abs((int) i - (int) j) > 1)
        return fZero;

    if (i == j) {
        if (i == 0 || i == fDimension - 1)
            return fFirstLastDiagonal;
        else
            return fDiagonal;
    }

    return fOffDiagonal;
}

class KIdentity : public KSquareMatrix<double>
{
  public:
    KIdentity(unsigned int dimension) : KSquareMatrix<double>(), fDimension(dimension), fOne(1.), fZero(0.) {}

    unsigned int Dimension() const
    {
        return fDimension;
    }

    const double& operator()(unsigned int i, unsigned int j) const;

  private:
    unsigned int fDimension;

    const double fOne;
    const double fZero;
};

const double& KIdentity::operator()(unsigned int i, unsigned int j) const
{
    return (i != j ? fZero : fOne);
}

class KKronickerProduct : public KSquareMatrix<double>
{
  public:
    KKronickerProduct(const KSquareMatrix<double>& A, const KSquareMatrix<double>& B) :
        KSquareMatrix<double>(),
        fA(A),
        fB(B),
        fValue(0.)
    {}

    unsigned int Dimension() const
    {
        return fA.Dimension() * fB.Dimension();
    }

    const double& operator()(unsigned int i, unsigned int j) const;

  private:
    const KSquareMatrix<double>& fA;
    const KSquareMatrix<double>& fB;
    mutable double fValue;
};

const double& KKronickerProduct::operator()(unsigned int i, unsigned int j) const
{
    unsigned int i_A = i / fA.Dimension();
    unsigned int j_A = j / fA.Dimension();
    unsigned int i_B = i % fB.Dimension();
    unsigned int j_B = j % fB.Dimension();

    fValue = fA(i_A, j_A) * fB(i_B, j_B);
    return fValue;
}

class KLaplacian : public KSquareMatrix<double>
{
  public:
    KLaplacian(unsigned int N_x, double xLen, unsigned int N_z, double zLen);
    virtual ~KLaplacian() {}

    unsigned int Dimension() const
    {
        return fN_x * fN_z;
    }

    const double& operator()(unsigned int i, unsigned int j) const;

    // void Multiply(const KVector<double>& x,KVector<double>& y) const;

    void PrintDx() const;
    void PrintDz() const;

  private:
    KDiscreteDirichlet2ndDerivative fDx;
    KDiscreteDirichlet2ndDerivative fDz;
    KIdentity fIx;
    KIdentity fIz;
    KKronickerProduct fDxKronProdIz;
    KKronickerProduct fIxKronProdDz;

    mutable double fValue;

    unsigned int fN_x;
    unsigned int fN_z;
};

KLaplacian::KLaplacian(unsigned int N_x, double xLen, unsigned int N_z, double zLen) :
    KSquareMatrix<double>(),
    fDx(N_x, xLen / N_x),
    fDz(N_z, zLen / N_z),
    fIx(N_x),
    fIz(N_z),
    fDxKronProdIz(fDx, fIz),
    fIxKronProdDz(fIx, fDz),
    fValue(0.),
    fN_x(N_x),
    fN_z(N_z)
{}

// void KLaplacian::Multiply(const KVector<double>& x,KVector<double>& y) const
// {
//   // Computes vector y in the equation A*x = y
//   for (unsigned int i=0;i<Dimension();i++)
//     y[i] = 0.;

//   for (unsigned int i=0;i<Dimension();i++)
//   {
//     if (((int)i) - ((int)fN_z) >= 0)
//       y[i] += this->operator()(i,i-fN_z)*x(i-fN_z);

//     if (i != 0)
//       y[i] += this->operator()(i,i-1)*x(i-1);
//     y[i] += this->operator()(i,i)*x(i);
//     if (i != Dimension()-1)
//       y[i] += this->operator()(i,i+1)*x(i+1);

//     if (i + fN_z < Dimension())
//       y[i] += this->operator()(i,i+fN_z)*x(i+fN_z);
//   }
// }

void KLaplacian::PrintDx() const
{
    std::cout << "Dx: " << std::endl;
    for (unsigned int i = 0; i < fDx.Dimension(); i++) {
        for (unsigned int j = 0; j < fDx.Dimension(); j++)
            std::cout << fDx(i, j) << " ";
        std::cout << "" << std::endl;
    }
}

void KLaplacian::PrintDz() const
{
    std::cout << "Dz: " << std::endl;
    for (unsigned int i = 0; i < fDz.Dimension(); i++) {
        for (unsigned int j = 0; j < fDz.Dimension(); j++)
            std::cout << fDz(i, j) << " ";
        std::cout << "" << std::endl;
    }
}

const double& KLaplacian::operator()(unsigned int i, unsigned int j) const
{
    fValue = fDxKronProdIz(i, j) + fIxKronProdDz(i, j);
    return fValue;
}

int main(int argc, char** argv)
{
    std::string usage = "\n"
                        "Usage: TestAnisotropicPoission <options>\n"
                        "\n"
                        "This program solves a matrix equation with a poor condition number.\n"
                        "\n"
                        "\tAvailable options:\n"
                        "\t -h, --help               (shows this message and exits)\n"
                        "\t -L, --length             (problem size in x)\n"
                        "\t -H, --height             (problem size in z)\n"
                        "\t -x, --N_x                (x discretization)\n"
                        "\t -z, --N_z                (z discretization)\n"
                        "\t -j, --x_frequency        (x frequency)\n"
                        "\t -k, --z_frequency        (z frequency)\n"
                        "\t -a, --accuracy           (relative accuracy of computation)\n"
                        "\t -c, --cache              (cache matrix elements)\n"
                        "\t -m, --method             (0: Gaussian elimination,\n"
                        "\t                           1: Gauss-Seidel,\n"
                        "\t                           2: Single-Element Robin Hood (default),\n"
                        "\t                           3: Multi-Element Robin Hood,\n"
                        "\t                           4: BiCGStab,\n"
                        "\t                           5: BiCGStab w/Jacobi preconditioner\n"
                        "\t                           6: Successive Subspace Correction)\n"
                        "\t -n, --rh_dimension       (dimension of subspace)\n";

    double length = 1.;
    double height = 1.;
    unsigned int N_x = 2;
    unsigned int N_z = 2;
    int j = 2;
    int k = 2;
    double accuracy = 1.e-8;
    //  bool cache = false;
    int method = 2;
    int rh_dimension = 2;

    static struct option longOptions[] = {
        {"help", no_argument, 0, 'h'},
        {"length", required_argument, 0, 'L'},
        {"height", required_argument, 0, 'H'},
        {"N_x", required_argument, 0, 'x'},
        {"N_z", required_argument, 0, 'z'},
        {"x_frequency", required_argument, 0, 'j'},
        {"z_frequency", required_argument, 0, 'k'},
        {"accuracy", required_argument, 0, 'a'},
        {"cache", no_argument, 0, 'c'},
        {"method", required_argument, 0, 'm'},
        {"rh_dimension", required_argument, 0, 'n'},
    };

    static const char* optString = "hL:H:x:z:j:k:a:cm:n:";

    while (1) {
        char optId = getopt_long(argc, argv, optString, longOptions, NULL);
        if (optId == -1)
            break;
        switch (optId) {
            case ('h'):  // help
                std::cout << usage << std::endl;
                return 0;
            case ('L'):
                length = atof(optarg);
                break;
            case ('H'):
                height = atof(optarg);
                break;
            case ('x'):
                N_x = atoi(optarg);
                break;
            case ('z'):
                N_z = atoi(optarg);
                break;
            case ('j'):
                j = atoi(optarg);
                break;
            case ('k'):
                k = atoi(optarg);
                break;
            case ('a'):
                accuracy = atof(optarg);
                break;
            case ('m'):
                method = atoi(optarg);
                break;
                //    case('c'):
                //      cache = true;
                //      break;
            case ('n'):
                rh_dimension = atoi(optarg);
                break;
            default:  // unrecognized option
                std::cout << usage << std::endl;
                return 1;
        }
    }

    KLaplacian A(N_x, length, N_z, height);

    // for (unsigned int i=0;i<N_x*N_z;i++)
    // {
    //   for (unsigned int j=0;j<N_x*N_z;j++)
    //   {
    //     std::cout<<"A("<<i<<","<<j<<") = "<<A(i,j)<<"\t";
    //   }
    //   std::cout<<""<<std::endl;
    // }

    KSimpleVector<double> xSolved(N_x * N_z);
    for (unsigned int idx = 0; idx < N_x; idx++)
        for (unsigned int jdx = 0; jdx < N_z; jdx++) {
            unsigned int index = idx * N_z + jdx;
            double x_ = (idx + .5) / double(N_x);
            double z_ = (jdx + .5) / double(N_z);
            xSolved[index] = sin(M_PI * j * x_) * sin(M_PI * k * z_);
        }

    std::cout << "|xSolved| = " << xSolved.InfinityNorm() << std::endl;

    KSimpleVector<double> b(N_x * N_z);
    KSimpleVector<double> b_analytic(N_x * N_z);

    A.Multiply(xSolved, b);

    std::cout << "|b| = " << b.InfinityNorm() << std::endl;

    double lambda = -((j * M_PI / length) * (j * M_PI / length) + (k * M_PI / height) * (k * M_PI / height));

    std::cout << "lambda = " << lambda << std::endl;

    b_analytic = xSolved;
    b_analytic *= lambda;

    double b_analytic_inftyNorm = b_analytic.InfinityNorm();

    b_analytic -= b;
    b_analytic *= 1. / b_analytic_inftyNorm;

    std::cout << "Laplacian discretization error: " << b_analytic.InfinityNorm() << std::endl;

    KSimpleVector<double> x(N_x * N_z);

    StartTimer();

    if (method == 0) {
        KGaussianElimination<double> gaussianElimination;
        gaussianElimination.Solve(A, x, b);
    }
    else if (method == 1) {
        KGaussSeidel<double> gaussSeidel;
        gaussSeidel.SetTolerance(accuracy);

        gaussSeidel.AddVisitor(new KIterationDisplay<double>());

#ifdef KEMFIELD_USE_VTK
        gaussSeidel.AddVisitor(new KVTKResidualGraph<double>());
#endif

        gaussSeidel.Solve(A, x, b);
    }
    else if (method == 2) {
        KRobinHood<double> robinHood;
        robinHood.SetTolerance(accuracy);

        robinHood.AddVisitor(new KIterationDisplay<double>());

#ifdef KEMFIELD_USE_VTK
        robinHood.AddVisitor(new KVTKResidualGraph<double>());
#endif

        robinHood.SetResidualCheckInterval(b.Dimension());

        robinHood.Solve(A, x, b);
    }
    else if (method == 3) {
        KMultiElementRobinHood<double, KMultiElementRobinHood_SingleThread> robinHood;
        robinHood.SetSubspaceDimension(rh_dimension);
        robinHood.SetTolerance(accuracy);

        robinHood.AddVisitor(new KIterationDisplay<double>());

#ifdef KEMFIELD_USE_VTK
        robinHood.AddVisitor(new KVTKResidualGraph<double>());
#endif

        robinHood.SetResidualCheckInterval(b.Dimension() / rh_dimension);

        robinHood.Solve(A, x, b);
    }
    else if (method == 4) {
        KSimpleIterativeKrylovSolver<double, KBiconjugateGradientStabilized> biCGStab;
        biCGStab.SetTolerance(accuracy);

        biCGStab.AddVisitor(new KIterationDisplay<double>());

#ifdef KEMFIELD_USE_VTK
        biCGStab.AddVisitor(new KVTKResidualGraph<double>());
#endif

        biCGStab.Solve(A, x, b);
    }
    else if (method == 5) {

        KPreconditionedIterativeKrylovSolver<double, KPreconditionedBiconjugateGradientStabilized> biCGStab;
        biCGStab.SetTolerance(accuracy);

        biCGStab.AddVisitor(new KIterationDisplay<double>());

#ifdef KEMFIELD_USE_VTK
        biCGStab.AddVisitor(new KVTKResidualGraph<double>());
#endif

        KJacobiPreconditioner<double> jacobi(A);

        biCGStab.Solve(A, jacobi, x, b);
    }
    else if (method == 6) {
        KSuccessiveSubspaceCorrection<double, KSuccessiveSubspaceCorrection_SingleThread> ssc;
        ssc.SetSubspaceDimension(rh_dimension);
        ssc.SetTolerance(accuracy);

        ssc.AddVisitor(new KIterationDisplay<double>());

#ifdef KEMFIELD_USE_VTK
        ssc.AddVisitor(new KVTKResidualGraph<double>());
#endif

        ssc.SetResidualCheckInterval(b.Dimension() / rh_dimension);

        ssc.Solve(A, x, b);
    }

    double computationTime = Time();

    std::cout << "Converged after " << computationTime << " seconds" << std::endl;

    // for (unsigned int i=0;i<N_x;i++)
    //   for (unsigned int j=0;j<N_z;j++)
    //   {
    //     unsigned int index = i*N_z + j;
    //     std::cout<<index<<": "<<x(index)<<" "<<xSolved(index)<<std::endl;
    //   }


    x -= xSolved;
    x *= 1. / xSolved.InfinityNorm();
    std::cout << "relative error of x is " << x.InfinityNorm() << std::endl;

    return 0;
}
