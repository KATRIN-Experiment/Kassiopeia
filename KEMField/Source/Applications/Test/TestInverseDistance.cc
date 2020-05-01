#include "KBiconjugateGradientStabilized.hh"
#include "KEMFieldCanvas.hh"
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
#include "KEMVTKFieldCanvas.hh"
#include "KVTKResidualGraph.hh"
#endif

#ifdef KEMFIELD_USE_ROOT
#include "KEMRootFieldCanvas.hh"
#endif

#ifndef DEFAULT_OUTPUT_DIR
#define DEFAULT_OUTPUT_DIR "."
#endif /* !DEFAULT_OUTPUT_DIR */

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

class KInverseDistance : public KSquareMatrix<double>
{
  public:
    KInverseDistance(unsigned int dimension, double diagonal, double offDiagonal, double power);
    virtual ~KInverseDistance() {}

    unsigned int Dimension() const
    {
        return fDimension;
    }

    const double& operator()(unsigned int i, unsigned int j) const;

  private:
    unsigned int fDimension;

    double fDiagonal;
    double fOffDiagonal;
    double fPower;
    mutable double fValue;
};

KInverseDistance::KInverseDistance(unsigned int dimension, double diagonal, double offDiagonal, double power) :
    KSquareMatrix<double>(),
    fDimension(dimension),
    fDiagonal(diagonal),
    fOffDiagonal(offDiagonal),
    fPower(power)
{}

const double& KInverseDistance::operator()(unsigned int i, unsigned int j) const
{
    if (i == j)
        return fDiagonal;
    double d1 = i;
    double d2 = j;
    // fValue = fOffDiagonal/fabs(d1-d2);
    // if (fabs(fPower-1.)<1.e-6)
    fValue = fDiagonal * (fDimension - 1. - fabs(d1 - d2)) / fDimension;
    // else
    //   fValue = fOffDiagonal*pow(fabs(d1-d2),fPower);
    // fValue = fOffDiagonal;
    // fValue = fOffDiagonal/fDimension*fabs(d1-d2);
    // std::cout<<i<<","<<j<<": "<<fValue<<std::endl;
    (void) fOffDiagonal;
    (void) fPower;
    return fValue;
}

int main(int argc, char** argv)
{
    std::string usage = "\n"
                        "Usage: TestInverseDistance <options>\n"
                        "\n"
                        "This program solves a matrix equation where the matrix elements are valued\n"
                        "by the inverse of their distance to the diagonal.\n"
                        "\n"
                        "\tAvailable options:\n"
                        "\t -h, --help               (shows this message and exits)\n"
                        "\t -D, --dimension          (problem dimension)\n"
                        "\t -d, --diagonal           (value of diagonal element)\n"
                        "\t -o, --off_diagonal       (maximum value of off-diagonal element)\n"
                        "\t -p, --power              (power by which off-diagonal elements scale)\n"
                        "\t -a, --accuracy           (relative accuracy of computation)\n"
                        "\t -c, --cache              (cache matrix elements)\n"
                        "\t -m, --method             (0: Gaussian elimination,\n"
                        "\t                           1: Gauss-Seidel,\n"
                        "\t                           2: Single-Element Robin Hood (default),\n"
                        "\t                           3: Multi-Element Robin Hood,\n"
                        "\t                           4: BiCGStab,\n"
                        "\t                           5: BiCGStab w/Jacobi preconditioner\n"
                        "\t                           6: Successive Subspace Correction)\n"
                        "\t -n, --rh_dimension       (dimension of subspace)\n"
                        "\t -l, --plot_matrix        (plot the matrix elements on a grid)\n"
                        "\t -r, --residual_graph     (graph residual during iterative solve)\n";

    unsigned int dimension = 2;
    double diagonal = 2.;
    double offDiagonal = 1.;
    double power = -1.;
    double accuracy = 1.e-8;
    //  bool cache = false;
    int method = 2;
    int rh_dimension = 2;
    bool plotMatrix = false;
    bool residualGraph = false;
    (void) residualGraph;

    static struct option longOptions[] = {
        {"help", no_argument, 0, 'h'},
        {"dimension", required_argument, 0, 'D'},
        {"diagonal", required_argument, 0, 'd'},
        {"off_diagonal", required_argument, 0, 'o'},
        {"power", required_argument, 0, 'p'},
        {"accuracy", required_argument, 0, 'a'},
        {"cache", no_argument, 0, 'c'},
        {"method", required_argument, 0, 'm'},
        {"rh_dimension", required_argument, 0, 'n'},
        {"plot_matrix", no_argument, 0, 'l'},
        {"residual_graph", required_argument, 0, 'r'},
    };

    static const char* optString = "hD:d:o:p:a:cm:n:lr";

    while (1) {
        char optId = getopt_long(argc, argv, optString, longOptions, NULL);
        if (optId == -1)
            break;
        switch (optId) {
            case ('h'):  // help
                std::cout << usage << std::endl;
                return 0;
            case ('D'):
                dimension = atoi(optarg);
                break;
            case ('d'):
                diagonal = atof(optarg);
                break;
            case ('o'):
                offDiagonal = atof(optarg);
                break;
            case ('p'):
                power = atof(optarg);
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
            case ('l'):
                plotMatrix = true;
                break;
            case ('r'):
                residualGraph = true;
                break;
            default:  // unrecognized option
                std::cout << usage << std::endl;
                return 1;
        }
    }

    KInverseDistance A(dimension, diagonal, offDiagonal, power);

    if (plotMatrix) {
        KEMFieldCanvas* fieldCanvas = NULL;

#if defined(KEMFIELD_USE_VTK)
        fieldCanvas = new KEMVTKFieldCanvas(0, dimension, 0, dimension, 1.e30, true);
#elif defined(KEMFIELD_USE_ROOT)
        fieldCanvas = new KEMRootFieldCanvas(0, dimension, 0, dimension, 1.e30, true);
#endif

        if (fieldCanvas) {
            std::vector<double> x_;
            std::vector<double> y_;
            std::vector<double> V_;

            for (unsigned int i = 0; i < dimension; i++) {
                x_.push_back(i);
                y_.push_back(i);

                for (unsigned int j = 0; j < dimension; j++) {
                    double value = A(i, j);
                    if (value > 1.e-16)
                        V_.push_back(log(value));
                    else
                        V_.push_back(-16.);
                }
            }

            fieldCanvas->DrawFieldMap(x_, y_, V_, false, 0.);
            fieldCanvas->LabelAxes("i", "j", "log (A_{ij})");
            std::string fieldCanvasName = DEFAULT_OUTPUT_DIR;
            fieldCanvas->SaveAs(fieldCanvasName + "/Matrix.gif");
        }
    }

    // KSimpleVector<double> xSolved(dimension);

    // for (unsigned int idx=0;idx<dimension;idx++)
    //   xSolved[idx] = 1.;

    KSimpleVector<double> b(dimension);

    // A.Multiply(xSolved,b);

    for (unsigned int i = 0; i < dimension; i++)
        b[i] = 1.;

    KSimpleVector<double> x(dimension);

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
        if (residualGraph)
            gaussSeidel.AddVisitor(new KVTKResidualGraph<double>());
#endif

        gaussSeidel.Solve(A, x, b);
    }
    else if (method == 2) {
        KRobinHood<double> robinHood;
        robinHood.SetTolerance(accuracy);

        robinHood.AddVisitor(new KIterationDisplay<double>());

        robinHood.SetResidualCheckInterval(b.Dimension());

#ifdef KEMFIELD_USE_VTK
        if (residualGraph) {
            robinHood.AddVisitor(new KVTKResidualGraph<double>());
            robinHood.SetResidualCheckInterval(1);
        }
#endif

        robinHood.Solve(A, x, b);
    }
    else if (method == 3) {
        KMultiElementRobinHood<double, KMultiElementRobinHood_SingleThread> robinHood;
        robinHood.SetSubspaceDimension(rh_dimension);
        robinHood.SetTolerance(accuracy);

        robinHood.AddVisitor(new KIterationDisplay<double>());

        robinHood.SetResidualCheckInterval(b.Dimension() / rh_dimension);

#ifdef KEMFIELD_USE_VTK
        if (residualGraph) {
            robinHood.AddVisitor(new KVTKResidualGraph<double>());
            robinHood.SetResidualCheckInterval(1);
        }
#endif

        robinHood.Solve(A, x, b);
    }
    else if (method == 4) {
        KSimpleIterativeKrylovSolver<double, KBiconjugateGradientStabilized> biCGStab;
        biCGStab.SetTolerance(accuracy);

        biCGStab.AddVisitor(new KIterationDisplay<double>());

#ifdef KEMFIELD_USE_VTK
        if (residualGraph)
            biCGStab.AddVisitor(new KVTKResidualGraph<double>());
#endif

        biCGStab.Solve(A, x, b);
    }
    else if (method == 5) {

        KPreconditionedIterativeKrylovSolver<double, KPreconditionedBiconjugateGradientStabilized> biCGStab;
        biCGStab.SetTolerance(accuracy);

        biCGStab.AddVisitor(new KIterationDisplay<double>());

#ifdef KEMFIELD_USE_VTK
        if (residualGraph)
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
        if (residualGraph)
            ssc.AddVisitor(new KVTKResidualGraph<double>());
#endif

        ssc.SetResidualCheckInterval(b.Dimension() / rh_dimension);

        ssc.Solve(A, x, b);
    }

    double computationTime = Time();

    std::cout << "Converged after " << computationTime << " seconds" << std::endl;

    std::cout << "Infinity norm of solution: " << x.InfinityNorm() << std::endl;

    // x -= xSolved;
    // x *= 1./xSolved.InfinityNorm();
    // std::cout<<"relative error of x is "<<x.InfinityNorm()<<std::endl;

    return 0;
}
