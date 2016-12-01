#include <getopt.h>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <iomanip>

#include "KMatrix.hh"
#include "KSquareMatrix.hh"
#include "KSimpleMatrix.hh"
#include "KSimpleSquareMatrix.hh"
#include "KVector.hh"
#include "KSimpleVector.hh"

#include "KGaussianElimination.hh"
#include "KRobinHood.hh"
#include "KRobinHood_SingleThread.hh"
#include "KSimpleIterativeKrylovSolver.hh"
#include "KBiconjugateGradientStabilized.hh"
#include "KGeneralizedMinimalResidual.hh"

#include "KPreconditionedIterativeKrylovSolver.hh"
#include "KJacobiPreconditioner.hh"
#include "KPreconditionedBiconjugateGradientStabilized.hh"
#include "KPreconditionedGeneralizedMinimalResidual.hh"


#ifdef KEMFIELD_USE_VTK
#include "KVTKResidualGraph.hh"
#include "KVTKIterationPlotter.hh"
#endif

using namespace KEMField;

int main(int argc, char** argv)
{
  std::string usage =
    "\n"
    "Usage: TestKrylovSolvers <options>\n"
    "\n"
    "This program computes compares the solution to a randomly generated linear equation A*x = b, compares the solution of various Krylov solvers to Gaussian elimination \n"
    "\tAvailable options:\n"
    "\t -h, --help               (shows this message and exits)\n"
    "\t -a, --accuracy           (absolute tolerance on residual norm)\n"
    "\t -s, --size               (number of unknowns in linear equation)\n"
    "\t -t, --type               (0: dense uniformly random matrix, \n"
    "\t                           1: diagonally dominant dense random matrix, \n"
    "\t -m, --method             (0: BiCGStab,\n"
    "\t                           1: GMRES, \n"
    "\t                           2: BiCGStab w/Jacobi preconditioner,\n"
    "\t                           3: GMRES w/Jacobi preconditioner)\n"
    "\t                           4: Robin Hood (for comparison) \n"
    "\t                           5: Gaussian elimination (for comparison) \n"
    ;


    double tolerance = 1e-6;
    int size = 100;
    int method = 0;
    int type = 0;

    static struct option longOptions[] =
    {
        {"help", no_argument, 0, 'h'},
        {"accuracy", required_argument, 0, 'a'},
        {"size", required_argument, 0, 's'},
        {"type", required_argument, 0, 't'},
        {"method", required_argument, 0, 'm'}
    };

    static const char *optString = "ha:t:s:m:";

    while(1)
    {
        char optId = getopt_long(argc, argv,optString, longOptions, NULL);
        if(optId == -1) break;
        switch(optId)
        {
            case('h'): // help
            std::cout<<usage<<std::endl;
            return 0;
            case('a'):
            tolerance = atof(optarg);
            break;
            case('s'):
            size = atoi(optarg);
            break;
            case('t'):
            type = atoi(optarg);
            break;
            case('m'):
            method = atoi(optarg);
            break;
            default:
            std::cout<<usage<<std::endl;
            return 1;
        }
    }

    //we have to generate a random matrix and some vectors
    KSimpleSquareMatrix<double> A(size);
    KSimpleVector<double> X(size,0.);
    KSimpleVector<double> X_original(size, 0.);
    KSimpleVector<double> B(size, 0.);

    //seed the random number generator
//    srand((unsigned)time(NULL));
    srand(12346);

    //zero out the matrix
    for(int i=0; i<size; i++)
    {
        for(int j=0; j<size; j++)
        {
            A(i,j) = 0.0;
        }
    }

    if(type == 0)
    {
        //dense uniformly random matrix
        for(int i=0; i<size; i++)
        {
            for(int j=0; j<size; j++)
            {
                double r = ((double)rand()/(double)RAND_MAX);
                A(i,j) += r;
            }
        }
    }
    else if(type >= 1)
    {
        //we want the diagonals of the matrix to be positive
        //and larger than any other elements in their row, similar to a BEM matrix
        double diag_val = std::sqrt((double)size);

        for(int i=0; i<size; i++)
        {
            A(i,i) = diag_val;
        }

        for(int i=0; i<size; i++)
        {
            for(int j=0; j<size; j++)
            {
                double r = ((double)rand()/(double)RAND_MAX);
                double diagonal_dist = 1.0;
                if(i != j)
                {
                    diagonal_dist = 1.0/( (double)i - (double)j );
                    diagonal_dist *= diagonal_dist;
                }
                A(i,j) += diagonal_dist*r;
            }
        }
    }

    //now we generate a random vector x_original
    for(int i=0; i<size; i++)
    {
        double r1 = ((double)rand()/(double)RAND_MAX);
        X_original[i] = r1;
    }

    //now we compute the right hand side B
    A.Multiply(X_original, B);


    //set X to zero
    for(int i=0; i<size; i++)
    {
        X[i] = 0.;
    }

    double l2_norm = 0.;

////////////////////////////////////////////////////////////////////////////////
    switch(method)
    {
        case 0:
        {
            //now solve A*X=B with bicgstab solver

            KSimpleIterativeKrylovSolver<double, KBiconjugateGradientStabilized> biCGStab;
            biCGStab.SetTolerance(tolerance);
            biCGStab.AddVisitor(new KIterationDisplay<double>());
            biCGStab.Solve(A,X,B);

            l2_norm = 0.;
            for(int i=0; i<size; i++)
            {
                l2_norm += (X_original[i] - X[i])*(X_original[i] - X[i]);
            }
            l2_norm = std::sqrt(l2_norm);

            std::cout<<"l2 norm of biCGStab solution is: "<<l2_norm<<std::endl;
        }
        break;
        case 1:
        {
            //now solve A*X=B with a gmres solver

            KSimpleIterativeKrylovSolver<double, KGeneralizedMinimalResidual> gmres;
            gmres.SetTolerance(tolerance);
            gmres.AddVisitor(new KIterationDisplay<double>());
            gmres.Solve(A,X,B);

            l2_norm = 0.;
            for(int i=0; i<size; i++)
            {
                l2_norm += (X_original[i] - X[i])*(X_original[i] - X[i]);
            }
            l2_norm = std::sqrt(l2_norm);

            std::cout<<"l2 norm of gmres solution is: "<<l2_norm<<std::endl;
        }
        break;
        case 2:
        {
            //create a preconditioner
            KJacobiPreconditioner<double> P(A);
            //now solve A*X=B with jacobi preconditioned bicgstab solver

            KPreconditionedIterativeKrylovSolver<double, KPreconditionedBiconjugateGradientStabilized> pbiCGStab;

            pbiCGStab.SetTolerance(tolerance);
            pbiCGStab.AddVisitor(new KIterationDisplay<double>());
            pbiCGStab.Solve(A,P,X,B);

            l2_norm = 0.;
            for(int i=0; i<size; i++)
            {
                l2_norm += (X_original[i] - X[i])*(X_original[i] - X[i]);
            }
            l2_norm = std::sqrt(l2_norm);

            std::cout<<"l2 norm of preconditioned biCGStab solution is: "<<l2_norm<<std::endl;
        }
        break;
        case 3:
        {
            //create a preconditioner
            KJacobiPreconditioner<double> P(A);

            //now solve A*X=B with preconditioned gmres solver

            KPreconditionedIterativeKrylovSolver<double, KPreconditionedGeneralizedMinimalResidual> pgmres;
            pgmres.SetTolerance(tolerance);
            pgmres.AddVisitor(new KIterationDisplay<double>());
            pgmres.Solve(A,P,X,B);

            l2_norm = 0.;
            for(int i=0; i<size; i++)
            {
                l2_norm += (X_original[i] - X[i])*(X_original[i] - X[i]);
            }
            l2_norm = std::sqrt(l2_norm);

            std::cout<<"l2 norm of preconditioned gmres solution is: "<<l2_norm<<std::endl;
        }
        break;
        case 4:
        {
            //now solve A*X=B with robin hood

            KRobinHood<double> robinHood;
            robinHood.SetTolerance(tolerance);
            robinHood.AddVisitor(new KIterationDisplay<double>());
            robinHood.SetResidualCheckInterval(100);
            robinHood.Solve(A,X,B);

            l2_norm = 0.;
            for(int i=0; i<size; i++)
            {
                l2_norm += (X_original[i] - X[i])*(X_original[i] - X[i]);
            }
            l2_norm = std::sqrt(l2_norm);

            std::cout<<"l2 norm of robin hood solution is: "<<l2_norm<<std::endl;
        }
        break;
        case 5:
        {
            //now we use gaussian elimination to solve for X
            KGaussianElimination<double> gaussianElimination;
            gaussianElimination.Solve(A,X,B);

            //compute L2 norm difference with X_original
            l2_norm = 0.;
            for(int i=0; i<size; i++)
            {
                l2_norm += (X_original[i] - X[i])*(X_original[i] - X[i]);
            }
            l2_norm = std::sqrt(l2_norm);

            std::cout<<"l2 norm of gaussian elimination solution is: "<<l2_norm<<std::endl;
        }
        break;
        default:
        break;
    }

    return 0;
}
