#include "KSVDSolver.hh"
#include "KSimpleMatrix.hh"
#include "KSimpleVector.hh"

#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>

#ifdef KEMFIELD_USE_ROOT
#include "KEMRootSVDSolver.hh"
#endif

using namespace KEMField;

int main()
{
    KSimpleMatrix<double> A(3, 4);
    A(0, 0) = 1.;
    A(0, 1) = 1.;
    A(0, 2) = 0.;
    A(0, 3) = 1.;
    A(1, 0) = -1.;
    A(1, 1) = 1.;
    A(1, 2) = 2.;
    A(1, 3) = 2.;
    A(2, 0) = 0.;
    A(2, 1) = 0.;
    A(2, 2) = 1.;
    A(2, 3) = 3.;

    KSimpleVector<double> x(A.Dimension(1));
    KSimpleVector<double> b(A.Dimension(0));
    b[0] = 1.;
    b[1] = 0.;
    b[2] = .5;

#ifdef KEMFIELD_USE_ROOT
    {
        std::cout << "Using Root's SVD solver:" << std::endl;

        KEMRootSVDSolver<double> solver;

        bool ok = solver.Solve(A, x, b);

        if (ok) {
            std::cout << "solution:";
            for (unsigned int i = 0; i < x.Dimension(); i++)
                std::cout << " " << x(i);
            std::cout << "" << std::endl;

            KSimpleVector<double> b_comp(b.Dimension());
            A.Multiply(x, b_comp);
            std::cout << "b:     ";
            for (unsigned int i = 0; i < b.Dimension(); i++)
                std::cout << " " << b(i);
            std::cout << "" << std::endl;

            std::cout << "b_comp:";
            for (unsigned int i = 0; i < b_comp.Dimension(); i++)
                std::cout << " " << b_comp(i);
            std::cout << "" << std::endl;
        }
        else {
            std::cout << "solution cannot be determined" << std::endl;
        }
    }
#endif

    {
        std::cout << "Using generic SVD solver:" << std::endl;

        KSVDSolver<double> solver;
        bool ok = solver.Solve(A, x, b);

        if (ok) {
            std::cout << "solution:";
            for (unsigned int i = 0; i < x.Dimension(); i++)
                std::cout << " " << x(i);
            std::cout << "" << std::endl;

            KSimpleVector<double> b_comp(b.Dimension());
            A.Multiply(x, b_comp);
            std::cout << "b:     ";
            for (unsigned int i = 0; i < b.Dimension(); i++)
                std::cout << " " << b(i);
            std::cout << "" << std::endl;

            std::cout << "b_comp:";
            for (unsigned int i = 0; i < b_comp.Dimension(); i++)
                std::cout << " " << b_comp(i);
            std::cout << "" << std::endl;
        }
        else {
            std::cout << "solution cannot be determined" << std::endl;
        }
    }

    return 0;
}
