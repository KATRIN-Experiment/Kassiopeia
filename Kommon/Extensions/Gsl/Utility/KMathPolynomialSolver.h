#ifndef KMATHPOLYNOMIALSOLVER_H_
#define KMATHPOLYNOMIALSOLVER_H_

#include "gsl/gsl_poly.h"

namespace katrin
{

    class KMathPolynomialSolver
    {
        public:
            enum
            {
                eMaxDegree = 10
            };

        public:
            KMathPolynomialSolver();
            ~KMathPolynomialSolver();

        public:
            void Solve( unsigned int aDegree, const double* aCoefficientSet, double* aSolutionSet ) const;

        private:
            class Workspaces
            {
                public:
                    Workspaces();
                    ~Workspaces();

                    gsl_poly_complex_workspace* operator[]( unsigned int aType );

                private:
                    gsl_poly_complex_workspace* fTypes[ eMaxDegree ];
            };

            static Workspaces sPolynomialTypes;
    };

    inline void KMathPolynomialSolver::Solve( unsigned int aDegree, const double aCoefficientSet[ eMaxDegree + 1 ], double aSolutionSet[ 2 * eMaxDegree ] ) const
    {
        gsl_poly_complex_workspace* tWorkspace = sPolynomialTypes[ aDegree ];
        gsl_poly_complex_solve( aCoefficientSet, aDegree + 1, tWorkspace, aSolutionSet );
        return;
    }

}

#endif
