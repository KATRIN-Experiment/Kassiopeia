#include "KMathPolynomialSolver.h"

namespace katrin
{

KMathPolynomialSolver::KMathPolynomialSolver() {}
KMathPolynomialSolver::~KMathPolynomialSolver() {}

KMathPolynomialSolver::Workspaces KMathPolynomialSolver::sPolynomialTypes = KMathPolynomialSolver::Workspaces();

KMathPolynomialSolver::Workspaces::Workspaces()
{
    for (unsigned int tType = 0; tType < eMaxDegree; tType++) {
        fTypes[tType] = gsl_poly_complex_workspace_alloc(tType + 1);
    }
}
KMathPolynomialSolver::Workspaces::~Workspaces()
{
    for (unsigned int tType = 0; tType < eMaxDegree; tType++) {
        gsl_poly_complex_workspace_free(fTypes[tType]);
    }
}

gsl_poly_complex_workspace* KMathPolynomialSolver::Workspaces::operator[](unsigned int aType)
{
    return fTypes[aType];
}

}  // namespace katrin
