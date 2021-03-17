#include "KMathPolynomialSolver.h"

namespace katrin
{

KMathPolynomialSolver::KMathPolynomialSolver() = default;
KMathPolynomialSolver::~KMathPolynomialSolver() = default;

KMathPolynomialSolver::Workspaces KMathPolynomialSolver::sPolynomialTypes = KMathPolynomialSolver::Workspaces();

KMathPolynomialSolver::Workspaces::Workspaces()
{
    for (unsigned int tType = 0; tType < eMaxDegree; tType++) {
        fTypes[tType] = gsl_poly_complex_workspace_alloc(tType + 1);
    }
}
KMathPolynomialSolver::Workspaces::~Workspaces()
{
    for (auto& type : fTypes) {
        gsl_poly_complex_workspace_free(type);
    }
}

gsl_poly_complex_workspace* KMathPolynomialSolver::Workspaces::operator[](unsigned int aType)
{
    return fTypes[aType];
}

}  // namespace katrin
