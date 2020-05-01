#include "KEMRootSVDSolver.hh"

#include "TDecompSVD.h"
#include "TMatrixDUtils.h"
#include "TROOT.h"
#include "TVectorD.h"

#include <iostream>

namespace KEMField
{
bool KEMRootSVDSolver<double>::Solve(const KMatrix<double>& A, KVector<double>& x, const KVector<double>& b) const
{
    UInt_t nRows = A.Dimension(0);
    if (nRows < A.Dimension(1))
        nRows = A.Dimension(1);
    TMatrixD A_root(nRows, A.Dimension(1));
    for (UInt_t i = 0; i < nRows; i++)
        for (UInt_t j = 0; j < A.Dimension(1); j++)
            A_root[i][j] = (i < A.Dimension(0) ? A(i)(j) : 0.);

    TVectorD b_root(nRows);
    for (UInt_t i = 0; i < nRows; i++)
        b_root[i] = (i < A.Dimension(0) ? b(i) : 0.);

    TDecompSVD svd(A_root);
    Bool_t ok;
    ok = svd.Decompose();
    if (!ok)
        return ok;
    TVectorD x_root(A.Dimension(1));
    x_root = svd.Solve(b_root, ok);
    if (!ok)
        return ok;

    for (UInt_t i = 0; i < A.Dimension(1); i++)
        x[i] = x_root[i];

    KSimpleVector<double> b_comp(b.Dimension());
    A.Multiply(x, b_comp);
    b_comp -= b;

    return b_comp.InfinityNorm() < fTolerance;
}
}  // namespace KEMField
