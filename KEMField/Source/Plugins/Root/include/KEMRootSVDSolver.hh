#ifndef KEMROOTSVDSOLVER_H
#define KEMROOTSVDSOLVER_H

#include "KMatrix.hh"
#include "KVector.hh"

namespace KEMField
{
template<typename ValueType> class KEMRootSVDSolver;

template<> class KEMRootSVDSolver<double>
{
  public:
    typedef double ValueType;
    using Matrix = KMatrix<ValueType>;
    using Vector = KVector<ValueType>;

    KEMRootSVDSolver() : fTolerance(1.e-14) {}
    virtual ~KEMRootSVDSolver() = default;

    bool Solve(const Matrix& A, Vector& x, const Vector& b) const;
    void SetTolerance(double tol)
    {
        fTolerance = tol;
    }

  private:
    double fTolerance;
};
}  // namespace KEMField

#endif
