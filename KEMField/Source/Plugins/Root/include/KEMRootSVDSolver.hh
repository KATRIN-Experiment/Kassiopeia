#ifndef KEMROOTSVDSOLVER_H
#define KEMROOTSVDSOLVER_H

#include "KMatrix.hh"
#include "KVector.hh"

namespace KEMField
{
  template <typename ValueType>
  class KEMRootSVDSolver;

  template <>
  class KEMRootSVDSolver<double>
  {
  public:
    typedef double ValueType;
    typedef KMatrix<ValueType> Matrix;
    typedef KVector<ValueType> Vector;

    KEMRootSVDSolver() : fTolerance(1.e-14) {}
    virtual ~KEMRootSVDSolver() {}

    bool Solve(const Matrix& A,Vector& x, const Vector& b) const;
    void SetTolerance(double tol) { fTolerance = tol; }

  private:
    double fTolerance;
  };
}

#endif
