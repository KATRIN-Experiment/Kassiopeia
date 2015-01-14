#ifndef KELECTROSTATICFIELDSOLVER_DEF
#define KELECTROSTATICFIELDSOLVER_DEF

#include "KEMThreeVector.hh"

namespace KEMField
{
  class KElectrostaticFieldSolver
  {
  public:
    KElectrostaticFieldSolver() {}
    virtual ~KElectrostaticFieldSolver() {}

    static std::string Name() { return "ElectrostaticFieldSolver"; }

    virtual double Potential(const KPosition& P) const = 0;
    virtual KEMThreeVector ElectricField(const KPosition&) const = 0;
  };

}

#endif /* KELECTROSTATICFIELDSOLVER_DEF */
