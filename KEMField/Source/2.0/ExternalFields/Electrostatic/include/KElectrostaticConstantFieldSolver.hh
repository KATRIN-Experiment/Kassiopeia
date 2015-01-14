#ifndef KELECTROSTATICCONSTANTFIELDSOLVER_DEF
#define KELECTROSTATICCONSTANTFIELDSOLVER_DEF

#include "KElectrostaticFieldSolver.hh"

namespace KEMField
{
  class KElectrostaticConstantFieldSolver : public KElectrostaticFieldSolver
  {
  public:
    KElectrostaticConstantFieldSolver() :
      KElectrostaticFieldSolver(),
      fField() {}
    KElectrostaticConstantFieldSolver(KEMThreeVector field) :
      KElectrostaticFieldSolver(),
      fField(field) {}
    virtual ~KElectrostaticConstantFieldSolver() {}

    static std::string Name() { return "ElectrostaticConstantFieldSolver"; }

    double Potential(const KPosition& P) const { return fField.Dot(P); }
    KEMThreeVector ElectricField(const KPosition&) const { return fField; }

    void SetField(KEMThreeVector field) { fField = field; }

  protected:

    KEMThreeVector fField;
  };

}

#endif /* KELECTROSTATICCONSTANTFIELDSOLVER_DEF */
