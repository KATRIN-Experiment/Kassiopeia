#ifndef KELECTROSTATICCONSTANTFIELD_DEF
#define KELECTROSTATICCONSTANTFIELD_DEF

#include "KElectrostaticField.hh"

namespace KEMField
{
  class KElectrostaticConstantField : public KElectrostaticField
  {
  public:
    KElectrostaticConstantField() :
      KElectrostaticField(),
      fField() {}

    KElectrostaticConstantField(const KEMThreeVector& field) :
      KElectrostaticField(),
      fField(field) {}

    virtual ~KElectrostaticConstantField() {}

    void SetField(KEMThreeVector field) { fField = field; }
    KEMThreeVector GetField() {return fField;}

    static std::string Name() { return "ElectrostaticConstantFieldSolver"; }

private:
    virtual double PotentialCore(const KPosition& P) const {
    	return fField.Dot(P);
    }

    virtual KEMThreeVector ElectricFieldCore(const KPosition&) const {
    	return fField;
    }

  protected:

    KEMThreeVector fField;
  };

}

#endif /* KELECTROSTATICCONSTANTFIELD_DEF */
