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

    KElectrostaticConstantField(const KThreeVector& field) :
      KElectrostaticField(),
      fField(field) {}

    virtual ~KElectrostaticConstantField() {}

    void SetField(KThreeVector field) { fField = field; }
    KThreeVector GetField() {return fField;}

    static std::string Name() { return "ElectrostaticConstantFieldSolver"; }

private:
    virtual double PotentialCore(const KPosition& P) const {
    	return fField.Dot(P);
    }

    virtual KThreeVector ElectricFieldCore(const KPosition&) const {
    	return fField;
    }

  protected:

    KThreeVector fField;
  };

}

#endif /* KELECTROSTATICCONSTANTFIELD_DEF */
