#ifndef KELECTROSTATICFIELD_DEF
#define KELECTROSTATICFIELD_DEF

#include "KElectricField.hh"

namespace KEMField
{
  class KElectrostaticField : public KElectricField
  {
  public:
    KElectrostaticField() {}
    virtual ~KElectrostaticField() {}

    static std::string Name() { return "ElectrostaticField"; }

    using KElectricField::Potential; // don't hide time specifying Potential call with the overload below

    double Potential(const KPosition& P) const {
    	return PotentialCore(P);
    }

    using KElectricField::ElectricField; // don't hide time specifying ElectricField call with the overload below

    KEMThreeVector ElectricField(const KPosition& P ) const {
    	return ElectricFieldCore(P);
    }

  private:

    virtual double PotentialCore(const KPosition& P, const double& /*time*/) const
    {
    	return PotentialCore(P);
    }

    virtual KEMThreeVector ElectricFieldCore(const KPosition& P, const double& /*time*/) const
    {
    	return ElectricFieldCore(P);
    }

    virtual std::pair<KEMThreeVector,double> ElectricFieldAndPotentialCore(const KPosition& P, const double& /*time*/) const
    {
        return ElectricFieldAndPotentialCore(P);
    }

    virtual double PotentialCore(const KPosition& P) const = 0;

    virtual KEMThreeVector ElectricFieldCore(const KPosition&) const = 0;

    virtual std::pair<KEMThreeVector,double> ElectricFieldAndPotentialCore(const KPosition& P) const
    {
        //the default behavior is just to call the field and potential separately

        //this routine can be overloaded to allow for additional efficiency in for some specific
        //field calculations methods which can produce the field and potential values
        //at the same time with minimal additional work (e.g. ZH and fast multipole).

        double potential = PotentialCore(P);
        KEMThreeVector field = ElectricFieldCore(P);

        return std::pair<KEMThreeVector,double>(field,potential);
    }


  };

}

#endif /* KELECTROSTATICFIELD_DEF */
