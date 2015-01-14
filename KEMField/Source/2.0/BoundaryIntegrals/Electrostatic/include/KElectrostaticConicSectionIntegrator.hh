#ifndef KELECTROSTATICCONICSECTIONINTEGRATOR_DEF
#define KELECTROSTATICCONICSECTIONINTEGRATOR_DEF

#include "KSurface.hh"
#include "KEMConstants.hh"

namespace KEMField
{
  class KElectrostaticConicSectionIntegrator
  {
  public:
    typedef KConicSection Shape;
    typedef KElectrostaticBasis::ValueType ValueType;

    KElectrostaticConicSectionIntegrator() {}
    ~KElectrostaticConicSectionIntegrator() {}

    double Potential(const KConicSection* source,
		       const KPosition& P) const;

    KEMThreeVector ElectricField(const KConicSection* source,
			       const KPosition& P) const;

    double Potential(const KSymmetryGroup<KConicSection>* source,
		       const KPosition& P) const;

    KEMThreeVector ElectricField(const KSymmetryGroup<KConicSection>* source,
			       const KPosition& P) const;
  };

}

#endif /* KELECTROSTATICCONICSECTIONINTEGRATOR_DEF */
