#ifndef KELECTROSTATICCONICSECTIONINTEGRATOR_DEF
#define KELECTROSTATICCONICSECTIONINTEGRATOR_DEF

#include "KSurface.hh"
#include "KEMConstants.hh"

#include "KElectrostaticElementIntegrator.hh"

namespace KEMField
{
  class KElectrostaticAnalyticConicSectionIntegrator :
          public KElectrostaticElementIntegrator<KConicSection>
  {
  public:
    typedef KConicSection Shape;
    typedef KElectrostaticBasis::ValueType ValueType;

    KElectrostaticAnalyticConicSectionIntegrator() {}
    ~KElectrostaticAnalyticConicSectionIntegrator() {}

    double Potential( const KConicSection* source, const KPosition& P ) const;
    KThreeVector ElectricField( const KConicSection* source, const KPosition& P ) const;
    using KElectrostaticElementIntegrator<KConicSection>::Potential;
    using KElectrostaticElementIntegrator<KConicSection>::ElectricField;
  };

}

#endif /* KELECTROSTATICCONICSECTIONINTEGRATOR_DEF */
