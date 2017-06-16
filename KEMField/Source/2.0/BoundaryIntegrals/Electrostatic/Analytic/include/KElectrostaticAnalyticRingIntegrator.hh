#ifndef KELECTROSTATICRINGINTEGRATOR_DEF
#define KELECTROSTATICRINGINTEGRATOR_DEF

#include "KSurface.hh"
#include "KEMConstants.hh"

#include "KElectrostaticElementIntegrator.hh"

namespace KEMField
{
  class KElectrostaticAnalyticRingIntegrator :
          public KElectrostaticElementIntegrator<KRing>
  {
  public:
    typedef KRing Shape;
    typedef KElectrostaticBasis::ValueType ValueType;

    friend class KElectrostaticAnalyticConicSectionIntegrator;

    KElectrostaticAnalyticRingIntegrator() {}
    ~KElectrostaticAnalyticRingIntegrator() {}

    double Potential( const KRing* source, const KPosition& P ) const;
    KEMThreeVector ElectricField( const KRing* source, const KPosition& P ) const;
    using KElectrostaticElementIntegrator<KRing>::Potential;
    using KElectrostaticElementIntegrator<KRing>::ElectricField;

  private:
    static double PotentialFromChargedRing(const double *P,double *par);
    static double EFieldRFromChargedRing(const double *P,double *par);
    static double EFieldZFromChargedRing(const double *P,double *par);
  };

}

#endif /* KELECTROSTATICRINGINTEGRATOR_DEF */
