#ifndef KELECTROSTATICRINGINTEGRATOR_DEF
#define KELECTROSTATICRINGINTEGRATOR_DEF

#include "KSurface.hh"
#include "KEMConstants.hh"

namespace KEMField
{
  class KElectrostaticRingIntegrator
  {
  public:
    typedef KRing Shape;
    typedef KElectrostaticBasis::ValueType ValueType;

    friend class KElectrostaticConicSectionIntegrator;

    KElectrostaticRingIntegrator() {}
    ~KElectrostaticRingIntegrator() {}

    double Potential(const KRing* source,
		       const KPosition& P) const;

    KEMThreeVector ElectricField(const KRing* source,
			       const KPosition& P) const;

    double Potential(const KSymmetryGroup<KRing>* source,
		       const KPosition& P) const;

    KEMThreeVector ElectricField(const KSymmetryGroup<KRing>* source,
			       const KPosition& P) const;

  private:
    static double PotentialFromChargedRing(const double *P,double *par);
    static double EFieldRFromChargedRing(const double *P,double *par);
    static double EFieldZFromChargedRing(const double *P,double *par);
  };

}

#endif /* KELECTROSTATICRINGINTEGRATOR_DEF */
