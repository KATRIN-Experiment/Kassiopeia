#ifndef KELECTROSTATICLINESEGMENTINTEGRATOR_DEF
#define KELECTROSTATICLINESEGMENTINTEGRATOR_DEF

#include "KSurface.hh"
#include "KEMConstants.hh"

namespace KEMField
{
  class KElectrostaticLineSegmentIntegrator
  {
  public:
    typedef KLineSegment Shape;
    typedef KElectrostaticBasis::ValueType ValueType;

    KElectrostaticLineSegmentIntegrator() {}
    ~KElectrostaticLineSegmentIntegrator() {}

    double Potential(const KLineSegment* source,
		       const KPosition& P) const;

    KEMThreeVector ElectricField(const KLineSegment* source,
			       const KPosition& P) const;

    double Potential(const KSymmetryGroup<KLineSegment>* source,
		       const KPosition& P) const;

    KEMThreeVector ElectricField(const KSymmetryGroup<KLineSegment>* source,
			       const KPosition& P) const;

  };

}

#endif /* KELECTROSTATICLINESEGMENTINTEGRATOR_DEF */
