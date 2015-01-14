#ifndef KGRADIALDEFORMATION_HH_
#define KGRADIALDEFORMATION_HH_

#include "KGDeformation.hh"

namespace KGeoBag
{

  class KGRadialDeformation : public KGDeformation
  {
  public:
    KGRadialDeformation() {}
    virtual ~KGRadialDeformation() {}

    void Apply(KThreeVector& point) const;

    virtual double RadialScale(double theta,double z) const = 0;
  };
}

#endif
