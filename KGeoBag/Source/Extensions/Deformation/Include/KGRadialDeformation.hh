#ifndef KGRADIALDEFORMATION_HH_
#define KGRADIALDEFORMATION_HH_

#include "KGDeformation.hh"

namespace KGeoBag
{

class KGRadialDeformation : public KGDeformation
{
  public:
    KGRadialDeformation() {}
    ~KGRadialDeformation() override {}

    void Apply(KThreeVector& point) const override;

    virtual double RadialScale(double theta, double z) const = 0;
};
}  // namespace KGeoBag

#endif
