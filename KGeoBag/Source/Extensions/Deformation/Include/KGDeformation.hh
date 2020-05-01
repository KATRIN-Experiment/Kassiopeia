#ifndef KGDEFORMATION_HH_
#define KGDEFORMATION_HH_

#include "KThreeVector.hh"

namespace KGeoBag
{
class KGDeformation
{
  public:
    KGDeformation() {}
    virtual ~KGDeformation() {}

    virtual void Apply(KThreeVector& point) const = 0;
};
}  // namespace KGeoBag

#endif
