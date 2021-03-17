#ifndef KGDEFORMATION_HH_
#define KGDEFORMATION_HH_

#include "KThreeVector.hh"

namespace KGeoBag
{
class KGDeformation
{
  public:
    KGDeformation() = default;
    virtual ~KGDeformation() = default;

    virtual void Apply(KThreeVector& point) const = 0;
};
}  // namespace KGeoBag

#endif
