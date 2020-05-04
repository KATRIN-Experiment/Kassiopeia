#ifndef KGeoBag_KGRotatedCircleSpaceAxialMesher_hh_
#define KGeoBag_KGRotatedCircleSpaceAxialMesher_hh_

#include "KGRotatedCircleSpace.hh"
#include "KGSimpleAxialMesher.hh"

namespace KGeoBag
{
class KGRotatedCircleSpaceAxialMesher : virtual public KGSimpleAxialMesher, public KGRotatedCircleSpace::Visitor
{
  public:
    using KGAxialMesherBase::VisitExtendedSpace;
    using KGAxialMesherBase::VisitExtendedSurface;

  public:
    KGRotatedCircleSpaceAxialMesher();
    ~KGRotatedCircleSpaceAxialMesher() override;

  protected:
    void VisitRotatedClosedPathSpace(KGRotatedCircleSpace* aRotatedCircleSpace) override;
};

}  // namespace KGeoBag

#endif
