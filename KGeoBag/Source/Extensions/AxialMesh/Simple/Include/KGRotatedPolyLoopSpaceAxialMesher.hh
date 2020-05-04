#ifndef KGeoBag_KGRotatedPolyLoopSpaceAxialMesher_hh_
#define KGeoBag_KGRotatedPolyLoopSpaceAxialMesher_hh_

#include "KGRotatedPolyLoopSpace.hh"
#include "KGSimpleAxialMesher.hh"

namespace KGeoBag
{
class KGRotatedPolyLoopSpaceAxialMesher : virtual public KGSimpleAxialMesher, public KGRotatedPolyLoopSpace::Visitor
{
  public:
    using KGAxialMesherBase::VisitExtendedSpace;
    using KGAxialMesherBase::VisitExtendedSurface;

  public:
    KGRotatedPolyLoopSpaceAxialMesher();
    ~KGRotatedPolyLoopSpaceAxialMesher() override;

  protected:
    void VisitRotatedClosedPathSpace(KGRotatedPolyLoopSpace* aRotatedPolyLoopSpace) override;
};

}  // namespace KGeoBag

#endif
