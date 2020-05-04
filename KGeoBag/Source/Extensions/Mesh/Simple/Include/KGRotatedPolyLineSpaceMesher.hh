#ifndef KGeoBag_KGRotatedPolyLineSpaceMesher_hh_
#define KGeoBag_KGRotatedPolyLineSpaceMesher_hh_

#include "KGRotatedPolyLineSpace.hh"
#include "KGSimpleMesher.hh"

namespace KGeoBag
{
class KGRotatedPolyLineSpaceMesher : virtual public KGSimpleMesher, public KGRotatedPolyLineSpace::Visitor
{
  public:
    using KGMesherBase::VisitExtendedSpace;
    using KGMesherBase::VisitExtendedSurface;

  public:
    KGRotatedPolyLineSpaceMesher();
    ~KGRotatedPolyLineSpaceMesher() override;

  protected:
    void VisitRotatedOpenPathSpace(KGRotatedPolyLineSpace* aRotatedPolyLineSpace) override;
};

}  // namespace KGeoBag

#endif
