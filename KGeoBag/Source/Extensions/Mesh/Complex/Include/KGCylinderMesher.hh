#ifndef KGeoBag_KGCylinderMesher_hh_
#define KGeoBag_KGCylinderMesher_hh_

#include "KGComplexMesher.hh"
#include "KGCylinder.hh"

namespace KGeoBag
{
class KGCylinderMesher : virtual public KGComplexMesher, public KGCylinder::Visitor
{
  public:
    using KGMesherBase::VisitExtendedSpace;
    using KGMesherBase::VisitExtendedSurface;

  public:
    KGCylinderMesher() = default;
    ~KGCylinderMesher() override = default;

  protected:
    void VisitCylinder(KGCylinder* cylinder) override;
};

}  // namespace KGeoBag

#endif
