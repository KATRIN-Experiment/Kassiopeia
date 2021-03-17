#ifndef KGeoBag_KGBoxMesher_hh_
#define KGeoBag_KGBoxMesher_hh_

#include "KGBox.hh"
#include "KGComplexMesher.hh"

namespace KGeoBag
{
class KGBoxMesher : virtual public KGComplexMesher, public KGBox::Visitor
{
  public:
    using KGMesherBase::VisitExtendedSpace;
    using KGMesherBase::VisitExtendedSurface;

  public:
    KGBoxMesher() = default;
    ~KGBoxMesher() override = default;

  protected:
    void VisitBox(KGBox* box) override;
};

}  // namespace KGeoBag

#endif
