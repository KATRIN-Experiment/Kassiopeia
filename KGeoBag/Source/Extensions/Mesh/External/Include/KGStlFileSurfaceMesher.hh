/**
 * @file KGStlFileSurfaceMesher.hh
 * @author Jan Behrens <jan.behrens@kit.edu>
 * @date 2021-07-02
 */

#ifndef KGeoBag_KGStlFileSurfaceMesher_hh_
#define KGeoBag_KGStlFileSurfaceMesher_hh_

#include "KGStlFileSurface.hh"
#include "KGComplexMesher.hh"

namespace KGeoBag
{

class KGStlFileSurfaceMesher : virtual public KGComplexMesher, public KGStlFileSurface::Visitor
{
  public:
    using KGMesherBase::VisitExtendedSurface;

  public:
    KGStlFileSurfaceMesher() = default;
    ~KGStlFileSurfaceMesher() override = default;

  protected:
    void VisitWrappedSurface(KGStlFileSurface* stlSurface) override;

};

}  // namespace KGeoBag

#endif /* KGBEAMSURFACEMESHER_HH_ */
