/**
 * @file KGPlyFileSurfaceMesher.hh
 * @author Jan Behrens <jan.behrens@kit.edu>
 * @date 2022-11-24
 */

#ifndef KGeoBag_KGPlyFileSurfaceMesher_hh_
#define KGeoBag_KGPlyFileSurfaceMesher_hh_

#include "KGPlyFileSurface.hh"
#include "KGComplexMesher.hh"

namespace KGeoBag
{

class KGPlyFileSurfaceMesher : virtual public KGComplexMesher, public KGPlyFileSurface::Visitor
{
  public:
    using KGMesherBase::VisitExtendedSurface;

  public:
    KGPlyFileSurfaceMesher() = default;
    ~KGPlyFileSurfaceMesher() override = default;

  protected:
    void VisitWrappedSurface(KGPlyFileSurface* PlySurface) override;

};

}  // namespace KGeoBag

#endif /* KGBEAMSURFACEMESHER_HH_ */
