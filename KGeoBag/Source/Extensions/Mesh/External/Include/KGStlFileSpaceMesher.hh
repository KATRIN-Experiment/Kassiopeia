/**
 * @file KGStlFileSpaceMesher.hh
 * @author Jan Behrens <jan.behrens@kit.edu>
 * @date 2021-07-02
 */

#ifndef KGeoBag_KGStlFileSpaceMesher_hh_
#define KGeoBag_KGStlFileSpaceMesher_hh_

#include "KGStlFileSpace.hh"
#include "KGComplexMesher.hh"

namespace KGeoBag
{

class KGStlFileSpaceMesher : virtual public KGComplexMesher, public KGStlFileSpace::Visitor
{
  public:
    using KGMesherBase::VisitExtendedSpace;

  public:
    KGStlFileSpaceMesher() = default;
    ~KGStlFileSpaceMesher() override = default;

  protected:
    void VisitWrappedSpace(KGStlFileSpace* stlSpace) override;
};

}  // namespace KGeoBag

#endif /* KGBEAMSpaceMESHER_HH_ */
