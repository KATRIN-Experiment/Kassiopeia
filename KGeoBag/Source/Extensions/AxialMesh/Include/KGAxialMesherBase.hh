#ifndef KGeoBag_KGAxialMesherBase_hh_
#define KGeoBag_KGAxialMesherBase_hh_

#include "KGAxialMesh.hh"
#include "KGCore.hh"

namespace KGeoBag
{

class KGAxialMesherBase :
    public KGVisitor,
    public KGExtendedSurface<KGAxialMesh>::Visitor,
    public KGExtendedSpace<KGAxialMesh>::Visitor
{
  protected:
    KGAxialMesherBase();

  public:
    ~KGAxialMesherBase() override;

  public:
    void VisitExtendedSurface(KGExtendedSurface<KGAxialMesh>* aSurface) override;
    void VisitExtendedSpace(KGExtendedSpace<KGAxialMesh>* aSpace) override;

  protected:
    KGAxialMeshElementVector* fCurrentElements;
};

}  // namespace KGeoBag

#endif
