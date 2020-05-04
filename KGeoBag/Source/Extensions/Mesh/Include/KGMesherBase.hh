#ifndef KGeoBag_KGMesherBase_hh_
#define KGeoBag_KGMesherBase_hh_

#include "KGCore.hh"
#include "KGMesh.hh"

namespace KGeoBag
{

class KGMesherBase :
    public KGVisitor,
    public KGExtendedSurface<KGMesh>::Visitor,
    public KGExtendedSpace<KGMesh>::Visitor
{
  protected:
    KGMesherBase();

  public:
    ~KGMesherBase() override;

  public:
    void VisitExtendedSurface(KGExtendedSurface<KGMesh>* aSurface) override;
    void VisitExtendedSpace(KGExtendedSpace<KGMesh>* aSpace) override;

  protected:
    KGMeshElementVector* fCurrentElements;
    KGExtendedSurface<KGMesh>* fCurrentSurface;
    KGExtendedSpace<KGMesh>* fCurrentSpace;
};
}  // namespace KGeoBag

#endif
