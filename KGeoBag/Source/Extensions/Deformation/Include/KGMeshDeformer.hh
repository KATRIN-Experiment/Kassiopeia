#ifndef KGMESHDEFORMER_DEF
#define KGMESHDEFORMER_DEF

#include "KGCore.hh"
#include "KGDeformation.hh"
#include "KGDeformed.hh"
#include "KGMesh.hh"

namespace KGeoBag
{
class KGMeshElement;
class KGMeshRectangle;
class KGMeshTriangle;
class KGMeshWire;

class KGMeshDeformer :
    public KGVisitor,
    public KGExtendedSpace<KGDeformed>::Visitor,
    public KGExtendedSurface<KGMesh>::Visitor
{
  public:
    KGMeshDeformer() = default;
    ~KGMeshDeformer() override = default;

    void VisitSurface(KGSurface*) {}
    void VisitExtendedSpace(KGExtendedSpace<KGDeformed>* deformedSpace) override;
    void VisitExtendedSurface(KGExtendedSurface<KGMesh>* meshSurface) override;

  private:
    void AddDeformed(KGMeshElement* e);
    void AddDeformed(KGMeshRectangle* r);
    void AddDeformed(KGMeshTriangle* t);
    void AddDeformed(KGMeshWire* w);

    std::shared_ptr<KGDeformation> fDeformation;
    KGMeshElementVector fDeformedVector;
};
}  // namespace KGeoBag

#endif /* KGMESHDEFORMER_DEF */
