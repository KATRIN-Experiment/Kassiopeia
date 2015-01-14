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
    public KGExtendedSpace< KGDeformed >::Visitor,
    public KGExtendedSurface< KGMesh >::Visitor
  {
  public:
    KGMeshDeformer() {}
    virtual ~KGMeshDeformer() {}

    void VisitSurface(KGSurface*) {}
    void VisitExtendedSpace( KGExtendedSpace< KGDeformed >* deformedSpace );
    void VisitExtendedSurface( KGExtendedSurface< KGMesh >* meshSurface );

  private:
    void AddDeformed(KGMeshElement* e);
    void AddDeformed(KGMeshRectangle* r);
    void AddDeformed(KGMeshTriangle* t);
    void AddDeformed(KGMeshWire* w);

    KSmartPointer<KGDeformation> fDeformation;
    KGMeshElementVector fDeformedVector;
  };
}

#endif /* KGMESHDEFORMER_DEF */
