#ifndef KGeoBag_KGDiscreteRotationalMesher_hh_
#define KGeoBag_KGDiscreteRotationalMesher_hh_

#include "KGAxialMesh.hh"
#include "KGConicalWireArrayDiscreteRotationalMesher.hh"
#include "KGCore.hh"

namespace KGeoBag
{
class KGAxialMeshElement;
class KGAxialMeshLoop;
class KGAxialMeshRing;

class KGDiscreteRotationalMesher :

    public KGSurface::Visitor,
    public KGSpace::Visitor,
    public KGVisitor

{
  public:
    void VisitSurface(KGSurface* aSurface) override;
    void VisitSpace(KGSpace* aSpace) override;

  public:
    KGDiscreteRotationalMesher();
    ~KGDiscreteRotationalMesher() override;

    void SetAxialAngle(double d)
    {
        fAxialAngle = d;
    }
    void SetAxialCount(unsigned int i)
    {
        fAxialCount = i;
    }


  private:
    void MeshAxialSurface(KGExtendedSurface<KGAxialMesh>* aSurface);
    void MeshAxialSpace(KGExtendedSpace<KGAxialMesh>* aSpace);
    void AddAxialMeshElement(KGAxialMeshElement* e);

    void AddAxialMeshLoop(KGAxialMeshLoop* l);
    void AddAxialMeshRing(KGAxialMeshRing* r);

    KGDiscreteRotationalMeshElementVector* fCurrentElements;


    double fAxialAngle;
    unsigned int fAxialCount;
};
}  // namespace KGeoBag

#endif
