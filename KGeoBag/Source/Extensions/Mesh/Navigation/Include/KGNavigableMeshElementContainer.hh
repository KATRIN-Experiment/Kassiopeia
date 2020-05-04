#ifndef KGNavigableMeshElementContainer_HH__
#define KGNavigableMeshElementContainer_HH__

#include "KGAxisAlignedBox.hh"
#include "KGBall.hh"
#include "KGBoundaryCalculator.hh"
#include "KGCube.hh"
#include "KGNavigableMeshElement.hh"
#include "KGPointCloud.hh"

#include <vector>

namespace KGeoBag
{

class KGNavigableMeshElementContainer
{
  public:
    KGNavigableMeshElementContainer();
    virtual ~KGNavigableMeshElementContainer();

    //access individual element data
    virtual void Add(KGNavigableMeshElement* element);
    virtual KGNavigableMeshElement* GetElement(unsigned int id);
    virtual KGBall<KGMESH_DIM> GetElementBoundingBall(unsigned int id);

    unsigned int GetNElements() const
    {
        return fMeshElements.size();
    };

    //cube enclosing all contained elements
    virtual KGCube<KGMESH_DIM> GetGlobalBoundingCube();

  protected:
    std::vector<KGNavigableMeshElement*> fMeshElements;
    std::vector<KGBall<KGMESH_DIM>> fMeshElementBoundingBalls;

    KGBoundaryCalculator<KGMESH_DIM> fBoundaryCalculator;

    bool fValidGlobalCube;
    KGCube<KGMESH_DIM> fGlobalBoundingCube;
};

}  // namespace KGeoBag

#endif /* end of include guard: KGNavigableMeshElementContainer_HH__ */
