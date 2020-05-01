#ifndef KGeoBag_KGDiscreteRotationalMesh_hh_
#define KGeoBag_KGDiscreteRotationalMesh_hh_

#include "KGCore.hh"
#include "KGDiscreteRotationalMeshElement.hh"

namespace KGeoBag
{

class KGDiscreteRotationalMeshData
{
  public:
    KGDiscreteRotationalMeshData() {}
    KGDiscreteRotationalMeshData(KGSpace*) {}
    KGDiscreteRotationalMeshData(KGSurface*) {}
    KGDiscreteRotationalMeshData(KGSpace*, const KGDiscreteRotationalMeshData& aCopy)
    {
        fElements = aCopy.fElements;
    }
    KGDiscreteRotationalMeshData(KGSurface*, const KGDiscreteRotationalMeshData& aCopy)
    {
        fElements = aCopy.fElements;
    }
    virtual ~KGDiscreteRotationalMeshData();

    void Clear();

    const KGDiscreteRotationalMeshElementVector* Elements() const;
    KGDiscreteRotationalMeshElementVector* Elements();

  private:
    KGDiscreteRotationalMeshElementVector fElements;
};

class KGDiscreteRotationalMesh
{
  public:
    typedef KGDiscreteRotationalMeshData Surface;
    typedef KGDiscreteRotationalMeshData Space;
};

typedef KGExtendedSurface<KGDiscreteRotationalMesh> KGDiscreteRotationalMeshSurface;
typedef KGExtendedSpace<KGDiscreteRotationalMesh> KGDiscreteRotationalMeshSpace;

}  // namespace KGeoBag

#endif
