#ifndef KGeoBag_KGDiscreteRotationalMesh_hh_
#define KGeoBag_KGDiscreteRotationalMesh_hh_

#include "KGCore.hh"
#include "KGDiscreteRotationalMeshElement.hh"

namespace KGeoBag
{

class KGDiscreteRotationalMeshData
{
  public:
    KGDiscreteRotationalMeshData() = default;
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
    using Space = KGDiscreteRotationalMeshData;
};

using KGDiscreteRotationalMeshSurface = KGExtendedSurface<KGDiscreteRotationalMesh>;
using KGDiscreteRotationalMeshSpace = KGExtendedSpace<KGDiscreteRotationalMesh>;

}  // namespace KGeoBag

#endif
