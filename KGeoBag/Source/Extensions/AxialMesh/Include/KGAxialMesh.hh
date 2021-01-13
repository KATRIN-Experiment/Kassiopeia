#ifndef KGeoBag_KGAxialMesh_hh_
#define KGeoBag_KGAxialMesh_hh_

#include "KGAxialMeshElement.hh"
#include "KGCore.hh"

namespace KGeoBag
{

class KGAxialMeshData
{
  public:
    KGAxialMeshData() = default;
    KGAxialMeshData(KGSurface*) {}
    KGAxialMeshData(KGSpace*) {}
    KGAxialMeshData(KGSurface*, const KGAxialMeshData& aCopy)
    {
        fAxialMeshElements = aCopy.fAxialMeshElements;
    }
    KGAxialMeshData(KGSpace*, const KGAxialMeshData& aCopy)
    {
        fAxialMeshElements = aCopy.fAxialMeshElements;
    }
    virtual ~KGAxialMeshData();

    void Clear();

    const KGAxialMeshElementVector* Elements() const;
    KGAxialMeshElementVector* Elements();

  private:
    KGAxialMeshElementVector fAxialMeshElements;
};

class KGAxialMesh
{
  public:
    typedef KGAxialMeshData Surface;
    using Space = KGAxialMeshData;
};

using KGAxialMeshSurface = KGExtendedSurface<KGAxialMesh>;
using KGAxialMeshSpace = KGExtendedSpace<KGAxialMesh>;

}  // namespace KGeoBag

#endif
