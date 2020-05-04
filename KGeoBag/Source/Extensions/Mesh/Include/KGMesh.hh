#ifndef KGeoBag_KGMesh_hh_
#define KGeoBag_KGMesh_hh_

#include "KGCore.hh"
#include "KGMeshElement.hh"

namespace KGeoBag
{

class KGMeshData
{
  public:
    KGMeshData() {}
    KGMeshData(KGSpace*) {}
    KGMeshData(KGSurface*) {}
    KGMeshData(KGSpace*, const KGMeshData& aCopy)
    {
        fElements = aCopy.fElements;
    }
    KGMeshData(KGSurface*, const KGMeshData& aCopy)
    {
        fElements = aCopy.fElements;
    }
    virtual ~KGMeshData();

    void Clear();

    bool HasData() const
    {
        if (fElements.size() != 0) {
            return true;
        }
        return false;
    };

    const KGMeshElementVector* Elements() const;
    KGMeshElementVector* Elements();

  private:
    KGMeshElementVector fElements;
};

class KGMesh
{
  public:
    typedef KGMeshData Surface;
    typedef KGMeshData Space;
};

typedef KGExtendedSurface<KGMesh> KGMeshSurface;
typedef KGExtendedSpace<KGMesh> KGMeshSpace;

}  // namespace KGeoBag

#endif
