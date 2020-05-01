#include "KGDiscreteRotationalMesh.hh"

namespace KGeoBag
{
KGDiscreteRotationalMeshData::~KGDiscreteRotationalMeshData()
{
    Clear();
}

void KGDiscreteRotationalMeshData::Clear()
{
    auto it = fElements.begin();
    while (it != fElements.end()) {
        if (*it)
            delete (*it);
        ++it;
    }
    fElements.clear();
}

const KGDiscreteRotationalMeshElementVector* KGDiscreteRotationalMeshData::Elements() const
{
    return &fElements;
}
KGDiscreteRotationalMeshElementVector* KGDiscreteRotationalMeshData::Elements()
{
    return &fElements;
}

}  // namespace KGeoBag
