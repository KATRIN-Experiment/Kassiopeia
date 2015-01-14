#include "KGMesh.hh"

namespace KGeoBag
{
    KGMeshData::~KGMeshData()
    {
        Clear();
    }

    void KGMeshData::Clear()
    {
        KGMeshElementIt it = fElements.begin();
        while( it != fElements.end() )
        {
            if( *it )
                delete (*it);
            ++it;
        }
        fElements.clear();
    }

    const KGMeshElementVector* KGMeshData::Elements() const
    {
        return &fElements;
    }
    KGMeshElementVector* KGMeshData::Elements()
    {
        return &fElements;
    }

}
