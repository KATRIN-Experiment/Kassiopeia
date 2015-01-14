#include "KGAxialMesh.hh"

namespace KGeoBag
{
    KGAxialMeshData::~KGAxialMeshData()
    {
        Clear();
    }

    void KGAxialMeshData::Clear()
    {
        KGAxialMeshElementIt it = fAxialMeshElements.begin();
        while( it != fAxialMeshElements.end() )
        {
            if( *it )
                delete (*it);
            ++it;
        }
        fAxialMeshElements.clear();
    }

    const KGAxialMeshElementVector* KGAxialMeshData::Elements() const
    {
        return &fAxialMeshElements;
    }
    KGAxialMeshElementVector* KGAxialMeshData::Elements()
    {
        return &fAxialMeshElements;
    }

}
