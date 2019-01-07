#include "KGDiscreteRotationalMesherBase.hh"

namespace KGeoBag
{

    KGDiscreteRotationalMesherBase::KGDiscreteRotationalMesherBase() :
            fCurrentElements( NULL )
    {
    }
    KGDiscreteRotationalMesherBase::~KGDiscreteRotationalMesherBase()
    {
    }

    void KGDiscreteRotationalMesherBase::SetMeshElementOutput( KGDiscreteRotationalMeshElementVector* aMeshElementVector)

    {
        fCurrentElements = aMeshElementVector;
        return;
    }

}
