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

    void KGDiscreteRotationalMesherBase::VisitExtendedSurface( KGExtendedSurface< KGDiscreteRotationalMesh >* aSurface )
    {
        fCurrentElements = aSurface->Elements();
        return;
    }

    void KGDiscreteRotationalMesherBase::VisitExtendedSpace( KGExtendedSpace< KGDiscreteRotationalMesh >* aSpace )
    {
        fCurrentElements = aSpace->Elements();
        return;
    }

}
