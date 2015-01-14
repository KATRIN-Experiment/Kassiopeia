#include "KGAxialMesherBase.hh"

namespace KGeoBag
{

    KGAxialMesherBase::KGAxialMesherBase() :
            fCurrentElements( NULL )
    {
    }
    KGAxialMesherBase::~KGAxialMesherBase()
    {
    }

    void KGAxialMesherBase::VisitExtendedSurface( KGExtendedSurface< KGAxialMesh >* aSurface )
    {
        fCurrentElements = aSurface->Elements();
        return;
    }

    void KGAxialMesherBase::VisitExtendedSpace( KGExtendedSpace< KGAxialMesh >* aSpace )
    {
        fCurrentElements = aSpace->Elements();
        return;
    }

}
