#include "KGMesherBase.hh"

namespace KGeoBag
{

    KGMesherBase::KGMesherBase() :
        fCurrentElements( NULL ),
        fCurrentSurface( NULL),
        fCurrentSpace( NULL )
    {
    }
    KGMesherBase::~KGMesherBase()
    {
    }

    void KGMesherBase::VisitExtendedSurface( KGExtendedSurface< KGMesh >* aSurface )
    {
        fCurrentElements = aSurface->Elements();
        fCurrentSurface = aSurface;
        fCurrentSpace = NULL;
        return;
    }
    void KGMesherBase::VisitExtendedSpace( KGExtendedSpace< KGMesh >* aSpace )
    {
        fCurrentElements = aSpace->Elements();
        fCurrentSurface = NULL;
        fCurrentSpace = aSpace;
        return;
    }

}
