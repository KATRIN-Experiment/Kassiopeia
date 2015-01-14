#include "KGMesherBase.hh"

namespace KGeoBag
{

    KGMesherBase::KGMesherBase() :
        fCurrentElements( NULL )
    {
    }
    KGMesherBase::~KGMesherBase()
    {
    }

    void KGMesherBase::VisitExtendedSurface( KGExtendedSurface< KGMesh >* aSurface )
    {
        fCurrentElements = aSurface->Elements();
        return;
    }
    void KGMesherBase::VisitExtendedSpace( KGExtendedSpace< KGMesh >* aSpace )
    {
        fCurrentElements = aSpace->Elements();
        return;
    }

}
