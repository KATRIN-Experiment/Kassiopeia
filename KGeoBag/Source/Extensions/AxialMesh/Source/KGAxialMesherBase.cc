#include "KGAxialMesherBase.hh"

namespace KGeoBag
{

KGAxialMesherBase::KGAxialMesherBase() : fCurrentElements(nullptr) {}
KGAxialMesherBase::~KGAxialMesherBase() = default;

void KGAxialMesherBase::VisitExtendedSurface(KGExtendedSurface<KGAxialMesh>* aSurface)
{
    fCurrentElements = aSurface->Elements();
    return;
}

void KGAxialMesherBase::VisitExtendedSpace(KGExtendedSpace<KGAxialMesh>* aSpace)
{
    fCurrentElements = aSpace->Elements();
    return;
}

}  // namespace KGeoBag
