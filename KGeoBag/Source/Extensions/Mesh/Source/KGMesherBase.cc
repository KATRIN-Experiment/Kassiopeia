#include "KGMesherBase.hh"

namespace KGeoBag
{

KGMesherBase::KGMesherBase() : fCurrentElements(nullptr), fCurrentSurface(nullptr), fCurrentSpace(nullptr) {}
KGMesherBase::~KGMesherBase() = default;

void KGMesherBase::VisitExtendedSurface(KGExtendedSurface<KGMesh>* aSurface)
{
    fCurrentElements = aSurface->Elements();
    fCurrentSurface = aSurface;
    fCurrentSpace = nullptr;
    return;
}
void KGMesherBase::VisitExtendedSpace(KGExtendedSpace<KGMesh>* aSpace)
{
    fCurrentElements = aSpace->Elements();
    fCurrentSurface = nullptr;
    fCurrentSpace = aSpace;
    return;
}

}  // namespace KGeoBag
