#include "KSSurface.h"

#include "KSSpace.h"

using namespace std;

namespace Kassiopeia
{

KSSurface::KSSurface() : fParent(nullptr) {}
KSSurface::~KSSurface() = default;

const KSSpace* KSSurface::GetParent() const
{
    return fParent;
}
KSSpace* KSSurface::GetParent()
{
    return fParent;
}
void KSSurface::SetParent(KSSpace* aParent)
{
    for (auto tSurfaceIt = aParent->fSurfaces.begin(); tSurfaceIt != aParent->fSurfaces.end(); tSurfaceIt++) {
        if ((*tSurfaceIt) == this) {
            aParent->fSurfaces.erase(tSurfaceIt);
            break;
        }
    }

    aParent->fSurfaces.push_back(this);

    this->fParent = aParent;

    return;
}

}  // namespace Kassiopeia
