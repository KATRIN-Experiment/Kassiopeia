#include "KSSpace.h"

#include "KSSide.h"
#include "KSSurface.h"

using namespace std;

namespace Kassiopeia
{

KSSpace::KSSpace() : fParent(nullptr), fSpaces(), fSurfaces(), fSides() {}
KSSpace::~KSSpace() {}

const KSSpace* KSSpace::GetParent() const
{
    return fParent;
}
KSSpace* KSSpace::GetParent()
{
    return fParent;
}
void KSSpace::SetParent(KSSpace* aParent)
{
    for (auto tSpaceIt = aParent->fSpaces.begin(); tSpaceIt != aParent->fSpaces.end(); tSpaceIt++) {
        if ((*tSpaceIt) == this) {
            aParent->fSpaces.erase(tSpaceIt);
            break;
        }
    }

    aParent->fSpaces.push_back(this);

    this->fParent = aParent;

    for (auto tSideIt = this->fSides.begin(); tSideIt != this->fSides.end(); tSideIt++) {
        (*tSideIt)->fOutsideParent = aParent;
    }

    return;
}

int KSSpace::GetSpaceCount() const
{
    return fSpaces.size();
}
const KSSpace* KSSpace::GetSpace(int anIndex) const
{
    return fSpaces.at(anIndex);
}
KSSpace* KSSpace::GetSpace(int anIndex)
{
    return fSpaces.at(anIndex);
}
void KSSpace::AddSpace(KSSpace* aChild)
{
    for (auto tSpaceIt = this->fSpaces.begin(); tSpaceIt != this->fSpaces.end(); tSpaceIt++) {
        if ((*tSpaceIt) == aChild) {
            this->fSpaces.erase(tSpaceIt);
            break;
        }
    }

    this->fSpaces.push_back(aChild);

    aChild->fParent = this;

    //set outsideparent pointer of sides inside child spaces correctly
    for (int tChildSideIndex = 0; tChildSideIndex < aChild->GetSideCount(); tChildSideIndex++) {
        KSSide* tSide = aChild->GetSide(tChildSideIndex);
        tSide->fOutsideParent = this;
    }

    return;
}
void KSSpace::RemoveSpace(KSSpace* aChild)
{
    for (auto tSpaceIt = this->fSpaces.begin(); tSpaceIt != this->fSpaces.end(); tSpaceIt++) {
        if ((*tSpaceIt) == aChild) {
            this->fSpaces.erase(tSpaceIt);
            break;
        }
    }

    aChild->fParent = nullptr;

    return;
}

int KSSpace::GetSurfaceCount() const
{
    return fSurfaces.size();
}
const KSSurface* KSSpace::GetSurface(int anIndex) const
{
    return fSurfaces.at(anIndex);
}
KSSurface* KSSpace::GetSurface(int anIndex)
{
    return fSurfaces.at(anIndex);
}
void KSSpace::AddSurface(KSSurface* aChild)
{
    for (auto tSurfaceIt = this->fSurfaces.begin(); tSurfaceIt != this->fSurfaces.end(); tSurfaceIt++) {
        if ((*tSurfaceIt) == aChild) {
            this->fSurfaces.erase(tSurfaceIt);
            break;
        }
    }

    this->fSurfaces.push_back(aChild);

    aChild->fParent = this;

    return;
}
void KSSpace::RemoveSurface(KSSurface* aChild)
{
    for (auto tSurfaceIt = this->fSurfaces.begin(); tSurfaceIt != this->fSurfaces.end(); tSurfaceIt++) {
        if (*tSurfaceIt == aChild) {
            this->fSurfaces.erase(tSurfaceIt);
            break;
        }
    }
    aChild->fParent = nullptr;
    return;
}

int KSSpace::GetSideCount() const
{
    return fSides.size();
}
const KSSide* KSSpace::GetSide(int anIndex) const
{
    return fSides.at(anIndex);
}
KSSide* KSSpace::GetSide(int anIndex)
{
    return fSides.at(anIndex);
}
void KSSpace::AddSide(KSSide* aChild)
{
    for (auto tSideIt = this->fSides.begin(); tSideIt != this->fSides.end(); tSideIt++) {
        if ((*tSideIt) == aChild) {
            this->fSides.erase(tSideIt);
            break;
        }
    }

    this->fSides.push_back(aChild);

    aChild->fInsideParent = this;

    aChild->fOutsideParent = this->fParent;

    return;
}
void KSSpace::RemoveSide(KSSide* aChild)
{
    for (auto tSideIt = this->fSides.begin(); tSideIt != this->fSides.end(); tSideIt++) {
        if ((*tSideIt) == aChild) {
            this->fSides.erase(tSideIt);
            break;
        }
    }

    aChild->fInsideParent = nullptr;

    aChild->fOutsideParent = nullptr;

    return;
}

}  // namespace Kassiopeia
