#include "KSRootSpace.h"

#include "KSException.h"
#include "KSGeometryMessage.h"

using namespace std;

namespace Kassiopeia
{

KSRootSpace::KSRootSpace() {}
KSRootSpace::KSRootSpace(const KSRootSpace&) : KSComponent() {}
KSRootSpace* KSRootSpace::Clone() const
{
    return new KSRootSpace(*this);
}
KSRootSpace::~KSRootSpace() {}

void KSRootSpace::Enter() const
{
    return;
}
void KSRootSpace::Exit() const
{
    return;
}

bool KSRootSpace::Outside(const KThreeVector&) const
{
    return false;
}
KThreeVector KSRootSpace::Point(const KThreeVector&) const
{
    return KThreeVector(0., 0., 1.e30);
}
KThreeVector KSRootSpace::Normal(const KThreeVector&) const
{
    return KThreeVector(0., 0., 1.);
}

void KSRootSpace::AddSpace(KSSpace* aSpace)
{
    this->KSSpace::AddSpace(aSpace);
    return;
}
void KSRootSpace::RemoveSpace(KSSpace* aSpace)
{
    this->KSSpace::RemoveSpace(aSpace);
    return;
}

void KSRootSpace::AddSurface(KSSurface* aSurface)
{
    this->KSSpace::AddSurface(aSurface);
    return;
}
void KSRootSpace::RemoveSurface(KSSurface* aSurface)
{
    this->KSSpace::RemoveSurface(aSurface);
    return;
}

void KSRootSpace::InitializeComponent()
{
    vector<KSSpace*>::iterator tGeoSpaceIt;
    for (tGeoSpaceIt = fSpaces.begin(); tGeoSpaceIt != fSpaces.end(); tGeoSpaceIt++) {
        (*tGeoSpaceIt)->Initialize();
    }
    vector<KSSurface*>::iterator tGeoSurfaceIt;
    for (tGeoSurfaceIt = fSurfaces.begin(); tGeoSurfaceIt != fSurfaces.end(); tGeoSurfaceIt++) {
        (*tGeoSurfaceIt)->Initialize();
    }
    vector<KSSide*>::iterator tGeoSideIt;
    for (tGeoSideIt = fSides.begin(); tGeoSideIt != fSides.end(); tGeoSideIt++) {
        (*tGeoSideIt)->Initialize();
    }

    return;
}
void KSRootSpace::DeinitializeComponent()
{
    vector<KSSpace*>::iterator tGeoSpaceIt;
    for (tGeoSpaceIt = fSpaces.begin(); tGeoSpaceIt != fSpaces.end(); tGeoSpaceIt++) {
        (*tGeoSpaceIt)->Deinitialize();
    }
    vector<KSSurface*>::iterator tGeoSurfaceIt;
    for (tGeoSurfaceIt = fSurfaces.begin(); tGeoSurfaceIt != fSurfaces.end(); tGeoSurfaceIt++) {
        (*tGeoSurfaceIt)->Deinitialize();
    }
    vector<KSSide*>::iterator tGeoSideIt;
    for (tGeoSideIt = fSides.begin(); tGeoSideIt != fSides.end(); tGeoSideIt++) {
        (*tGeoSideIt)->Deinitialize();
    }

    return;
}

STATICINT sKSRootSpaceDict =
    KSDictionary<KSRootSpace>::AddCommand(&KSRootSpace::AddSpace, &KSRootSpace::RemoveSpace, "add_space",
                                          "remove_space") +
    KSDictionary<KSRootSpace>::AddCommand(&KSRootSpace::AddSurface, &KSRootSpace::RemoveSurface, "add_surface",
                                          "remove_surface");

}  // namespace Kassiopeia
