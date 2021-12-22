/*
 * KSGenPositionSurfaceRandom.cxx
 *
 *  Created on: 17.09.2014
 *      Author: J. Behrens
 */

#include "KSGenPositionSurfaceRandom.h"

using namespace std;

namespace Kassiopeia
{
KSGenPositionSurfaceRandom::KSGenPositionSurfaceRandom() = default;
KSGenPositionSurfaceRandom::KSGenPositionSurfaceRandom(const KSGenPositionSurfaceRandom& aCopy) :
    KSComponent(aCopy),
    fSurfaces(aCopy.fSurfaces)
{}

KSGenPositionSurfaceRandom* KSGenPositionSurfaceRandom::Clone() const
{
    return new KSGenPositionSurfaceRandom(*this);
}

KSGenPositionSurfaceRandom::~KSGenPositionSurfaceRandom() = default;

void KSGenPositionSurfaceRandom::AddSurface(KGeoBag::KGSurface* aSurface)
{
    fSurfaces.push_back(aSurface);
}

bool KSGenPositionSurfaceRandom::RemoveSurface(KGeoBag::KGSurface* aSurface)
{
    for (auto s = fSurfaces.begin(); s != fSurfaces.end(); ++s) {
        if ((*s) == aSurface) {
            fSurfaces.erase(s);
            return true;
        }
    }

    return false;
}

void KSGenPositionSurfaceRandom::Dice(KSParticleQueue* aPrimaries)
{
    for (auto& aPrimarie : *aPrimaries) {
        katrin::KThreeVector pos = random.Random(fSurfaces);
        genmsg_debug("surface random position generator <" << GetName() << "> diced position " << pos << eom);
        aPrimarie->SetPosition(pos);
    }
}

void KSGenPositionSurfaceRandom::InitializeComponent()
{
    if (fSurfaces.size() == 0) {
        genmsg(eError) << "trying to initialize surface random position generator <" << GetName()
                       << "> without any defined surfaces" << eom;
    }
    return;
}
void KSGenPositionSurfaceRandom::DeinitializeComponent() {}

}  // namespace Kassiopeia
