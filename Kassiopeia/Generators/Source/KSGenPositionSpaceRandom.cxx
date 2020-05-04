/*
 * KSGenPositionSpaceRandom.cxx
 *
 *  Created on: 21.05.2014
 *      Author: user
 */

#include "KSGenPositionSpaceRandom.h"

namespace Kassiopeia
{
KSGenPositionSpaceRandom::KSGenPositionSpaceRandom() {}
KSGenPositionSpaceRandom::KSGenPositionSpaceRandom(const KSGenPositionSpaceRandom& aCopy) :
    KSComponent(),
    fSpaces(aCopy.fSpaces)
{}

KSGenPositionSpaceRandom* KSGenPositionSpaceRandom::Clone() const
{
    return new KSGenPositionSpaceRandom(*this);
}

KSGenPositionSpaceRandom::~KSGenPositionSpaceRandom() {}

void KSGenPositionSpaceRandom::InitializeComponent() {}
void KSGenPositionSpaceRandom::DeinitializeComponent() {}

void KSGenPositionSpaceRandom::AddSpace(KGeoBag::KGSpace* aSpace)
{
    fSpaces.push_back(aSpace);
}

bool KSGenPositionSpaceRandom::RemoveSpace(KGeoBag::KGSpace* aSpace)
{
    for (auto s = fSpaces.begin(); s != fSpaces.end(); ++s) {
        if ((*s) == aSpace) {
            fSpaces.erase(s);
            return true;
        }
    }

    return false;
}

void KSGenPositionSpaceRandom::Dice(KSParticleQueue* aPrimaries)
{
    for (auto p = aPrimaries->begin(); p != aPrimaries->end(); ++p) {
        (*p)->SetPosition(random.Random(fSpaces));
    }
}
}  // namespace Kassiopeia
