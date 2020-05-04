/*
 * KSGenPositionMask.h
 *
 *  Created on: 25.04.2015
 *      Author: Jan Behrens
 */

#include "KSGenPositionMask.h"


namespace Kassiopeia
{
KSGenPositionMask::KSGenPositionMask() : fGenerator(nullptr), fMaxRetries(10000) {}
KSGenPositionMask::KSGenPositionMask(const KSGenPositionMask& aCopy) :
    KSComponent(),
    fAllowedSpaces(aCopy.fAllowedSpaces),
    fForbiddenSpaces(aCopy.fForbiddenSpaces),
    fGenerator(aCopy.fGenerator),
    fMaxRetries(aCopy.fMaxRetries)
{}
KSGenPositionMask* KSGenPositionMask::Clone() const
{
    return new KSGenPositionMask(*this);
}
KSGenPositionMask::~KSGenPositionMask() {}

void KSGenPositionMask::Dice(KSParticleQueue* aPrimaries)
{
    auto* tTempQueue = new KSParticleQueue();
    tTempQueue->push_back(new KSParticle());

    for (auto tParticleIt = aPrimaries->begin(); tParticleIt != aPrimaries->end(); ++tParticleIt) {
        KThreeVector tPosition;
        bool tPositionValid = false;
        unsigned int tNumRetries = 0;

        // loop until valid position has been found
        while (!tPositionValid) {
            tNumRetries++;
            if (tNumRetries >= fMaxRetries) {
                genmsg(eError) << "failed to find valid position after " << tNumRetries
                               << " retries, error in generator configuration?" << eom;
            }

            fGenerator->Dice(tTempQueue);
            tPosition = tTempQueue->back()->GetPosition();
            tPositionValid = true;

            // check if position is not inside any of the forbidden spaces
            if (fForbiddenSpaces.size() > 0) {
                tPositionValid = true;
                for (auto tSpaceIt = fForbiddenSpaces.begin(); tSpaceIt != fForbiddenSpaces.end(); ++tSpaceIt) {
                    if ((*tSpaceIt)->Outside(tPosition) == false)  // inside forbidden space
                    {
                        tPositionValid = false;
                        break;
                    }
                }

                if (!tPositionValid) {
                    genmsg_debug("diced position "
                                 << tPosition << " is inside at least one of the forbidden spaces, will retry" << eom);
                    continue;
                }
            }

            // check if position is inside at least one of the allowed spaces
            if (fAllowedSpaces.size() > 0) {
                tPositionValid = false;
                for (auto tSpaceIt = fAllowedSpaces.begin(); tSpaceIt != fAllowedSpaces.end(); ++tSpaceIt) {
                    if ((*tSpaceIt)->Outside(tPosition) == false)  // inside allowed space
                    {
                        tPositionValid = true;
                        break;
                    }
                }

                if (!tPositionValid) {
                    genmsg_debug("diced position " << tPosition
                                                   << " is not inside any of the allowed spaces, will retry" << eom);
                    continue;
                }
            }
        }

        genmsg(eNormal) << "found valid position " << tPosition << " after " << tNumRetries
                        << (tNumRetries > 1 ? " tries" : " try") << eom;
        (*tParticleIt)->SetPosition(tPosition);
    }

    delete tTempQueue;
}

void KSGenPositionMask::AddAllowedSpace(const KGeoBag::KGSpace* aSpace)
{
    fAllowedSpaces.push_back(aSpace);
}

bool KSGenPositionMask::RemoveAllowedSpace(const KGeoBag::KGSpace* aSpace)
{
    for (auto tSpaceIt = fAllowedSpaces.begin(); tSpaceIt != fAllowedSpaces.end(); ++tSpaceIt) {
        if ((*tSpaceIt) == aSpace) {
            fAllowedSpaces.erase(tSpaceIt);
            return true;
        }
    }
    return false;
}

void KSGenPositionMask::AddForbiddenSpace(const KGeoBag::KGSpace* aSpace)
{
    fForbiddenSpaces.push_back(aSpace);
}

bool KSGenPositionMask::RemoveForbiddenSpace(const KGeoBag::KGSpace* aSpace)
{
    for (auto tSpaceIt = fForbiddenSpaces.begin(); tSpaceIt != fForbiddenSpaces.end(); ++tSpaceIt) {
        if ((*tSpaceIt) == aSpace) {
            fForbiddenSpaces.erase(tSpaceIt);
            return true;
        }
    }
    return false;
}

void KSGenPositionMask::SetGenerator(KSGenCreator* aGenerator)
{
    fGenerator = aGenerator;
}

void KSGenPositionMask::SetMaxRetries(unsigned int& aNumber)
{
    fMaxRetries = aNumber;
}

void KSGenPositionMask::InitializeComponent()
{
    if (fGenerator == nullptr)
        genmsg(eWarning) << "no generator defined to apply position mask" << eom;
    else
        fGenerator->Initialize();

    if (fForbiddenSpaces.size() + fAllowedSpaces.size() == 0)
        genmsg(eWarning) << "no forbidden or allowed spaces defined, position masking will not be effective" << eom;
}
void KSGenPositionMask::DeinitializeComponent() {}

}  // namespace Kassiopeia
