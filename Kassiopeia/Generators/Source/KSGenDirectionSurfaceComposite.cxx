/*
 * KSGenDirectionSurfaceComposite.cxx
 *
 *  Created on: 17.09.2014
 *      Author: J. Behrens
 */

#include "KSGenDirectionSurfaceComposite.h"

#include "KSGeneratorsMessage.h"
#include "KSNumerical.h"
#include "KTransformation.hh"

using namespace std;
using KGeoBag::KThreeVector;

namespace Kassiopeia
{

KSGenDirectionSurfaceComposite::KSGenDirectionSurfaceComposite() :
    fThetaValue(nullptr),
    fPhiValue(nullptr),
    fOutside(false)
{}
KSGenDirectionSurfaceComposite::KSGenDirectionSurfaceComposite(const KSGenDirectionSurfaceComposite& aCopy) :
    KSComponent(aCopy),
    fSurfaces(aCopy.fSurfaces),
    fThetaValue(aCopy.fThetaValue),
    fPhiValue(aCopy.fPhiValue),
    fOutside(aCopy.fOutside)
{}
KSGenDirectionSurfaceComposite* KSGenDirectionSurfaceComposite::Clone() const
{
    return new KSGenDirectionSurfaceComposite(*this);
}
KSGenDirectionSurfaceComposite::~KSGenDirectionSurfaceComposite() = default;

void KSGenDirectionSurfaceComposite::Dice(KSParticleQueue* aPrimaries)
{
    if (!fThetaValue || !fPhiValue)
        genmsg(eError) << "theta or phi value undefined in surface composite direction creator <" << this->GetName()
                       << ">" << eom;

    KThreeVector tMomentum;

    KSParticle* tParticle;
    KSParticleIt tParticleIt;
    KSParticleQueue tParticles;

    KGeoBag::KGSurface* tSurface;
    std::vector<KGeoBag::KGSurface*>::iterator tSurfaceIt;

    double tSmallestSurfaceDistance;
    KThreeVector tSurfacePoint;
    KThreeVector tSurfaceNormal;

    double tThetaValue;
    vector<double> tThetaValues;
    vector<double>::iterator tThetaValueIt;

    double tPhiValue;
    vector<double> tPhiValues;
    vector<double>::iterator tPhiValueIt;

    fThetaValue->DiceValue(tThetaValues);
    fPhiValue->DiceValue(tPhiValues);

    for (tThetaValueIt = tThetaValues.begin(); tThetaValueIt != tThetaValues.end(); tThetaValueIt++) {
        tThetaValue = (katrin::KConst::Pi() / 180.) * (*tThetaValueIt);
        for (tPhiValueIt = tPhiValues.begin(); tPhiValueIt != tPhiValues.end(); tPhiValueIt++) {
            tPhiValue = (katrin::KConst::Pi() / 180.) * (*tPhiValueIt);
            for (tParticleIt = aPrimaries->begin(); tParticleIt != aPrimaries->end(); tParticleIt++) {
                tParticle = new KSParticle(**tParticleIt);
                tSurface = fSurfaces.front();

                // find surface which is closest to the particle
                tSmallestSurfaceDistance = KSNumerical<double>::Maximum();
                for (tSurfaceIt = fSurfaces.begin(); tSurfaceIt != fSurfaces.end(); tSurfaceIt++) {
                    tSurfacePoint =
                        (*tSurfaceIt)->Point(tParticle->GetPosition());  // nearest point on surface to given point
                    double tSurfaceDistance = (tParticle->GetPosition() - tSurfacePoint).Magnitude();
                    if (tSurfaceDistance < tSmallestSurfaceDistance) {
                        tSmallestSurfaceDistance = tSurfaceDistance;
                        tSurface = (*tSurfaceIt);
                    }
                }
                tSurfaceNormal = tSurface->Normal(tParticle->GetPosition());  // nearest normal to given point

                // rotate direction vector by phi and theta
                KThreeVector tOrthogonalOne = tSurfaceNormal.Unit().Orthogonal();
                KThreeVector tOrthogonalTwo = tSurfaceNormal.Unit().Cross(tOrthogonalOne);
                KThreeVector tFinalDirection = 1. * (sin(tThetaValue) * (cos(tPhiValue) * tOrthogonalOne.Unit() +
                                                                         sin(tPhiValue) * tOrthogonalTwo.Unit()) +
                                                     cos(tThetaValue) * tSurfaceNormal.Unit());

                // apply rotation to the momentum vector, mirror it for inward direction
                tMomentum = tParticle->GetMomentum().Magnitude() * (fOutside ? 1.0 : -1.0) * tFinalDirection.Unit();
                tParticle->SetMomentum(tMomentum);
                tParticles.push_back(tParticle);
            }
        }
    }

    for (tParticleIt = aPrimaries->begin(); tParticleIt != aPrimaries->end(); tParticleIt++) {
        tParticle = *tParticleIt;
        delete tParticle;
    }

    aPrimaries->assign(tParticles.begin(), tParticles.end());

    return;
}

void KSGenDirectionSurfaceComposite::AddSurface(KGeoBag::KGSurface* aSurface)
{
    fSurfaces.push_back(aSurface);
}

bool KSGenDirectionSurfaceComposite::RemoveSurface(KGeoBag::KGSurface* aSurface)
{
    for (auto s = fSurfaces.begin(); s != fSurfaces.end(); ++s) {
        if ((*s) == aSurface) {
            fSurfaces.erase(s);
            return true;
        }
    }
    return false;
}

void KSGenDirectionSurfaceComposite::SetThetaValue(KSGenValue* anThetaValue)
{
    if (fThetaValue == nullptr) {
        fThetaValue = anThetaValue;
        return;
    }
    genmsg(eError) << "cannot set theta value <" << anThetaValue->GetName()
                   << "> to surface composite direction creator <" << this->GetName() << ">" << eom;
    return;
}
void KSGenDirectionSurfaceComposite::ClearThetaValue(KSGenValue* anThetaValue)
{
    if (fThetaValue == anThetaValue) {
        fThetaValue = nullptr;
        return;
    }
    genmsg(eError) << "cannot clear theta value <" << anThetaValue->GetName()
                   << "> from surface composite direction creator <" << this->GetName() << ">" << eom;
    return;
}

void KSGenDirectionSurfaceComposite::SetPhiValue(KSGenValue* aPhiValue)
{
    if (fPhiValue == nullptr) {
        fPhiValue = aPhiValue;
        return;
    }
    genmsg(eError) << "cannot set phi value <" << aPhiValue->GetName() << "> to surface composite direction creator <"
                   << this->GetName() << ">" << eom;
    return;
}
void KSGenDirectionSurfaceComposite::ClearPhiValue(KSGenValue* anPhiValue)
{
    if (fPhiValue == anPhiValue) {
        fPhiValue = nullptr;
        return;
    }
    genmsg(eError) << "cannot clear phi value <" << anPhiValue->GetName()
                   << "> from surface composite direction creator <" << this->GetName() << ">" << eom;
    return;
}

void KSGenDirectionSurfaceComposite::SetSide(bool aSide)
{
    fOutside = aSide;
    return;
}

void KSGenDirectionSurfaceComposite::InitializeComponent()
{
    if (fSurfaces.size() == 0) {
        genmsg(eError) << "trying to initialize surface composite direction generator <" << GetName()
                       << "> without any defined surfaces" << eom;
    }

    if (fThetaValue != nullptr) {
        fThetaValue->Initialize();
    }
    if (fPhiValue != nullptr) {
        fPhiValue->Initialize();
    }
    return;
}
void KSGenDirectionSurfaceComposite::DeinitializeComponent()
{
    if (fThetaValue != nullptr) {
        fThetaValue->Deinitialize();
    }
    if (fPhiValue != nullptr) {
        fPhiValue->Deinitialize();
    }
    return;
}

STATICINT sKSGenDirectionSurfaceCompositeDict =
    KSDictionary<KSGenDirectionSurfaceComposite>::AddCommand(&KSGenDirectionSurfaceComposite::SetThetaValue,
                                                             &KSGenDirectionSurfaceComposite::ClearThetaValue,
                                                             "set_theta", "clear_theta") +
    KSDictionary<KSGenDirectionSurfaceComposite>::AddCommand(&KSGenDirectionSurfaceComposite::SetPhiValue,
                                                             &KSGenDirectionSurfaceComposite::ClearPhiValue, "set_phi",
                                                             "clear_phi");

}  // namespace Kassiopeia
