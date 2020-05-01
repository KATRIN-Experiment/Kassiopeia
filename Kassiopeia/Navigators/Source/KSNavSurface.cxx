#include "KSNavSurface.h"

#include "KSNavigatorsMessage.h"

namespace Kassiopeia
{

KSNavSurface::KSNavSurface() : fTransmissionSplit(false), fReflectionSplit(false) {}
KSNavSurface::KSNavSurface(const KSNavSurface& aCopy) :
    KSComponent(),
    fTransmissionSplit(aCopy.fTransmissionSplit),
    fReflectionSplit(aCopy.fReflectionSplit)
{}
KSNavSurface* KSNavSurface::Clone() const
{
    return new KSNavSurface(*this);
}
KSNavSurface::~KSNavSurface() {}

void KSNavSurface::SetTransmissionSplit(const bool& aTransmissionSplit)
{
    fTransmissionSplit = aTransmissionSplit;
    return;
}
const bool& KSNavSurface::GetTransmissionSplit() const
{
    return fTransmissionSplit;
}

void KSNavSurface::SetReflectionSplit(const bool& aReflectionSplit)
{
    fReflectionSplit = aReflectionSplit;
    return;
}
const bool& KSNavSurface::GetReflectionSplit() const
{
    return fReflectionSplit;
}

void KSNavSurface::ExecuteNavigation(const KSParticle& anInitialParticle, const KSParticle& aNavigationParticle,
                                     KSParticle& aFinalParticle, KSParticleQueue& aParticleQueue) const
{
    navmsg_debug("navigation surface <" << this->GetName() << "> executing navigation:" << eom);

    KSSide* tCurrentSide = aNavigationParticle.GetCurrentSide();
    KSSurface* tCurrentSurface = aNavigationParticle.GetCurrentSurface();
    KSSpace* tCurrentSpace = aNavigationParticle.GetCurrentSpace();

    if (tCurrentSurface != nullptr) {
        KThreeVector tInitialMomentum = anInitialParticle.GetMomentum();
        KThreeVector tFinalMomentum = aNavigationParticle.GetMomentum();
        KThreeVector tNormal = tCurrentSurface->Normal(aNavigationParticle.GetPosition());

        //check if momentum changed its sign relative to the normal of the surface (by the surface interaction)
        if ((tInitialMomentum.Dot(tNormal) > 0.) == (tFinalMomentum.Dot(tNormal) > 0.)) {
            navmsg(eNormal) << "  transmission occurred on child surface <" << tCurrentSurface->GetName() << ">" << eom;

            aFinalParticle = aNavigationParticle;
            aFinalParticle.SetLabel(GetName());
            aFinalParticle.AddLabel(tCurrentSurface->GetName());
            aFinalParticle.AddLabel("transmission");

            if (fTransmissionSplit == true) {
                auto* tTransmissionSplitParticle = new KSParticle(aFinalParticle);
                tTransmissionSplitParticle->SetCurrentSurface(nullptr);
                tTransmissionSplitParticle->SetLastStepSurface(tCurrentSurface);
                tTransmissionSplitParticle->ResetFieldCaching();
                aParticleQueue.push_back(tTransmissionSplitParticle);
                aFinalParticle.SetActive(false);
            }

            return;
        }
        else {
            navmsg(eNormal) << "  reflection occurred on child surface <" << tCurrentSurface->GetName() << ">" << eom;

            aFinalParticle = aNavigationParticle;
            aFinalParticle.SetLabel(GetName());
            aFinalParticle.AddLabel(tCurrentSurface->GetName());
            aFinalParticle.AddLabel("reflection");

            if (fReflectionSplit == true) {
                auto* tTransmissionSplitParticle = new KSParticle(aFinalParticle);
                tTransmissionSplitParticle->SetCurrentSurface(nullptr);
                tTransmissionSplitParticle->SetLastStepSurface(tCurrentSurface);
                tTransmissionSplitParticle->ResetFieldCaching();
                aParticleQueue.push_back(tTransmissionSplitParticle);
                aFinalParticle.SetActive(false);
            }
            return;
        }
    }

    KThreeVector tMomentum = aNavigationParticle.GetMomentum();
    KThreeVector tNormal = tCurrentSide->Normal(aNavigationParticle.GetPosition());

    if (tCurrentSpace == tCurrentSide->GetInsideParent()) {
        if (tMomentum.Dot(tNormal) > 0.) {
            navmsg(eNormal) << "  transmission occurred on boundary <" << tCurrentSide->GetName()
                            << "> of parent space <" << tCurrentSide->GetInsideParent()->GetName() << ">" << eom;

            aFinalParticle = aNavigationParticle;
            aFinalParticle.SetLabel(GetName());
            aFinalParticle.AddLabel(tCurrentSide->GetName());
            aFinalParticle.AddLabel("transmission");
            aFinalParticle.AddLabel("outbound");

            if (fTransmissionSplit == true) {
                auto* tTransmissionSplitParticle = new KSParticle(aFinalParticle);
                tTransmissionSplitParticle->SetCurrentSide(nullptr);
                tTransmissionSplitParticle->SetCurrentSpace(tCurrentSide->GetOutsideParent());
                tTransmissionSplitParticle->ResetFieldCaching();
                aParticleQueue.push_back(tTransmissionSplitParticle);
                aFinalParticle.SetActive(false);
            }

            return;
        }
        else {
            navmsg(eNormal) << "  reflection occurred on boundary <" << tCurrentSide->GetName() << "> of parent space <"
                            << tCurrentSide->GetInsideParent()->GetName() << ">" << eom;

            aFinalParticle = aNavigationParticle;
            aFinalParticle.SetLabel(GetName());
            aFinalParticle.AddLabel(tCurrentSide->GetName());
            aFinalParticle.AddLabel("reflection");
            aFinalParticle.AddLabel("outbound");

            if (fReflectionSplit == true) {
                auto* tReflectionSplitParticle = new KSParticle(aFinalParticle);
                tReflectionSplitParticle->SetCurrentSide(nullptr);
                tReflectionSplitParticle->ResetFieldCaching();
                aParticleQueue.push_back(tReflectionSplitParticle);
                aFinalParticle.SetActive(false);
            }

            return;
        }
    }

    if (tCurrentSpace == tCurrentSide->GetOutsideParent()) {
        if (tMomentum.Dot(tNormal) < 0.) {
            navmsg(eNormal) << "  transmission occurred on boundary <" << tCurrentSide->GetName()
                            << "> of child space <" << tCurrentSide->GetInsideParent()->GetName() << ">" << eom;

            aFinalParticle = aNavigationParticle;
            aFinalParticle.SetLabel(GetName());
            aFinalParticle.AddLabel(tCurrentSide->GetName());
            aFinalParticle.AddLabel("transmission");
            aFinalParticle.AddLabel("inbound");

            if (fTransmissionSplit == true) {
                auto* tTransmissionSplitParticle = new KSParticle(aFinalParticle);
                tTransmissionSplitParticle->SetCurrentSide(nullptr);
                tTransmissionSplitParticle->SetCurrentSpace(tCurrentSide->GetInsideParent());
                tTransmissionSplitParticle->ResetFieldCaching();
                aParticleQueue.push_back(tTransmissionSplitParticle);
                aFinalParticle.SetActive(false);
            }

            return;
        }
        else {
            navmsg(eNormal) << "  reflection occurred on boundary <" << tCurrentSide->GetName() << "> of child space <"
                            << tCurrentSide->GetInsideParent()->GetName() << ">" << eom;

            aFinalParticle = aNavigationParticle;
            aFinalParticle.SetLabel(GetName());
            aFinalParticle.AddLabel(tCurrentSide->GetName());
            aFinalParticle.AddLabel("reflection");
            aFinalParticle.AddLabel("inbound");

            if (fReflectionSplit == true) {
                auto* tReflectionSplitParticle = new KSParticle(aFinalParticle);
                tReflectionSplitParticle->SetCurrentSide(nullptr);
                tReflectionSplitParticle->ResetFieldCaching();
                aParticleQueue.push_back(tReflectionSplitParticle);
                aFinalParticle.SetActive(false);
            }

            return;
        }
    }

    navmsg(eError) << "could not determine surface navigation" << eom;
    return;
}


void KSNavSurface::FinalizeNavigation(KSParticle& aFinalParticle) const
{
    navmsg_debug("navigation surface <" << this->GetName() << "> finalizing navigation:" << eom);

    KSSide* tCurrentSide = aFinalParticle.GetCurrentSide();
    KSSurface* tCurrentSurface = aFinalParticle.GetCurrentSurface();
    KSSpace* tCurrentSpace = aFinalParticle.GetCurrentSpace();

    if (tCurrentSurface != nullptr) {
        navmsg_debug("  finalizing child surface <" << tCurrentSurface->GetName() << ">" << eom);
        aFinalParticle.SetCurrentSurface(nullptr);
        aFinalParticle.ResetFieldCaching();
        tCurrentSurface->Off();
        return;
    }

    KThreeVector tMomentum = aFinalParticle.GetMomentum();
    KThreeVector tNormal = tCurrentSide->Normal(aFinalParticle.GetPosition());

    if (tCurrentSpace == tCurrentSide->GetInsideParent()) {
        if (tMomentum.Dot(tNormal) > 0.) {
            navmsg_debug("  finalizing transmission on boundary <" << tCurrentSide->GetName() << "> of parent space <"
                                                                   << tCurrentSide->GetInsideParent()->GetName() << ">"
                                                                   << eom);

            aFinalParticle.SetCurrentSide(nullptr);
            aFinalParticle.SetCurrentSpace(tCurrentSide->GetOutsideParent());
            aFinalParticle.ResetFieldCaching();
            tCurrentSide->Off();
            tCurrentSide->GetInsideParent()->Exit();
            return;
        }
        else {
            navmsg_debug("  finalizing reflection on boundary <" << tCurrentSide->GetName() << "> of parent space <"
                                                                 << tCurrentSide->GetInsideParent()->GetName() << ">"
                                                                 << eom);

            aFinalParticle.SetCurrentSide(nullptr);
            aFinalParticle.ResetFieldCaching();
            tCurrentSide->Off();
            return;
        }
    }

    if (tCurrentSpace == tCurrentSide->GetOutsideParent()) {
        if (tMomentum.Dot(tNormal) < 0.) {
            navmsg_debug("  finalizing transmission on boundary <" << tCurrentSide->GetName() << "> of child space <"
                                                                   << tCurrentSide->GetInsideParent()->GetName() << ">"
                                                                   << eom);

            aFinalParticle.SetCurrentSide(nullptr);
            aFinalParticle.SetCurrentSpace(tCurrentSide->GetInsideParent());
            aFinalParticle.ResetFieldCaching();
            tCurrentSide->Off();
            tCurrentSide->GetInsideParent()->Enter();
            return;
        }
        else {
            navmsg_debug("  finalizing reflection on boundary <" << tCurrentSide->GetName() << "> of child space <"
                                                                 << tCurrentSide->GetInsideParent()->GetName() << ">"
                                                                 << eom);

            aFinalParticle.SetCurrentSide(nullptr);
            aFinalParticle.ResetFieldCaching();
            tCurrentSide->Off();
            return;
        }
    }

    navmsg(eError) << "could not determine surface navigation" << eom;
    return;
}
}  // namespace Kassiopeia
