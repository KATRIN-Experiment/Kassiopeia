#include "KSRootGenerator.h"

#include "KSException.h"
#include "KSGeneratorsMessage.h"
#include "KSNumerical.h"

namespace Kassiopeia
{

KSRootGenerator::KSRootGenerator() : fGenerator(nullptr), fEvent(nullptr) {}
KSRootGenerator::KSRootGenerator(const KSRootGenerator& aCopy) :
    KSComponent(aCopy),
    fGenerator(aCopy.fGenerator),
    fEvent(aCopy.fEvent)
{}
KSRootGenerator* KSRootGenerator::Clone() const
{
    return new KSRootGenerator(*this);
}
KSRootGenerator::~KSRootGenerator() = default;

void KSRootGenerator::SetGenerator(KSGenerator* aGenerator)
{
    if (fGenerator != nullptr) {
        genmsg(eError) << "tried to set generator <" << aGenerator->GetName() << "> with generator <"
                       << fGenerator->GetName() << "> already set" << eom;
        return;
    }
    genmsg_debug("setting root generator <" << aGenerator->GetName() << ">" << eom);
    fGenerator = aGenerator;
    return;
}
void KSRootGenerator::ClearGenerator(KSGenerator* aGenerator)
{
    if (fGenerator != aGenerator) {
        genmsg(eError) << "tried to remove generator <" << aGenerator->GetName() << "> with generator <"
                       << fGenerator->GetName() << "> already set" << eom;
        return;
    }
    genmsg_debug("clearing root generator" << eom);
    fGenerator = nullptr;
    return;
}

void KSRootGenerator::ExecuteGeneration(KSParticleQueue& anInitialStates)
{
    if (fGenerator == nullptr) {
        genmsg(eError) << "root generator cannot generate with no generator set" << eom;
        return;
    }
    try {
        fGenerator->ExecuteGeneration(anInitialStates);
    }
    catch (KSException const& e) {
        throw KSGeneratorError().Nest(e) << "Failed to execute generator <" << fGenerator->GetName() << ">";
    }
    return;
}

void KSRootGenerator::SetEvent(KSEvent* anEvent)
{
    fEvent = anEvent;
    return;
}
KSEvent* KSRootGenerator::GetEvent() const
{
    return fEvent;
}

void KSRootGenerator::ExecuteGeneration()
{
    ExecuteGeneration(fEvent->ParticleQueue());

    KGeoBag::KThreeVector tPosition;
    KGeoBag::KThreeVector tCenterPosition(0., 0., 0.);
    double tEnergy;
    double tTotalEnergy = 0;
    double tTime;
    double tMinTime = KSNumerical<double>::Maximum();
    double tMaxTime = -1. * KSNumerical<double>::Maximum();
    double tRadius;
    double tMaxRadius = 0.;

    KSParticleCIt tIt;
    KSParticle* tParticle;
    for (tIt = fEvent->ParticleQueue().begin(); tIt != fEvent->ParticleQueue().end(); tIt++) {
        tParticle = *tIt;

        if (!tParticle->IsValid()) {
            tParticle->Print();
            throw KSGeneratorError() << "Generator error: invalid particle state";
            continue;
        }

        tPosition = tParticle->GetPosition();
        tCenterPosition += tPosition;

        tEnergy = tParticle->GetKineticEnergy_eV();
        tTotalEnergy += tEnergy;

        tTime = tParticle->GetTime();
        if (tTime < tMinTime) {
            tMinTime = tTime;
        }
        if (tTime > tMaxTime) {
            tMaxTime = tTime;
        }
    }

    tCenterPosition = (1. / (double) (fEvent->ParticleQueue().size())) * tCenterPosition;

    for (tIt = fEvent->ParticleQueue().begin(); tIt != fEvent->ParticleQueue().end(); tIt++) {
        tParticle = *tIt;
        tRadius = (tParticle->GetPosition() - tCenterPosition).Magnitude();
        if (tRadius > tMaxRadius) {
            tMaxRadius = tRadius;
        }
    }

    fEvent->GeneratorFlag() = true;
    fEvent->GeneratorName() = fGenerator->GetName();
    fEvent->GeneratorPrimaries() = fEvent->ParticleQueue().size();
    fEvent->GeneratorEnergy() = tTotalEnergy;
    fEvent->GeneratorMinTime() = tMinTime;
    fEvent->GeneratorMaxTime() = tMaxTime;
    fEvent->GeneratorLocation() = tCenterPosition;
    fEvent->GeneratorRadius() = tMaxRadius;

    genmsg_debug("event: " << ret);
    genmsg_debug("  generator name: <" << fEvent->GetGeneratorName() << ">" << ret);
    genmsg_debug("  generator primaries: <" << fEvent->GetGeneratorPrimaries() << ">" << ret);
    genmsg_debug("  generator total energy: <" << fEvent->GetGeneratorEnergy() << ">" << ret);
    genmsg_debug("  generator min time: <" << fEvent->GetGeneratorMinTime() << ">" << ret);
    genmsg_debug("  generator max time: <" << fEvent->GetGeneratorMaxTime() << ">" << ret);
    genmsg_debug("  generator location: <" << fEvent->GetGeneratorLocation().X() << ", "
                                           << fEvent->GetGeneratorLocation().Y() << ", "
                                           << fEvent->GetGeneratorLocation().Z() << ">" << ret);
    genmsg_debug("  generator radius: <" << fEvent->GetGeneratorRadius() << ">" << eom);

    return;
}

STATICINT sKSRootGeneratorDict = KSDictionary<KSRootGenerator>::AddCommand(
    &KSRootGenerator::SetGenerator, &KSRootGenerator::ClearGenerator, "set_generator", "clear_generator");

}  // namespace Kassiopeia
