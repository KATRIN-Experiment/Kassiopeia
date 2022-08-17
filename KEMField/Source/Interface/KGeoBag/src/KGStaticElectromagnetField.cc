/*
 * KGStaticElectromagnetField.cc
 *
 *  Created on: 25 Mar 2016
 *      Author: wolfgang
 */

#include "KGStaticElectromagnetField.hh"

#include "KEMCoreMessage.hh"

using namespace KGeoBag;
using namespace std;

namespace KEMField
{

KGStaticElectromagnetField::KGStaticElectromagnetField() :
    fSystem(nullptr),
    fConverter(nullptr),
    fSaveMagfield3(true),
    fDirectoryMagfield3(SCRATCH_DEFAULT_DIR)
{}

KGStaticElectromagnetField::~KGStaticElectromagnetField() = default;

void KGStaticElectromagnetField::SetSystem(KGeoBag::KGSpace* aSystem)
{
    fSystem = aSystem;
}

void KGStaticElectromagnetField::AddSurface(KGeoBag::KGSurface* aSurface)
{
    fSurfaces.push_back(aSurface);
}

void KGStaticElectromagnetField::AddSpace(KGeoBag::KGSpace* aSpace)
{
    fSpaces.push_back(aSpace);
}

void KGStaticElectromagnetField::SetSaveMagfield3(bool aFlag)
{
    fSaveMagfield3 = aFlag;
}

void KGStaticElectromagnetField::SetDirectoryMagfield3(const string& aDirectory)
{
    fDirectoryMagfield3 = aDirectory;
}

KSmartPointer<KGeoBag::KGElectromagnetConverter> KGStaticElectromagnetField::GetConverter()
{
    return fConverter;
}

void KGStaticElectromagnetField::InitializeCore()
{
    CheckSolverExistance();
    ConfigureSurfaceContainer();
    KStaticElectromagnetField::InitializeCore();
}

KFieldVector KGStaticElectromagnetField::MagneticPotentialCore(const KPosition& aSamplePoint) const
{
    KPosition internal = fConverter->GlobalToInternalPosition(aSamplePoint);
    KDirection internalPotential = KStaticElectromagnetField::MagneticPotentialCore(internal);
    return fConverter->InternalToGlobalVector(internalPotential);
}

KFieldVector KGStaticElectromagnetField::MagneticFieldCore(const KPosition& aSamplePoint) const
{
    KPosition internal = fConverter->GlobalToInternalPosition(aSamplePoint);
    KDirection internalField = KStaticElectromagnetField::MagneticFieldCore(internal);
    return fConverter->InternalToGlobalVector(internalField);
}

KGradient KGStaticElectromagnetField::MagneticGradientCore(const KPosition& aSamplePoint) const
{
    KPosition internal = fConverter->GlobalToInternalPosition(aSamplePoint);
    KGradient internalGradient = KStaticElectromagnetField::MagneticGradientCore(internal);
    return fConverter->InternalTensorToGlobal(internalGradient);
}

void KGStaticElectromagnetField::ConfigureSurfaceContainer()
{
    auto* container = new KElectromagnetContainer();

    fConverter = new KGElectromagnetConverter();
    fConverter->SetElectromagnetContainer(container);

    if (fSaveMagfield3) {
        string tFileName = GetName() + string(".mag3");
        GetConverter()->SetDumpMagfield3ToFile(fDirectoryMagfield3, tFileName);
    }

    if (fSystem != nullptr) {
        fConverter->SetSystem(fSystem->GetOrigin(), fSystem->GetXAxis(), fSystem->GetYAxis(), fSystem->GetZAxis());
    }

    for (auto& surface : fSurfaces) {
        surface->AcceptNode(&(*fConverter));
    }

    for (auto& space : fSpaces) {
        space->AcceptNode(&(*fConverter));
    }

    SetContainer(container);
}

} /* namespace KEMField */
