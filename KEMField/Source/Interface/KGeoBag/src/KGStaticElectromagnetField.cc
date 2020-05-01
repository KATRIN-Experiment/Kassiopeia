/*
 * KGStaticElectromagnetField.cc
 *
 *  Created on: 25 Mar 2016
 *      Author: wolfgang
 */

#include "KGStaticElectromagnetField.hh"

using namespace KGeoBag;
using namespace std;

namespace KEMField
{

KGStaticElectromagnetField::KGStaticElectromagnetField() :
    KStaticElectromagnetField(),
    fSystem(nullptr),
    fSurfaces(),
    fSpaces(),
    fConverter(nullptr)
{}

KGStaticElectromagnetField::~KGStaticElectromagnetField() {}

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

KThreeVector KGStaticElectromagnetField::MagneticPotentialCore(const KPosition& aSamplePoint) const
{
    KPosition internal = fConverter->GlobalToInternalPosition(aSamplePoint);
    KDirection internalPotential = KStaticElectromagnetField::MagneticPotentialCore(internal);
    return fConverter->InternalToGlobalVector(internalPotential);
}

KThreeVector KGStaticElectromagnetField::MagneticFieldCore(const KPosition& aSamplePoint) const
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

    if (fSystem != nullptr) {
        fConverter->SetSystem(fSystem->GetOrigin(), fSystem->GetXAxis(), fSystem->GetYAxis(), fSystem->GetZAxis());
    }

    for (auto tSurfaceIt = fSurfaces.begin(); tSurfaceIt != fSurfaces.end(); tSurfaceIt++) {
        (*tSurfaceIt)->AcceptNode(&(*fConverter));
    }

    for (auto tSpaceIt = fSpaces.begin(); tSpaceIt != fSpaces.end(); tSpaceIt++) {
        (*tSpaceIt)->AcceptNode(&(*fConverter));
    }

    if (container->empty()) {
        cout << "ERROR:"
             << "electromagnet field solver <" /*<< GetName()*/ << "> has zero surface or space elements" << endl;
        std::exit(-1);
    }

    SetContainer(container);
}

} /* namespace KEMField */
