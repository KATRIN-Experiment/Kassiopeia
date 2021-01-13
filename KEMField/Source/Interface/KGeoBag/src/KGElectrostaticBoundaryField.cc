/*
 * KGElectrostaticBoundaryField.cc
 *
 *  Created on: 15 Jun 2015
 *      Author: wolfgang
 */

#include "KGElectrostaticBoundaryField.hh"

#include "KEMCoreMessage.hh"
#include "KFile.h"
#include "KSADataStreamer.hh"

using namespace KGeoBag;

namespace KEMField
{

KGElectrostaticBoundaryField::KGElectrostaticBoundaryField() :
    fMinimumElementArea(0.0),
    fMaximumElementAspectRatio(1e100),
    fSystem(nullptr),
    fSymmetry(NoSymmetry),
    fConverter(nullptr)
{}

KGElectrostaticBoundaryField::~KGElectrostaticBoundaryField() = default;

double KGElectrostaticBoundaryField::PotentialCore(const KPosition& P) const
{
    KPosition internal = fConverter->GlobalToInternalPosition(P);
    double aPotential = KElectrostaticBoundaryField::PotentialCore(internal);
    //bindingmsg_debug( "potential at " << P << " is " << aPotential <<eom);
    return aPotential;
}

KFieldVector KGElectrostaticBoundaryField::ElectricFieldCore(const KPosition& P) const
{
    KPosition internal = fConverter->GlobalToInternalPosition(P);
    KFieldVector internalField = KElectrostaticBoundaryField::ElectricFieldCore(internal);
    KFieldVector aField = fConverter->InternalToGlobalVector(internalField);
    //bindingmsg_debug( "electric field at " << P << " is " << aField <<eom);
    return aField;
}

void KGElectrostaticBoundaryField::SetMinimumElementArea(const double& aArea)
{
    fMinimumElementArea = aArea;
}

void KGElectrostaticBoundaryField::SetMaximumElementAspectRatio(const double& aAspect)
{
    fMaximumElementAspectRatio = aAspect;
}

void KGElectrostaticBoundaryField::SetSystem(KGeoBag::KGSpace* aSystem)
{
    fSystem = aSystem;
}

void KGElectrostaticBoundaryField::AddSurface(KGeoBag::KGSurface* aSurface)
{
    fSurfaces.push_back(aSurface);
}

void KGElectrostaticBoundaryField::AddSpace(KGeoBag::KGSpace* aSpace)
{
    fSpaces.push_back(aSpace);
}


void KGElectrostaticBoundaryField::SetSymmetry(const Symmetry& aSymmetry)
{
    fSymmetry = aSymmetry;
}

KSmartPointer<KGBEMConverter> KGElectrostaticBoundaryField::GetConverter()
{
    return fConverter;
}

void KGElectrostaticBoundaryField::InitializeCore()
{
    CheckSolverExistance();
    ConfigureSurfaceContainer();
    KElectrostaticBoundaryField::InitializeCore();
}

void KGElectrostaticBoundaryField::ConfigureSurfaceContainer()
{
    auto* container = new KSurfaceContainer();
    fConverter = nullptr;

    switch (fSymmetry) {
        case NoSymmetry:
            fConverter = new KGBEMMeshConverter();
            break;

        case AxialSymmetry:
            fConverter = new KGBEMAxialMeshConverter();
            break;

        case DiscreteAxialSymmetry:
            fConverter = new KGBEMDiscreteRotationalMeshConverter();
            break;

        default:
            kem_cout(eError) << "ERROR: boundary field got unknown symmetry flag <" << fSymmetry << ">" << eom;
            break;
    }
    fConverter->SetMinimumArea(fMinimumElementArea);
    fConverter->SetMaximumAspectRatio(fMaximumElementAspectRatio);

    fConverter->SetSurfaceContainer(container);

    if (fSystem != nullptr) {
        fConverter->SetSystem(fSystem->GetOrigin(), fSystem->GetXAxis(), fSystem->GetYAxis(), fSystem->GetZAxis());
    }

    for (auto& surface : fSurfaces) {
        surface->AcceptNode(&(*fConverter));
    }

    for (auto& space : fSpaces) {
        space->AcceptNode(&(*fConverter));
    }

    if (container->empty()) {
        kem_cout(eWarning) << "WARNING:"
                           << "electrostatic field solver <" << GetName() << "> has zero surface elements" << eom;
        //std::exit(-1);
    }

    SetContainer(container);
}

}  // namespace KEMField
