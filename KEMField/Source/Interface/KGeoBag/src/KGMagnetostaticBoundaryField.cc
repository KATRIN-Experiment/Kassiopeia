/*
 * KGMagnetostaticBoundaryField.cc
 *
 *  Created on: 7 May 2025
 *      Author: pslocum
 */

#include "KGMagnetostaticBoundaryField.hh"

#include "KEMCoreMessage.hh"
#include "KFile.h"
#include "KSADataStreamer.hh"

using namespace KGeoBag;

namespace KEMField
{

KGMagnetostaticBoundaryField::KGMagnetostaticBoundaryField() :
    fMinimumElementArea(0.0),
    fMaximumElementAspectRatio(1e100),
    fSystem(nullptr),
    fSymmetry(NoSymmetry),
    fConverter(nullptr)
{}

KGMagnetostaticBoundaryField::~KGMagnetostaticBoundaryField() = default;


KFieldVector KGMagnetostaticBoundaryField::MagneticFieldCore(const KPosition& P) const
{
    KPosition internal = fConverter->GlobalToInternalPosition(P);
    KFieldVector internalField = KMagnetostaticBoundaryField::MagneticFieldCore(internal);
    KFieldVector aField = fConverter->InternalToGlobalVector(internalField);
    //bindingmsg_debug( "Magnetic field at " << P << " is " << aField <<eom);
    return aField;
}

void KGMagnetostaticBoundaryField::SetMinimumElementArea(const double& aArea)
{
    fMinimumElementArea = aArea;
}

void KGMagnetostaticBoundaryField::SetMaximumElementAspectRatio(const double& aAspect)
{
    fMaximumElementAspectRatio = aAspect;
}

void KGMagnetostaticBoundaryField::SetSystem(KGeoBag::KGSpace* aSystem)
{
    fSystem = aSystem;
}

void KGMagnetostaticBoundaryField::AddSurface(KGeoBag::KGSurface* aSurface)
{
    fSurfaces.push_back(aSurface);
}

void KGMagnetostaticBoundaryField::AddSpace(KGeoBag::KGSpace* aSpace)
{
    fSpaces.push_back(aSpace);
}


void KGMagnetostaticBoundaryField::SetSymmetry(const Symmetry& aSymmetry)
{
    fSymmetry = aSymmetry;
}

std::shared_ptr<KGBEMConverter> KGMagnetostaticBoundaryField::GetConverter()
{
    return fConverter;
}

void KGMagnetostaticBoundaryField::InitializeCore()
{
    CheckSolverExistance();
    ConfigureSurfaceContainer();
    KMagnetostaticBoundaryField::InitializeCore();
}

void KGMagnetostaticBoundaryField::ConfigureSurfaceContainer()
{
    auto container = std::make_shared<KSurfaceContainer>();
    fConverter = nullptr;

    switch (fSymmetry) {
        case NoSymmetry:
            fConverter = std::make_shared<KGBEMMeshConverter>();
            break;

        case AxialSymmetry:
            fConverter = std::make_shared<KGBEMAxialMeshConverter>();
            break;

        case DiscreteAxialSymmetry:
            fConverter = std::make_shared<KGBEMDiscreteRotationalMeshConverter>();
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
                           << "Magnetostatic field solver <" << GetName() << "> has zero surface elements" << eom;
        //std::exit(-1);
    }

    SetContainer(container);
}

}  // namespace KEMField
