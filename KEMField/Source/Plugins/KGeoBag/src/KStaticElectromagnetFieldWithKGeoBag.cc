/*
 * KStaticElectromagnetFieldWithKGeoBag.cc
 *
 *  Created on: 25 Mar 2016
 *      Author: wolfgang
 */

#include "KStaticElectromagnetFieldWithKGeoBag.hh"
#include "KEMVectorConverters.hh"

using namespace KGeoBag;
using namespace std;

namespace KEMField {

KStaticElectromagnetFieldWithKGeoBag::KStaticElectromagnetFieldWithKGeoBag() :
        KStaticElectromagnetField(),
        fSystem(NULL),
        fSurfaces(),
        fSpaces(),
        fConverter(NULL)
{
}

KStaticElectromagnetFieldWithKGeoBag::~KStaticElectromagnetFieldWithKGeoBag()
{
}

void KStaticElectromagnetFieldWithKGeoBag::SetSystem(KGeoBag::KGSpace* aSystem)
{
    fSystem = aSystem;
}

void KStaticElectromagnetFieldWithKGeoBag::AddSurface(
        KGeoBag::KGSurface* aSurface)
{
    fSurfaces.push_back( aSurface );
}

void KStaticElectromagnetFieldWithKGeoBag::AddSpace(KGeoBag::KGSpace* aSpace)
{
    fSpaces.push_back( aSpace );
}

KSmartPointer<KGeoBag::KGElectromagnetConverter> KStaticElectromagnetFieldWithKGeoBag::GetConverter() {
    return fConverter;
}

void KStaticElectromagnetFieldWithKGeoBag::InitializeCore()
{
    CheckSolverExistance();
    ConfigureSurfaceContainer();
    KStaticElectromagnetField::InitializeCore();
}

KEMThreeVector KStaticElectromagnetFieldWithKGeoBag::MagneticPotentialCore(
        const KPosition& aSamplePoint) const {
    KPosition internal = fConverter->GlobalToInternalPosition(KEM2KThreeVector(aSamplePoint));
    KDirection internalPotential = KStaticElectromagnetField::MagneticPotentialCore(internal);
    return K2KEMThreeVector(fConverter->InternalToGlobalVector(internalPotential));
}

KEMThreeVector KStaticElectromagnetFieldWithKGeoBag::MagneticFieldCore(
        const KPosition& aSamplePoint) const {
    KPosition internal = fConverter->GlobalToInternalPosition(KEM2KThreeVector(aSamplePoint));
    KDirection internalField = KStaticElectromagnetField::MagneticFieldCore(internal);
    return K2KEMThreeVector(fConverter->InternalToGlobalVector(internalField));
}

KGradient KStaticElectromagnetFieldWithKGeoBag::MagneticGradientCore(
        const KPosition& aSamplePoint) const {
    KPosition internal = fConverter->GlobalToInternalPosition(KEM2KThreeVector(aSamplePoint));
    KGradient internalGradient = KStaticElectromagnetField::MagneticGradientCore(internal);
    return K2KEMThreeMatrix(fConverter->InternalTensorToGlobal(internalGradient));
}

void KStaticElectromagnetFieldWithKGeoBag::ConfigureSurfaceContainer() {

    KElectromagnetContainer* container = new KElectromagnetContainer();

    fConverter = new KGElectromagnetConverter();

    fConverter->SetElectromagnetContainer( container );

    if( fSystem != NULL )
    {
        fConverter->SetSystem( fSystem->GetOrigin(), fSystem->GetXAxis(), fSystem->GetYAxis(), fSystem->GetZAxis() );
    }

    for( vector< KGSurface* >::iterator tSurfaceIt = fSurfaces.begin(); tSurfaceIt != fSurfaces.end(); tSurfaceIt++ )
    {
        (*tSurfaceIt)->AcceptNode( &(*fConverter) );
    }

    for( vector< KGSpace* >::iterator tSpaceIt = fSpaces.begin(); tSpaceIt != fSpaces.end(); tSpaceIt++ )
    {
        (*tSpaceIt)->AcceptNode( &(*fConverter) );
    }

    if( container->empty() )
    {
        cout << "ERROR:" << "electromagnet field solver <" /*<< GetName()*/ << "> has zero surface elements" << endl;
        std::exit(-1);
    }

    SetContainer(container);
}

} /* namespace KEMField */
