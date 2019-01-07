/*
 * KGElectrostaticBoundaryField.cc
 *
 *  Created on: 15 Jun 2015
 *      Author: wolfgang
 */

#include "KGElectrostaticBoundaryField.hh"

#include "KFile.h"

#include "KSADataStreamer.hh"

using namespace KGeoBag;

namespace KEMField{

KGElectrostaticBoundaryField::KGElectrostaticBoundaryField() :
					KElectrostaticBoundaryField(),
					fMinimumElementArea(0.0),
					fMaximumElementAspectRatio(1e100),
					fSystem( NULL ),
					fSurfaces(),
					fSpaces(),
					fSymmetry( NoSymmetry ),
					fConverter( NULL )
{
}

KGElectrostaticBoundaryField::~KGElectrostaticBoundaryField()
{
}

double KGElectrostaticBoundaryField::PotentialCore( const KPosition& P) const
{
	KPosition internal = fConverter->GlobalToInternalPosition(P);
	double aPotential = KElectrostaticBoundaryField::PotentialCore( internal );
	//bindingmsg_debug( "potential at " << P << " is " << aPotential <<eom);
	return aPotential;
}

KThreeVector KGElectrostaticBoundaryField::ElectricFieldCore( const KPosition& P) const
{
	KPosition internal = fConverter->GlobalToInternalPosition(P);
	KThreeVector internalField = KElectrostaticBoundaryField::ElectricFieldCore(internal);
	KThreeVector aField = fConverter->InternalToGlobalVector(internalField);
	//bindingmsg_debug( "electric field at " << P << " is " << aField <<eom);
    return aField;
}

void KGElectrostaticBoundaryField::SetMinimumElementArea(
		const double& aArea) {
	fMinimumElementArea = aArea;
}

void KGElectrostaticBoundaryField::SetMaximumElementAspectRatio(
		const double& aAspect) {
	fMaximumElementAspectRatio = aAspect;
}

void KGElectrostaticBoundaryField::SetSystem(
		KGeoBag::KGSpace* aSystem) {
    fSystem = aSystem;
}

void KGElectrostaticBoundaryField::AddSurface(
		KGeoBag::KGSurface* aSurface) {
	fSurfaces.push_back( aSurface );
}

void KGElectrostaticBoundaryField::AddSpace( KGeoBag::KGSpace* aSpace)
{
	fSpaces.push_back( aSpace );
}


void KGElectrostaticBoundaryField::SetSymmetry( const Symmetry& aSymmetry)
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

void KGElectrostaticBoundaryField::ConfigureSurfaceContainer() {
	KSurfaceContainer* container = new KSurfaceContainer();
	fConverter = NULL;

        switch( fSymmetry )
        {
            case NoSymmetry :
                fConverter = new KGBEMMeshConverter();
                break;

            case AxialSymmetry :
                fConverter = new KGBEMAxialMeshConverter();
                break;

            case DiscreteAxialSymmetry :
                fConverter = new KGBEMDiscreteRotationalMeshConverter();
                break;

            default :
                cout << "ERROR: boundary field got unknown symmetry flag <" << fSymmetry << ">" << endl;
                break;
        }
        fConverter->SetMinimumArea(fMinimumElementArea);
        fConverter->SetMaximumAspectRatio(fMaximumElementAspectRatio);

        fConverter->SetSurfaceContainer( container );

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
        	// TODO find alternative to KNamed::GetName which is not available here
            cout << "ERROR:" << "electrostatic field solver <" /*<< GetName()*/ << "> has zero surface elements" << endl;
            std::exit(-1);
        }

        SetContainer(container);
}

} // KEMField
