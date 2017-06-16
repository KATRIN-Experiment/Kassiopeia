/*
 * KElectrostaticBoundaryFieldWithKGeoBag.cc
 *
 *  Created on: 15 Jun 2015
 *      Author: wolfgang
 */

#include "KElectrostaticBoundaryFieldWithKGeoBag.hh"

//#ifdef KEMFIELD_USE_VTK
//#include "KEMVTKViewer.hh"
//#include "KEMVTKElectromagnetViewer.hh"
//#include "KVTKIterationPlotter.hh"
//using KEMField::KEMVTKViewer;
//using KEMField::KEMVTKElectromagnetViewer;
//using KEMField::KVTKIterationPlotter;
//#endif

#include "KEMVectorConverters.hh"

#include "KFile.h"

#include "KSADataStreamer.hh"

using namespace KGeoBag;

namespace KEMField{

KElectrostaticBoundaryFieldWithKGeoBag::KElectrostaticBoundaryFieldWithKGeoBag() :
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

KElectrostaticBoundaryFieldWithKGeoBag::~KElectrostaticBoundaryFieldWithKGeoBag()
{
}

double KElectrostaticBoundaryFieldWithKGeoBag::PotentialCore( const KPosition& P) const
{
	KPosition internal = fConverter->GlobalToInternalPosition(KEM2KThreeVector(P));
	double aPotential = KElectrostaticBoundaryField::PotentialCore( internal );
	//bindingmsg_debug( "potential at " << KEM2KThreeVector(P) << " is " << aPotential <<eom);
	return aPotential;
}

KEMThreeVector KElectrostaticBoundaryFieldWithKGeoBag::ElectricFieldCore( const KPosition& P) const
{
	KPosition internal = fConverter->GlobalToInternalPosition(KEM2KThreeVector(P));
	KEMThreeVector internalField = KElectrostaticBoundaryField::ElectricFieldCore(internal);
	KEMThreeVector aField = K2KEMThreeVector (fConverter->InternalToGlobalVector(internalField));
	//bindingmsg_debug( "electric field at " << KEM2KThreeVector(P) << " is " << aField <<eom);
    return aField;
}

void KElectrostaticBoundaryFieldWithKGeoBag::SetMinimumElementArea(
		const double& aArea) {
	fMinimumElementArea = aArea;
}

void KElectrostaticBoundaryFieldWithKGeoBag::SetMaximumElementAspectRatio(
		const double& aAspect) {
	fMaximumElementAspectRatio = aAspect;
}

void KElectrostaticBoundaryFieldWithKGeoBag::SetSystem(
		KGeoBag::KGSpace* aSystem) {
    fSystem = aSystem;
}

void KElectrostaticBoundaryFieldWithKGeoBag::AddSurface(
		KGeoBag::KGSurface* aSurface) {
	fSurfaces.push_back( aSurface );
}

void KElectrostaticBoundaryFieldWithKGeoBag::AddSpace( KGeoBag::KGSpace* aSpace)
{
	fSpaces.push_back( aSpace );
}


void KElectrostaticBoundaryFieldWithKGeoBag::SetSymmetry( const Symmetry& aSymmetry)
{
	fSymmetry = aSymmetry;
}

KSmartPointer<KGBEMConverter> KElectrostaticBoundaryFieldWithKGeoBag::GetConverter()
{
	return fConverter;
}

void KElectrostaticBoundaryFieldWithKGeoBag::InitializeCore()
{
	CheckSolverExistance();
	ConfigureSurfaceContainer();
	KElectrostaticBoundaryField::InitializeCore();

}

void KElectrostaticBoundaryFieldWithKGeoBag::ConfigureSurfaceContainer() {
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
