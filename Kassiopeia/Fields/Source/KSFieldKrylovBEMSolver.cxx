/*
 * KSFieldKrylovBEMSolver.cxx
 *
 *  Created on: 27.04.2015
 *      Author: gosda
 */

#include "KSFieldKrylovBEMSolver.h"
#include "KSFieldsMessage.h"
using Kassiopeia::fieldmsg; // pulls the resulting function of the fieldmsg_debug macro into scope
#include "KFMMessaging.hh"

#ifdef KEMFIELD_USE_MPI
#include "KMPIInterface.hh"
using KEMField::KMPIInterface;
#endif


#ifdef KEMFIELD_USE_MPI
    #ifndef MPI_SINGLE_PROCESS
        #define MPI_SINGLE_PROCESS if ( KEMField::KMPIInterface::GetInstance()->GetProcess()==0 )
    #endif
#else
    #ifndef MPI_SINGLE_PROCESS
        #define MPI_SINGLE_PROCESS if( true )
    #endif
#endif

namespace KEMField {

KrylovBEMSolver::KrylovBEMSolver():
		fSolver(new KFMElectrostaticFastMultipoleBoundaryValueSolver())
{
};

KrylovBEMSolver::~KrylovBEMSolver()
{
};

void KrylovBEMSolver::Initialize( KSurfaceContainer& surfaceContainer )
{
	if(fKrylovConfig.GetFFTMParams() == NULL) {
		kfmout << "ABORTING no multiplication method set for"
				" krylov bem solver" << kfmendl;
		kfmexit(1);
	}
	fSolver->SetSolverElectrostaticParameters(*fKrylovConfig.GetFFTMParams());
	fSolver->SetConfigurationObject(&fKrylovConfig);
	if(fKrylovConfig.GetPreconditionerFFTMParams() != NULL)
		fSolver->SetPreconditionerElectrostaticParameters(
				*fKrylovConfig.GetPreconditionerFFTMParams());

	//kfmout << fSolver->GetParameterInformation() << kfmendl;

	std::vector< std::string > info = fSolver->GetParameterInformationVector();
	for(unsigned int i=0; i<info.size(); i++)
	{
		MPI_SINGLE_PROCESS
		{
        	fieldmsg( eNormal ) << info[i] << eom;
		}
	}


	if( !FindSolution( fSolver->GetTolerance(), surfaceContainer ) )
	{
		//solve the boundary value problem
		fSolver->Solve(surfaceContainer);
		SaveSolution( fSolver->GetResidualNorm(), surfaceContainer );
	}

	delete fSolver;
	fSolver = NULL;
}

    bool KrylovBEMSolver::FindSolution( double aThreshold, KSurfaceContainer& aContainer )
    {
        // compute shape hash
        KMD5HashGenerator tShapeHashGenerator;
        tShapeHashGenerator.MaskedBits( fHashMaskedBits );
        tShapeHashGenerator.Threshold( fHashThreshold );
        tShapeHashGenerator.Omit( KEMField::Type2Type< KElectrostaticBasis >() );
        tShapeHashGenerator.Omit( Type2Type< KBoundaryType< KElectrostaticBasis, KDirichletBoundary > >() );
        string tShapeHash = tShapeHashGenerator.GenerateHash( aContainer );

        fieldmsg_debug( "<shape> hash is <" << tShapeHash << ">" << eom )

        // compute shape+boundary hash
        KMD5HashGenerator tShapeBoundaryHashGenerator;
        tShapeBoundaryHashGenerator.MaskedBits( fHashMaskedBits );
        tShapeBoundaryHashGenerator.Threshold( fHashThreshold );
        tShapeBoundaryHashGenerator.Omit( Type2Type< KElectrostaticBasis >() );
        string tShapeBoundaryHash = tShapeBoundaryHashGenerator.GenerateHash( aContainer );

        fieldmsg_debug( "<shape+boundary> hash is <" << tShapeBoundaryHash << ">" << eom )

        vector< string > tLabels;
        unsigned int tCount;
        bool tSolution;

        // compose residual threshold labels for shape and shape+boundary
        tLabels.clear();
        tLabels.push_back( KResidualThreshold::Name() );
        tLabels.push_back( tShapeHash );
        tLabels.push_back( tShapeBoundaryHash );

        // find matching residual thresholds
        tCount = KEMFileInterface::GetInstance()->NumberWithLabels( tLabels );

        fieldmsg_debug( "found <" << tCount << "> that match <shape> and <shape+boundary> hashes" << eom )

        if( tCount > 0 )
        {
            KResidualThreshold tResidualThreshold;
            KResidualThreshold tMinResidualThreshold;

            for( unsigned int i = 0; i < tCount; i++ )
            {
                KEMFileInterface::GetInstance()->FindByLabels( tResidualThreshold, tLabels, i );

                fieldmsg_debug( "found threshold <" << tResidualThreshold.fResidualThreshold << ">" << eom )

                if( tResidualThreshold < tMinResidualThreshold )
                {
                    fieldmsg_debug( "found minimum solution <" << tResidualThreshold.fGeometryHash << "> with threshold <" << tResidualThreshold.fResidualThreshold << ">" << eom )

                    tMinResidualThreshold = tResidualThreshold;
                }

            }

            fieldmsg_debug( "global minimum solution <" << tMinResidualThreshold.fGeometryHash << "> with threshold <" << tResidualThreshold.fResidualThreshold << ">" << eom )

            KEMFileInterface::GetInstance()->FindByHash( aContainer, tMinResidualThreshold.fGeometryHash );

            tSolution = false;
            if( tMinResidualThreshold.fResidualThreshold <= aThreshold )
            {
                kfmout << "previously computed solution found" << kfmendl;
                tSolution = true;
            }

            if( tSolution == true )
            {
                return true;
            }
        }
        return false;
    }
}//KEMField
