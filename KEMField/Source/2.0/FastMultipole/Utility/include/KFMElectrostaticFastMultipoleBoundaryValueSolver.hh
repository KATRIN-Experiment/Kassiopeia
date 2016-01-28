#ifndef KFMElectrostaticFastMultipoleBoundaryValueSolver_HH__
#define KFMElectrostaticFastMultipoleBoundaryValueSolver_HH__

#include <cstdlib>
#include <vector>
#include <string>

#ifdef KEMFIELD_USE_REALTIME_CLOCK
#include <time.h>
#endif

#include "KSurfaceContainer.hh"
#include "KSortedSurfaceContainer.hh"
#include "KDataDisplay.hh"

#include "KBoundaryIntegralMatrix.hh"
#include "KBoundaryIntegralVector.hh"
#include "KBoundaryIntegralSolutionVector.hh"

#include "KBiconjugateGradientStabilized.hh"
#include "KGeneralizedMinimalResidual.hh"
#include "KPreconditionedBiconjugateGradientStabilized.hh"
#include "KPreconditionedGeneralizedMinimalResidual.hh"
#include "KJacobiPreconditioner.hh"
#include "KImplicitKrylovPreconditioner.hh"

#include "KIterativeKrylovRestartCondition.hh"
#include "KIterativeKrylovSolver.hh"
#include "KIterativeKrylovStateWriter.hh"
#include "KIterativeKrylovStateReader.hh"
#include "KPreconditionedIterativeKrylovSolver.hh"
#include "KPreconditionedIterativeKrylovStateWriter.hh"
#include "KPreconditionedIterativeKrylovStateReader.hh"

#include "KFMBoundaryIntegralMatrix.hh"
#include "KFMDenseBoundaryIntegralMatrix.hh"

#include "KFMDenseBlockSparseMatrix.hh"
#include "KFMSparseBoundaryIntegralMatrix_BlockCompressedRow.hh"

#include "KFMElectrostaticBoundaryIntegrator.hh"
#include "KFMElectrostaticBoundaryIntegratorEngine_SingleThread.hh"

#include "KFMElectrostaticTree.hh"
#include "KFMElectrostaticParameters.hh"
#include "KFMElectrostaticTreeConstructor.hh"

#ifdef KEMFIELD_USE_OPENCL
#include "KFMSparseElectrostaticBoundaryIntegratorEngine_OpenCL.hh"
#endif

#ifdef KEMFIELD_USE_VTK
#include "KVTKResidualGraph.hh"
#include "KVTKIterationPlotter.hh"
#endif


#ifdef KEMFIELD_USE_MPI
    #include "KMPIInterface.hh"
    #include "KFMDenseBlockSparseMatrix_MPI.hh"
    #include "KFMElectrostaticBoundaryIntegrator_MPI.hh"
    #include "KGeneralizedMinimalResidual_MPI.hh"
    #include "KPreconditionedGeneralizedMinimalResidual_MPI.hh"
#endif

#include "KSAStructuredASCIIHeaders.hh"

#include "KFMElectrostaticFastMultipoleBoundaryValueSolverConfiguration.hh"
#include "KFMElectrostaticParametersConfiguration.hh"

namespace KEMField
{

/*
*
*@file KFMElectrostaticFastMultipoleBoundaryValueSolver.hh
*@class KFMElectrostaticFastMultipoleBoundaryValueSolver
*@brief class to simplify user interface to electrostatic
* fast multipole boundary value problems for various compilation options
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Fri Feb 6 13:54:55 EST 2015 J. Barrett (barrettj@mit.edu) First Version
*
*/


class KFMElectrostaticFastMultipoleBoundaryValueSolver
{
    public:
        //whole bunch of typedefs to simply which objects are constructed
        #ifdef KEMFIELD_USE_OPENCL
            #ifdef KEMFIELD_USE_MPI //mpi+opencl solver engines
                //#pragma message("Using MPI+OpenCL")
                typedef KFMElectrostaticBoundaryIntegrator_MPI<KFMSparseElectrostaticBoundaryIntegratorEngine_OpenCL>
                FastMultipoleEBI;
                typedef KFMSparseBoundaryIntegralMatrix_BlockCompressedRow<KFMElectrostaticNodeObjects,
                    FastMultipoleEBI, KFMDenseBlockSparseMatrix_MPI<FastMultipoleEBI::ValueType> >
                FastMultipoleSparseMatrix;
            #else //mpi not enabled, default to single threaded opencl
                //#pragma message("Using OpenCL only")
                typedef KFMElectrostaticBoundaryIntegrator<KFMSparseElectrostaticBoundaryIntegratorEngine_OpenCL>
                FastMultipoleEBI;
                typedef KFMSparseBoundaryIntegralMatrix_BlockCompressedRow<KFMElectrostaticNodeObjects,
                    FastMultipoleEBI, KFMDenseBlockSparseMatrix<FastMultipoleEBI::ValueType> >
                FastMultipoleSparseMatrix;
            #endif
        #else //opencl not enabled, default to mpi/single threaded only
            #if KEMFIELD_USE_MPI
                //#pragma message("Using MPI Only")
                typedef KFMElectrostaticBoundaryIntegrator_MPI<KFMElectrostaticBoundaryIntegratorEngine_SingleThread>
                FastMultipoleEBI;
                typedef KFMSparseBoundaryIntegralMatrix_BlockCompressedRow<KFMElectrostaticNodeObjects,
                    FastMultipoleEBI, KFMDenseBlockSparseMatrix_MPI<FastMultipoleEBI::ValueType> >
                FastMultipoleSparseMatrix;
            #else //nothing enabled, single threaded only
                //#pragma message("Using single thread")
                typedef KFMElectrostaticBoundaryIntegrator<KFMElectrostaticBoundaryIntegratorEngine_SingleThread>
                FastMultipoleEBI;
                typedef KFMSparseBoundaryIntegralMatrix_BlockCompressedRow<KFMElectrostaticNodeObjects,
                    FastMultipoleEBI, KFMDenseBlockSparseMatrix<FastMultipoleEBI::ValueType> >
                FastMultipoleSparseMatrix;
            #endif
        #endif

        typedef KFMDenseBoundaryIntegralMatrix<FastMultipoleEBI>
        FastMultipoleDenseMatrix;
        typedef KFMBoundaryIntegralMatrix< FastMultipoleDenseMatrix, FastMultipoleSparseMatrix>
        FastMultipoleMatrix;

        KFMElectrostaticFastMultipoleBoundaryValueSolver();
        virtual ~KFMElectrostaticFastMultipoleBoundaryValueSolver();

        //read the parameters from a configuration file
        void ReadConfigurationFile(std::string config_file);

        //parameters used to contruct the region tree
        void SetSolverElectrostaticParameters(KFMElectrostaticParameters params)
        {
            fSolverParameters = params;
            fHaveRecievedSolverParameters = true;
        };

        //ability to set independent tree parameters for the preconditioner
        //only used in the case where the preconditioner type is independent_implicit_krylov
        void SetPreconditionerElectrostaticParameters(KFMElectrostaticParameters params){fPreconditionerParameters = params;};

        //relative tolerance before convergence is reached
        void SetTolerance(double tol){fSolverTolerance = tol;};
        double GetTolerance() const {return fSolverTolerance;};

        //returns the relative l2 norm current solutions residual
        double GetResidualNorm() const {return fResidualNorm;};

        //maximum number of iterations executed by solver before termination
        void SetMaxIterations(unsigned int max_iter){fMaxSolverIterations = max_iter;};

        //set the name of the solver (gmres or bicgstab)
        void SetSolverType(std::string solver_name){fSolverName = solver_name;};

        //set the name of the preconditioner (none, jacobi, implicit_krylov)
        void SetPreconditionerType(std::string preconditioner_name){fPreconditionerName = preconditioner_name;};

        //number of iterations before the solver/preconditioner is restarted
        void SetRestart(unsigned int restart){fIterationsBetweenRestart = restart;};

        //if the preconditioner is implicit_krylov, number of iterations
        //before the preconditioner is restarted
        void SetMaxPreconditionerIterations(unsigned int max_iter){fMaxPreconditionerIterations = max_iter;};

        //if the preconditioner is implicit_krylov, the relative tolerance
        //required before the preconditioned multiply is declared converged
        void SetPreconditionerTolerance(double p_tol){fPreconditionerTolerance = p_tol;};

        //if the preconditioner is implicit_krylov, the expansion degree
        //used in the far field multipole expansion
        void SetPreconditionerDegree(unsigned int p_degree){fPreconditionerDegree = p_degree;};

        //enable/disable checkpoints of the krylov solver state
        //in case of interruptions, default frequency is once per iteration
        //but can be set to be less frequent to avoid excess disk access
        void EnableCheckpoints(){fUseCheckpoints = true;};
        void DisableCheckpoints(){fUseCheckpoints = false;};
        void SetCheckpointFrequency(unsigned int n_iter){fCheckpointFrequency = n_iter;};

        //enable/disable display of the progress towards convergence
        void EnableIterationDisplay(){fUseDisplay = true;};
        void DisableIterationDisplay(){fUseDisplay = false;};

        //enable/disable dynamic plot of convergence (requires VTK)
        void EnableDynamicPlot(){fUsePlot = true;};
        void DisableDynamicPlot(){fUsePlot = false;};

        //enable/disable tracking of time taken to converge
        void EnableTiming(){fUseTimer = true;};
        void DisableTiming(){fUseTimer = false;};
        void SetMaxTimeAllowed(double t){fMaxTimeAllowed = t;};
        void SetTimeCheckFrequency(int i){fTimeCheckFrequency = i;};

        //set configuration
        void SetConfigurationObject(KFMElectrostaticFastMultipoleBoundaryValueSolverConfiguration* config);

        //solve the boundary value problem presented by this surface container
        void Solve(KSurfaceContainer& surfaceContainer);

        std::string GetParameterInformation();
        std::vector< std::string > GetParameterInformationVector();

    private:

        KFMElectrostaticParameters fSolverParameters;
        KFMElectrostaticParameters fPreconditionerParameters;
        bool fHaveRecievedSolverParameters;

        std::string fSolverName;
        std::string fPreconditionerName;

        double fSolverTolerance;
        double fResidualNorm;
        unsigned int fMaxSolverIterations;
        unsigned int fIterationsBetweenRestart;
        double fPreconditionerTolerance;
        unsigned int fMaxPreconditionerIterations;
        unsigned int fPreconditionerDegree;

        bool fUseCheckpoints;
        unsigned int fCheckpointFrequency;

        bool fUseDisplay;
        bool fUsePlot;

        bool fUseTimer;
        double fMaxTimeAllowed; //seconds
        unsigned int fTimeCheckFrequency;

        //determine which solver type to use
        unsigned int DetermineSolverType();

        //profiling
        #ifdef KEMFIELD_USE_REALTIME_CLOCK
        timespec TimeDifference(timespec start, timespec end);
        #endif

        //solving is delegated to these functions
        void SolveGMRES(KSurfaceContainer& surfaceContainer); //unpreconditioned
        void SolveGMRES_Jacobi(KSurfaceContainer& surfaceContainer); //jacobi preconditioner
        void SolveGMRES_ImplicitKrylov(KSurfaceContainer& surfaceContainer); //preconditioned with implicit kylov solver (GMRES)
        void SolveGMRES_IndependentImplicitKrylov(KSurfaceContainer& surfaceContainer); //preconditioned with implicit kylov solver (GMRES)
        void SolveBICGSTAB(KSurfaceContainer& surfaceContainer); //unpreconditioned
        void SolveBICGSTAB_Jacobi(KSurfaceContainer& surfaceContainer); //jacobi preconditioner
        void SolveBICGSTAB_ImplicitKrylov(KSurfaceContainer& surfaceContainer); //preconditioned with implicit kylov solver (GMRES)
        void SolveBICGSTAB_IndependentImplicitKrylov(KSurfaceContainer& surfaceContainer); //preconditioned with implicit kylov solver (GMRES)
};






}//end of kemfield namespace



#endif /* KFMElectrostaticFastMultipoleBoundaryValueSolver_HH__ */
