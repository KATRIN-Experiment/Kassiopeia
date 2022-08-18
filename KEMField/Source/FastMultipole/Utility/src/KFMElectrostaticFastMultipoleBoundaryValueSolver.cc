#include "KFMElectrostaticFastMultipoleBoundaryValueSolver.hh"

#include "KBiconjugateGradientStabilized.hh"
#include "KBoundaryIntegralMatrix.hh"
#include "KBoundaryIntegralSolutionVector.hh"
#include "KBoundaryIntegralVector.hh"
#include "KDataDisplay.hh"
#include "KElectrostaticBoundaryIntegratorFactory.hh"
#include "KFMElectrostaticParametersConfiguration.hh"
#include "KGeneralizedMinimalResidual.hh"
#include "KImplicitKrylovPreconditioner.hh"
#include "KIterativeKrylovRestartCondition.hh"
#include "KIterativeKrylovStateReader.hh"
#include "KIterativeKrylovStateWriter.hh"
#include "KJacobiPreconditioner.hh"
#include "KMPIEnvironment.hh"
#include "KPreconditionedBiconjugateGradientStabilized.hh"
#include "KPreconditionedGeneralizedMinimalResidual.hh"
#include "KPreconditionedIterativeKrylovSolver.hh"
#include "KPreconditionedIterativeKrylovStateReader.hh"
#include "KPreconditionedIterativeKrylovStateWriter.hh"
#include "KSimpleIterativeKrylovSolver.hh"
#include "KSortedSurfaceContainer.hh"
#include "KTimeTerminator.hh"

using namespace KEMField::KFMElectrostaticTypes;

namespace KEMField
{

KFMElectrostaticFastMultipoleBoundaryValueSolver::KFMElectrostaticFastMultipoleBoundaryValueSolver() :
    fDirectIntegrator(KEBIFactory::MakeDefaultForFFTM())
{
    fSolverName = std::string("gmres");
    fPreconditionerName = std::string("none");

    fSolverTolerance = 0.1;
    fResidualNorm = 1.0;
    fMaxSolverIterations = UINT_MAX;
    fIterationsBetweenRestart = UINT_MAX;

    fPreconditionerTolerance = 0.1;
    fMaxPreconditionerIterations = UINT_MAX;
    fPreconditionerDegree = 0;

    fUseCheckpoints = false;
    fCheckpointFrequency = 1;

    fUseDisplay = false;
    fUsePlot = false;

    fUseTimer = false;
    fMaxTimeAllowed = 3e10;
    fTimeCheckFrequency = 1;

#if defined(KEMFIELD_USE_MPI) && defined(KEMFIELD_USE_OPENCL)
    //assign devices according to the number available and local process rank
    unsigned int proc_id = KMPIInterface::GetInstance()->GetProcess();
    int n_dev = KOpenCLInterface::GetInstance()->GetNumberOfDevices();
    int dev_id = proc_id % n_dev;  //fallback to global process rank if local is unavailable
   int local_rank = KMPIInterface::GetInstance()->GetLocalRank();
   if (local_rank != -1) {
       if (KMPIInterface::GetInstance()->SplitMode()) {
           dev_id = (local_rank / 2) % n_dev;
       }
       else {
           dev_id = (local_rank) % n_dev;
       }
   }
    KOpenCLInterface::GetInstance()->SetGPU(dev_id);
#endif

    fHaveRecievedSolverParameters = false;
}

KFMElectrostaticFastMultipoleBoundaryValueSolver::~KFMElectrostaticFastMultipoleBoundaryValueSolver() = default;
;

void KFMElectrostaticFastMultipoleBoundaryValueSolver::ReadConfigurationFile(const std::string& config_file)
{
    bool solver_parameters_present = false;
    auto* solver_parameter_input =
        new KSAObjectInputNode<KFMElectrostaticParametersConfiguration>(std::string("SolverElectrostaticParameters"));

    bool bvp_parameters_present = false;
    auto* bvp_config_input = new KSAObjectInputNode<KFMElectrostaticFastMultipoleBoundaryValueSolverConfiguration>(
        std::string("BoundaryValueSolverConfiguration"));

    bool preconditioner_parameters_present = false;
    auto* preconditioner_parameter_input = new KSAObjectInputNode<KFMElectrostaticParametersConfiguration>(
        std::string("PreconditionerElectrostaticParameters"));

    if (!config_file.empty()) {
        //need to read in tree parameters
        KEMFileInterface::GetInstance()->ReadKSAFile(solver_parameter_input, config_file, solver_parameters_present);

        //need to read in boundary value problem solver parameters
        KEMFileInterface::GetInstance()->ReadKSAFile(bvp_config_input, config_file, bvp_parameters_present);

        //optionally present in config file
        KEMFileInterface::GetInstance()->ReadKSAFile(preconditioner_parameter_input,
                                                     config_file,
                                                     preconditioner_parameters_present);
    }

    if (solver_parameters_present && bvp_parameters_present) {
        SetConfigurationObject(bvp_config_input->GetObject());
        SetSolverElectrostaticParameters(solver_parameter_input->GetObject()->GetParameters());
        //will not be used unless the preconditioner is the independent_implicit_krylov type
        if (preconditioner_parameters_present) {
            SetPreconditionerElectrostaticParameters(preconditioner_parameter_input->GetObject()->GetParameters());
        }

        MPI_SINGLE_PROCESS
        {
            if (fSolverParameters.verbosity > 2) {
                kfmout << "KFMElectrostaticFastMultipoleBoundaryValueSolver::ReadConfigurationFile(): ";
                kfmout << "Configuration file read successfully." << kfmendl;
            }
        }
    }
    else {
        MPI_SINGLE_PROCESS
        {
            kfmout << "KFMElectrostaticFastMultipoleBoundaryValueSolver::ReadConfigurationFile(): ";
            kfmout << config_file << " read failed." << kfmendl;
        }
#ifdef KEMFIELD_USE_MPI
        KMPIInterface::GetInstance()->Finalize();
#endif
        kfmexit(1);
    }

    delete bvp_config_input;
    delete solver_parameter_input;
    delete preconditioner_parameter_input;
}

void KFMElectrostaticFastMultipoleBoundaryValueSolver::SetConfigurationObject(
    KFMElectrostaticFastMultipoleBoundaryValueSolverConfiguration* config)
{
    SetTolerance(config->GetSolverTolerance());
    SetMaxIterations(config->GetMaxSolverIterations());
    SetSolverType(config->GetSolverName());
    SetPreconditionerType(config->GetPreconditionerName());
    SetRestart(config->GetIterationsBetweenRestart());

    SetMaxPreconditionerIterations(config->GetMaxPreconditionerIterations());
    SetPreconditionerTolerance(config->GetPreconditionerTolerance());
    SetPreconditionerDegree(config->GetPreconditionerDegree());

    int use_checkpoints = config->GetUseCheckpoints();
    if (use_checkpoints == 0) {
        DisableCheckpoints();
    }
    else {
        EnableCheckpoints();
        SetCheckpointFrequency(config->GetCheckpointFrequency());
    }

    int use_display = config->GetUseDisplay();
    if (use_display == 0) {
        DisableIterationDisplay();
    }
    else {
        EnableIterationDisplay();
    }

    int use_plot = config->GetUsePlot();
    if (use_plot == 0) {
        DisableDynamicPlot();
    }
    else {
        EnableDynamicPlot();
    }

    int use_timer = config->GetUseTimer();
    if (use_timer == 0) {
        DisableTiming();
    }
    else {
        EnableTiming();
        SetMaxTimeAllowed(config->GetTimeLimitSeconds());
        SetTimeCheckFrequency(config->GetTimeCheckFrequency());
    }
}

void KFMElectrostaticFastMultipoleBoundaryValueSolver::Solve(KSurfaceContainer& surfaceContainer)
{
    if (!fHaveRecievedSolverParameters) {
        MPI_SINGLE_PROCESS
        {
            kfmout << "KFMElectrostaticFastMultipoleBoundaryValueSolver::Solve():";
            kfmout << " Error, cannot initialized solver without parameters first being set." << kfmendl;
            kfmexit(1);
        }
    }

    unsigned int type = DetermineSolverType();

    switch (type) {
        case 0:
            SolveGMRES(surfaceContainer);
            break;
        case 1:
            SolveGMRES_Jacobi(surfaceContainer);
            break;
        case 2:
            SolveGMRES_ImplicitKrylov(surfaceContainer);
            break;
        case 3:
            SolveBICGSTAB(surfaceContainer);
            break;
        case 4:
            SolveBICGSTAB_Jacobi(surfaceContainer);
            break;
        case 5:
            SolveBICGSTAB_ImplicitKrylov(surfaceContainer);
            break;
        case 6:
            SolveGMRES_IndependentImplicitKrylov(surfaceContainer);
            break;
        case 7:
            SolveBICGSTAB_IndependentImplicitKrylov(surfaceContainer);
            break;
        default:
            SolveGMRES(surfaceContainer);
            break;
    }
}

std::string KFMElectrostaticFastMultipoleBoundaryValueSolver::GetParameterInformation()
{
    std::stringstream output;

    output << "Krylov Solver Parameters: \n";
    output << " krylov type: " << fSolverName << "\n";
    output << " tolerance: " << fSolverTolerance << "\n";
    output << " max iterations: " << fMaxSolverIterations << "\n";
    output << " iterations between restarts: " << fIterationsBetweenRestart << "\n";

    if (fUseCheckpoints) {
        output << " use_checkpoints: true \n";
        output << " checkpoint_interval: " << fCheckpointFrequency << "\n";
    }
    else {
        output << "use_checkpoints: false \n";
    }

#ifdef KEMFIELD_USE_REALTIME_CLOCK
    if (fUseTimer) {
        output << " use_timer: true \n";
    }
    else {
        output << "use_timer: false \n";
    }
#else
    output << " use_timer: disabled \n";
#endif

    output << "Solver Fast Multipole Parameters: \n";
    output << " top level divisions: " << fSolverParameters.top_level_divisions << "\n";
    output << " tree level divisions: " << fSolverParameters.divisions << "\n";
    output << " degree: " << fSolverParameters.degree << "\n";
    output << " zeromask size: " << fSolverParameters.zeromask << "\n";
    output << " maximum tree depth: " << fSolverParameters.maximum_tree_depth << "\n";
    output << " insertion_ratio: " << fSolverParameters.insertion_ratio << "\n";
    output << " verbosity: " << fSolverParameters.verbosity << "\n";

    if (fSolverParameters.use_region_estimation) {
        output << " use region estimation: true \n";
        output << " region expansion factor " << fSolverParameters.region_expansion_factor << "\n";
    }
    else {
        output << " use region estimation: false \n";
        output << " world center x " << fSolverParameters.world_center_x << "\n";
        output << " world center y " << fSolverParameters.world_center_y << "\n";
        output << " world center z " << fSolverParameters.world_center_z << "\n";
        output << " world length " << fSolverParameters.world_length << "\n";
    }

    if (fSolverParameters.use_caching) {
        output << " use caching: true";
    }
    else {
        output << " use caching: false";
    }


    if (fPreconditionerName != std::string("none")) {
        output << "\n";

        output << "Preconditioner Parameters: \n";
        output << " preconditioner type: " << fPreconditionerName << "\n";
        output << " tolerance: " << fPreconditionerTolerance << "\n";
        output << " max iterations: " << fMaxPreconditionerIterations << "\n";
        output << " iterations between restarts: " << fIterationsBetweenRestart << "\n";

        if (fPreconditionerName == std::string("independent_implicit_krylov")) {
            output << "Preconditioner Fast Multipole Parameters: \n";
            output << " top level divisions: " << fPreconditionerParameters.top_level_divisions << "\n";
            output << " tree level divisions: " << fPreconditionerParameters.divisions << "\n";
            output << " degree: " << fPreconditionerParameters.degree << "\n";
            output << " zeromask size: " << fPreconditionerParameters.zeromask << "\n";
            output << " maximum tree depth: " << fPreconditionerParameters.maximum_tree_depth << "\n";
            output << " insertion_ratio: " << fPreconditionerParameters.insertion_ratio << "\n";
            output << " verbosity: " << fPreconditionerParameters.verbosity << "\n";

            if (fPreconditionerParameters.use_region_estimation) {
                output << " use region estimation: true \n";
                output << " region expansion factor " << fPreconditionerParameters.region_expansion_factor << "\n";
            }
            else {
                output << " use region estimation: false \n";
                output << " world center x " << fPreconditionerParameters.world_center_x << "\n";
                output << " world center y " << fPreconditionerParameters.world_center_y << "\n";
                output << " world center z " << fPreconditionerParameters.world_center_z << "\n";
                output << " world length " << fPreconditionerParameters.world_length << "\n";
            }

            if (fPreconditionerParameters.use_caching) {
                output << " use caching: true \n";
            }
            else {
                output << " use caching: false \n";
            }
        }
        else {
            output << " degree: " << fPreconditionerDegree;
        }
    }

    return output.str();
}


std::vector<std::string> KFMElectrostaticFastMultipoleBoundaryValueSolver::GetParameterInformationVector()
{
    std::vector<std::string> info_vec;

    std::stringstream output;

    output << "Krylov Solver Parameters: ";
    info_vec.push_back(output.str());
    output.str("");
    output << " krylov type: " << fSolverName;
    info_vec.push_back(output.str());
    output.str("");
    output << " tolerance: " << fSolverTolerance;
    info_vec.push_back(output.str());
    output.str("");
    output << " max iterations: " << fMaxSolverIterations;
    info_vec.push_back(output.str());
    output.str("");
    output << " iterations between restarts: " << fIterationsBetweenRestart;
    info_vec.push_back(output.str());
    output.str("");

    if (fUseCheckpoints) {
        output << " use_checkpoints: true";
        info_vec.push_back(output.str());
        output.str("");
        output << " checkpoint_interval: " << fCheckpointFrequency;
        info_vec.push_back(output.str());
        output.str("");
    }
    else {
        output << "use_checkpoints: false ";
        info_vec.push_back(output.str());
        output.str("");
    }

#ifdef KEMFIELD_USE_REALTIME_CLOCK
    if (fUseTimer) {
        output << " use_timer: true";
        info_vec.push_back(output.str());
        output.str("");
    }
    else {
        output << "use_timer: false";
        info_vec.push_back(output.str());
        output.str("");
    }
#else
    output << " use_timer: disabled";
    info_vec.push_back(output.str());
    output.str("");
#endif

    output << "Solver Fast Multipole Parameters: ";
    info_vec.push_back(output.str());
    output.str("");
    output << " top level divisions: " << fSolverParameters.top_level_divisions;
    info_vec.push_back(output.str());
    output.str("");
    output << " tree level divisions: " << fSolverParameters.divisions;
    info_vec.push_back(output.str());
    output.str("");
    output << " degree: " << fSolverParameters.degree;
    info_vec.push_back(output.str());
    output.str("");
    output << " zeromask size: " << fSolverParameters.zeromask;
    info_vec.push_back(output.str());
    output.str("");
    output << " maximum tree depth: " << fSolverParameters.maximum_tree_depth;
    info_vec.push_back(output.str());
    output.str("");
    output << " insertion_ratio: " << fSolverParameters.insertion_ratio;
    info_vec.push_back(output.str());
    output.str("");
    output << " verbosity: " << fSolverParameters.verbosity;
    info_vec.push_back(output.str());
    output.str("");

    if (fSolverParameters.use_region_estimation) {
        output << " use region estimation: true ";
        info_vec.push_back(output.str());
        output.str("");
        output << " region expansion factor " << fSolverParameters.region_expansion_factor;
        info_vec.push_back(output.str());
        output.str("");
    }
    else {
        output << " use region estimation: false ";
        info_vec.push_back(output.str());
        output.str("");
        output << " world center x " << fSolverParameters.world_center_x;
        info_vec.push_back(output.str());
        output.str("");
        output << " world center y " << fSolverParameters.world_center_y;
        info_vec.push_back(output.str());
        output.str("");
        output << " world center z " << fSolverParameters.world_center_z;
        info_vec.push_back(output.str());
        output.str("");
        output << " world length " << fSolverParameters.world_length;
        info_vec.push_back(output.str());
        output.str("");
    }

    if (fSolverParameters.use_caching) {
        output << " use caching: true";
        info_vec.push_back(output.str());
        output.str("");
    }
    else {
        output << " use caching: false";
        info_vec.push_back(output.str());
        output.str("");
    }


    if (fPreconditionerName != std::string("none")) {
        output << "Preconditioner Parameters:";
        info_vec.push_back(output.str());
        output.str("");
        output << " preconditioner type: " << fPreconditionerName;
        info_vec.push_back(output.str());
        output.str("");
        output << " tolerance: " << fPreconditionerTolerance;
        info_vec.push_back(output.str());
        output.str("");
        output << " max iterations: " << fMaxPreconditionerIterations;
        info_vec.push_back(output.str());
        output.str("");
        output << " iterations between restarts: " << fIterationsBetweenRestart;
        info_vec.push_back(output.str());
        output.str("");

        if (fPreconditionerName == std::string("independent_implicit_krylov")) {
            output << "Preconditioner Fast Multipole Parameters: ";
            info_vec.push_back(output.str());
            output.str("");
            output << " top level divisions: " << fPreconditionerParameters.top_level_divisions;
            info_vec.push_back(output.str());
            output.str("");
            output << " tree level divisions: " << fPreconditionerParameters.divisions;
            info_vec.push_back(output.str());
            output.str("");
            output << " degree: " << fPreconditionerParameters.degree;
            info_vec.push_back(output.str());
            output.str("");
            output << " zeromask size: " << fPreconditionerParameters.zeromask;
            info_vec.push_back(output.str());
            output.str("");
            output << " maximum tree depth: " << fPreconditionerParameters.maximum_tree_depth;
            info_vec.push_back(output.str());
            output.str("");
            output << " insertion_ratio: " << fPreconditionerParameters.insertion_ratio;
            info_vec.push_back(output.str());
            output.str("");
            output << " verbosity: " << fPreconditionerParameters.verbosity;
            info_vec.push_back(output.str());
            output.str("");

            if (fPreconditionerParameters.use_region_estimation) {
                output << " use region estimation: true";
                info_vec.push_back(output.str());
                output.str("");
                output << " region expansion factor " << fPreconditionerParameters.region_expansion_factor;
                info_vec.push_back(output.str());
                output.str("");
            }
            else {
                output << " use region estimation: false";
                info_vec.push_back(output.str());
                output.str("");
                output << " world center x " << fPreconditionerParameters.world_center_x;
                info_vec.push_back(output.str());
                output.str("");
                output << " world center y " << fPreconditionerParameters.world_center_y;
                info_vec.push_back(output.str());
                output.str("");
                output << " world center z " << fPreconditionerParameters.world_center_z;
                info_vec.push_back(output.str());
                output.str("");
                output << " world length " << fPreconditionerParameters.world_length;
                info_vec.push_back(output.str());
                output.str("");
            }

            if (fPreconditionerParameters.use_caching) {
                output << " use caching: true";
                info_vec.push_back(output.str());
                output.str("");
            }
            else {
                output << " use caching: false";
                info_vec.push_back(output.str());
                output.str("");
            }
        }
        else {
            output << " degree: " << fPreconditionerDegree;
            info_vec.push_back(output.str());
            output.str("");
        }
    }

    return info_vec;
}

//solving is delegated to these functions
void KFMElectrostaticFastMultipoleBoundaryValueSolver::SolveGMRES(KSurfaceContainer& surfaceContainer)
{
    std::shared_ptr<FastMultipoleEBI> fm_integrator(new FastMultipoleEBI(fDirectIntegrator, surfaceContainer));
    std::shared_ptr<FastMultipoleMatrix> fmA = CreateMatrix(surfaceContainer, fm_integrator);
    KBoundaryIntegralSolutionVector<FastMultipoleEBI> fmx(surfaceContainer, *fm_integrator);
    KBoundaryIntegralVector<FastMultipoleEBI> fmb(surfaceContainer, *fm_integrator);

    auto restart_cond = std::make_shared<KIterativeKrylovRestartCondition>();
    restart_cond->SetNumberOfIterationsBetweenRestart(fIterationsBetweenRestart);

    KSimpleIterativeKrylovSolver<FastMultipoleEBI::ValueType, KGeneralizedMinimalResidual> gmres;
    gmres.SetTolerance(fSolverTolerance);
    gmres.SetMaximumIterations(fMaxSolverIterations);
    gmres.SetRestartCondition(restart_cond);

    if (fUseCheckpoints) {
        //do not need to delete this visitor, as it is deleted by the solver
        auto* checkpoint_reader =
            new KIterativeKrylovStateReader<FastMultipoleEBI::ValueType,
                                            KGeneralizedMinimalResidual,
                                            KGeneralizedMinimalResidualState>(fm_integrator->GetLabels());
        checkpoint_reader->SetVerbosity(fSolverParameters.verbosity);
        gmres.AddVisitor(checkpoint_reader);
    }

    if (fUseTimer) {
        gmres.AddVisitor(new KTimeTerminator<FastMultipoleEBI::ValueType>(fMaxTimeAllowed, fTimeCheckFrequency));
    }

    MPI_SINGLE_PROCESS
    {
        if (fUseCheckpoints) {
            //do not need to delete this visitor, as it is deleted by the solver
            auto* checkpoint =
                new KIterativeKrylovStateWriter<FastMultipoleEBI::ValueType, KGeneralizedMinimalResidual>(
                    fm_integrator->GetLabels());
            checkpoint->Interval(fCheckpointFrequency);
            gmres.AddVisitor(checkpoint);
        }

        if (fUseDisplay) {
            gmres.AddVisitor(new KIterationDisplay<double>(std::string("GMRES: ")));
        }
#ifdef KEMFIELD_USE_VTK
        if (fUsePlot) {
            gmres.AddVisitor(new KVTKIterationPlotter<double>());
        }
#endif
    }

#ifdef KEMFIELD_USE_REALTIME_CLOCK
    timespec start, end;
#endif
    clock_t cstart = clock();
    clock_t cend = clock();
    double time;

    MPI_SINGLE_PROCESS
    {
        if (fUseTimer) {
//timer
#ifdef KEMFIELD_USE_REALTIME_CLOCK
            clock_gettime(CLOCK_REALTIME, &start);
#endif
            cstart = clock();
        }
    }

    gmres.Solve(*fmA, fmx, fmb);
    fResidualNorm = gmres.ResidualNorm();

    MPI_SINGLE_PROCESS
    {
        if (fUseTimer) {
#ifdef KEMFIELD_USE_REALTIME_CLOCK
            clock_gettime(CLOCK_REALTIME, &end);
            timespec temp = TimeDifference(start, end);
            kfmout << "Real time required to solve (sec): " << temp.tv_sec << "." << temp.tv_nsec << kfmendl;
#endif
            cend = clock();
            time = ((double) (cend - cstart)) / CLOCKS_PER_SEC;  // time in seconds
            kfmout << "Process/CPU time required to solve (sec): " << time << kfmendl;
        }
    }
}

void KFMElectrostaticFastMultipoleBoundaryValueSolver::SolveGMRES_Jacobi(KSurfaceContainer& surfaceContainer)
{
    auto* fm_integrator = new FastMultipoleEBI(fDirectIntegrator, surfaceContainer);
    fm_integrator->Initialize(fSolverParameters);

    FastMultipoleDenseMatrix denseA(*fm_integrator);
    FastMultipoleSparseMatrix sparseA(surfaceContainer, *fm_integrator);
    FastMultipoleMatrix fmA(denseA, sparseA);

    KBoundaryIntegralSolutionVector<FastMultipoleEBI> fmx(surfaceContainer, *fm_integrator);
    KBoundaryIntegralVector<FastMultipoleEBI> fmb(surfaceContainer, *fm_integrator);

    auto restart_cond = std::make_shared<KIterativeKrylovRestartCondition>();
    restart_cond->SetNumberOfIterationsBetweenRestart(fIterationsBetweenRestart);

    KPreconditionedIterativeKrylovSolver<FastMultipoleEBI::ValueType, KPreconditionedGeneralizedMinimalResidual> pgmres;
    pgmres.SetTolerance(fSolverTolerance);
    pgmres.SetRestartCondition(restart_cond);
    pgmres.SetMaximumIterations(fMaxSolverIterations);

    if (fUseTimer) {
        pgmres.AddVisitor(new KTimeTerminator<FastMultipoleEBI::ValueType>(fMaxTimeAllowed, fTimeCheckFrequency));
    }

    KJacobiPreconditioner<FastMultipoleEBI::ValueType> P(fmA);

    if (fUseCheckpoints) {
        //do not need to delete this visitor, as it is deleted by the solver
        auto* checkpoint_reader =
            new KPreconditionedIterativeKrylovStateReader<FastMultipoleEBI::ValueType,
                                                          KPreconditionedGeneralizedMinimalResidual,
                                                          KPreconditionedGeneralizedMinimalResidualState>(
                fm_integrator->GetLabels());
        checkpoint_reader->SetVerbosity(fSolverParameters.verbosity);
        pgmres.AddVisitor(checkpoint_reader);
    }

    MPI_SINGLE_PROCESS
    {
        if (fUseCheckpoints) {
            //do not need to delete this visitor, as it is deleted by the solver
            auto* checkpoint = new KPreconditionedIterativeKrylovStateWriter<FastMultipoleEBI::ValueType,
                                                                             KPreconditionedGeneralizedMinimalResidual>(
                fm_integrator->GetLabels());
            checkpoint->Interval(fCheckpointFrequency);
            pgmres.AddVisitor(checkpoint);
        }

        if (fUseDisplay) {
            pgmres.AddVisitor(new KIterationDisplay<double>(std::string("GMRES: ")));
        }
#ifdef KEMFIELD_USE_VTK
        if (fUsePlot) {
            pgmres.AddVisitor(new KVTKIterationPlotter<double>());
        }
#endif
    }

#ifdef KEMFIELD_USE_REALTIME_CLOCK
    timespec start, end;
#endif
    clock_t cstart = clock();
    clock_t cend = clock();
    double time;

    MPI_SINGLE_PROCESS
    {
        if (fUseTimer) {
//timer
#ifdef KEMFIELD_USE_REALTIME_CLOCK
            clock_gettime(CLOCK_REALTIME, &start);
#endif
            cstart = clock();
        }
    }

    pgmres.Solve(fmA, P, fmx, fmb);
    fResidualNorm = pgmres.ResidualNorm();

    MPI_SINGLE_PROCESS
    {
        if (fUseTimer) {
#ifdef KEMFIELD_USE_REALTIME_CLOCK
            clock_gettime(CLOCK_REALTIME, &end);
            timespec temp = TimeDifference(start, end);
            kfmout << "Real time required to solve (sec): " << temp.tv_sec << "." << temp.tv_nsec << kfmendl;
#endif
            cend = clock();
            time = ((double) (cend - cstart)) / CLOCKS_PER_SEC;  // time in seconds
            kfmout << "Process/CPU time required to solve (sec): " << time << kfmendl;
        }
    }

    delete fm_integrator;
}

void KFMElectrostaticFastMultipoleBoundaryValueSolver::SolveGMRES_ImplicitKrylov(KSurfaceContainer& surfaceContainer)
{
    auto* fm_integrator = new FastMultipoleEBI(fDirectIntegrator, surfaceContainer);
    fm_integrator->Initialize(fSolverParameters);

    KFMElectrostaticParameters alt_params = fSolverParameters;
    alt_params.degree = fPreconditionerDegree;
    auto* alt_fm_integrator = new FastMultipoleEBI(fDirectIntegrator, surfaceContainer);
    alt_fm_integrator->Initialize(alt_params, fm_integrator->GetTree());

    FastMultipoleDenseMatrix denseA(*fm_integrator);
    FastMultipoleDenseMatrix alt_denseA(*alt_fm_integrator);
    FastMultipoleSparseMatrix sparseA(surfaceContainer, *fm_integrator);
    FastMultipoleMatrix fmA(denseA, sparseA);
    std::shared_ptr<FastMultipoleMatrix> alt_fmA(new FastMultipoleMatrix(alt_denseA, sparseA));

    KBoundaryIntegralSolutionVector<FastMultipoleEBI> fmx(surfaceContainer, *fm_integrator);
    KBoundaryIntegralVector<FastMultipoleEBI> fmb(surfaceContainer, *fm_integrator);

    auto restart_cond = std::make_shared<KIterativeKrylovRestartCondition>();
    restart_cond->SetNumberOfIterationsBetweenRestart(fIterationsBetweenRestart);

    KPreconditionedIterativeKrylovSolver<FastMultipoleEBI::ValueType, KPreconditionedGeneralizedMinimalResidual> pgmres;
    pgmres.SetTolerance(fSolverTolerance);
    pgmres.SetRestartCondition(restart_cond);
    pgmres.SetMaximumIterations(fMaxSolverIterations);

    if (fUseTimer) {
        pgmres.AddVisitor(new KTimeTerminator<FastMultipoleEBI::ValueType>(fMaxTimeAllowed, fTimeCheckFrequency));
    }

    KKrylovSolverConfiguration config;
    config.SetSolverName("gmres");
    config.SetMaxIterations(fMaxPreconditionerIterations);
    config.SetTolerance(fPreconditionerTolerance);
    config.SetUseDisplay(fUseDisplay);
    config.SetDisplayName("Preconditioner: ");
    auto IKP = KBuildKrylovPreconditioner<FastMultipoleEBI::ValueType>(config, alt_fmA);

    if (fUseCheckpoints) {
        //do not need to delete this visitor, as it is deleted by the solver
        auto* checkpoint_reader =
            new KPreconditionedIterativeKrylovStateReader<FastMultipoleEBI::ValueType,
                                                          KPreconditionedGeneralizedMinimalResidual,
                                                          KPreconditionedGeneralizedMinimalResidualState>(
                fm_integrator->GetLabels());
        checkpoint_reader->SetVerbosity(fSolverParameters.verbosity);
        pgmres.AddVisitor(checkpoint_reader);
    }

    MPI_SINGLE_PROCESS
    {
        if (fUseCheckpoints) {
            //do not need to delete this visitor, as it is deleted by the solver
            auto* checkpoint = new KPreconditionedIterativeKrylovStateWriter<FastMultipoleEBI::ValueType,
                                                                             KPreconditionedGeneralizedMinimalResidual>(
                fm_integrator->GetLabels());
            checkpoint->Interval(fCheckpointFrequency);
            pgmres.AddVisitor(checkpoint);
        }

        if (fUseDisplay) {
            pgmres.AddVisitor(new KIterationDisplay<double>("GMRES: "));
        }
#ifdef KEMFIELD_USE_VTK
        if (fUsePlot) {
            pgmres.AddVisitor(new KVTKIterationPlotter<double>());
        }
#endif
    }

#ifdef KEMFIELD_USE_REALTIME_CLOCK
    timespec start, end;
#endif
    clock_t cstart = clock();
    clock_t cend = clock();
    double time;

    MPI_SINGLE_PROCESS
    {
        if (fUseTimer) {
//timer
#ifdef KEMFIELD_USE_REALTIME_CLOCK
            clock_gettime(CLOCK_REALTIME, &start);
#endif
            cstart = clock();
        }
    }

    pgmres.Solve(fmA, *IKP, fmx, fmb);
    fResidualNorm = pgmres.ResidualNorm();

    MPI_SINGLE_PROCESS
    {
        if (fUseTimer) {
#ifdef KEMFIELD_USE_REALTIME_CLOCK
            clock_gettime(CLOCK_REALTIME, &end);
            timespec temp = TimeDifference(start, end);
            kfmout << "Real time required to solve (sec): " << temp.tv_sec << "." << temp.tv_nsec << kfmendl;
#endif
            cend = clock();
            time = ((double) (cend - cstart)) / CLOCKS_PER_SEC;  // time in seconds
            kfmout << "Process/CPU time required to solve (sec): " << time << kfmendl;
        }
    }

    delete fm_integrator;
    delete alt_fm_integrator;
}

void KFMElectrostaticFastMultipoleBoundaryValueSolver::SolveGMRES_IndependentImplicitKrylov(
    KSurfaceContainer& surfaceContainer)
{
    auto* fm_integrator = new FastMultipoleEBI(fDirectIntegrator, surfaceContainer);
    fm_integrator->Initialize(fSolverParameters);

    auto* alt_fm_integrator = new FastMultipoleEBI(fDirectIntegrator, surfaceContainer);
    alt_fm_integrator->Initialize(fPreconditionerParameters);

    FastMultipoleDenseMatrix denseA(*fm_integrator);
    FastMultipoleSparseMatrix sparseA(surfaceContainer, *fm_integrator);

    FastMultipoleDenseMatrix alt_denseA(*alt_fm_integrator);
    FastMultipoleSparseMatrix alt_sparseA(surfaceContainer, *alt_fm_integrator);

    FastMultipoleMatrix fmA(denseA, sparseA);
    std::shared_ptr<FastMultipoleMatrix> alt_fmA(new FastMultipoleMatrix(alt_denseA, alt_sparseA));

    KBoundaryIntegralSolutionVector<FastMultipoleEBI> fmx(surfaceContainer, *fm_integrator);
    KBoundaryIntegralVector<FastMultipoleEBI> fmb(surfaceContainer, *fm_integrator);

    auto restart_cond = std::make_shared<KIterativeKrylovRestartCondition>();
    restart_cond->SetNumberOfIterationsBetweenRestart(fIterationsBetweenRestart);

    KPreconditionedIterativeKrylovSolver<FastMultipoleEBI::ValueType, KPreconditionedGeneralizedMinimalResidual> pgmres;
    pgmres.SetTolerance(fSolverTolerance);
    pgmres.SetRestartCondition(restart_cond);
    pgmres.SetMaximumIterations(fMaxSolverIterations);

    if (fUseTimer) {
        pgmres.AddVisitor(new KTimeTerminator<FastMultipoleEBI::ValueType>(fMaxTimeAllowed, fTimeCheckFrequency));
    }

    KKrylovSolverConfiguration config;
    config.SetSolverName("gmres");
    config.SetMaxIterations(fMaxPreconditionerIterations);
    config.SetTolerance(fPreconditionerTolerance);
    config.SetUseDisplay(fUseDisplay);
    config.SetDisplayName("Preconditioner: ");
    auto IKP = KBuildKrylovPreconditioner<FastMultipoleEBI::ValueType>(config, alt_fmA);

    if (fUseCheckpoints) {
        //do not need to delete this visitor, as it is deleted by the solver
        auto* checkpoint_reader =
            new KPreconditionedIterativeKrylovStateReader<FastMultipoleEBI::ValueType,
                                                          KPreconditionedGeneralizedMinimalResidual,
                                                          KPreconditionedGeneralizedMinimalResidualState>(
                fm_integrator->GetLabels());
        checkpoint_reader->SetVerbosity(fSolverParameters.verbosity);
        pgmres.AddVisitor(checkpoint_reader);
    }

    MPI_SINGLE_PROCESS
    {
        if (fUseCheckpoints) {
            //do not need to delete this visitor, as it is deleted by the solver
            auto* checkpoint = new KPreconditionedIterativeKrylovStateWriter<FastMultipoleEBI::ValueType,
                                                                             KPreconditionedGeneralizedMinimalResidual>(
                fm_integrator->GetLabels());
            checkpoint->Interval(fCheckpointFrequency);
            pgmres.AddVisitor(checkpoint);
        }

        if (fUseDisplay) {
            pgmres.AddVisitor(new KIterationDisplay<double>("GMRES: "));
        }
#ifdef KEMFIELD_USE_VTK
        if (fUsePlot) {
            pgmres.AddVisitor(new KVTKIterationPlotter<double>());
        }
#endif
    }

#ifdef KEMFIELD_USE_REALTIME_CLOCK
    timespec start, end;
#endif
    clock_t cstart = clock();
    clock_t cend = clock();
    double time;

    MPI_SINGLE_PROCESS
    {
        if (fUseTimer) {
//timer
#ifdef KEMFIELD_USE_REALTIME_CLOCK
            clock_gettime(CLOCK_REALTIME, &start);
#endif
            cstart = clock();
        }
    }

    pgmres.Solve(fmA, *IKP, fmx, fmb);
    fResidualNorm = pgmres.ResidualNorm();

    MPI_SINGLE_PROCESS
    {
        if (fUseTimer) {
#ifdef KEMFIELD_USE_REALTIME_CLOCK
            clock_gettime(CLOCK_REALTIME, &end);
            timespec temp = TimeDifference(start, end);
            kfmout << "Real time required to solve (sec): " << temp.tv_sec << "." << temp.tv_nsec << kfmendl;
#endif
            cend = clock();
            time = ((double) (cend - cstart)) / CLOCKS_PER_SEC;  // time in seconds
            kfmout << "Process/CPU time required to solve (sec): " << time << kfmendl;
        }
    }

    delete fm_integrator;
    delete alt_fm_integrator;
}

void KFMElectrostaticFastMultipoleBoundaryValueSolver::SolveBICGSTAB(KSurfaceContainer& surfaceContainer)
{
    auto* fm_integrator = new FastMultipoleEBI(fDirectIntegrator, surfaceContainer);
    fm_integrator->Initialize(fSolverParameters);

    FastMultipoleDenseMatrix denseA(*fm_integrator);
    FastMultipoleSparseMatrix sparseA(surfaceContainer, *fm_integrator);
    FastMultipoleMatrix fmA(denseA, sparseA);
    KBoundaryIntegralSolutionVector<FastMultipoleEBI> fmx(surfaceContainer, *fm_integrator);
    KBoundaryIntegralVector<FastMultipoleEBI> fmb(surfaceContainer, *fm_integrator);

    auto restart_cond = std::make_shared<KIterativeKrylovRestartCondition>();
    restart_cond->SetNumberOfIterationsBetweenRestart(fIterationsBetweenRestart);

    KSimpleIterativeKrylovSolver<FastMultipoleEBI::ValueType, KBiconjugateGradientStabilized> bicgstab;
    bicgstab.SetTolerance(fSolverTolerance);
    bicgstab.SetRestartCondition(restart_cond);
    bicgstab.SetMaximumIterations(fMaxSolverIterations);

    if (fUseTimer) {
        bicgstab.AddVisitor(new KTimeTerminator<FastMultipoleEBI::ValueType>(fMaxTimeAllowed, fTimeCheckFrequency));
    }

    if (fUseCheckpoints) {
        //do not need to delete this visitor, as it is deleted by the solver
        auto* checkpoint_reader =
            new KIterativeKrylovStateReader<FastMultipoleEBI::ValueType,
                                            KBiconjugateGradientStabilized,
                                            KBiconjugateGradientStabilizedState>(fm_integrator->GetLabels());
        checkpoint_reader->SetVerbosity(fSolverParameters.verbosity);
        bicgstab.AddVisitor(checkpoint_reader);
    }

    MPI_SINGLE_PROCESS
    {
        if (fUseCheckpoints) {
            //do not need to delete this visitor, as it is deleted by the solver
            auto* checkpoint =
                new KIterativeKrylovStateWriter<FastMultipoleEBI::ValueType, KBiconjugateGradientStabilized>(
                    fm_integrator->GetLabels());
            checkpoint->Interval(fCheckpointFrequency);
            bicgstab.AddVisitor(checkpoint);
        }

        if (fUseDisplay) {
            bicgstab.AddVisitor(new KIterationDisplay<double>(std::string("BICGSTAB: ")));
        }
#ifdef KEMFIELD_USE_VTK
        if (fUsePlot) {
            bicgstab.AddVisitor(new KVTKIterationPlotter<double>());
        }
#endif
    }

#ifdef KEMFIELD_USE_REALTIME_CLOCK
    timespec start, end;
#endif
    clock_t cstart = clock();
    clock_t cend = clock();
    double time;

    MPI_SINGLE_PROCESS
    {
        if (fUseTimer) {
//timer
#ifdef KEMFIELD_USE_REALTIME_CLOCK
            clock_gettime(CLOCK_REALTIME, &start);
#endif
            cstart = clock();
        }
    }

    bicgstab.Solve(fmA, fmx, fmb);
    fResidualNorm = bicgstab.ResidualNorm();

    MPI_SINGLE_PROCESS
    {
        if (fUseTimer) {
#ifdef KEMFIELD_USE_REALTIME_CLOCK
            clock_gettime(CLOCK_REALTIME, &end);
            timespec temp = TimeDifference(start, end);
            kfmout << "Real time required to solve (sec): " << temp.tv_sec << "." << temp.tv_nsec << kfmendl;
#endif
            cend = clock();
            time = ((double) (cend - cstart)) / CLOCKS_PER_SEC;  // time in seconds
            kfmout << "Process/CPU time required to solve (sec): " << time << kfmendl;
        }
    }

    delete fm_integrator;
}

void KFMElectrostaticFastMultipoleBoundaryValueSolver::SolveBICGSTAB_Jacobi(KSurfaceContainer& surfaceContainer)
{
    auto* fm_integrator = new FastMultipoleEBI(fDirectIntegrator, surfaceContainer);
    fm_integrator->Initialize(fSolverParameters);

    FastMultipoleDenseMatrix denseA(*fm_integrator);
    FastMultipoleSparseMatrix sparseA(surfaceContainer, *fm_integrator);
    FastMultipoleMatrix fmA(denseA, sparseA);

    KBoundaryIntegralSolutionVector<FastMultipoleEBI> fmx(surfaceContainer, *fm_integrator);
    KBoundaryIntegralVector<FastMultipoleEBI> fmb(surfaceContainer, *fm_integrator);

    auto restart_cond = std::make_shared<KIterativeKrylovRestartCondition>();
    restart_cond->SetNumberOfIterationsBetweenRestart(fIterationsBetweenRestart);

    KPreconditionedIterativeKrylovSolver<FastMultipoleEBI::ValueType, KPreconditionedBiconjugateGradientStabilized>
        pbicgstab;
    pbicgstab.SetTolerance(fSolverTolerance);
    pbicgstab.SetRestartCondition(restart_cond);
    pbicgstab.SetMaximumIterations(fMaxSolverIterations);

    if (fUseTimer) {
        pbicgstab.AddVisitor(new KTimeTerminator<FastMultipoleEBI::ValueType>(fMaxTimeAllowed, fTimeCheckFrequency));
    }


    KJacobiPreconditioner<FastMultipoleEBI::ValueType> P(fmA);

    if (fUseCheckpoints) {
        //do not need to delete this visitor, as it is deleted by the solver
        auto* checkpoint_reader =
            new KPreconditionedIterativeKrylovStateReader<FastMultipoleEBI::ValueType,
                                                          KPreconditionedBiconjugateGradientStabilized,
                                                          KBiconjugateGradientStabilizedState>(
                fm_integrator->GetLabels());
        checkpoint_reader->SetVerbosity(fSolverParameters.verbosity);
        pbicgstab.AddVisitor(checkpoint_reader);
    }

    MPI_SINGLE_PROCESS
    {
        if (fUseCheckpoints) {
            //do not need to delete this visitor, as it is deleted by the solver
            auto* checkpoint =
                new KPreconditionedIterativeKrylovStateWriter<FastMultipoleEBI::ValueType,
                                                              KPreconditionedBiconjugateGradientStabilized>(
                    fm_integrator->GetLabels());
            checkpoint->Interval(fCheckpointFrequency);
            pbicgstab.AddVisitor(checkpoint);
        }

        if (fUseDisplay) {
            pbicgstab.AddVisitor(new KIterationDisplay<double>(std::string("BICGSTAB")));
        }
#ifdef KEMFIELD_USE_VTK
        if (fUsePlot) {
            pbicgstab.AddVisitor(new KVTKIterationPlotter<double>());
        }
#endif
    }

#ifdef KEMFIELD_USE_REALTIME_CLOCK
    timespec start, end;
#endif
    clock_t cstart = clock();
    clock_t cend = clock();
    double time;

    MPI_SINGLE_PROCESS
    {
        if (fUseTimer) {
//timer
#ifdef KEMFIELD_USE_REALTIME_CLOCK
            clock_gettime(CLOCK_REALTIME, &start);
#endif
            cstart = clock();
        }
    }

    pbicgstab.Solve(fmA, P, fmx, fmb);
    fResidualNorm = pbicgstab.ResidualNorm();

    MPI_SINGLE_PROCESS
    {
        if (fUseTimer) {
#ifdef KEMFIELD_USE_REALTIME_CLOCK
            clock_gettime(CLOCK_REALTIME, &end);
            timespec temp = TimeDifference(start, end);
            kfmout << "Real time required to solve (sec): " << temp.tv_sec << "." << temp.tv_nsec << kfmendl;
#endif
            cend = clock();
            time = ((double) (cend - cstart)) / CLOCKS_PER_SEC;  // time in seconds
            kfmout << "Process/CPU time required to solve (sec): " << time << kfmendl;
        }
    }

    delete fm_integrator;
}

void KFMElectrostaticFastMultipoleBoundaryValueSolver::SolveBICGSTAB_ImplicitKrylov(KSurfaceContainer& surfaceContainer)
{
    auto* fm_integrator = new FastMultipoleEBI(fDirectIntegrator, surfaceContainer);
    fm_integrator->Initialize(fSolverParameters);

    KFMElectrostaticParameters alt_params = fSolverParameters;
    alt_params.degree = fPreconditionerDegree;
    auto* alt_fm_integrator = new FastMultipoleEBI(fDirectIntegrator, surfaceContainer);
    alt_fm_integrator->Initialize(alt_params, fm_integrator->GetTree());

    FastMultipoleDenseMatrix denseA(*fm_integrator);
    FastMultipoleDenseMatrix alt_denseA(*alt_fm_integrator);
    FastMultipoleSparseMatrix sparseA(surfaceContainer, *fm_integrator);
    FastMultipoleMatrix fmA(denseA, sparseA);
    std::shared_ptr<FastMultipoleMatrix> alt_fmA(new FastMultipoleMatrix(alt_denseA, sparseA));

    KBoundaryIntegralSolutionVector<FastMultipoleEBI> fmx(surfaceContainer, *fm_integrator);
    KBoundaryIntegralVector<FastMultipoleEBI> fmb(surfaceContainer, *fm_integrator);

    auto restart_cond = std::make_shared<KIterativeKrylovRestartCondition>();
    restart_cond->SetNumberOfIterationsBetweenRestart(fIterationsBetweenRestart);

    KPreconditionedIterativeKrylovSolver<FastMultipoleEBI::ValueType, KPreconditionedBiconjugateGradientStabilized>
        pbicgstab;
    pbicgstab.SetTolerance(fSolverTolerance);
    pbicgstab.SetRestartCondition(restart_cond);
    pbicgstab.SetMaximumIterations(fMaxSolverIterations);

    if (fUseTimer) {
        pbicgstab.AddVisitor(new KTimeTerminator<FastMultipoleEBI::ValueType>(fMaxTimeAllowed, fTimeCheckFrequency));
    }


    KKrylovSolverConfiguration config;
    config.SetSolverName("gmres");  // that's the way it was, I suspected bicgstab also, but there you go
    config.SetMaxIterations(fMaxPreconditionerIterations);
    config.SetTolerance(fPreconditionerTolerance);
    config.SetUseDisplay(fUseDisplay);
    config.SetDisplayName("Preconditioner: ");
    auto IKP = KBuildKrylovPreconditioner<FastMultipoleEBI::ValueType>(config, alt_fmA);

    if (fUseCheckpoints) {
        //do not need to delete this visitor, as it is deleted by the solver
        auto* checkpoint_reader =
            new KPreconditionedIterativeKrylovStateReader<FastMultipoleEBI::ValueType,
                                                          KPreconditionedBiconjugateGradientStabilized,
                                                          KBiconjugateGradientStabilizedState>(
                fm_integrator->GetLabels());
        checkpoint_reader->SetVerbosity(fSolverParameters.verbosity);
        pbicgstab.AddVisitor(checkpoint_reader);
    }

    MPI_SINGLE_PROCESS
    {
        if (fUseCheckpoints) {
            //do not need to delete this visitor, as it is deleted by the solver
            auto* checkpoint =
                new KPreconditionedIterativeKrylovStateWriter<FastMultipoleEBI::ValueType,
                                                              KPreconditionedBiconjugateGradientStabilized>(
                    fm_integrator->GetLabels());
            checkpoint->Interval(fCheckpointFrequency);
            pbicgstab.AddVisitor(checkpoint);
        }

        if (fUseDisplay) {
            pbicgstab.AddVisitor(new KIterationDisplay<double>("BICGSTAB: "));
        }
#ifdef KEMFIELD_USE_VTK
        if (fUsePlot) {
            pbicgstab.AddVisitor(new KVTKIterationPlotter<double>());
        }
#endif
    }

#ifdef KEMFIELD_USE_REALTIME_CLOCK
    timespec start, end;
#endif
    clock_t cstart = clock();
    clock_t cend = clock();
    double time;

    MPI_SINGLE_PROCESS
    {
        if (fUseTimer) {
//timer
#ifdef KEMFIELD_USE_REALTIME_CLOCK
            clock_gettime(CLOCK_REALTIME, &start);
#endif
            cstart = clock();
        }
    }

    pbicgstab.Solve(fmA, *IKP, fmx, fmb);
    fResidualNorm = pbicgstab.ResidualNorm();

    MPI_SINGLE_PROCESS
    {
        if (fUseTimer) {
#ifdef KEMFIELD_USE_REALTIME_CLOCK
            clock_gettime(CLOCK_REALTIME, &end);
            timespec temp = TimeDifference(start, end);
            kfmout << "Real time required to solve (sec): " << temp.tv_sec << "." << temp.tv_nsec << kfmendl;
#endif
            cend = clock();
            time = ((double) (cend - cstart)) / CLOCKS_PER_SEC;  // time in seconds
            kfmout << "Process/CPU time required to solve (sec): " << time << kfmendl;
        }
    }

    delete fm_integrator;
    delete alt_fm_integrator;
}

void KFMElectrostaticFastMultipoleBoundaryValueSolver::SolveBICGSTAB_IndependentImplicitKrylov(
    KSurfaceContainer& surfaceContainer)
{
    auto* fm_integrator = new FastMultipoleEBI(fDirectIntegrator, surfaceContainer);
    fm_integrator->Initialize(fSolverParameters);

    auto* alt_fm_integrator = new FastMultipoleEBI(fDirectIntegrator, surfaceContainer);
    alt_fm_integrator->Initialize(fPreconditionerParameters);

    FastMultipoleDenseMatrix denseA(*fm_integrator);
    FastMultipoleSparseMatrix sparseA(surfaceContainer, *fm_integrator);

    FastMultipoleDenseMatrix alt_denseA(*alt_fm_integrator);
    FastMultipoleSparseMatrix alt_sparseA(surfaceContainer, *alt_fm_integrator);

    FastMultipoleMatrix fmA(denseA, sparseA);
    std::shared_ptr<FastMultipoleMatrix> alt_fmA(new FastMultipoleMatrix(alt_denseA, alt_sparseA));

    KBoundaryIntegralSolutionVector<FastMultipoleEBI> fmx(surfaceContainer, *fm_integrator);
    KBoundaryIntegralVector<FastMultipoleEBI> fmb(surfaceContainer, *fm_integrator);

    auto restart_cond = std::make_shared<KIterativeKrylovRestartCondition>();
    restart_cond->SetNumberOfIterationsBetweenRestart(fIterationsBetweenRestart);

    KPreconditionedIterativeKrylovSolver<FastMultipoleEBI::ValueType, KPreconditionedBiconjugateGradientStabilized>
        pbicgstab;
    pbicgstab.SetTolerance(fSolverTolerance);
    pbicgstab.SetRestartCondition(restart_cond);
    pbicgstab.SetMaximumIterations(fMaxSolverIterations);

    if (fUseTimer) {
        pbicgstab.AddVisitor(new KTimeTerminator<FastMultipoleEBI::ValueType>(fMaxTimeAllowed, fTimeCheckFrequency));
    }

    KKrylovSolverConfiguration config;
    config.SetSolverName("gmres");  // sorry, that's the way it was
    config.SetMaxIterations(fMaxPreconditionerIterations);
    config.SetTolerance(fPreconditionerTolerance);
    config.SetUseDisplay(fUseDisplay);
    config.SetDisplayName("Preconditioner: ");
    auto IKP = KBuildKrylovPreconditioner<FastMultipoleEBI::ValueType>(config, alt_fmA);

    if (fUseCheckpoints) {
        //do not need to delete this visitor, as it is deleted by the solver
        auto* checkpoint_reader =
            new KPreconditionedIterativeKrylovStateReader<FastMultipoleEBI::ValueType,
                                                          KPreconditionedBiconjugateGradientStabilized,
                                                          KBiconjugateGradientStabilizedState>(
                fm_integrator->GetLabels());
        checkpoint_reader->SetVerbosity(fSolverParameters.verbosity);
        pbicgstab.AddVisitor(checkpoint_reader);
    }

    MPI_SINGLE_PROCESS
    {
        if (fUseCheckpoints) {
            //do not need to delete this visitor, as it is deleted by the solver
            auto* checkpoint =
                new KPreconditionedIterativeKrylovStateWriter<FastMultipoleEBI::ValueType,
                                                              KPreconditionedBiconjugateGradientStabilized>(
                    fm_integrator->GetLabels());
            checkpoint->Interval(fCheckpointFrequency);
            pbicgstab.AddVisitor(checkpoint);
        }

        if (fUseDisplay) {
            pbicgstab.AddVisitor(new KIterationDisplay<double>("BICGSTAB: "));
        }
#ifdef KEMFIELD_USE_VTK
        if (fUsePlot) {
            pbicgstab.AddVisitor(new KVTKIterationPlotter<double>());
        }
#endif
    }

#ifdef KEMFIELD_USE_REALTIME_CLOCK
    timespec start, end;
#endif
    clock_t cstart = clock();
    clock_t cend = clock();
    double time;

    MPI_SINGLE_PROCESS
    {
        if (fUseTimer) {
//timer
#ifdef KEMFIELD_USE_REALTIME_CLOCK
            clock_gettime(CLOCK_REALTIME, &start);
#endif
            cstart = clock();
        }
    }

    pbicgstab.Solve(fmA, *IKP, fmx, fmb);
    fResidualNorm = pbicgstab.ResidualNorm();

    MPI_SINGLE_PROCESS
    {
        if (fUseTimer) {
#ifdef KEMFIELD_USE_REALTIME_CLOCK
            clock_gettime(CLOCK_REALTIME, &end);
            timespec temp = TimeDifference(start, end);
            kfmout << "Real time required to solve (sec): " << temp.tv_sec << "." << temp.tv_nsec << kfmendl;
#endif
            cend = clock();
            time = ((double) (cend - cstart)) / CLOCKS_PER_SEC;  // time in seconds
            kfmout << "Process/CPU time required to solve (sec): " << time << kfmendl;
        }
    }

    delete fm_integrator;
    delete alt_fm_integrator;
}


unsigned int KFMElectrostaticFastMultipoleBoundaryValueSolver::DetermineSolverType()
{

    if (fSolverName == std::string("gmres") && fPreconditionerName == std::string("none")) {
        return 0;
    }

    if (fSolverName == std::string("gmres") && fPreconditionerName == std::string("jacobi")) {
        return 1;
    }

    if (fSolverName == std::string("gmres") && fPreconditionerName == std::string("implicit_krylov")) {
        return 2;
    }

    if (fSolverName == std::string("bicgstab") && fPreconditionerName == std::string("none")) {
        return 3;
    }

    if (fSolverName == std::string("bicgstab") && fPreconditionerName == std::string("jacobi")) {
        return 4;
    }

    if (fSolverName == std::string("bicgstab") && fPreconditionerName == std::string("implicit_krylov")) {
        return 5;
    }

    if (fSolverName == std::string("gmres") && fPreconditionerName == std::string("independent_implicit_krylov")) {
        return 6;
    }

    if (fSolverName == std::string("bicgstab") && fPreconditionerName == std::string("independent_implicit_krylov")) {
        return 7;
    }

    //some invalid name set, issue warning and use default solver
    MPI_SINGLE_PROCESS
    {
        kfmout << "KFMElectrostaticFastMultipoleBoundaryValueSolver::DetermineSolverType: ";
        kfmout << "Warning, invalid options set, using default.";
        kfmout << "No combination with solver name: " << fSolverName << " ";
        kfmout << "and preconditioner name: " << fPreconditionerName << " available." << kfmendl;
    }
    return 0;
}


#ifdef KEMFIELD_USE_REALTIME_CLOCK
timespec KFMElectrostaticFastMultipoleBoundaryValueSolver::TimeDifference(timespec start, timespec end)
{

    timespec temp;
    if ((end.tv_nsec - start.tv_nsec) < 0) {
        temp.tv_sec = end.tv_sec - start.tv_sec - 1;
        temp.tv_nsec = 1000000000 + end.tv_nsec - start.tv_nsec;
    }
    else {
        temp.tv_sec = end.tv_sec - start.tv_sec;
        temp.tv_nsec = end.tv_nsec - start.tv_nsec;
    }
    return temp;
}
#endif

std::shared_ptr<FastMultipoleMatrix> KFMElectrostaticFastMultipoleBoundaryValueSolver::CreateMatrix(
    const KSurfaceContainer& surfaceContainer, const std::shared_ptr<FastMultipoleEBI>& fm_integrator) const
{
    fm_integrator->Initialize(fSolverParameters);

    std::shared_ptr<FastMultipoleDenseMatrix> denseA(new FastMultipoleDenseMatrix(fm_integrator));
    std::shared_ptr<FastMultipoleSparseMatrix> sparseA(new FastMultipoleSparseMatrix(surfaceContainer, fm_integrator));

    return std::make_shared<FastMultipoleMatrix>(denseA, sparseA);
}

}  // namespace KEMField
