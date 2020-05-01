#include "KFMElectrostaticBoundaryIntegratorEngine_OpenCL.hh"

#include "KEMChunkedFileInterface.hh"
#include "KEMFileInterface.hh"
#include "KFMBatchedMultidimensionalFastFourierTransform_OpenCL.hh"
#include "KFMDenseBlockSparseMatrix.hh"
#include "KFMElectrostaticMultipoleBatchCalculator.hh"
#include "KFMElectrostaticMultipoleBatchCalculator_OpenCL.hh"
#include "KFMEmptyIdentitySetRemover.hh"
#include "KFMLeafConditionActor.hh"
#include "KFMMatrixOperations.hh"
#include "KFMMatrixVectorOperations.hh"
#include "KFMNodeFlagValueInspector.hh"
#include "KFMReducedScalarMomentRemoteToLocalConverter_OpenCL.hh"
#include "KFMScalarMomentLocalToLocalConverter_OpenCL.hh"
#include "KFMScalarMomentRemoteToLocalConverter.hh"
#include "KFMScalarMomentRemoteToLocalConverter_OpenCL.hh"
#include "KFMScalarMomentRemoteToRemoteConverter_OpenCL.hh"
#include "KFMSparseReducedScalarMomentRemoteToLocalConverter_OpenCL.hh"
#include "KFMVectorOperations.hh"
#include "KFMWorkLoadBalanceWeights.hh"

#ifdef KEMFIELD_USE_REALTIME_CLOCK
#include <time.h>
#endif

namespace KEMField
{


#ifdef USE_REDUCED_M2L
typedef KFMSparseReducedScalarMomentRemoteToLocalConverter_OpenCL<
    KFMElectrostaticNodeObjects, KFMElectrostaticMultipoleSet, KFMElectrostaticLocalCoefficientSet,
    KFMResponseKernel_3DLaplaceM2L, KFMELECTROSTATICS_DIM, KFMELECTROSTATICS_FLAGS>
    KFMElectrostaticRemoteToLocalConverter_OpenCL;

//typedef KFMReducedScalarMomentRemoteToLocalConverter_OpenCL<KFMElectrostaticNodeObjects, KFMElectrostaticMultipoleSet, KFMElectrostaticLocalCoefficientSet, KFMResponseKernel_3DLaplaceM2L, 3>
//KFMElectrostaticRemoteToLocalConverter_OpenCL;

#else

typedef KFMScalarMomentRemoteToLocalConverter_OpenCL<KFMElectrostaticNodeObjects, KFMElectrostaticMultipoleSet,
                                                     KFMElectrostaticLocalCoefficientSet,
                                                     KFMResponseKernel_3DLaplaceM2L, KFMELECTROSTATICS_DIM>
    KFMElectrostaticRemoteToLocalConverter_OpenCL;
#endif

typedef KFMScalarMomentLocalToLocalConverter_OpenCL<KFMElectrostaticNodeObjects, KFMElectrostaticLocalCoefficientSet,
                                                    KFMResponseKernel_3DLaplaceL2L, KFMELECTROSTATICS_DIM>
    KFMElectrostaticLocalToLocalConverter_OpenCL;

typedef KFMScalarMomentRemoteToRemoteConverter_OpenCL<KFMElectrostaticNodeObjects, KFMElectrostaticMultipoleSet,
                                                      KFMResponseKernel_3DLaplaceM2M, KFMELECTROSTATICS_DIM>
    KFMElectrostaticRemoteToRemoteConverter_OpenCL;


const std::string KFMElectrostaticBoundaryIntegratorEngine_OpenCL::fWeightFilePrefix = std::string("oclwlw_");

KFMElectrostaticBoundaryIntegratorEngine_OpenCL::KFMElectrostaticBoundaryIntegratorEngine_OpenCL()
{
    fTree = NULL;
    fContainer = NULL;

    fDegree = 0;
    fDivisions = 0;
    fZeroMaskSize = 0;
    fMaximumTreeDepth = 0;
    fVerbosity = 0;

    fBatchCalc = new KFMElectrostaticMultipoleBatchCalculator_OpenCL();
    //    fBatchCalc = new KFMElectrostaticMultipoleBatchCalculator();

    fMultipoleDistributor = new KFMElectrostaticElementMultipoleDistributor();
    fMultipoleDistributor->SetBatchCalculator(fBatchCalc);
    fElementNodeAssociator = new KFMElectrostaticElementNodeAssociator();

    fLocalCoeffInitializer = new KFMElectrostaticLocalCoefficientInitializer();
    fMultipoleInitializer = new KFMElectrostaticMultipoleInitializer();

    fLocalCoeffResetter = new KFMElectrostaticLocalCoefficientResetter();
    fMultipoleResetter = new KFMElectrostaticMultipoleResetter();

    fM2MConverter = new KFMElectrostaticRemoteToRemoteConverter_OpenCL();
    fM2LConverter = new KFMElectrostaticRemoteToLocalConverter_OpenCL();
    fL2LConverter = new KFMElectrostaticLocalToLocalConverter_OpenCL();

    //    fM2MConverter = new KFMElectrostaticRemoteToRemoteConverter();
    //    fM2LConverter = new KFMElectrostaticRemoteToLocalConverter();
    //    fL2LConverter = new KFMElectrostaticLocalToLocalConverter();
};

KFMElectrostaticBoundaryIntegratorEngine_OpenCL::~KFMElectrostaticBoundaryIntegratorEngine_OpenCL()
{
    delete fBatchCalc;
    delete fMultipoleDistributor;
    delete fElementNodeAssociator;
    delete fLocalCoeffInitializer;
    delete fMultipoleInitializer;
    delete fLocalCoeffResetter;
    delete fMultipoleResetter;
    delete fM2MConverter;
    delete fM2LConverter;
    delete fL2LConverter;
}

void KFMElectrostaticBoundaryIntegratorEngine_OpenCL::SetTree(KFMElectrostaticTree* tree)
{
    fTree = tree;
    KFMCube<KFMELECTROSTATICS_DIM>* world_cube;
    world_cube = KFMObjectRetriever<KFMElectrostaticNodeObjects, KFMCube<KFMELECTROSTATICS_DIM>>::GetNodeObject(
        tree->GetRootNode());
    fWorldLength = world_cube->GetLength();
};

//set parameters
void KFMElectrostaticBoundaryIntegratorEngine_OpenCL::SetParameters(KFMElectrostaticParameters params)
{
    fDegree = params.degree;
    fNTerms = (fDegree + 1) * (fDegree + 1);
    fDivisions = params.divisions;
    fZeroMaskSize = params.zeromask;
    fMaximumTreeDepth = params.maximum_tree_depth;
    fVerbosity = params.verbosity;

    if (fVerbosity > 4) {
        //print the parameters
        kfmout << "KFMElectrostaticBoundaryIntegratorEngine_OpenCL::SetParameters: divisions set to "
               << params.divisions << kfmendl;
        kfmout << "KFMElectrostaticBoundaryIntegratorEngine_OpenCL::SetParameters: degree set to " << params.degree
               << kfmendl;
        kfmout << "KFMElectrostaticBoundaryIntegratorEngine_OpenCL::SetParameters: zero mask size set to "
               << params.zeromask << kfmendl;
        kfmout << "KFMElectrostaticBoundaryIntegratorEngine_OpenCL::SetParameters: max tree depth set to "
               << params.maximum_tree_depth << kfmendl;
    }
}


void KFMElectrostaticBoundaryIntegratorEngine_OpenCL::InitializeMultipoleMoments()
{
    //the multipole coefficient initializer
    KFMElectrostaticMultipoleInitializer multipoleInitializer;
    multipoleInitializer.SetNumberOfTermsInSeries(fNTerms);

    //remove any pre-existing multipole expansions
    KFMNodeObjectRemover<KFMElectrostaticNodeObjects, KFMElectrostaticMultipoleSet> remover;
    fTree->ApplyCorecursiveAction(&remover);

    //condition for a node to have a multipole expansion is based on the non-zero multipole moment flag
    KFMNodeFlagValueInspector<KFMElectrostaticNodeObjects, KFMELECTROSTATICS_FLAGS> multipole_flag_condition;
    multipole_flag_condition.SetFlagIndex(1);
    multipole_flag_condition.SetFlagValue(1);

    //now we constuct the conditional actor
    KFMConditionalActor<KFMElectrostaticNode> conditional_actor;

    conditional_actor.SetInspectingActor(&multipole_flag_condition);
    conditional_actor.SetOperationalActor(&multipoleInitializer);

    //initialize multipole expansions for appropriate nodes
    fTree->ApplyRecursiveAction(&conditional_actor);

    if (fVerbosity > 4) {
        kfmout
            << "KFMElectrostaticBoundaryIntegratorEngine_OpenCL::InitializeMultipoleMoments: Done initializing multipole moment expansions."
            << kfmendl;
    }
}


void KFMElectrostaticBoundaryIntegratorEngine_OpenCL::InitializeLocalCoefficientsForPrimaryNodes()
{
    //sparse initialization only for primary nodes
    KFMElectrostaticLocalCoefficientInitializer localCoeffInitializer;
    localCoeffInitializer.SetNumberOfTermsInSeries(fNTerms);

    //delete all pre-existing local coefficient expansions
    KFMNodeObjectRemover<KFMElectrostaticNodeObjects, KFMElectrostaticLocalCoefficientSet> remover;
    fTree->ApplyCorecursiveAction(&remover);

    //the primacy flag inspector
    KFMNodeFlagValueInspector<KFMElectrostaticNodeObjects, KFMELECTROSTATICS_FLAGS> primacy_condition;
    primacy_condition.SetFlagIndex(0);
    primacy_condition.SetFlagValue(1);

    //now we constuct the conditional actor
    KFMConditionalActor<KFMElectrostaticNode> conditional_actor;
    conditional_actor.SetInspectingActor(&primacy_condition);
    conditional_actor.SetOperationalActor(&localCoeffInitializer);

    //initialize the local coefficient expansions of the primary nodes
    fTree->ApplyCorecursiveAction(&conditional_actor);

    if (fVerbosity > 4) {
        kfmout
            << "KFMElectrostaticBoundaryIntegratorEngine_OpenCL::InitializeLocalCoefficientsForPrimaryNodes: Done initializing local coefficient expansions."
            << kfmendl;
    }
}


void KFMElectrostaticBoundaryIntegratorEngine_OpenCL::Initialize()
{

    if (fVerbosity > 4) {
        kfmout
            << "KFMElectrostaticBoundaryIntegratorEngine_OpenCL::Initialize: Initializing the element multipole moment batch calculator. ";
    }

    fBatchCalc->SetDegree(fDegree);
    fBatchCalc->SetElectrostaticElementContainer(fContainer);
    fBatchCalc->Initialize();

    if (fVerbosity > 4) {
        kfmout << "Done." << kfmendl;
    }

    fLocalCoeffInitializer->SetNumberOfTermsInSeries(fNTerms);
    fMultipoleInitializer->SetNumberOfTermsInSeries(fNTerms);

    fLocalCoeffResetter->SetNumberOfTermsInSeries(fNTerms);
    fMultipoleResetter->SetNumberOfTermsInSeries(fNTerms);

    if (fVerbosity > 4) {
        kfmout
            << "KFMElectrostaticBoundaryIntegratorEngine_OpenCL::Initialize: Initializing the multipole to multipole translator. ";
    }

    fM2MConverter->SetNumberOfTermsInSeries(fNTerms);
    fM2MConverter->SetDivisions(fDivisions);
    fM2MConverter->Initialize();

    if (fVerbosity > 4) {
        kfmout << "Done." << kfmendl;
    }

    if (fVerbosity > 4) {
        kfmout
            << "KFMElectrostaticBoundaryIntegratorEngine_OpenCL::Initialize: Initializing the multipole to local translator. ";
    }

    fM2LConverter->SetLength(fWorldLength);
    fM2LConverter->SetMaxTreeDepth(fMaximumTreeDepth);
    fM2LConverter->SetNumberOfTermsInSeries(fNTerms);
    fM2LConverter->SetZeroMaskSize(fZeroMaskSize);
    fM2LConverter->SetDivisions(fDivisions);
    fM2LConverter->Initialize();

    if (fVerbosity > 4) {
        kfmout << "Done." << kfmendl;
    }

    if (fVerbosity > 4) {
        kfmout
            << "KFMElectrostaticBoundaryIntegratorEngine_OpenCL::Initialize: Initializing the local to local translator. ";
    }

    fL2LConverter->SetNumberOfTermsInSeries(fNTerms);
    fL2LConverter->SetDivisions(fDivisions);
    fL2LConverter->Initialize();

    if (fVerbosity > 4) {
        kfmout << "Done." << kfmendl;
    }

    AssociateElementsAndNodes();
}


void KFMElectrostaticBoundaryIntegratorEngine_OpenCL::MapField()
{
    ResetMultipoleMoments();
    ResetLocalCoefficients();
    ComputeMultipoleMoments();
    ComputeLocalCoefficients();
}

void KFMElectrostaticBoundaryIntegratorEngine_OpenCL::AssociateElementsAndNodes()
{
    fElementNodeAssociator->Clear();
    fTree->ApplyRecursiveAction(fElementNodeAssociator);

    fMultipoleDistributor->SetElementIDList(fElementNodeAssociator->GetElementIDList());
    fMultipoleDistributor->SetNodeList(fElementNodeAssociator->GetNodeList());
    fMultipoleDistributor->SetOriginList(fElementNodeAssociator->GetOriginList());

    if (fVerbosity > 4) {
        kfmout
            << "KFMElectrostaticBoundaryIntegratorEngine_OpenCL::AssociateElementsAndNodes: Done making element to node association. For "
            << fElementNodeAssociator->GetElementIDList()->size() << " elements." << kfmendl;
    }
}

void KFMElectrostaticBoundaryIntegratorEngine_OpenCL::ResetMultipoleMoments()
{
    //reset all pre-existing multipole expansions
    fTree->ApplyCorecursiveAction(fMultipoleResetter);

    if (fVerbosity > 4) {
        kfmout
            << "KFMElectrostaticBoundaryIntegratorEngine_OpenCL::ResetMultipoleMoments: Done reseting pre-exisiting multipole moments."
            << kfmendl;
    }
}

void KFMElectrostaticBoundaryIntegratorEngine_OpenCL::ComputeMultipoleMoments()
{
    //compute the individual multipole moments of each node due to owned electrodes
    fMultipoleDistributor->ProcessAndDistributeMoments();

    if (fVerbosity > 4) {
        kfmout
            << "KFMElectrostaticBoundaryIntegratorEngine_OpenCL::ComputeMultipoleMoments: Done processing and distributing boundary element moments."
            << kfmendl;
    }

    //now we perform the upward pass to collect child nodes' moments into their parents' moments
    fTree->ApplyRecursiveAction(fM2MConverter, false);  //false indicates this visitation proceeds from child to parent

    if (fVerbosity > 4) {
        kfmout
            << "KFMElectrostaticBoundaryIntegratorEngine_OpenCL::ComputeMultipoleMoments: Done performing the multipole to multipole (M2M) translations."
            << kfmendl;
    }
}

void KFMElectrostaticBoundaryIntegratorEngine_OpenCL::ResetLocalCoefficients()
{
    //reset all existing local coefficient expansions to zero
    fTree->ApplyCorecursiveAction(fLocalCoeffResetter);

    if (fVerbosity > 4) {
        kfmout << "KFMElectrostaticTreeManager::ResetLocalCoefficients: Done resetting local coefficients." << kfmendl;
    }
}

void KFMElectrostaticBoundaryIntegratorEngine_OpenCL::ComputeMultipoleToLocal()
{
    //compute the local coefficients due to neighbors at the same tree level
    fM2LConverter->Prepare(fTree);
    do {
        fTree->ApplyCorecursiveAction(fM2LConverter);
    } while (!(fM2LConverter->IsFinished()));
    fM2LConverter->Finalize(fTree);


    if (fVerbosity > 4) {
        kfmout
            << "KFMElectrostaticBoundaryIntegratorEngine_OpenCL::ComputeLocalCoefficients: Done performing the multipole to local (M2L) translations."
            << kfmendl;
    }
}

void KFMElectrostaticBoundaryIntegratorEngine_OpenCL::ComputeLocalToLocal()
{
    //now form the downward distributions of the local coefficients
    fTree->ApplyRecursiveAction(fL2LConverter);

    if (fVerbosity > 4) {
        kfmout
            << "KFMElectrostaticBoundaryIntegratorEngine_OpenCL::ComputeLocalCoefficients: Done performing the local to local (L2L) translations."
            << kfmendl;
    }
}


void KFMElectrostaticBoundaryIntegratorEngine_OpenCL::ComputeLocalCoefficients()
{
    ComputeMultipoleToLocal();
    ComputeLocalToLocal();
}

void KFMElectrostaticBoundaryIntegratorEngine_OpenCL::EvaluateWorkLoads(unsigned int divisions, unsigned int zeromask)
{
    //check if we have computed these parameters before
    //we need save and re-use the weight factors because if we don't
    //the tree structure construction will become non-deterministic
    //(since the weights might change slightly depending on system load)
    //if the tree structure isn't the same then any saved matrix files we
    //may have previously computed are invalid

    std::stringstream ss;
    ss << "d" << divisions;
    ss << "z" << zeromask;

    std::string filename = fWeightFilePrefix + ss.str() + std::string(".ksa");
    bool on_file = KEMFileInterface::GetInstance()->DoesFileExist(filename);

    if (on_file) {
        //read the weight file from disk
        bool result = false;
        KSAObjectInputNode<KFMWorkLoadBalanceWeights>* in_node;
        in_node = new KSAObjectInputNode<KFMWorkLoadBalanceWeights>(KSAClassName<KFMWorkLoadBalanceWeights>::name());
        KEMFileInterface::GetInstance()->ReadKSAFileFromActiveDirectory(in_node, filename, result);
        if (!result) {
            on_file = false;
        }
        else {
            fDiskWeight = in_node->GetObject()->GetDiskMatrixVectorProductWeight();
            fRamWeight = in_node->GetObject()->GetRamMatrixVectorProductWeight();
            fFFTWeight = in_node->GetObject()->GetFFTWeight();
        }
        delete in_node;
    }

    if (!on_file) {
        fDiskWeight = ComputeDiskMatrixVectorProductWeight();
        fRamWeight = ComputeRamMatrixVectorProductWeight();
        fFFTWeight = ComputeFFTWeight(divisions, zeromask);

        KFMWorkLoadBalanceWeights weights;
        weights.SetDivisions(divisions);
        weights.SetZeroMaskSize(zeromask);
        weights.SetDiskMatrixVectorProductWeight(fDiskWeight);
        weights.SetRamMatrixVectorProductWeight(fRamWeight);
        weights.SetFFTWeight(fFFTWeight);

        //now write out the values to disk so we can reuse them next time if needed
        KSAObjectOutputNode<KFMWorkLoadBalanceWeights>* out_node =
            new KSAObjectOutputNode<KFMWorkLoadBalanceWeights>(KSAClassName<KFMWorkLoadBalanceWeights>::name());
        out_node->AttachObjectToNode(&weights);
        bool result;
        KEMFileInterface::GetInstance()->SaveKSAFileToActiveDirectory(out_node, filename, result);
        delete out_node;
    }
}


double KFMElectrostaticBoundaryIntegratorEngine_OpenCL::ComputeDiskMatrixVectorProductWeight()
{
    //evaluate the time for a matrix-like chuck of multiply-adds
    //read off of a disk

    unsigned int n_samples = 100;
    unsigned int dim = KFMDenseBlockSparseMatrix<double>::GetSuggestedMaximumRowWidth();

    //create a temporary file
    std::string filename("disk_matrix_temp.wle");
    KEMChunkedFileInterface* fElementFileInterface = new KEMChunkedFileInterface();
    fElementFileInterface->OpenFileForWriting(filename);

    std::vector<double> tmp;
    size_t sz = n_samples * dim * dim;
    tmp.resize(sz, 1);
    fElementFileInterface->Write(sz, &(tmp[0]));
    fElementFileInterface->CloseFile();

    fElementFileInterface->OpenFileForReading(filename);

    kfm_vector* lhs = kfm_vector_alloc(dim);
    kfm_vector* rhs = kfm_vector_alloc(dim);
    kfm_matrix* mx = kfm_matrix_alloc(dim, dim);

    double time;
#ifdef KEMFIELD_USE_REALTIME_CLOCK
    timespec start, end;
    clock_gettime(CLOCK_REALTIME, &start);
#else
    clock_t cstart, cend;
    cstart = clock();
#endif

    for (unsigned int i = 0; i < n_samples; i++) {
        fElementFileInterface->Read(dim * dim, &(tmp[0]));
        kfm_matrix_vector_product(mx, rhs, lhs);
    }

#ifdef KEMFIELD_USE_REALTIME_CLOCK
    clock_gettime(CLOCK_REALTIME, &end);
    timespec temp = diff(start, end);
    time = (double) temp.tv_sec + (double) (temp.tv_nsec * 1e-9);
#else
    cend = clock();
    time = ((double) (cend - cstart)) / CLOCKS_PER_SEC;  // time in seconds
#endif

    //scale by number of samples and number of matrix elements
    double weight = time / (double) n_samples;
    weight /= ((double) (dim * dim));

    kfm_matrix_free(mx);
    kfm_vector_free(rhs);
    kfm_vector_free(lhs);

    fElementFileInterface->CloseFile();
    KEMFileInterface::GetInstance()->RemoveFileFromActiveDirectory(filename);

    return weight;
}

double KFMElectrostaticBoundaryIntegratorEngine_OpenCL::ComputeRamMatrixVectorProductWeight()
{
    //now we evaluate the time for a matrix like chuck of multiply-adds
    unsigned int n_samples = 100;
    unsigned int dim = KFMDenseBlockSparseMatrix<double>::GetSuggestedMaximumRowWidth();

    kfm_vector* lhs = kfm_vector_alloc(dim);
    kfm_vector* rhs = kfm_vector_alloc(dim);
    kfm_matrix* mx = kfm_matrix_alloc(dim, dim);

    double time;
#ifdef KEMFIELD_USE_REALTIME_CLOCK
    timespec start, end;
    clock_gettime(CLOCK_REALTIME, &start);
#else
    clock_t cstart, cend;
    cstart = clock();
#endif

    for (unsigned int i = 0; i < n_samples; i++) {
        kfm_matrix_vector_product(mx, rhs, lhs);
    }

#ifdef KEMFIELD_USE_REALTIME_CLOCK
    clock_gettime(CLOCK_REALTIME, &end);
    timespec temp = diff(start, end);
    time = (double) temp.tv_sec + (double) (temp.tv_nsec * 1e-9);
#else
    cend = clock();
    time = ((double) (cend - cstart)) / CLOCKS_PER_SEC;  // time in seconds
#endif

    //scale by number of samples and number of matrix elements
    double weight = time / (double) n_samples;
    weight /= ((double) (dim * dim));

    kfm_matrix_free(mx);
    kfm_vector_free(rhs);
    kfm_vector_free(lhs);

    return weight;
}

double KFMElectrostaticBoundaryIntegratorEngine_OpenCL::ComputeFFTWeight(unsigned int divisions, unsigned int zeromask)
{
    unsigned int n_samples = 100;
    unsigned int fft_batch_size = 100;
    unsigned int spatial = 2 * divisions * (zeromask + 1);
    unsigned int dim_size[4] = {fft_batch_size, spatial, spatial, spatial};

    //evaluate time taken for opencl fft's
    unsigned int total_size_gpu = dim_size[0] * dim_size[1] * dim_size[2] * dim_size[3];
    std::complex<double>* raw_data_gpu = new std::complex<double>[total_size_gpu];
    KFMArrayWrapper<std::complex<double>, 4> input_gpu(raw_data_gpu, dim_size);

    KFMBatchedMultidimensionalFastFourierTransform_OpenCL<3>* fft_gpu =
        new KFMBatchedMultidimensionalFastFourierTransform_OpenCL<3>();

    fft_gpu->SetForward();
    fft_gpu->SetInput(&input_gpu);
    fft_gpu->SetOutput(&input_gpu);
    fft_gpu->Initialize();

    double time;
#ifdef KEMFIELD_USE_REALTIME_CLOCK
    timespec start, end;
    clock_gettime(CLOCK_REALTIME, &start);
#else
    clock_t cstart, cend;
    cstart = clock();
#endif

    for (unsigned int s = 0; s < n_samples; s++) {
        fft_gpu->ExecuteOperation();
    }

#ifdef KEMFIELD_USE_REALTIME_CLOCK
    clock_gettime(CLOCK_REALTIME, &end);
    timespec temp = diff(start, end);
    time = (double) temp.tv_sec + (double) (temp.tv_nsec * 1e-9);
#else
    cend = clock();
    time = ((double) (cend - cstart)) / CLOCKS_PER_SEC;  // time in seconds
#endif

    double weight = time / ((double) (n_samples * fft_batch_size));

    delete fft_gpu;
    delete[] raw_data_gpu;

    return weight;
}


}  // namespace KEMField
