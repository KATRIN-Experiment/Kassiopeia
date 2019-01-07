#ifndef KFMElectrostaticBoundaryIntegrator_MPI_HH__
#define KFMElectrostaticBoundaryIntegrator_MPI_HH__

#include "KBoundaryIntegralVector.hh"

#include "KElectrostaticBoundaryIntegratorFactory.hh"
#include "KElectrostaticIntegratingFieldSolver.hh"

#include "KFMElectrostaticTree.hh"
#include "KFMElectrostaticTreeBuilder_MPI.hh"

#include "KFMElectrostaticBoundaryIntegratorEngine_SingleThread.hh"

#include "KFMElectrostaticFastMultipoleFieldSolver.hh"
#include "KFMElectrostaticLocalCoefficientFieldCalculator.hh"

#include "KFMElectrostaticSurfaceConverter.hh"
#include "KFMElectrostaticElementContainer.hh"

#include "KFMElectrostaticParameters.hh"
#include "KFMElectrostaticTreeInformationExtractor.hh"

#include "KFMNodeFlagValueInspector.hh"
#include "KFMInsertionCondition.hh"
#include "KFMSubdivisionCondition.hh"
#include "KFMSubdivisionConditionAggressive.hh"
#include "KFMSubdivisionConditionBalanced.hh"
#include "KFMSubdivisionConditionGuided.hh"

#include "KFMDenseBlockSparseMatrixGenerator.hh"

#include <utility>

#include "KMD5HashGenerator.hh"
#include "KMPIEnvironment.hh"

namespace KEMField
{


/*
*
*@file KFMElectrostaticBoundaryIntegrator_MPI.hh
*@class KFMElectrostaticBoundaryIntegrator_MPI
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Fri Jan 31 11:33:06 EST 2014 J. Barrett (barrettj@mit.edu) First Version
*
*/

template<typename ParallelTrait = KFMElectrostaticBoundaryIntegratorEngine_SingleThread>
class KFMElectrostaticBoundaryIntegrator_MPI: public KElectrostaticBoundaryIntegrator
{
    public:



        KFMElectrostaticBoundaryIntegrator_MPI(KElectrostaticBoundaryIntegrator directIntegrator, const KSurfaceContainer& surfaceContainer):
            KElectrostaticBoundaryIntegrator(directIntegrator),
            fInitialized(false),
            fSurfaceContainer(surfaceContainer),
            fTrait(NULL)
        {
            fUniqueID = "INVALID_ID";
            fTree = NULL;
            fElementContainer = NULL;
            fTreeIsOwned = true;
            fDimension = fSurfaceContainer.size();
            fSubdivisionCondition = NULL;

            if(KMPIInterface::GetInstance()->SplitMode())
            {
                if( KMPIInterface::GetInstance()->IsEvenGroupMember() )
                {
                    fTrait = new ParallelTrait();
                }
            }
            else
            {
                fTrait = new ParallelTrait();
            }
        };


        KFMElectrostaticBoundaryIntegrator_MPI(const KSurfaceContainer& surfaceContainer):
            KElectrostaticBoundaryIntegrator(KEBIFactory::MakeDefaultForFFTM()),
            fInitialized(false),
            fSurfaceContainer(surfaceContainer),
            fTrait(NULL)
        {
            fUniqueID = "INVALID_ID";
            fTree = NULL;
            fElementContainer = NULL;
            fTreeIsOwned = true;
            fDimension = fSurfaceContainer.size();
            fSubdivisionCondition = NULL;

            if(KMPIInterface::GetInstance()->SplitMode())
            {
                if( KMPIInterface::GetInstance()->IsEvenGroupMember() )
                {
                    fTrait = new ParallelTrait();
                }
            }
            else
            {
                fTrait = new ParallelTrait();
            }
        };


        virtual ~KFMElectrostaticBoundaryIntegrator_MPI()
        {
            if(fTreeIsOwned)
            {
                //reset the node's ptr to the element container to null
                KFMNodeObjectNullifier<KFMElectrostaticNodeObjects, KFMElectrostaticElementContainerBase<3,1> > elementContainerNullifier;
                fTree->ApplyCorecursiveAction(&elementContainerNullifier);
                delete fTree;
                delete fElementContainer;
                delete fSubdivisionCondition;
            }
            delete fTrait;
        };

        unsigned int GetVerbosity() const { return fParameters.verbosity;};

        //for hash identification
        std::string GetUniqueIDString() const {return fUniqueID;};
        std::string GetGeometryHash(){return fGeometryHash;};
        std::string GetBoundaryConditionHash(){return fBoundaryConditionHash;}
        std::string GetTreeParameterHash(){return fTreeParameterHash;};
        std::vector< std::string > GetLabels()
        {
            std::vector< std::string > labels;
            labels.push_back(fGeometryHash);
            labels.push_back(fBoundaryConditionHash);
            labels.push_back(fTreeParameterHash);
            return labels;
        }

        //size of the surface container
        unsigned int Dimension() const {return fDimension;};

        //initialize and construct new tree
        void Initialize(const KFMElectrostaticParameters& params)
        {
            if(!fInitialized)
            {
                fParameters = params;
                InitializeSubdivisionCondition();

                //resize the boundary conditioner vector to the dimension of the problem
                fBCIn.resize(fDimension);
                fBCOut.resize(fDimension);

                ComputeUniqueHash(fSurfaceContainer, params);

                fTree = new KFMElectrostaticTree();
                fTreeIsOwned = true;
                fTree->SetParameters(fParameters);
                fTree->SetUniqueID(fUniqueID);

                if(fParameters.verbosity > 4)
                {
                    MPI_SINGLE_PROCESS
                    {
                    kfmout<<"KFMElectrostaticBoundaryIntegrator_MPI::Initialize: Initializing electrostatic fast multipole boundary integrator."<<kfmendl;
                    }
                }

                if(fParameters.verbosity > 2)
                {
                    MPI_SINGLE_PROCESS
                    {
                    kfmout<<"KFMElectrostaticBoundaryIntegrator_MPI::Initialize: Extracting surface container data."<<kfmendl;
                    }
                }

                //we have a surface container with a bunch of electrode discretizations
                //we just want to convert these into point clouds, and then bounding balls
                //so extract the information we want
                fElementContainer = new KFMElectrostaticElementContainer<3,1>();
                fSurfaceConverter.SetSurfaceContainer(&fSurfaceContainer);
                fSurfaceConverter.SetElectrostaticElementContainer(fElementContainer);
                fSurfaceConverter.Extract();

                //create the tree builder
                fTreeBuilder.SetSubdivisionCondition(fSubdivisionCondition);
                fTreeBuilder.SetFFTWeight(fFFTWeight);
                fTreeBuilder.SetSparseMatrixWeight(fSparseMatrixWeight);
                fTreeBuilder.SetElectrostaticElementContainer(fElementContainer);
                fTreeBuilder.SetTree(fTree);

                //first we construct the tree's global structure
                fTreeBuilder.ConstructRootNode();
                fTreeBuilder.PerformSpatialSubdivision();
                fTreeBuilder.FlagNonZeroMultipoleNodes();
                fTreeBuilder.PerformAdjacencySubdivision();
                fTreeBuilder.FlagPrimaryNodes();

                //now determine the set of nodes which are relevant to this mpi process
                fTreeBuilder.DetermineSourceNodes();
                fTreeBuilder.DetermineTargetNodes();

                //now we eliminate unneeded data in the tree
                //to concentrate only on data relevant to this process
                //this also unflags nodes that this process is not responsible for
                fTreeBuilder.RemoveExtraneousData();

                //remove the unneeded bounding balls from the element container
                fElementContainer->ClearBoundingBalls();

                //proceed with the collection of the needed identities
                fTreeBuilder.CollectDirectCallIdentitiesForPrimaryNodes();

                //get source node indexes
                fNChildren = fTree->GetRootNode()->GetNChildren();
                fTreeBuilder.GetSourceNodeIndexes(&fSourceNodeIndexes);
                for(unsigned int i=0; i<fNChildren; i++)
                {
                    bool is_present = false;
                    for(unsigned int j=0; j<fSourceNodeIndexes.size(); j++)
                    {
                        if(i == fSourceNodeIndexes[j])
                        {
                            is_present = true;
                        }
                    }

                    if(!is_present)
                    {
                        fNonSourceNodeIndexes.push_back(i);
                    }
                }

                fTreeBuilder.GetTargetNodeIndexes(&fTargetNodeIndexes);
                for(unsigned int i=0; i<fNChildren; i++)
                {
                    bool is_present = false;
                    for(unsigned int j=0; j<fTargetNodeIndexes.size(); j++)
                    {
                        if(i == fTargetNodeIndexes[j])
                        {
                            is_present = true;
                        }
                    }

                    if(!is_present)
                    {
                        fNonTargetNodeIndexes.push_back(i);
                    }
                }

                //create a list of all the primary nodes at the top level
                std::vector<unsigned int> tlpni; //top level primary node indexes

                //flag inspector determines if a node is primary or not
                KFMNodeFlagValueInspector<KFMElectrostaticNodeObjects, KFMELECTROSTATICS_FLAGS> primary_flag_condition;
                primary_flag_condition.SetFlagIndex(0);
                primary_flag_condition.SetFlagValue(1);

                for(unsigned int i=0; i<fNChildren; i++)
                {
                    KFMElectrostaticNode* node = fTree->GetRootNode()->GetChild(i);
                    if(primary_flag_condition.ConditionIsSatisfied(node) )
                    {
                        tlpni.push_back(node->GetIndex());
                    }
                }

                //now we have to determine which primary nodes
                //we want to assign to this process to recieve far-field contributions
                unsigned int n_nodes = tlpni.size();
                unsigned int n_processes = 0;
                unsigned int process_id = 0;
                MPI_Comm* subgroup_comm = NULL;
                if(KMPIInterface::GetInstance()->SplitMode())
                {
                    n_processes = KMPIInterface::GetInstance()->GetNSubGroupProcesses();
                    process_id = KMPIInterface::GetInstance()->GetSubGroupRank();
                    subgroup_comm = KMPIInterface::GetInstance()->GetSubGroupCommunicator();
                }
                else
                {
                    n_processes = KMPIInterface::GetInstance()->GetNProcesses();
                    process_id = KMPIInterface::GetInstance()->GetProcess();
                }

                for(unsigned int i=0; i<n_processes; i++)
                {
                    if(process_id == i)
                    {
                        for(unsigned int x=0; x<n_nodes; x++)
                        {
                            for(unsigned int y=0; y<fTargetNodeIndexes.size(); y++)
                            {
                                if(tlpni[x] == fTargetNodeIndexes[y])
                                {
                                    fOwnedTargetNodeIndexes.push_back(tlpni[x]);
                                    tlpni[x] = fNChildren+1;
                                    break;
                                }
                            }
                        }
                    }

                    if(KMPIInterface::GetInstance()->SplitMode())
                    {
                        MPI_Bcast(&(tlpni[0]), n_nodes, MPI_UNSIGNED, i, *subgroup_comm);
                    }
                    else
                    {
                        MPI_Bcast(&(tlpni[0]), n_nodes, MPI_UNSIGNED, i, MPI_COMM_WORLD);
                    }
                }

                //allocate space for the buffers
                fMomentSize = KFMScalarMultipoleExpansion::TriangleNumber(fParameters.degree+1);
                fBufferSize = fNChildren*fMomentSize;
                fLocalCoeffRealIn.resize(fBufferSize);
                fLocalCoeffRealOut.resize(fBufferSize);
                fLocalCoeffImagIn.resize(fBufferSize);
                fLocalCoeffImagOut.resize(fBufferSize);

                fTree->RestrictActionBehavior(false);

                //the parallel trait is responsible for computing
                //local coefficient field map everywhere it is needed (primary nodes)
                if( KMPIInterface::GetInstance()->SplitMode() )
                {
                    if(KMPIInterface::GetInstance()->IsEvenGroupMember())
                    {
                        fTrait->SetElectrostaticElementContainer(fElementContainer);
                        fTrait->SetParameters(params); //always set the parameters before setting the tree
                        fTrait->SetTree(fTree);
                        fTrait->InitializeMultipoleMoments();
                        fTrait->InitializeLocalCoefficientsForPrimaryNodes();
                        fTrait->Initialize();
                    }
                }
                else
                {
                    fTrait->SetElectrostaticElementContainer(fElementContainer);
                    fTrait->SetParameters(params); //always set the parameters before setting the tree
                    fTrait->SetTree(fTree);
                    fTrait->InitializeMultipoleMoments();
                    fTrait->InitializeLocalCoefficientsForPrimaryNodes();
                    fTrait->Initialize();
                }

                //extract information
                if(fParameters.verbosity > 2)
                {
                    std::stringstream header;
                    std::string msg;
                    if( KMPIInterface::GetInstance()->SplitMode() )
                    {
                        if(KMPIInterface::GetInstance()->IsEvenGroupMember())
                        {
                            KFMElectrostaticTreeInformationExtractor extractor;
                            extractor.SetDegree(fParameters.degree);
                            fTree->ApplyCorecursiveAction(&extractor);
                            header<< "****************************** Tree statistics from process #";
                            header<<KMPIInterface::GetInstance()->GetProcess()<<" ******************************"<<std::endl;
                            msg = header.str() + extractor.GetStatistics();
                        }
                    }
                    else
                    {
                        KFMElectrostaticTreeInformationExtractor extractor;
                        extractor.SetDegree(fParameters.degree);
                        fTree->ApplyCorecursiveAction(&extractor);
                        header<< "****************************** Tree statistics from process #";
                        header<<KMPIInterface::GetInstance()->GetProcess()<<" ******************************"<<std::endl;
                        msg = header.str() + extractor.GetStatistics();
                    }
                    KMPIInterface::GetInstance()->PrintMessage(msg);
                }


                //fast field solver (from local coeff)
                fFastFieldSolver.SetDegree(params.degree);

                ConstructElementNodeAssociation();

                if(fParameters.verbosity > 4)
                {
                    MPI_SINGLE_PROCESS
                    {
                    kfmout<<"KFMElectrostaticBoundaryIntegrator_MPI::Initialize: Done fast multipole boundary integrator intialization."<<kfmendl;
                    }
                }

                fInitialized = true;
            }
        }

        //initialize with an externally constructed tree
        void Initialize(const KFMElectrostaticParameters& params, KFMElectrostaticTree* tree)
        {
            if(!fInitialized)
            {
                fTree = tree;
                fTreeIsOwned = false;

                fParameters = params;
                //check to make sure parameters are compatible with the pre-constructed tree
                KFMElectrostaticParameters tree_params = fTree->GetParameters();

                bool isValid = true;
                if(params.top_level_divisions != tree_params.top_level_divisions){isValid = false;};
                if(params.divisions != tree_params.divisions){isValid = false;};
                if(params.degree > tree_params.degree){isValid = false;}; //this is the only meaningful parameter that is allowed to differ
                if(params.zeromask != tree_params.zeromask){isValid = false;};

                if( !(tree_params.use_region_estimation) )
                {
                    if( params.world_center_x != tree_params.world_center_x){isValid = false;};
                    if( params.world_center_y != tree_params.world_center_y){isValid = false;};
                    if( params.world_center_z != tree_params.world_center_z){isValid = false;};
                    if( params.world_length != tree_params.world_length){isValid = false;};
                }

                if(!isValid)
                {
                    kfmout<<"KFMElectrostaticBoundaryIntegrator_MPI::Initialize: Error, attempted to reused pre-constructed tree, but there is a parameter mismatch. "<<kfmendl;
                    KMPIInterface::GetInstance()->Finalize();
                    kfmexit(1);
                }

                ComputeUniqueHash(fSurfaceContainer, params);

                //resize the boundary conditioner vector to the dimension of the problem
                fBCIn.resize(fDimension);
                fBCOut.resize(fDimension);

                if(fParameters.verbosity > 4)
                {
                    MPI_SINGLE_PROCESS
                    {
                    kfmout<<"KFMElectrostaticBoundaryIntegrator_MPI::Initialize: Initializing electrostatic fast multipole boundary integrator."<<kfmendl;
                    }
                }

                if(fParameters.verbosity > 2)
                {
                    MPI_SINGLE_PROCESS
                    {
                    kfmout<<"KFMElectrostaticBoundaryIntegrator_MPI::Initialize: Extracting surface container data."<<kfmendl;
                    }
                }


                //get a pointer to the pre-existing element container
                fElementContainer =
                KFMObjectRetriever<KFMElectrostaticNodeObjects, KFMElectrostaticElementContainerBase<3,1> >::GetNodeObject(fTree->GetRootNode());

                //set up the surface converter w/ pre-existing element container
                //no need to extract the data, as this has already been done
                fSurfaceConverter.SetSurfaceContainer(&fSurfaceContainer);
                fSurfaceConverter.SetElectrostaticElementContainer(fElementContainer);

                //get number of children
                fNChildren = fTree->GetRootNode()->GetNChildren();

                //determine the indices of the 'source' nodes
                //the source nodes for this process should already be flagged
                KFMNodeFlagValueInspector<KFMElectrostaticNodeObjects, KFMELECTROSTATICS_FLAGS> multipole_flag_condition;
                multipole_flag_condition.SetFlagIndex(1);
                multipole_flag_condition.SetFlagValue(1);
                //determine the number of source nodes at the top level of the tree
                fSourceNodeIndexes.clear();
                for(unsigned int i=0; i<fNChildren; i++)
                {
                    if( multipole_flag_condition.ConditionIsSatisfied( fTree->GetRootNode()->GetChild(i) ) )
                    {
                        fSourceNodeIndexes.push_back(i);
                    }
                }

                //determine the indices of the 'target nodes'
                //create the neighbor finder
                KFMCubicSpaceNodeNeighborFinder<KFMELECTROSTATICS_DIM, KFMElectrostaticNodeObjects> neighbor_finder;

                //the primacy flag inspector (check if a node contains target points)
                KFMNodeFlagValueInspector<KFMElectrostaticNodeObjects, KFMELECTROSTATICS_FLAGS> primacy_inspector;
                primacy_inspector.SetFlagIndex(0);
                primacy_inspector.SetFlagValue(1);

                std::vector< KFMElectrostaticNode* > neighbors;

                //in this function we collect the neighbors of the source nodes
                fTargetNodeIndexes.clear();
                for(unsigned int i=0; i<fSourceNodeIndexes.size(); i++)
                {
                    neighbors.clear();
                    KFMElectrostaticNode* node = fTree->GetRootNode()->GetChild(fSourceNodeIndexes[i]);
                    neighbor_finder.GetAllNeighbors(node, fParameters.zeromask, &neighbors);
                    for(unsigned int j=0; j<neighbors.size(); j++)
                    {
                        if(neighbors[j] != NULL)
                        {
                            unsigned int index = neighbors[j]->GetIndex();
                            bool is_present = false;
                            for(unsigned int n=0; n < fTargetNodeIndexes.size(); n++)
                            {
                                if(index == fTargetNodeIndexes[n])
                                {
                                    is_present = true;
                                    break;
                                }
                            }

                            if(!is_present)
                            {
                                fTargetNodeIndexes.push_back(index);
                            }
                        }
                    }
                }

                //now determine the non-target node indexes
                for(unsigned int i=0; i<fNChildren; i++)
                {
                    bool is_present = false;
                    for(unsigned int j=0; j<fTargetNodeIndexes.size(); j++)
                    {
                        if(i == fTargetNodeIndexes[j])
                        {
                            is_present = true;
                        }
                    }

                    if(!is_present)
                    {
                        fNonTargetNodeIndexes.push_back(i);
                    }
                }

                //create a list of all the primary nodes at the top level
                std::vector<unsigned int> tlpni; //top level primary node indexes

                //flag inspector determines if a node is primary or not
                KFMNodeFlagValueInspector<KFMElectrostaticNodeObjects, KFMELECTROSTATICS_FLAGS> primary_flag_condition;
                primary_flag_condition.SetFlagIndex(0);
                primary_flag_condition.SetFlagValue(1);

                for(unsigned int i=0; i<fNChildren; i++)
                {
                    KFMElectrostaticNode* node = fTree->GetRootNode()->GetChild(i);
                    if(primary_flag_condition.ConditionIsSatisfied(node) )
                    {
                        tlpni.push_back(node->GetIndex());
                    }
                }

                //now we have to determine which primary nodes
                //we want to assign to this process to recieve far-field contributions
                unsigned int n_nodes = tlpni.size();
                unsigned int n_processes = 0;
                unsigned int process_id = 0;
                MPI_Comm* subgroup_comm = NULL;

                if(KMPIInterface::GetInstance()->SplitMode())
                {
                    n_processes = KMPIInterface::GetInstance()->GetNSubGroupProcesses();
                    process_id = KMPIInterface::GetInstance()->GetSubGroupRank();
                    subgroup_comm = KMPIInterface::GetInstance()->GetSubGroupCommunicator();
                }
                else
                {
                    n_processes = KMPIInterface::GetInstance()->GetNProcesses();
                    process_id = KMPIInterface::GetInstance()->GetProcess();
                }

                for(unsigned int i=0; i<n_processes; i++)
                {
                    if(process_id == i)
                    {
                        for(unsigned int x=0; x<n_nodes; x++)
                        {
                            for(unsigned int y=0; y<fTargetNodeIndexes.size(); y++)
                            {
                                if(tlpni[x] == fTargetNodeIndexes[y])
                                {
                                    fOwnedTargetNodeIndexes.push_back(tlpni[x]);
                                    tlpni[x] = fNChildren+1;
                                    break;
                                }
                            }
                        }
                    }

                    if(KMPIInterface::GetInstance()->SplitMode())
                    {
                        MPI_Bcast(&(tlpni[0]), n_nodes, MPI_UNSIGNED, i, *subgroup_comm);
                    }
                    else
                    {
                        MPI_Bcast(&(tlpni[0]), n_nodes, MPI_UNSIGNED, i, MPI_COMM_WORLD);
                    }
                }

                //allocate space for the buffers
                fMomentSize = KFMScalarMultipoleExpansion::TriangleNumber(fParameters.degree + 1);
                fBufferSize = fNChildren*fMomentSize;
                fLocalCoeffRealIn.resize(fBufferSize);
                fLocalCoeffRealOut.resize(fBufferSize);
                fLocalCoeffImagIn.resize(fBufferSize);
                fLocalCoeffImagOut.resize(fBufferSize);

                fTree->RestrictActionBehavior(false);

                //the parallel trait is responsible for computing
                //local coefficient field map everywhere it is needed (primary nodes)

                if(KMPIInterface::GetInstance()->SplitMode())
                {
                    if(KMPIInterface::GetInstance()->IsEvenGroupMember() )
                    {
                        fTrait->SetElectrostaticElementContainer(fElementContainer);
                        fTrait->SetParameters(fParameters); //always set parameters before setting the tree
                        fTrait->SetTree(fTree);
                        fTrait->Initialize();
                    }
                }
                else
                {
                    fTrait->SetElectrostaticElementContainer(fElementContainer);
                    fTrait->SetParameters(fParameters); //always set parameters before setting the tree
                    fTrait->SetTree(fTree);
                    fTrait->Initialize();
                }

                //extract information
                if(fParameters.verbosity > 4)
                {
                    std::stringstream header;
                    std::string msg;
                    if(KMPIInterface::GetInstance()->SplitMode())
                    {
                        if(KMPIInterface::GetInstance()->IsEvenGroupMember() )
                        {
                            KFMElectrostaticTreeInformationExtractor extractor;
                            extractor.SetDegree(fParameters.degree);
                            fTree->ApplyCorecursiveAction(&extractor);
                            header<< "****************************** Tree statistics from process #";
                            header<<KMPIInterface::GetInstance()->GetProcess()<<" ******************************"<<std::endl;
                            msg = header.str() + extractor.GetStatistics();
                        }
                    }
                    else
                    {
                        KFMElectrostaticTreeInformationExtractor extractor;
                        extractor.SetDegree(fParameters.degree);
                        fTree->ApplyCorecursiveAction(&extractor);
                        header<< "****************************** Tree statistics from process #";
                        header<<KMPIInterface::GetInstance()->GetProcess()<<" ******************************"<<std::endl;
                        msg = header.str() + extractor.GetStatistics();
                    }
                    KMPIInterface::GetInstance()->PrintMessage(msg);
                }

                //fast field solver (from local coeff)
                fFastFieldSolver.SetDegree(params.degree);

                ConstructElementNodeAssociation();

                if(fParameters.verbosity > 4)
                {
                    MPI_SINGLE_PROCESS
                    {
                    kfmout<<"KFMElectrostaticBoundaryIntegrator_MPI::Initialize: Done fast multipole boundary integrator intialization."<<kfmendl;
                    }
                }

                fInitialized = true;
            }

        }

        KFMElectrostaticTree* GetTree(){return fTree;};

        void Update(const KVector<ValueType>& x)
        {
            if(KMPIInterface::GetInstance()->SplitMode())
            {
                if(KMPIInterface::GetInstance()->IsEvenGroupMember() )
                {
                    fSurfaceConverter.UpdateBasisData(x);
                    //field mapping
                    fTrait->ResetMultipoleMoments();
                    fTrait->ResetLocalCoefficients();
                    fTrait->ComputeMultipoleMoments();
                    fTrait->ComputeMultipoleToLocal();
                    ReduceLocalCoefficients();
                    fTrait->ComputeLocalToLocal();
                    //update b.c. at all collocation points across processes
                    UpdateBoundaryConditions();
                }
            }
            else
            {
                fSurfaceConverter.UpdateBasisData(x);
                //field mapping
                fTrait->ResetMultipoleMoments();
                fTrait->ResetLocalCoefficients();
                fTrait->ComputeMultipoleMoments();
                fTrait->ComputeMultipoleToLocal();
                ReduceLocalCoefficients();
                fTrait->ComputeLocalToLocal();
                //update b.c. at all collocation points across processes
                UpdateBoundaryConditions();
            }
        }

        using KElectrostaticBoundaryIntegrator::BoundaryIntegral;

        ValueType BoundaryIntegral(unsigned int sourceIndex, unsigned int targetIndex)
        {
            return KElectrostaticBoundaryIntegrator::BoundaryIntegral( fSurfaceContainer[sourceIndex], sourceIndex, fSurfaceContainer[targetIndex], targetIndex );
        }

        ValueType BoundaryIntegral(KSurfacePrimitive* /*target*/, unsigned int targetIndex)
        {
            //get the updated value
            return fBCOut[targetIndex];
        }

        ValueType BoundaryIntegral(unsigned int targetIndex)
        {
            //get the updated value
            return fBCOut[targetIndex];
        }

    protected:

    void
    InitializeSubdivisionCondition()
    {
        //construct the subdivision condition
        if(fParameters.strategy == KFMSubdivisionStrategy::Balanced )
        {
            KFMSubdivisionConditionBalanced<KFMELECTROSTATICS_DIM, KFMElectrostaticNodeObjects>* balancedSubdivision = new KFMSubdivisionConditionBalanced<KFMELECTROSTATICS_DIM, KFMElectrostaticNodeObjects>();

            double disk_weight = 0;
            double ram_weight = 0;
            double fft_weight = 0;

            //only root process determines the weights, we assume all nodes are similar
            //the weights must be the same for all nodes so that the tree is constructed consistently
            MPI_SINGLE_PROCESS
            {
                //determine how to weight the work load contributions
                fTrait->EvaluateWorkLoads(fParameters.divisions, fParameters.zeromask);
                disk_weight = fTrait->GetDiskWeight();
                ram_weight = fTrait->GetRamWeight();
                fft_weight = fTrait->GetFFTWeight();
            }

            //broadcast weight factors to all nodes
            MPI_Bcast(&disk_weight, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Bcast(&ram_weight, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Bcast(&fft_weight, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

            fFFTWeight = fft_weight;
            //use average as approximation since we do not
            //a priori know if the matrix will be kept in ram or on disk
            fSparseMatrixWeight = (ram_weight + disk_weight)/2.0;

            //set the work load weights
            balancedSubdivision->SetDiskWeight(disk_weight);
            balancedSubdivision->SetRamWeight(ram_weight);
            balancedSubdivision->SetFFTWeight(fft_weight);
            balancedSubdivision->SetBiasDegree(fParameters.bias_degree);
            balancedSubdivision->SetDegree(fParameters.degree);
            fSubdivisionCondition = balancedSubdivision;
        }
        else if( fParameters.strategy == KFMSubdivisionStrategy::Guided )
        {
            KFMSubdivisionConditionGuided<KFMELECTROSTATICS_DIM, KFMElectrostaticNodeObjects>* guidedSubdivision = new KFMSubdivisionConditionGuided<KFMELECTROSTATICS_DIM, KFMElectrostaticNodeObjects>();
            guidedSubdivision->SetFractionForDivision(fParameters.allowed_fraction);
            guidedSubdivision->SetAllowedNumberOfElements(fParameters.allowed_number);
            fSubdivisionCondition = guidedSubdivision;
        }
        else
        {
            fSubdivisionCondition = new KFMSubdivisionConditionAggressive<KFMELECTROSTATICS_DIM, KFMElectrostaticNodeObjects>();
        }
    }

////////////////////////////////////////////////////////////////////////////////

        void ReduceLocalCoefficients()
        {
            //zero out the buffers
            for(unsigned int i=0; i<fBufferSize; i++)
            {
                fLocalCoeffRealIn[i] = 0.0;
                fLocalCoeffRealOut[i] = 0.0;
                fLocalCoeffImagIn[i] = 0.0;
                fLocalCoeffImagOut[i] = 0.0;
            }

            //refresh the top level local coefficients in the tree
            //this is necessary if we are using OpenCL
            fTrait->RecieveTopLevelLocalCoefficients();

            //get top level local coefficients from tree
            //ONLY FROM NODES THAT ARE NOT IN THE TARGET REGION!! (avoid double counting)
            for(unsigned int i=0; i<fNonTargetNodeIndexes.size(); i++)
            {
                unsigned int index = fNonTargetNodeIndexes[i];
                KFMElectrostaticNode* node = fTree->GetRootNode()->GetChild(index);

                KFMElectrostaticLocalCoefficientSet* set = NULL;
                set = KFMObjectRetriever<KFMElectrostaticNodeObjects, KFMElectrostaticLocalCoefficientSet>::GetNodeObject(node);

                if(set != NULL)
                {
                    unsigned int offset = index*fMomentSize;
                    for(unsigned int n=0; n<fMomentSize; n++)
                    {
                        fLocalCoeffRealIn[offset + n] = (*(set->GetRealMoments()))[n];
                        fLocalCoeffImagIn[offset + n] = (*(set->GetImaginaryMoments()))[n];
                    }
                }
            }

            //perform a reduction over all processes' local coefficient moments
            if(KMPIInterface::GetInstance()->SplitMode())
            {
                if(KMPIInterface::GetInstance()->IsEvenGroupMember() )
                {
                    MPI_Comm* subgroup_comm = KMPIInterface::GetInstance()->GetSubGroupCommunicator();
                    MPI_Allreduce( &(fLocalCoeffRealIn[0]), &(fLocalCoeffRealOut[0]), fBufferSize, MPI_DOUBLE, MPI_SUM, *subgroup_comm);
                    MPI_Allreduce( &(fLocalCoeffImagIn[0]), &(fLocalCoeffImagOut[0]), fBufferSize, MPI_DOUBLE, MPI_SUM, *subgroup_comm);
                }
            }
            else
            {
                MPI_Allreduce( &(fLocalCoeffRealIn[0]), &(fLocalCoeffRealOut[0]), fBufferSize, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                MPI_Allreduce( &(fLocalCoeffImagIn[0]), &(fLocalCoeffImagOut[0]), fBufferSize, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            }

            for(unsigned int i=0; i<fOwnedTargetNodeIndexes.size(); i++)
            {
                unsigned int index = fOwnedTargetNodeIndexes[i];
                KFMElectrostaticNode* node = fTree->GetRootNode()->GetChild(index);

                KFMElectrostaticLocalCoefficientSet* set = NULL;
                set = KFMObjectRetriever<KFMElectrostaticNodeObjects, KFMElectrostaticLocalCoefficientSet>::GetNodeObject(node);

                if(set != NULL)
                {
                    unsigned int offset = index*fMomentSize;
                    for(unsigned int n=0; n<fMomentSize; n++)
                    {
                        (*(set->GetRealMoments()))[n] += fLocalCoeffRealOut[offset + n];
                        (*(set->GetImaginaryMoments()))[n] += fLocalCoeffImagOut[offset + n];
                    }
                }
            }

            fTrait->SendTopLevelLocalCoefficients();
        }




////////////////////////////////////////////////////////////////////////////////

        void ConstructElementNodeAssociation()
        {
            //construct the target volume from the appropriate nodes
            fTargetVolume.Clear();
            for(unsigned int i=0; i<fTargetNodeIndexes.size(); i++)
            {
                KFMElectrostaticNode* node = fTree->GetRootNode()->GetChild(fTargetNodeIndexes[i]);

                KFMCube<KFMELECTROSTATICS_DIM>* cube = NULL;
                cube = KFMObjectRetriever<KFMElectrostaticNodeObjects, KFMCube<KFMELECTROSTATICS_DIM> >::GetNodeObject(node);

                if(cube != NULL)
                {
                    fTargetVolume.AddCube(cube);
                }
            }

            //we need to create of list of the collocation points that this process
            //is responsible for evaluating and then
            //associate each element's centroid with the node containing it

            //tree navigator
            KFMElectrostaticTreeNavigator navigator;

            unsigned int n_elements = fElementContainer->GetNElements();
            fNodes.clear();
            fValidCollocationPointIDs.clear();
            for(unsigned int i=0; i<n_elements; i++)
            {
                KFMPoint<KFMELECTROSTATICS_DIM>* centroid = fElementContainer->GetCentroid(i);

                if( fTargetVolume.PointIsInside( *(centroid) ) )
                {
                    fValidCollocationPointIDs.push_back(i);
                    navigator.SetPoint( centroid );
                    navigator.ApplyAction(fTree->GetRootNode());

                    if(navigator.Found())
                    {
                        fNodes.push_back( navigator.GetLeafNode() );
                    }
                    else
                    {
                        kfmout<<"KFMElectrostaticBoundaryIntegrator_MPI::ConstructElementNodeAssociation: Error, element centroid not found in region."<<kfmendl;

                        KMPIInterface::GetInstance()->Finalize();
                        kfmexit(1);
                    }
                }
            }

        }


        void UpdateBoundaryConditions()
        {
            //first set all the b.c's to zero
            unsigned int n_elements = fElementContainer->GetNElements();
            for(unsigned int i=0; i<n_elements; i++)
            {
                fBCIn[i] = 0.;
                fBCOut[i] = 0.;
            }

            //loop over all the collocation points that are in the target volume
            //and update their values due to the local coefficients of the nodes
            //the contain them

            unsigned int n_points = fValidCollocationPointIDs.size();
            for(unsigned int i=0; i<n_points; i++)
            {
                //element index
                unsigned int id = fValidCollocationPointIDs[i];

                //look up the node corresponding to this target
                KFMElectrostaticNode* node = fNodes[i];

                //retrieve the expansion origin
                KFMCube<3>* cube = KFMObjectRetriever<KFMElectrostaticNodeObjects, KFMCube<3> >::GetNodeObject(node);
                KFMPoint<3> origin = cube->GetCenter();

                //retrieve the local coefficients
                KFMElectrostaticLocalCoefficientSet* set;
                set = KFMObjectRetriever<KFMElectrostaticNodeObjects, KFMElectrostaticLocalCoefficientSet>::GetNodeObject(node);
                fFastFieldSolver.SetLocalCoefficients(set);
                fFastFieldSolver.SetExpansionOrigin(origin);

                //figure out the boundary element's type
                fSurfaceContainer.at(id)->Accept(fBoundaryVisitor);
                double ret_val = 0;
                if(fBoundaryVisitor.IsDirichlet())
                {
                    ret_val = fFastFieldSolver.Potential(fSurfaceContainer.at(id)->GetShape()->Centroid());
                }
                else
                {
                    KThreeVector field;
                    fFastFieldSolver.ElectricField(fSurfaceContainer.at(id)->GetShape()->Centroid(),field);
                    ret_val = field.Dot(fSurfaceContainer.at(id)->GetShape()->Normal());
                }

                fBCIn[id] = ret_val;
            }

            //now we reduce the boundary conditions across all processes

//            //DEBUG
//            double norm = 0.0;
//            for(unsigned int i=0; i<fDimension; i++)
//            {
//                norm += fBCIn[i]*fBCIn[i];
//            }
//            std::stringstream s;
//            s <<"Process # "<<KMPIInterface::GetInstance()->GetProcess()<<" dense mat-vec norm = "<<std::sqrt(norm)<<std::endl;
//            KMPIInterface::GetInstance()->PrintMessage(s.str());

            if(KMPIInterface::GetInstance()->SplitMode())
            {
                if(KMPIInterface::GetInstance()->IsEvenGroupMember() )
                {
                    MPI_Comm* subgroup_comm = KMPIInterface::GetInstance()->GetSubGroupCommunicator();
                    MPI_Allreduce( &(fBCIn[0]), &(fBCOut[0]), fDimension, MPI_DOUBLE, MPI_SUM, *subgroup_comm);
                }
            }
            else
            {
                MPI_Allreduce( &(fBCIn[0]), &(fBCOut[0]), fDimension, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            }
        }


////////////////////////////////////////////////////////////////////////////////

        void
        ComputeUniqueHash(const KSurfaceContainer& surfaceContainer, const KFMElectrostaticParameters& parameters)
        {
            int HashMaskedBits = 20;
            double HashThreshold = 1.e-14;

            // compute hash of the bare geometry
            KMD5HashGenerator tShapeHashGenerator;
            tShapeHashGenerator.MaskedBits( HashMaskedBits );
            tShapeHashGenerator.Threshold( HashThreshold );
            tShapeHashGenerator.Omit( Type2Type< KElectrostaticBasis >() );
            tShapeHashGenerator.Omit( Type2Type< KBoundaryType< KElectrostaticBasis, KDirichletBoundary > >() );
            tShapeHashGenerator.Omit( Type2Type< KBoundaryType< KElectrostaticBasis, KNeumannBoundary > >() );
            fGeometryHash = tShapeHashGenerator.GenerateHash( surfaceContainer );

            // compute hash of right hand size of the equation (boundary conditions)
            KBoundaryIntegralVector< KFMElectrostaticBoundaryIntegrator_MPI<ParallelTrait> > b(surfaceContainer, *this);
            KMD5HashGenerator tBCHashGenerator;
            tBCHashGenerator.MaskedBits( HashMaskedBits );
            tBCHashGenerator.Threshold( HashThreshold );
            fBoundaryConditionHash = tShapeHashGenerator.GenerateHash( b );

            // compute hash of the parameter values w/o the multipole expansion degree included
            KFMElectrostaticParameters params = parameters;

            //normally we set the degree parameter to zero so as to not affect the hash,
            //however, in the case of MPI the tree structure (and thus the sparse matrix files)
            //is dependent on the degree parameter through the load-balancing algorithm
            //so we need to include it when computing the hash

            KMD5HashGenerator parameterHashGenerator;
            parameterHashGenerator.MaskedBits( HashMaskedBits );
            parameterHashGenerator.Threshold( HashThreshold );
            fTreeParameterHash = parameterHashGenerator.GenerateHash( params );

            //construct a unique id by stripping the first 6 characters from the shape and parameter hashes
            std::stringstream ss;
            ss << fGeometryHash.substr(0,6);
            ss << fTreeParameterHash.substr(0,6);

            //now add the process specific parameters
            ss << "_mpi_";

            //get process id and number of processes
            unsigned int process_id = KMPIInterface::GetInstance()->GetProcess();
            unsigned int n_processes = KMPIInterface::GetInstance()->GetNProcesses();

            ss << n_processes;
            ss << "_";
            ss << process_id;

            fUniqueID = ss.str();

        }


////////////////////////////////////////////////////////////////////////////////
////////data and state
////////////////////////////////////////////////////////////////////////////////

        bool fInitialized;
        unsigned int fDimension;

        const KSurfaceContainer& fSurfaceContainer;
        KFMElectrostaticSurfaceConverter fSurfaceConverter;
        KFMElectrostaticElementContainerBase<3,1>* fElementContainer;

        std::string fUniqueID;
        std::string fGeometryHash;
        std::string fBoundaryConditionHash;
        std::string fTreeParameterHash;

        KFMElectrostaticTree* fTree;
        bool fTreeIsOwned;
        KFMElectrostaticParameters fParameters;
        ParallelTrait* fTrait;

        KFMSubdivisionCondition<KFMELECTROSTATICS_DIM, KFMElectrostaticNodeObjects>* fSubdivisionCondition;
        KFMElectrostaticTreeBuilder_MPI fTreeBuilder;
        double fFFTWeight;
        double fSparseMatrixWeight;

        //MPI Buffers
        unsigned int fNChildren;
        unsigned int fMomentSize;
        unsigned int fBufferSize;
        std::vector< unsigned int > fSourceNodeIndexes;
        std::vector< unsigned int > fNonSourceNodeIndexes;
        std::vector< unsigned int > fTargetNodeIndexes;
        std::vector< unsigned int > fNonTargetNodeIndexes;
        std::vector< unsigned int > fOwnedTargetNodeIndexes;
        std::vector< double > fLocalCoeffRealIn;
        std::vector< double > fLocalCoeffRealOut;
        std::vector< double > fLocalCoeffImagIn;
        std::vector< double > fLocalCoeffImagOut;
        std::vector< double > fBCIn;
        std::vector< double > fBCOut;


        //fast look-up for the node which contains the centroid of each element
        std::vector< KFMElectrostaticNode* > fNodes;
        std::vector< unsigned int > fValidCollocationPointIDs;
        KFMCubicVolumeCollection<KFMELECTROSTATICS_DIM> fTargetVolume;

        //compute the field from the local coefficients
        KFMElectrostaticLocalCoefficientFieldCalculator fFastFieldSolver;
};


}

#endif /* KFMElectrostaticBoundaryIntegrator_MPI_H__ */
