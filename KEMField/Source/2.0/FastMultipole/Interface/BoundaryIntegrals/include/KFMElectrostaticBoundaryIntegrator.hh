#ifndef KFMElectrostaticBoundaryIntegrator_HH__
#define KFMElectrostaticBoundaryIntegrator_HH__

#include "KSurfaceContainer.hh"

#include "KElectrostaticBoundaryIntegratorFactory.hh"
#include "KElectrostaticIntegratingFieldSolver.hh"

#include "KFMElectrostaticTree.hh"
#include "KFMElectrostaticTreeBuilder.hh"
#include "KFMElectrostaticBoundaryIntegratorEngine_SingleThread.hh"

#include "KFMElectrostaticFastMultipoleFieldSolver.hh"
#include "KFMElectrostaticLocalCoefficientFieldCalculator.hh"

#include "KFMElectrostaticSurfaceConverter.hh"
#include "KFMElectrostaticElementContainer.hh"

#include "KFMElectrostaticParameters.hh"
#include "KFMElectrostaticTreeInformationExtractor.hh"

#include "KFMInsertionCondition.hh"
#include "KFMSubdivisionCondition.hh"
#include "KFMSubdivisionConditionAggressive.hh"
#include "KFMSubdivisionConditionBalanced.hh"
#include "KFMSubdivisionConditionGuided.hh"


#include "KFMDenseBlockSparseMatrixGenerator.hh"

#include <utility>

#include "KBoundaryIntegralVector.hh"
#include "KMD5HashGenerator.hh"

namespace KEMField
{


/*
*
*@file KFMElectrostaticBoundaryIntegrator.hh
*@class KFMElectrostaticBoundaryIntegrator
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Fri Jan 31 11:33:06 EST 2014 J. Barrett (barrettj@mit.edu) First Version
*
*/

template<typename ParallelTrait = KFMElectrostaticBoundaryIntegratorEngine_SingleThread>
class KFMElectrostaticBoundaryIntegrator: public KElectrostaticBoundaryIntegrator
{
    public:

        KFMElectrostaticBoundaryIntegrator(const KSurfaceContainer& surfaceContainer):
        KElectrostaticBoundaryIntegrator(KEBIFactory::MakeDefaultForFFTM()),
        fInitialized(false),
        fSurfaceContainer(surfaceContainer)
        {
            fUniqueID = "INVALID_ID";
            fTree = NULL;
            fElementContainer = NULL;
            fTreeIsOwned = true;
            fSubdivisionCondition = NULL;
        };

        KFMElectrostaticBoundaryIntegrator(KElectrostaticBoundaryIntegrator directIntegrator, const KSurfaceContainer& surfaceContainer):
        KElectrostaticBoundaryIntegrator(directIntegrator),
        fInitialized(false),
        fSurfaceContainer(surfaceContainer)
        {
            fUniqueID = "INVALID_ID";
            fTree = NULL;
            fElementContainer = NULL;
            fTreeIsOwned = true;
            fSubdivisionCondition = NULL;
        };

        virtual ~KFMElectrostaticBoundaryIntegrator()
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
        unsigned int Dimension() const {return fSurfaceContainer.size();};

        //initialize and construct new tree
        void Initialize(const KFMElectrostaticParameters& params)
        {
            if(!fInitialized)
            {
                fParameters = params;

                InitializeSubdivisionCondition();

                ComputeUniqueHash(params);

                fTree = new KFMElectrostaticTree();
                fTreeIsOwned = true;

                fTree->SetParameters(fParameters);
                fTree->SetUniqueID(fUniqueID);

                if(fParameters.verbosity > 0)
                {
                    kfmout<<"KFMElectrostaticBoundaryIntegrator::Initialize: Initializing electrostatic fast multipole boundary integrator."<<kfmendl;
                }

                if(fParameters.verbosity > 2)
                {
                    kfmout<<"KFMElectrostaticBoundaryIntegrator::Initialize: Extracting surface container data."<<kfmendl;
                }

                //we have a surface container with a bunch of electrode discretizations
                //we just want to convert these into point clouds, and then bounding balls
                //so extract the information we want
                fElementContainer = new KFMElectrostaticElementContainer<3,1>();

                fSurfaceConverter.SetSurfaceContainer(&fSurfaceContainer);

                fSurfaceConverter.SetElectrostaticElementContainer(fElementContainer);
                fSurfaceConverter.Extract();

                //set up the tree builder
                fTreeBuilder.SetSubdivisionCondition(fSubdivisionCondition);
                fTreeBuilder.SetElectrostaticElementContainer(fElementContainer);
                fTreeBuilder.SetTree(fTree);

                //now we construct the tree's structure
                fTreeBuilder.ConstructRootNode();
                fTreeBuilder.PerformSpatialSubdivision();
                fTreeBuilder.FlagNonZeroMultipoleNodes();
                fTreeBuilder.PerformAdjacencySubdivision();
                fTreeBuilder.FlagPrimaryNodes();

                //remove the unneeded bounding balls from the element container
                fElementContainer->ClearBoundingBalls();

                //determine element ids for direct evaluation
                fTreeBuilder.CollectDirectCallIdentitiesForPrimaryNodes();

                //the parallel trait is responsible for computing
                //local coefficient field map everywhere it is needed (primary nodes)
                fTrait.SetElectrostaticElementContainer(fElementContainer);
                fTrait.SetParameters(params); //always set the parameters before setting the tree
                fTrait.SetTree(fTree);
                fTrait.InitializeMultipoleMoments();
                fTrait.InitializeLocalCoefficientsForPrimaryNodes();
                fTrait.Initialize();

                //extract information
                if(fParameters.verbosity > 1)
                {
                   KFMElectrostaticTreeInformationExtractor extractor;
                   extractor.SetDegree(fParameters.degree);
                   fTree->ApplyCorecursiveAction(&extractor);
                   extractor.PrintStatistics();
                }

                //fast field solver (from local coeff)
                fFastFieldSolver.SetDegree(params.degree);

                ConstructElementNodeAssociation();

                if(fParameters.verbosity > 0)
                {
                    kfmout<<"KFMElectrostaticBoundaryIntegrator::Initialize: Done fast multipole boundary integrator intialization."<<kfmendl;
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
                    kfmout<<"KFMElectrostaticBoundaryIntegrator::Initialize: Error, attempted to reused pre-constructed tree, but there is a parameter mis-match. "<<kfmendl;
                    kfmexit(1);
                }

                ComputeUniqueHash(fParameters);

                if(fParameters.verbosity > 0)
                {
                    kfmout<<"KFMElectrostaticBoundaryIntegrator::Initialize: Initializing electrostatic fast multipole boundary integrator."<<kfmendl;
                }

                //get a pointer to the pre-existing element container
                fElementContainer =
                KFMObjectRetriever<KFMElectrostaticNodeObjects, KFMElectrostaticElementContainerBase<3,1> >::GetNodeObject(fTree->GetRootNode());
                //set up the surface converter w/ pre-existing element container
                //no need to extract the data, as this has already been done
                fSurfaceConverter.SetSurfaceContainer(&fSurfaceContainer);
                fSurfaceConverter.SetElectrostaticElementContainer(fElementContainer);

                //the parallel trait is responsible for computing
                //local coefficient field map everywhere it is needed (primary nodes)
                fTrait.SetElectrostaticElementContainer(fElementContainer);
                fTrait.SetParameters(params); //always set parameters before setting the tree
                fTrait.SetTree(fTree);
                fTrait.Initialize();

                //fast field solver (from local coeff)
                fFastFieldSolver.SetDegree(params.degree);

                ConstructElementNodeAssociation();

                //compute the representation of the sparse matrix
                if(fParameters.verbosity > 2)
                {
                    kfmout<<"KFMElectrostaticBoundaryIntegrator::Initialize: Constructing sparse matrix representation."<<kfmendl;
                }

                if(fParameters.verbosity > 0)
                {
                    kfmout<<"KFMElectrostaticBoundaryIntegrator::Initialize: Done fast multipole boundary integrator intialization."<<kfmendl;
                }

                fInitialized = true;
            }
        }

        KFMElectrostaticTree* GetTree(){return fTree;};

        void Update(const KVector<ValueType>& x)
        {
            fSurfaceConverter.UpdateBasisData(x);
            //recompute the multipole moments and the local coefficients to update the field
            fTrait.MapField();
        }

        using KElectrostaticBoundaryIntegrator::BoundaryIntegral;

        ValueType BoundaryIntegral(unsigned int sourceIndex, unsigned int targetIndex)
        {
            return KElectrostaticBoundaryIntegrator::BoundaryIntegral( fSurfaceContainer[sourceIndex], sourceIndex, fSurfaceContainer[targetIndex], targetIndex );
        }

        ValueType BoundaryIntegral(KSurfacePrimitive* target, unsigned int targetIndex)
        {
            //for piecewise constant collocation, we do not use the target index
            //evaluation is always at a single point, the centroid

            //look up the node corresponding to this target
            KFMElectrostaticNode* node = fNodes[targetIndex];

            //retrieve the expansion origin
            KFMCube<3>* cube = KFMObjectRetriever<KFMElectrostaticNodeObjects, KFMCube<3> >::GetNodeObject(node);
            KFMPoint<3> origin = cube->GetCenter();

            //retrieve the local coefficients
            KFMElectrostaticLocalCoefficientSet* set;
            set = KFMObjectRetriever<KFMElectrostaticNodeObjects, KFMElectrostaticLocalCoefficientSet>::GetNodeObject(node);
            fFastFieldSolver.SetLocalCoefficients(set);
            fFastFieldSolver.SetExpansionOrigin(origin);

            //figure out the boundary element's type
            target->Accept(fBoundaryVisitor);
            double ret_val = 0;
            if(fBoundaryVisitor.IsDirichlet())
            {
                ret_val = fFastFieldSolver.Potential(target->GetShape()->Centroid());
            }
            else
            {
                KEMThreeVector field;
                fFastFieldSolver.ElectricField(target->GetShape()->Centroid(),field);
                ret_val = field.Dot(target->GetShape()->Normal());
            }
            return ret_val;
        }

        ValueType BoundaryIntegral( unsigned int targetIndex)
        {
            KSurfacePrimitive* target = fSurfaceContainer.at(targetIndex);

            //for piecewise constant collocation, we do not use the target index
            //evaluation is always at a single point, the centroid

            //look up the node corresponding to this target
            KFMElectrostaticNode* node = fNodes[targetIndex];

            //retrieve the expansion origin
            KFMCube<3>* cube = KFMObjectRetriever<KFMElectrostaticNodeObjects, KFMCube<3> >::GetNodeObject(node);
            KFMPoint<3> origin = cube->GetCenter();

            //retrieve the local coefficients
            KFMElectrostaticLocalCoefficientSet* set;
            set = KFMObjectRetriever<KFMElectrostaticNodeObjects, KFMElectrostaticLocalCoefficientSet>::GetNodeObject(node);
            fFastFieldSolver.SetLocalCoefficients(set);
            fFastFieldSolver.SetExpansionOrigin(origin);

            //figure out the boundary element's type
            target->Accept(fBoundaryVisitor);
            double ret_val = 0;
            if(fBoundaryVisitor.IsDirichlet())
            {
                ret_val = fFastFieldSolver.Potential(target->GetShape()->Centroid());
            }
            else
            {
                KEMThreeVector field;
                fFastFieldSolver.ElectricField(target->GetShape()->Centroid(),field);
                ret_val = field.Dot(target->GetShape()->Normal());
            }
            return ret_val;
        }

    protected:

        void
        InitializeSubdivisionCondition()
        {
            //construct the subdivision condition
            if(fParameters.strategy == KFMSubdivisionStrategy::Balanced )
            {
                KFMSubdivisionConditionBalanced<KFMELECTROSTATICS_DIM, KFMElectrostaticNodeObjects>* balancedSubdivision = new KFMSubdivisionConditionBalanced<KFMELECTROSTATICS_DIM, KFMElectrostaticNodeObjects>();

                //determine how to weight the work load contributions
                fTrait.EvaluateWorkLoads(fParameters.divisions, fParameters.zeromask);

                //set the work load weights
                balancedSubdivision->SetDiskWeight(fTrait.GetDiskWeight());
                balancedSubdivision->SetRamWeight(fTrait.GetRamWeight());
                balancedSubdivision->SetFFTWeight(fTrait.GetFFTWeight());
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

        void ConstructElementNodeAssociation()
        {
            //here we associate each element's centroid with the node containing it
            KFMElectrostaticTreeNavigator navigator;

            unsigned int n_elem = fElementContainer->GetNElements();
            fNodes.resize(n_elem);
            //loop over all elements of surface container
            for(unsigned int i=0; i<n_elem; i++)
            {
                fNodes[i] = NULL;
                navigator.SetPoint( fElementContainer->GetCentroid(i) );
                navigator.ApplyAction(fTree->GetRootNode());

                if(navigator.Found())
                {
                    fNodes[i] = navigator.GetLeafNode();
                }
                else
                {
                    fNodes[i] = NULL;
                    kfmout<<"KFMElectrostaticBoundaryIntegrator::ConstructElementNodeAssociation: Error, element centroid not found in region."<<kfmendl;
                    kfmexit(1);
                }
            }
        }

////////////////////////////////////////////////////////////////////////////////

        void
        ComputeUniqueHash( const KFMElectrostaticParameters& parameters)
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
            fGeometryHash = tShapeHashGenerator.GenerateHash( fSurfaceContainer );

            // compute hash of right hand size of the equation (boundary conditions)
            KBoundaryIntegralVector< KFMElectrostaticBoundaryIntegrator<ParallelTrait> > b(fSurfaceContainer, *this);
            KMD5HashGenerator tBCHashGenerator;
            tBCHashGenerator.MaskedBits( HashMaskedBits );
            tBCHashGenerator.Threshold( HashThreshold );
            fBoundaryConditionHash = tShapeHashGenerator.GenerateHash( b );

            // compute hash of the parameter values w/o the multipole expansion degree included
            KFMElectrostaticParameters params = parameters;
            params.degree = 0; //must always be set to zero when computing the hash
            KMD5HashGenerator parameterHashGenerator;
            parameterHashGenerator.MaskedBits( HashMaskedBits );
            parameterHashGenerator.Threshold( HashThreshold );
            fTreeParameterHash = parameterHashGenerator.GenerateHash( params );

            //construct a unique id by stripping the first 6 characters from the shape and parameter hashes
            std::string unique_id = fGeometryHash.substr(0,6) + fTreeParameterHash.substr(0,6);
            fUniqueID = unique_id;
        }


////////////////////////////////////////////////////////////////////////////////
////////data and state
////////////////////////////////////////////////////////////////////////////////

        bool fInitialized;

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
        ParallelTrait fTrait;
        KFMSubdivisionCondition<KFMELECTROSTATICS_DIM, KFMElectrostaticNodeObjects>* fSubdivisionCondition;

        KFMElectrostaticTreeBuilder fTreeBuilder;

        //fast look-up for the node which contains the centroid of each element
        std::vector< KFMElectrostaticNode* > fNodes;

        //compute the field from the local coefficients
        KFMElectrostaticLocalCoefficientFieldCalculator fFastFieldSolver;
};


}

#endif /* KFMElectrostaticBoundaryIntegrator_H__ */
