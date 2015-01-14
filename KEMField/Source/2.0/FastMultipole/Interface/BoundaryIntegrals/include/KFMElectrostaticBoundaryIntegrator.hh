#ifndef KFMElectrostaticBoundaryIntegrator_HH__
#define KFMElectrostaticBoundaryIntegrator_HH__

#include "KElectrostaticBoundaryIntegrator.hh"
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
        KFMElectrostaticBoundaryIntegrator(KSurfaceContainer& surfaceContainer):
        KElectrostaticBoundaryIntegrator(),
        fInitialized(false),
        fSurfaceContainer(surfaceContainer)
        {
            fUniqueID = "nonunique";
        };

        virtual ~KFMElectrostaticBoundaryIntegrator()
        {
            //reset the node's ptr to the element container to null
            KFMNodeObjectNullifier<KFMElectrostaticNodeObjects, KFMElectrostaticElementContainerBase<3,1> > elementContainerNullifier;
            fTree.ApplyCorecursiveAction(&elementContainerNullifier);

            delete fElementContainer;
        };

        void Initialize(const KFMElectrostaticParameters& params)
        {
            if(!fInitialized)
            {
                fParameters = params;

                fTree.SetParameters(fParameters);

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

                //create the tree builder
                KFMElectrostaticTreeBuilder treeBuilder;
                treeBuilder.SetElectrostaticElementContainer(fElementContainer);
                treeBuilder.SetTree(&fTree);

                //now we construct the tree's structure
                treeBuilder.ConstructRootNode();
                treeBuilder.PerformSpatialSubdivision();
                treeBuilder.FlagNonZeroMultipoleNodes();
                treeBuilder.PerformAdjacencySubdivision();
                treeBuilder.FlagPrimaryNodes();
                treeBuilder.CollectDirectCallIdentitiesForPrimaryNodes();

                //the parallel trait is responsible for computing
                //local coefficient field map everywhere it is needed (primary nodes)
                fTrait.SetElectrostaticElementContainer(fElementContainer);
                fTrait.SetTree(&fTree);
                fTrait.Initialize();
                fTrait.MapField();

                //extract information
                if(fParameters.verbosity > 1)
                {
                   KFMElectrostaticTreeInformationExtractor extractor;
                   fTree.ApplyCorecursiveAction(&extractor);
                   extractor.PrintStatistics();
                }

                //fast field solver (from local coeff)
                fFastFieldSolver.SetDegree(params.degree);

                //compute the representation of the sparse matrix
                //this must be done regardless of whether or not we are caching the matrix elements

                if(fParameters.verbosity > 2)
                {
                    kfmout<<"KFMElectrostaticBoundaryIntegrator::Initialize: Constructing sparse matrix representation."<<kfmendl;
                }

                ConstructElementNodeAssociation();
                ConstructSparseMatrixRepresentation();

                if(fParameters.verbosity > 0)
                {
                    kfmout<<"KFMElectrostaticBoundaryIntegrator::Initialize: Done fast multipole boundary integrator intialization."<<kfmendl;
                }

                fInitialized = true;
            }
        }

        //for hash identification
        void SetUniqueIDString(std::string unique_id){fUniqueID = unique_id;};
        std::string GetUniqueIDString() const {return fUniqueID;};

        KFMElectrostaticTree* GetTree(){return &fTree;};

        void Update(const KVector<ValueType>& x)
        {
            fSurfaceConverter.UpdateBasisData(x);

            //recompute the multipole moments and the local coefficients to update the field
            fTrait.MapField();
        }

        using KElectrostaticBoundaryIntegrator::BoundaryIntegral;

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

        unsigned int GetNMatrixElementsToCache()
        {
            return fNumberNonZeroSparseMatrixElements;
        }

        const std::vector< const std::vector<unsigned int>* >& GetCachedMatrixElementColumnIndexListPointers() const
        {
            return fColumnIndexListPointers;
        };

        const std::vector< unsigned int >& GetCachedMatrixElementRowOffsetList() const
        {
            return fMatrixElementRowOffsets;
        };


    protected:

        void ConstructElementNodeAssociation()
        {
            KFMElectrostaticTreeNavigator navigator;
            navigator.SetDivisions(fParameters.divisions);

            unsigned int n_elem = fElementContainer->GetNElements();
            fNodes.resize(n_elem);
            //loop over all elements of surface container
            for(unsigned int i=0; i<n_elem; i++)
            {
                fNodes[i] = NULL;
                navigator.SetPoint( fElementContainer->GetCentroid(i) );
                navigator.ApplyAction(fTree.GetRootNode());

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



        void ConstructSparseMatrixRepresentation()
        {
            //computes a representation of the local sparse matrix
            //does not compute the matrix elements themselves, just indexes of the non-zero entries

            //first compute the number of non-zero matrix elements
            unsigned int n_mx_elements = 0;
            unsigned int n_elem = fSurfaceContainer.size();

            //first we determine the non-zero column element indices
            //loop over all elements of surface container
            for(unsigned int i=0; i<n_elem; i++)
            {
                //look up the node corresponding to this target
                KFMElectrostaticNode* leaf_node = fNodes[i];
                KFMExternalIdentitySet* eid_set = KFMObjectRetriever<KFMElectrostaticNodeObjects, KFMExternalIdentitySet>::GetNodeObject(leaf_node);

                if(eid_set != NULL)
                {
                    n_mx_elements += eid_set->GetSize();
                }

            }

            fNumberNonZeroSparseMatrixElements = n_mx_elements;


            //now we retrieve pointers to lists of the non-zero column entries for each row
            //this list is redundant for many rows, hence why we only store the pointers to a common list
            fColumnIndexListPointers.resize(n_elem);

            //first we determine the non-zero column element indices
            //loop over all elements of surface container
            for(unsigned int i=0; i<n_elem; i++)
            {
                //look up the node corresponding to this target
                KFMElectrostaticNode* leaf_node = fNodes[i];
                KFMExternalIdentitySet* eid_set = KFMObjectRetriever<KFMElectrostaticNodeObjects, KFMExternalIdentitySet>::GetNodeObject(leaf_node);

                if(eid_set != NULL)
                {
                    fColumnIndexListPointers[i] = eid_set->GetRawIDList();
                }
                else
                {
                    std::stringstream ss;
                    ss <<"KFMElectrostaticBoundaryIntegrator::FillSparseColumnIndexLists: Error, leaf node ";
                    ss << leaf_node->GetID();
                    ss <<" does not contain external element id list.";
                    kfmout<< ss.str() <<kfmendl;
                    kfmexit(1);
                }
            }


            //if one were to store all of the non-zero sparse matrix elements compressed into a single block
            //of memory, we would need to index the start position of the data corresponding to each row
            //we compute these offsets here

            fMatrixElementRowOffsets.resize(n_elem);
            //the offset of the first row from the beginning is zero
            unsigned int offset = 0;
            for(unsigned int target_id=0; target_id<n_elem; target_id++)
            {
                fMatrixElementRowOffsets.at(target_id) = offset;
                unsigned int n_mx = fColumnIndexListPointers.at(target_id)->size();
                offset += n_mx;
            }

        }


        //data and state

        bool fInitialized;

        KSurfaceContainer& fSurfaceContainer;
        KFMElectrostaticSurfaceConverter fSurfaceConverter;
        KFMElectrostaticElementContainerBase<3,1>* fElementContainer;

        std::string fUniqueID;

        KFMElectrostaticTree fTree;
        KFMElectrostaticParameters fParameters;
        ParallelTrait fTrait;

        //fast look-up for the node which contains the centroid of each element
        std::vector< KFMElectrostaticNode* > fNodes;

        //compute the field from the local coefficients
        KFMElectrostaticLocalCoefficientFieldCalculator fFastFieldSolver;

        //matrix element caching
        unsigned int fNumberNonZeroSparseMatrixElements;
        std::vector< const std::vector<unsigned int>* > fColumnIndexListPointers;
        std::vector< unsigned int > fMatrixElementRowOffsets;

};


}

#endif /* KFMElectrostaticBoundaryIntegrator_H__ */
