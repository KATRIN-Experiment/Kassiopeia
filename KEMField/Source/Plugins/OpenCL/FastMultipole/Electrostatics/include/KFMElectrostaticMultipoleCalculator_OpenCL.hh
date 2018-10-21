#ifndef KFMElectrostaticMultipoleCalculator_OpenCL_HH__
#define KFMElectrostaticMultipoleCalculator_OpenCL_HH__

#include <sstream>

//core (opencl)
#include "KOpenCLInterface.hh"
#include "KOpenCLKernelBuilder.hh"


//kernel
#include "KFMScalarMultipoleExpansion.hh"

//math
#include "KFMPointCloud.hh"
#include "KFMMath.hh"
#include "KFMGaussLegendreQuadratureTableCalculator.hh"

//tree
#include "KFMSpecialNodeSet.hh"
#include "KFMElementMomentBatchCalculator.hh"

//electrostatics
#include "KFMElectrostaticNode.hh"
#include "KFMElectrostaticTree.hh"
#include "KFMElectrostaticParameters.hh"
#include "KFMElectrostaticElementContainer.hh"
#include "KFMElectrostaticMultipoleCalculatorAnalytic.hh"

namespace KEMField{

/**
*
*@file KFMElectrostaticMultipoleCalculator_OpenCL.hh
*@class KFMElectrostaticMultipoleCalculator_OpenCL
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Fri Jun  7 10:06:57 EDT 2013 J. Barrett (barrettj@mit.edu) First Version
*
*/

class KFMElectrostaticMultipoleCalculator_OpenCL
{
    public:
        KFMElectrostaticMultipoleCalculator_OpenCL(bool standalone = false);
        virtual ~KFMElectrostaticMultipoleCalculator_OpenCL();

        void SetElectrostaticElementContainer(const KFMElectrostaticElementContainerBase<3,1>* container){fContainer = container;};
        void SetParameters(KFMElectrostaticParameters params);
        void SetTree(KFMElectrostaticTree* tree){fTree = tree;};

        void SetNodeMomentBuffer(cl::Buffer* node_moments){fNodeMomentBufferCL = node_moments;};
        void SetMultipoleNodeSet(KFMSpecialNodeSet<KFMElectrostaticNodeObjects>* multipole_node_set){fMultipoleNodes = multipole_node_set;};

        void Initialize();
        void ComputeMoments();

        //use gpu to compute the moments of an individual boundary element
        //this is for testing/debugging purposes (can only use with fStandAlone flag turned on/true)
        virtual bool ConstructExpansion(double* target_origin, const KFMPointCloud<3>* vertex_cloud, KFMScalarMultipoleExpansion* moments) const;

        //selects whether to use analytic vs numerical integration
        //defaults uses analytic when possible, but reverts to numerical integration for bad aspect ratios
        void UseDefault(){fPrimaryIntegrationMode = -1; fSecondaryIntegrationMode = 1;};
        //force use of numerical integrator for all shapes
        void ForceNumerical(){fPrimaryIntegrationMode = 1; fSecondaryIntegrationMode = 1;};
        //force use of analytic integrator for all shapes
        void ForceAnalytic(){fPrimaryIntegrationMode = -1; fSecondaryIntegrationMode = -1;};

    protected:

        void BuildElementNodeIndex();
        void ConstructOpenCLKernels();
        void BuildBuffers();
        void AssignBuffers();
        void FillTemporaryBuffers();
        void ComputeCurrentMoments();
        void DistributeCurrentMoments();


        KFMElectrostaticMultipoleCalculatorAnalytic* fAnalyticCalc;

        const KFMElectrostaticElementContainerBase<3,1>* fContainer;
        KFMElectrostaticTree* fTree;


        ////////////////////////////////////////////////////////////////////////
        //list of elements and their associated nodes (extracted from node-element associator)

        const std::vector< unsigned int>* fElementIDList;
        const std::vector< KFMElectrostaticNode* >* fNodePtrList;
        const std::vector< unsigned int >* fNodeIDList;
        const std::vector< KFMPoint<KFMELECTROSTATICS_DIM> >* fOriginList;

        //list of the multipole-set id of each relevant node
        std::vector<unsigned int > fMultipoleNodeIDList;

        ////////////////////////////////////////////////////////////////////////
        bool fStandAlone; //turns on debugging mode if true
        int fPrimaryIntegrationMode;
        int fSecondaryIntegrationMode;

        long fMaxBufferSizeInBytes;
        bool fInitialized;
        int fDegree;
        int fVerbosity;
        int fDim;
        unsigned int fNMaxItems;
        unsigned int fStride;
        unsigned int fValidSize;

        int fScratchStride;
        int fJSize;

        unsigned int fNElements;
        unsigned int fTotalElementsToProcess;
        unsigned int fCurrentElementIndex;
        unsigned int fRemainingElementsToProcess;
        unsigned int fNumberOfElementsToProcessOnThisPass;

        ////////////////////////////////////////////////////////////////////////
        KFMGaussLegendreQuadratureTableCalculator fQuadratureTableCalc;
        std::vector< double > fAbscissaVector;
        std::vector< double > fWeightsVector;

        CL_TYPE* fAbscissa;
        CL_TYPE* fWeights;
        cl::Buffer* fAbscissaBufferCL;
        cl::Buffer* fWeightsBufferCL;

        ////////////////////////////////////////////////////////////////////////

        std::string fOpenCLFlags;

        CL_TYPE* fJMatrix;
        CL_TYPE* fAxialPlm;
        CL_TYPE* fEquatorialPlm;
        CL_TYPE* fACoefficient;

        CL_TYPE* fBasisData;

        CL_TYPE4* fIntermediateOriginData;
        CL_TYPE16* fVertexData;
        unsigned int* fNodeIDData;
        unsigned int fNGroupUniqueNodes;
        unsigned int* fNodeIndexData;
        unsigned int* fStartIndexData;
        unsigned int* fSizeData;

        //multipole calculaion kernel //////////////////////////////////////////
        mutable cl::Kernel* fMultipoleKernel;

        cl::Buffer* fOriginBufferCL; //expansion origin associated with each element (double)
        cl::Buffer* fVertexDataBufferCL; //the positions of the vertices of each element (double)
        cl::Buffer* fBasisDataBufferCL; //the basis data associated with each element (double)
        cl::Buffer* fMomentBufferCL; //the moments of each element (double)

        cl::Buffer* fACoefficientBufferCL; //normalization coefficients A(n,m) (double)
        cl::Buffer* fEquatorialPlmBufferCL; //the associated legendre polynomials evaluated at zero (double)
        cl::Buffer* fAxialPlmBufferCL; //the associated legendre polynomials evaluated at one (double)
        cl::Buffer* fJMatrixBufferCL; //the pinchon j-matrices (double)

        unsigned int fNLocal;
        unsigned int fNMaxWorkgroups;


        //distribution kernel //////////////////////////////////////////////////
        KFMElectrostaticElementNodeAssociator fElementNodeAssociator;
        KFMSpecialNodeSet<KFMElectrostaticNodeObjects>* fMultipoleNodes;

        mutable cl::Kernel* fMultipoleDistributionKernel;

        //buffer to store all of the multipole moments of the nodes w/ non-zero moments
        unsigned int fNMultipoleNodes;
        const unsigned int* fElementToNodeMap;
        cl::Buffer* fNodeIDBufferCL; //the id's of the nodes that own each element (unsigned int)

        cl::Buffer* fNodeIndexBufferCL;
        cl::Buffer* fStartIndexBufferCL;
        cl::Buffer* fSizeBufferCL;

        cl::Buffer* fNodeMomentBufferCL;
        unsigned fNDistributeLocal;

        //array zero-ing kernel ////////////////////////////////////////////////
        //we need a kernel to zero out the multipole buffer
        //because the OpenCL 1.1 specification lacks the clEnqueueFillBuffer command
        mutable cl::Kernel* fZeroKernel;
        unsigned int fNZeroLocal;
        unsigned int fMultipoleBufferSize;


};


}//end of KEMField

#endif /* KFMElectrostaticMultipoleCalculator_OpenCL_H__ */
