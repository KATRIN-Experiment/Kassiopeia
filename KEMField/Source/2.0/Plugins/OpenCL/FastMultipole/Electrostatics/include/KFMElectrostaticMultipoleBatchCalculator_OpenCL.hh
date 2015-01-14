#ifndef KFMElectrostaticMultipoleBatchCalculator_OpenCL_HH__
#define KFMElectrostaticMultipoleBatchCalculator_OpenCL_HH__

#include <sstream>

//core (opencl)
#include "KOpenCLInterface.hh"


//kernel
#include "KFMScalarMultipoleExpansion.hh"

//math
#include "KFMPointCloud.hh"
#include "KFMMath.hh"

//electrostatics
#include "KFMElectrostaticElementContainer.hh"
#include "KFMElectrostaticMultipoleBatchCalculatorBase.hh"

#include "KFMElectrostaticMultipoleCalculatorAnalytic.hh"

namespace KEMField{

/**
*
*@file KFMElectrostaticMultipoleBatchCalculator_OpenCL.hh
*@class KFMElectrostaticMultipoleBatchCalculator_OpenCL
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Fri Jun  7 10:06:57 EDT 2013 J. Barrett (barrettj@mit.edu) First Version
*
*/

class KFMElectrostaticMultipoleBatchCalculator_OpenCL: public KFMElectrostaticMultipoleBatchCalculatorBase
{
    public:
        KFMElectrostaticMultipoleBatchCalculator_OpenCL();
        virtual ~KFMElectrostaticMultipoleBatchCalculator_OpenCL();

        virtual void SetDegree(int degree);

        virtual void Initialize();

        //execute the operation to fill the multipole buffer
        virtual void ComputeMoments();

        std::string GetOpenCLFlags() const {return fOpenCLFlags;};

    protected:

        void ConstructOpenCLKernels();
        void BuildBuffers();
        void AssignBuffers();
        void FillTemporaryBuffers();

        KFMElectrostaticMultipoleCalculatorAnalytic* fAnalyticCalc;

        ////////////////////////////////////////////////////////////////////////

        //inherited from base
        //long fMaxBufferSizeInBytes;
        //bool fInitialized;
        //int fDegree;
        //int fDim;
        //unsigned int fNMaxItems;
        //unsigned int fStride;
        //unsigned int fValidSize;
        //int* fIDBuffer;//size = fNMaxItems
        //double* fOriginBuffer;   //size = fDim*fNMaxItems
        //double* fMomentBuffer; //size = 2*fNMaxItems*fStride

        int fComplexStride;
        int fJSize;

        ////////////////////////////////////////////////////////////////////////

        std::string fOpenCLFlags;

        CL_TYPE* fJMatrix;
        CL_TYPE* fAxialPlm;
        CL_TYPE* fEquatorialPlm;
        CL_TYPE* fACoefficient;

        CL_TYPE* fBasisData;
        CL_TYPE4* fIntermediateOriginData;
        CL_TYPE16* fVertexData;
        CL_TYPE2* fIntermediateMomentData;

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

        ////////////////////////////////////////////////////////////////////////

};


}//end of KEMField

#endif /* KFMElectrostaticMultipoleCalculatorBatch_H__ */
