#ifndef __KFMElectrostaticMultipoleDistributor_OpenCL_H__
#define __KFMElectrostaticMultipoleDistributor_OpenCL_H__


#include <sstream>

//core (opencl)
#include "KOpenCLInterface.hh"
#include "KOpenCLKernelBuilder.hh"


//kernel
#include "KFMScalarMultipoleExpansion.hh"

//math
#include "KFMMath.hh"
#include "KFMPointCloud.hh"

//tree
#include "KFMSpecialNodeSet.hh"

//electrostatics
#include "KFMElectrostaticElementContainer.hh"
#include "KFMElectrostaticParameters.hh"
#include "KFMElectrostaticTree.hh"


namespace KEMField
{

/**
*
*@file KFMElectrostaticMultipoleDistributor_OpenCL.hh
*@class KFMElectrostaticMultipoleDistributor_OpenCL
*@brief
*@details
*<b>Revision History:<b>
*Date Name Brief Description
*Mon Jul 21 16:48:34 EDT 2014 J. Barrett (barrettj@mit.edu) First Version
*
*/

class KFMElectrostaticMultipoleDistributor_OpenCL
{
  public:
    KFMElectrostaticMultipoleDistributor_OpenCL();
    virtual ~KFMElectrostaticMultipoleDistributor_OpenCL();

    void SetNodeMomentBuffer(cl::Buffer* node_moments)
    {
        fNodeMomentBufferCL = node_moments;
    };
    void SetMultipoleNodeSet(KFMSpecialNodeSet<KFMElectrostaticNodeObjects>* multipole_node_set)
    {
        fMultipoleNodes = multipole_node_set;
    };

    void SetDegree(unsigned int degree);
    void Initialize();
    void DistributeMoments();

  protected:
    unsigned int fDegree;
    unsigned int fNTerms;
    unsigned int fStride;
    unsigned int fNMultipoleNodes;

    KFMSpecialNodeSet<KFMElectrostaticNodeObjects>* fMultipoleNodes;

    CL_TYPE2* fNodeMomentData;
    cl::Buffer* fNodeMomentBufferCL;

    KFMElectrostaticMultipoleSet fTempMoments;
    KFMScalarMomentDistributor<KFMElectrostaticNodeObjects, KFMElectrostaticMultipoleSet> fDistributor;
};


}  // namespace KEMField


#endif /* __KFMElectrostaticMultipoleDistributor_OpenCL_H__ */
