#ifndef KFMElectrostaticFastMultipoleFieldSolver_OpenCL_HH__
#define KFMElectrostaticFastMultipoleFieldSolver_OpenCL_HH__

#include "KFMElectrostaticLocalCoefficientFieldCalculator.hh"
#include "KFMElectrostaticTree.hh"
#include "KOpenCLElectrostaticBoundaryIntegrator.hh"
#include "KOpenCLElectrostaticIntegratingFieldSolver.hh"
#include "KOpenCLSurfaceContainer.hh"
#include "KSurfaceContainer.hh"

namespace KEMField
{

/*
*
*@file KFMElectrostaticFastMultipoleFieldSolver_OpenCL.hh
*@class KFMElectrostaticFastMultipoleFieldSolver_OpenCL
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Thu Jan 23 16:56:53 EST 2014 J. Barrett (barrettj@mit.edu) First Version
*
*/

class KFMElectrostaticFastMultipoleFieldSolver_OpenCL
{
  public:
    KFMElectrostaticFastMultipoleFieldSolver_OpenCL(const KoclEBIConfig& config, KOpenCLSurfaceContainer& container,
                                                    KFMElectrostaticTree& tree);

    virtual ~KFMElectrostaticFastMultipoleFieldSolver_OpenCL();

    //computes the potential and field at a given point
    double Potential(const KPosition& P) const;
    KThreeVector ElectricField(const KPosition& P) const;

    //for debugging and information purposes
    int GetSubsetSize(const KPosition& P) const
    {
        SetPoint(P);
        return fSubsetSize;
    };
    int GetTreeLevel(const KPosition& P) const
    {
        SetPoint(P);
        return fNodeList->size() - 1;
    };

  protected:
    void SetPoint(const double* p) const;


    ////////////////////////////////////////////////////////////////////////////

    KOpenCLSurfaceContainer& fSurfaceContainer;

    KFMElectrostaticTree& fTree;
    KFMElectrostaticNode* fRootNode;
    KFMElectrostaticParameters fParameters;
    bool fUseCaching;

    //direct field evaluation
    mutable KOpenCLElectrostaticBoundaryIntegrator fDirectIntegrator;
    KIntegratingFieldSolver<KOpenCLElectrostaticBoundaryIntegrator>* fDirectFieldSolver;

    //needed for field evaluation
    mutable KFMElectrostaticLocalCoefficientFieldCalculator fFastFieldSolver;

    mutable KFMElectrostaticTreeNavigator fNavigator;
    mutable KFMPoint<3> fEvaluationPoint;
    mutable KFMElectrostaticNode* fLeafNode;
    mutable KFMCube<3>* fCube;
    mutable KFMPoint<3> fExpansionOrigin;
    mutable KFMElectrostaticLocalCoefficientSet* fLocalCoeff;

    mutable std::vector<KFMElectrostaticNode*>* fNodeList;
    mutable unsigned int fSubsetSize;
    mutable unsigned int* fDirectCallIDs;

    mutable std::vector<KFMIdentitySetList*> fListOfDirectCallSetLists;

    mutable bool fFallback;
};


}  // namespace KEMField


#endif /* KFMElectrostaticFastMultipoleFieldSolver_OpenCL_H__ */
