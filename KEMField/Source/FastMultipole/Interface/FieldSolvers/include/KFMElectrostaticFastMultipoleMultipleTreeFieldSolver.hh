#ifndef KFMElectrostaticFastMultipoleMultipleTreeFieldSolver_HH__
#define KFMElectrostaticFastMultipoleMultipleTreeFieldSolver_HH__

#include "KFMElectrostaticTree.hh"
#include "KFMElectrostaticLocalCoefficientFieldCalculator.hh"

#include "KSurfaceContainer.hh"
#include "KElectrostaticIntegratingFieldSolver.hh"

namespace KEMField
{

/*
*
*@file KFMElectrostaticFastMultipoleMultipleTreeFieldSolver.hh
*@class KFMElectrostaticFastMultipoleMultipleTreeFieldSolver
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Thu Jan 23 16:56:53 EST 2014 J. Barrett (barrettj@mit.edu) First Version
*
*/

class KFMElectrostaticFastMultipoleMultipleTreeFieldSolver
{
    public:

        KFMElectrostaticFastMultipoleMultipleTreeFieldSolver(const KSurfaceContainer& container);
        virtual ~KFMElectrostaticFastMultipoleMultipleTreeFieldSolver();

        void AddTree(KFMElectrostaticTree* tree);

        //computes the potential and field at a given point
        double Potential(const KPosition& P) const;
        KThreeVector ElectricField(const KPosition& P) const;

    protected:

        void SetPoint(const double* p) const;


    ////////////////////////////////////////////////////////////////////////////

        const KSurfaceContainer& fSurfaceContainer;

        unsigned int fNTrees;
        std::vector< KFMElectrostaticTree* > fTreeVector;
        std::vector< KFMElectrostaticNode* > fRootNodeVector;
        std::vector< KFMElectrostaticParameters > fParameterVector;

        //direct field evaluation
        mutable KElectrostaticBoundaryIntegrator fDirectIntegrator;
        KIntegratingFieldSolver<KElectrostaticBoundaryIntegrator> fDirectFieldSolver;

        //needed for field evaluation
        mutable KFMElectrostaticLocalCoefficientFieldCalculator fFastFieldSolver;

        mutable std::vector< KFMElectrostaticTreeNavigator* > fNavigatorVector;
        mutable KFMPoint<3> fEvaluationPoint;
        mutable std::vector< KFMElectrostaticNode* > fLeafNodeVector;
        mutable std::vector< KFMCube<3>* > fCubeVector;
        mutable std::vector< KFMPoint<3> > fExpansionOriginVector;
        mutable std::vector< KFMElectrostaticLocalCoefficientSet* > fLocalCoeffVector;
        mutable std::vector< KFMExternalIdentitySet* > fDirectCallIDSetVector;
        mutable std::vector< bool > fUseageIndicatorVector;
        mutable std::vector< double > fWeightVector;
        mutable std::vector< double > fFastPotential;
        mutable std::vector< double > fDirectPotential;
        mutable std::vector< double > fTotalPotential;
        mutable std::vector< KThreeVector > fFastField;
        mutable std::vector< KThreeVector > fDirectField;
        mutable std::vector< KThreeVector > fTotalField;

};


}//end of KEMField namespace



#endif /* KFMElectrostaticFastMultipoleMultipleTreeFieldSolver_H__ */
