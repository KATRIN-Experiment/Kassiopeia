#include "KFMElectrostaticFastMultipoleFieldSolver_OpenCL.hh"

#include <cmath>

#define KFM_MIN_DIRECT_CALLS 16

namespace KEMField
{

KFMElectrostaticFastMultipoleFieldSolver_OpenCL::KFMElectrostaticFastMultipoleFieldSolver_OpenCL(KOpenCLSurfaceContainer& container, KFMElectrostaticTree& tree):
fSurfaceContainer(container),
fTree(tree),
fDirectIntegrator(fSurfaceContainer),
fDirectFieldSolver(fSurfaceContainer, fDirectIntegrator, fTree.GetMaxDirectCalls(), KFM_MIN_DIRECT_CALLS),
fFastFieldSolver(),
fNavigator()
{
    fDirectFieldSolver.Initialize();

    fRootNode = fTree.GetRootNode();
    fParameters = fTree.GetParameters();
    fUseCaching = fParameters.use_caching;

    fNavigator.SetDivisions(fParameters.divisions);
    fFastFieldSolver.SetDegree(fParameters.degree);

    fLeafNode = NULL;
    fCube = NULL;
    fLocalCoeff = NULL;

    fFallback = false;
};

KFMElectrostaticFastMultipoleFieldSolver_OpenCL::~KFMElectrostaticFastMultipoleFieldSolver_OpenCL(){};

double
KFMElectrostaticFastMultipoleFieldSolver_OpenCL::Potential(const KPosition& P) const
{
    SetPoint(P);

    if(!fFallback)
    {
        double fast_potential = fFastFieldSolver.Potential(P);

        if(fDirectCallIDSet != NULL)
        {
            if(fDirectCallIDSet->GetSize() != 0)
            {
                double direct_potential = fDirectFieldSolver.Potential(fDirectCallIDSet->GetRawIDList(), P);
                fast_potential += direct_potential;
                return fast_potential;
            }
            else
            {
                return fast_potential;
            }
        }
        else
        {
            return fast_potential;
        }
    }
    else
    {
        //fallback mode
        return fDirectFieldSolver.Potential(P);
    }
}

KEMThreeVector
KFMElectrostaticFastMultipoleFieldSolver_OpenCL::ElectricField(const KPosition& P) const
{
    SetPoint(P);

    if(!fFallback)
    {
        double fast_f[3];
        KEMThreeVector f;
        fFastFieldSolver.ElectricField(P, fast_f);
        f[0] = fast_f[0];
        f[1] = fast_f[1];
        f[2] = fast_f[2];

        KEMThreeVector direct_f;

        if(fDirectCallIDSet != NULL)
        {
            if(fDirectCallIDSet->GetSize() != 0)
            {
                direct_f = fDirectFieldSolver.ElectricField(fDirectCallIDSet->GetRawIDList(), P);
                f += direct_f;
            }
            return f;
        }
        else
        {
            return f;
        }

    }
    else
    {
        //fallback mode, use direct solver
        return fDirectFieldSolver.ElectricField( P );
    }

}


void
KFMElectrostaticFastMultipoleFieldSolver_OpenCL::SetPoint(const double* p) const
{
    if(fUseCaching)
    {
        if(fCube != NULL)
        {
            if( fCube->PointIsInside(p) )
            {
                //already have located this point, and its associated node
                return;
            }
        }
    }

    //no caching, or the point is in a new node which we must locate
    fEvaluationPoint = KFMPoint<3>(p);
    fNavigator.SetPoint( &fEvaluationPoint );
    fNavigator.ApplyAction(fRootNode);

    if(fNavigator.Found())
    {
        fFallback = false;

        fLeafNode = fNavigator.GetLeafNode();
        fCube = KFMObjectRetriever<KFMElectrostaticNodeObjects, KFMCube<3> >::GetNodeObject(fLeafNode);
        fLocalCoeff = KFMObjectRetriever<KFMElectrostaticNodeObjects, KFMElectrostaticLocalCoefficientSet >::GetNodeObject(fLeafNode);
        fDirectCallIDSet = KFMObjectRetriever<KFMElectrostaticNodeObjects, KFMExternalIdentitySet >::GetNodeObject(fLeafNode);

        if(fLocalCoeff == NULL || fCube == NULL)
        {
            kfmout<<"KFMElectrostaticFastMultipoleFieldSolver_OpenCL::SetPoint: Warning, tree node located for point: ("<<p[0]<<", "<<p[1]<<", "<<p[2]<<") has incomplete data!"<<kfmendl;
            fFallback = true;
        }

        fFastFieldSolver.SetExpansionOrigin(fCube->GetCenter());
        fFastFieldSolver.SetLocalCoefficients(fLocalCoeff);
    }
    else
    {
        kfmout<<"KFMElectrostaticFastMultipoleFieldSolver_OpenCL::SetPoint: Warning, point: ("<<p[0]<<", "<<p[1]<<", "<<p[2]<<") not located in region!"<<kfmendl;
        fFallback = true;
    }
}

}//end of KEMField namespace
