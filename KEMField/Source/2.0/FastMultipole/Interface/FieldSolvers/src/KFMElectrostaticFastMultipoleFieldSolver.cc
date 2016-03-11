#include "KFMElectrostaticFastMultipoleFieldSolver.hh"

#include <cmath>

#include "KFMDirectCallCounter.hh"

namespace KEMField
{

KFMElectrostaticFastMultipoleFieldSolver::KFMElectrostaticFastMultipoleFieldSolver(const KSurfaceContainer& container, KFMElectrostaticTree& tree):
fSurfaceContainer(container),
fTree(tree),
fDirectIntegrator(),
fDirectFieldSolver(fSurfaceContainer, fDirectIntegrator),
fFastFieldSolver(),
fNavigator()
{
    fRootNode = fTree.GetRootNode();
    fParameters = fTree.GetParameters();
    fUseCaching = fParameters.use_caching;

    fFastFieldSolver.SetDegree(fParameters.degree);

    fSubsetSize = 0;

    //compute the maximum number of direct calls that occurs in this tree
    KFMDirectCallCounter<KFMElectrostaticNodeObjects> direct_call_counter;
    fTree.ApplyRecursiveAction(&direct_call_counter);
    unsigned int max_direct_calls = direct_call_counter.GetMaxDirectCalls();
    fDirectCallIDs = new unsigned int[max_direct_calls];

    fLeafNode = NULL;
    fCube = NULL;
    fLocalCoeff = NULL;
    fFallback = false;
}

KFMElectrostaticFastMultipoleFieldSolver::~KFMElectrostaticFastMultipoleFieldSolver()
{
    delete[] fDirectCallIDs;
}

double
KFMElectrostaticFastMultipoleFieldSolver::Potential(const KPosition& P) const
{
    SetPoint(P);

    if(!fFallback)
    {
        double fast_potential = fFastFieldSolver.Potential(P);

        if(fSubsetSize != 0)
        {
            double direct_potential = fDirectFieldSolver.Potential(fDirectCallIDs, fSubsetSize, P);
            return fast_potential + direct_potential;
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
KFMElectrostaticFastMultipoleFieldSolver::ElectricField(const KPosition& P) const
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

        if(fSubsetSize != 0)
        {
            direct_f = fDirectFieldSolver.ElectricField(fDirectCallIDs, fSubsetSize, P);
            f += direct_f;
        }
        return f;
    }
    else
    {
        //fallback mode, use direct solver
        return fDirectFieldSolver.ElectricField( P );
    }
}


void
KFMElectrostaticFastMultipoleFieldSolver::SetPoint(const double* p) const
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
        fNodeList = fNavigator.GetNodeList();

        fCube = KFMObjectRetriever<KFMElectrostaticNodeObjects, KFMCube<3> >::GetNodeObject(fLeafNode);
        fLocalCoeff = KFMObjectRetriever<KFMElectrostaticNodeObjects, KFMElectrostaticLocalCoefficientSet >::GetNodeObject(fLeafNode);

        //loop over the node list and collect the direct call elements from their id set lists
        fSubsetSize = 0;
        unsigned int n_nodes = fNodeList->size();
        for(unsigned int i=0; i<n_nodes; i++)
        {
            KFMElectrostaticNode* node = (*fNodeList)[i];
            if(node != NULL)
            {
                KFMIdentitySetList* id_set_list = KFMObjectRetriever<KFMElectrostaticNodeObjects, KFMIdentitySetList >::GetNodeObject(node);
                if(id_set_list != NULL)
                {
                    unsigned int n_sets = id_set_list->GetNumberOfSets();
                    for(unsigned int j=0; j<n_sets; j++)
                    {
                        const std::vector< unsigned int >* set = id_set_list->GetSet(j);
                        unsigned int set_size = set->size();
                        for(unsigned int k=0; k<set_size; k++)
                        {
                            fDirectCallIDs[fSubsetSize] = (*set)[k];
                            fSubsetSize++;
                        }
                    }
                }
            }
        }

        if(fLocalCoeff == NULL || fCube == NULL)
        {
            kfmout<<"KFMElectrostaticFastMultipoleFieldSolver::SetPoint: Warning, tree node located for point: ("<<p[0]<<", "<<p[1]<<", "<<p[2]<<") has incomplete data! Falling back to direct integrator. "<<kfmendl;

            fFallback = true;
        }

        fFastFieldSolver.SetExpansionOrigin(fCube->GetCenter());
        fFastFieldSolver.SetLocalCoefficients(fLocalCoeff);
    }
    else
    {
        kfmout<<"KFMElectrostaticFastMultipoleFieldSolver::SetPoint: Warning, point: ("<<p[0]<<", "<<p[1]<<", "<<p[2]<<") not located in region tree! Falling back to direct integrator. "<<kfmendl;
        fFallback = true;
    }
}

}//end of KEMField namespace
