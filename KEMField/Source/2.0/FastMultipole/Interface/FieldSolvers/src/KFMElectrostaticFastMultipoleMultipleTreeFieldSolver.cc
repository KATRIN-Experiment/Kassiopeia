#include "KFMElectrostaticFastMultipoleMultipleTreeFieldSolver.hh"

#include <cmath>

namespace KEMField
{

KFMElectrostaticFastMultipoleMultipleTreeFieldSolver::KFMElectrostaticFastMultipoleMultipleTreeFieldSolver(const KSurfaceContainer& container):
fSurfaceContainer(container),
fDirectIntegrator(),
fDirectFieldSolver(fSurfaceContainer, fDirectIntegrator),
fFastFieldSolver()
{
    fNTrees = 0;
}

KFMElectrostaticFastMultipoleMultipleTreeFieldSolver::~KFMElectrostaticFastMultipoleMultipleTreeFieldSolver()
{
    for(unsigned int i=0; i<fNavigatorVector.size(); i++)
    {
        delete fNavigatorVector[i];
    }
}

void
KFMElectrostaticFastMultipoleMultipleTreeFieldSolver::AddTree(KFMElectrostaticTree* tree)
{
    fNTrees++;
    fTreeVector.push_back(tree);
    fRootNodeVector.push_back(tree->GetRootNode());
    fParameterVector.push_back(tree->GetParameters());

    KFMElectrostaticTreeNavigator* nav = new KFMElectrostaticTreeNavigator();
    nav->SetDivisions(tree->GetParameters().divisions);
    fNavigatorVector.push_back(nav);

    fFastFieldSolver.SetDegree(tree->GetParameters().degree);

    fLeafNodeVector.push_back(NULL);
    fCubeVector.push_back(NULL);
    fExpansionOriginVector.push_back( KFMPoint<3>() );
    fLocalCoeffVector.push_back(NULL);
    fDirectCallIDSetVector.push_back(NULL);
    fUseageIndicatorVector.push_back(false);
    fWeightVector.push_back(0.0);
    fFastPotential.push_back(0.0);
    fDirectPotential.push_back(0.0);
    fTotalPotential.push_back(0.0);
    fFastField.push_back(KEMThreeVector());
    fDirectField.push_back(KEMThreeVector());
    fTotalField.push_back(KEMThreeVector());
}


double
KFMElectrostaticFastMultipoleMultipleTreeFieldSolver::Potential(const KPosition& P) const
{
    //weighted average of different tree potentials

    SetPoint(P);

    double numer = 0.0;
    double denom = 0.0;

    for(unsigned int i=0; i<fNTrees; i++)
    {
        if(fUseageIndicatorVector[i])
        {
            double L2 = (fCubeVector[i]->GetLength())/2.0;
            KPosition origin(fExpansionOriginVector[i]);
            KPosition del = P - origin;

            double wx = (1.0 - std::fabs(del[0])/(L2)) + 1e-9;
            double wy = (1.0 - std::fabs(del[1])/(L2)) + 1e-9;
            double wz = (1.0 - std::fabs(del[2])/(L2)) + 1e-9;
            double inv = 1.0/wx + 1.0/wy + 1.0/wz;
            double weight = 1.0/inv;

            fFastFieldSolver.SetExpansionOrigin(fExpansionOriginVector[i]);
            fFastFieldSolver.SetLocalCoefficients(fLocalCoeffVector[i]);
            fFastPotential[i] = fFastFieldSolver.Potential(P);

            if(fDirectCallIDSetVector[i] != NULL)
            {
                if(fDirectCallIDSetVector[i]->GetSize() != 0)
                {
                    fDirectPotential[i] = fDirectFieldSolver.Potential(fDirectCallIDSetVector[i]->GetRawIDList(), P);
                    fTotalPotential[i] = fFastPotential[i] + fDirectPotential[i];
                }
                else
                {
                    fTotalPotential[i] = fFastPotential[i];
                }
            }
            else
            {
                fTotalPotential[i] = fFastPotential[i];
            }
            numer += weight*fTotalPotential[i];
            denom += weight;
        }
    }
    return numer/denom;

}

KEMThreeVector
KFMElectrostaticFastMultipoleMultipleTreeFieldSolver::ElectricField(const KPosition& P) const
{
    //weighted average of different tree fields

    SetPoint(P);

    double numer_x = 0.0;
    double numer_y = 0.0;
    double numer_z = 0.0;
    double denom = 0.0;

    for(unsigned int i=0; i<fNTrees; i++)
    {
        if(fUseageIndicatorVector[i])
        {
            double L2 = (fCubeVector[i]->GetLength())/2.0;
            KPosition origin(fExpansionOriginVector[i]);
            KPosition del = P - origin;

            double wx = (1.0 - std::fabs(del[0])/(L2)) + 1e-9;
            double wy = (1.0 - std::fabs(del[1])/(L2)) + 1e-9;
            double wz = (1.0 - std::fabs(del[2])/(L2)) + 1e-9;
            double inv = 1.0/wx + 1.0/wy + 1.0/wz;
            double weight = 1.0/inv;

            fFastFieldSolver.SetExpansionOrigin(fExpansionOriginVector[i]);
            fFastFieldSolver.SetLocalCoefficients(fLocalCoeffVector[i]);
            fFastFieldSolver.ElectricField(P, fFastField[i]);

            if(fDirectCallIDSetVector[i] != NULL)
            {
                if(fDirectCallIDSetVector[i]->GetSize() != 0)
                {
                    fDirectField[i] = fDirectFieldSolver.ElectricField(fDirectCallIDSetVector[i]->GetRawIDList(), P);
                    fTotalField[i] = fFastField[i] + fDirectField[i];
                }
                else
                {
                    fTotalField[i] = fFastField[i];
                }
            }
            else
            {
                fTotalField[i] = fFastField[i];
            }

            numer_x += weight*fTotalField[i][0];
            numer_y += weight*fTotalField[i][1];
            numer_z += weight*fTotalField[i][2];
            denom += weight;
        }
    }

    KEMThreeVector val;
    val[0] = numer_x/denom;
    val[1] = numer_y/denom;
    val[2] = numer_z/denom;
    return val;
}


void
KFMElectrostaticFastMultipoleMultipleTreeFieldSolver::SetPoint(const double* p) const
{

    //no caching, must locate point across all trees
    fEvaluationPoint = KFMPoint<3>(p);

    for(unsigned int i = 0; i<fNTrees; i++)
    {
        fNavigatorVector[i]->SetPoint( &fEvaluationPoint );
        fNavigatorVector[i]->ApplyAction( fRootNodeVector[i] );
        fUseageIndicatorVector[i] = fNavigatorVector[i]->Found();

        if(fUseageIndicatorVector[i])
        {
            fLeafNodeVector[i] = fNavigatorVector[i]->GetLeafNode();
            fCubeVector[i] =  KFMObjectRetriever<KFMElectrostaticNodeObjects, KFMCube<3> >::GetNodeObject(fLeafNodeVector[i]);
            fLocalCoeffVector[i] = KFMObjectRetriever<KFMElectrostaticNodeObjects, KFMElectrostaticLocalCoefficientSet >::GetNodeObject(fLeafNodeVector[i]);

            if(fLocalCoeffVector[i] == NULL || fCubeVector[i] == NULL)
            {
                kfmout<<"KFMElectrostaticFastMultipoleFieldSolver::SetPoint: Warning, tree node located for point: ("<<p[0]<<", "<<p[1]<<", "<<p[2]<<") has incomplete data!"<<kfmendl;
            }

            fDirectCallIDSetVector[i] = KFMObjectRetriever<KFMElectrostaticNodeObjects, KFMExternalIdentitySet >::GetNodeObject(fLeafNodeVector[i]);
            fExpansionOriginVector[i] = fCubeVector[i]->GetCenter();
        }

    }
}

}//end of KEMField namespace
