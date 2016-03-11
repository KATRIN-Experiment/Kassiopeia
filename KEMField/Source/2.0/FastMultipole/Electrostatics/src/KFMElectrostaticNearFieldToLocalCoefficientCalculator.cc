#include "KFMElectrostaticNearFieldToLocalCoefficientCalculator.hh"

#include "KFMObjectRetriever.hh"

#include "KFMBall.hh"
#include "KFMCube.hh"

namespace KEMField
{

KFMElectrostaticNearFieldToLocalCoefficientCalculator::KFMElectrostaticNearFieldToLocalCoefficientCalculator():
fDegree(0),
fNQuadrature(0),
fLocalCoeffCalc(NULL),
fElementContainer(NULL)
{
    fConversionFactor = std::sqrt(3.0)/2.0; // 2.0/std::sqrt(3.0);
}

KFMElectrostaticNearFieldToLocalCoefficientCalculator::~KFMElectrostaticNearFieldToLocalCoefficientCalculator()
{
    delete fLocalCoeffCalc;
}

void
KFMElectrostaticNearFieldToLocalCoefficientCalculator::SetDegree(int l_max)
{
    fDegree = std::fabs(l_max);
    fTempMoments.SetDegree(fDegree);
}

void
KFMElectrostaticNearFieldToLocalCoefficientCalculator::SetNumberOfQuadratureTerms(unsigned int n)
{
    fNQuadrature = std::fabs(n);
}

void
KFMElectrostaticNearFieldToLocalCoefficientCalculator::Initialize()
{
    fLocalCoeffCalc = new KFMElectrostaticLocalCoefficientCalculatorNumeric();
    fLocalCoeffCalc->SetDegree(fDegree);
    fLocalCoeffCalc->SetNumberOfQuadratureTerms(fNQuadrature);
}


void
KFMElectrostaticNearFieldToLocalCoefficientCalculator::ApplyAction(KFMElectrostaticNode* node)
{
    if(node != NULL && !( node->HasChildren()) )
    {
        KFMExternalIdentitySet* eid_set = KFMObjectRetriever<KFMElectrostaticNodeObjects, KFMExternalIdentitySet>::GetNodeObject(node);
        KFMCube<3>* cube = KFMObjectRetriever<KFMElectrostaticNodeObjects, KFMCube<3> >::GetNodeObject(node);

        if(eid_set != NULL && cube != NULL)
        {
            KFMElectrostaticLocalCoefficientSet* l_set =  KFMObjectRetriever<KFMElectrostaticNodeObjects, KFMElectrostaticLocalCoefficientSet>::GetNodeObject(node);

            if(l_set != NULL )
            {
                //clear list of elements to remove from the eid_set
                fElementsToRemove.clear();
                fElementsToKeep.clear();

                //compute the bounding ball of the cube
                KFMPoint<3> cube_bball_center = cube->GetCenter();
                KFMBall<3> cube_bball(cube_bball_center, fConversionFactor*(cube->GetLength()) );

                //loop over elements specified by the external id set, if they satisfy the admissibility criteria
                //then we compute and add their contribution to this nodes local coefficient expansion
                unsigned int n_elem = eid_set->GetSize();
                for(unsigned int i=0; i<n_elem; i++)
                {
                    unsigned int id = eid_set->GetID(i);
                    const KFMBall<3>* ball = fElementContainer->GetBoundingBall(id);
                    if( cube_bball.BallIsOutside(ball) )
                    {
                        fElementsToRemove.push_back( id );

//                            std::cout<<"ptr to container = "<<fElementContainer<<std::endl;
//                            std::cout<<"id = "<<id<<std::endl;

                        fLocalCoeffCalc->ConstructExpansion(cube->GetCenter(), fElementContainer->GetPointCloud(id), &fTempMoments);

                        KFMBasisData<1>* basis = fElementContainer->GetBasisData(id);
                        double charge = (*basis)[0];


//                        fTempMoments *= (*basis)[0];

//                        std::cout<<"previous moments: "<<std::endl;
//                        l_set->PrintMoments();

//                        std::cout<<"new contrib:"<<std::endl;
//                        fTempMoments.PrintMoments();

                       // *l_set += fTempMoments;

                        std::vector< double >* real1 = l_set->GetRealMoments();
                        std::vector< double >* imag1 = l_set->GetImaginaryMoments();
                        const std::vector< double >* real2 = fTempMoments.GetRealMoments();
                        const std::vector< double >* imag2 = fTempMoments.GetImaginaryMoments();

                        unsigned int s = real1->size();
                        for(unsigned int i=0; i < s; ++i)
                        {
                            (*real1)[i] += charge*(*real2)[i];
                            (*imag1)[i] += charge*(*imag2)[i];
                        }


//                        std::cout<<"new moments:"<<std::endl;
//                        l_set->PrintMoments();
                    }
                    else
                    {
                        fElementsToKeep.push_back(id);
                    }
                }

//                //now remove the elements from the eid set which we added to the local coeff
//                for(unsigned int i=0; i<fElementsToRemove.size(); i++)
//                {
//                    eid_set->RemoveID(fElementsToRemove[i]);
//                }
//
                eid_set->Clear();
                eid_set->SetIDs(&fElementsToKeep);

                //resort eid_set
                eid_set->Sort();

            }
        }
    }
}














}
