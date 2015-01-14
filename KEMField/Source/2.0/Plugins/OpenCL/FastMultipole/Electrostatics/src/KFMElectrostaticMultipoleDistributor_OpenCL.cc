#include "KFMElectrostaticMultipoleDistributor_OpenCL.hh"

namespace KEMField
{



KFMElectrostaticMultipoleDistributor_OpenCL::KFMElectrostaticMultipoleDistributor_OpenCL()
{
    fMultipoleNodes = NULL;
    fNodeMomentData = NULL;
}

KFMElectrostaticMultipoleDistributor_OpenCL::~KFMElectrostaticMultipoleDistributor_OpenCL()
{
    delete[] fNodeMomentData;
}

void KFMElectrostaticMultipoleDistributor_OpenCL::SetDegree(unsigned int degree)
{
    fDegree = degree;
}

void KFMElectrostaticMultipoleDistributor_OpenCL::Initialize()
{
    fNTerms = (fDegree+1)*(fDegree+1);
    fStride = (fDegree+1)*(fDegree+2)/2;

    fDistributor.SetNumberOfTermsInSeries(fNTerms);
    fTempMoments.SetNumberOfTermsInSeries(fNTerms);

    fNMultipoleNodes = fMultipoleNodes->GetSize();
    fNodeMomentData = new CL_TYPE2[fNMultipoleNodes*fStride];
}

void KFMElectrostaticMultipoleDistributor_OpenCL::DistributeMoments()
{
    //read the data off of the gpu
    KOpenCLInterface::GetInstance()->GetQueue().enqueueReadBuffer(*fNodeMomentBufferCL, CL_TRUE, 0, fStride*fNMultipoleNodes*sizeof(CL_TYPE2), fNodeMomentData);

    //loop over the appropriate nodes and write out their multipole moments

    for(unsigned int i=0; i<fNMultipoleNodes; i++)
    {
        KFMElectrostaticNode* node = fMultipoleNodes->GetNodeFromSpecializedID(i);

        CL_TYPE2 moment;

        std::vector<double>* real_mom = fTempMoments.GetRealMoments();
        std::vector<double>* imag_mom = fTempMoments.GetImaginaryMoments();
        for(unsigned int j=0; j<fStride; j++)
        {
            moment = fNodeMomentData[i*fStride + j];

            (*real_mom)[j] = moment.s0;
            (*imag_mom)[j] = moment.s1;
        }

        fDistributor.SetExpansionToSet(&fTempMoments);
        fDistributor.ApplyAction(node);
    }
}













}
