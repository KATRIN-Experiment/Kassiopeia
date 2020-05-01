#include "KFMElectrostaticMultipoleBatchCalculator.hh"

#include "KFMMessaging.hh"

#include <cstdlib>

namespace KEMField
{

KFMElectrostaticMultipoleBatchCalculator::KFMElectrostaticMultipoleBatchCalculator() :
    KFMElectrostaticMultipoleBatchCalculatorBase()
{
    fDegree = 0;
    fNMaxItems = 0;
    fStride = 0;
    fValidSize = 0;

    fInitialized = false;
    fAnalyticCalc = new KFMElectrostaticMultipoleCalculatorAnalytic();
    fNumericCalc = new KFMElectrostaticMultipoleCalculatorNumeric();
}

KFMElectrostaticMultipoleBatchCalculator::~KFMElectrostaticMultipoleBatchCalculator()
{
    delete fAnalyticCalc;
    delete fNumericCalc;
}

void KFMElectrostaticMultipoleBatchCalculator::SetDegree(int l_max)
{
    if (!fInitialized)  //one time deal
    {
        fDegree = std::abs(l_max);
        fStride = (fDegree + 1) * (fDegree + 2) / 2;
        fAnalyticCalc->SetDegree(fDegree);
        fNumericCalc->SetDegree(fDegree);
        fNumericCalc->SetNumberOfQuadratureTerms(fDegree / 2 + 1);

        fMoments.resize((fDegree + 1) * (fDegree + 1));
        fConvertedMoments.resize((fDegree + 1) * (fDegree + 1));
        fTempExpansion.SetDegree(fDegree);
    }
}

void KFMElectrostaticMultipoleBatchCalculator::Initialize()
{

    if (!fInitialized) {
        //first lets figure out how many elements we can process at a time
        unsigned int bytes_per_element = fStride * 2 * sizeof(double);
        fNMaxItems = fMaxBufferSizeInBytes / bytes_per_element;

        if (fNMaxItems != 0) {
            fIDBuffer = new int[fNMaxItems];
            fMomentBuffer = new double[2 * fStride * fNMaxItems];
            fOriginBuffer = new double[3 * fNMaxItems];
        }
        else {
            //warning
            std::stringstream ss;
            ss << "Buffer size of ";
            ss << fMaxBufferSizeInBytes;
            ss << " bytes is not large enough for a single element. ";
            ss << "Required bytes per element = " << bytes_per_element << ". Aborting.";
            kfmout << ss.str() << std::endl;
        }

        fInitialized = true;
    }
}


void KFMElectrostaticMultipoleBatchCalculator::ComputeMoments()
{
    for (unsigned int i = 0; i < fValidSize; i++) {
        int tempID = fIDBuffer[i];

        fTargetOrigin[0] = fOriginBuffer[3 * i];
        fTargetOrigin[1] = fOriginBuffer[3 * i + 1];
        fTargetOrigin[2] = fOriginBuffer[3 * i + 2];

        //computes the multipole moments of the primitive about its center

        const KFMPointCloud<3>* point_cloud = fContainer->GetPointCloud(tempID);

        if (point_cloud->GetNPoints() > 2 && fContainer->GetAspectRatio(tempID) > KFM_MAX_ASPECT_RATIO) {
            fNumericCalc->ConstructExpansion(fTargetOrigin, point_cloud, &fTempExpansion);
        }
        else {
            fAnalyticCalc->ConstructExpansion(fTargetOrigin, point_cloud, &fTempExpansion);
        }

        //get the charge density
        const KFMBasisData<1>* charge_density = fContainer->GetBasisData(tempID);
        double cd = (*charge_density)[0];

        //fill the spot in the buffer and rescale by the charge density
        int si;
        for (int l = 0; l <= fDegree; l++) {
            for (int k = 0; k <= l; k++) {
                si = KFMScalarMultipoleExpansion::RealBasisIndex(l, k);
                fMomentBuffer[i * 2 * fStride + 2 * si] = cd * (fTempExpansion.GetRealMoments()->at(si));
                fMomentBuffer[i * 2 * fStride + 2 * si + 1] = cd * (fTempExpansion.GetImaginaryMoments()->at(si));
            }
        }
    }
}


}  // namespace KEMField
