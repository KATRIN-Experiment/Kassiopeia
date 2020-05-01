#ifndef KFMElementScalarMomentDistributor_HH__
#define KFMElementScalarMomentDistributor_HH__

#include "KFMElementMomentBatchCalculator.hh"
#include "KFMMessaging.hh"
#include "KFMNode.hh"
#include "KFMNodeActor.hh"
#include "KFMObjectRetriever.hh"
#include "KFMPoint.hh"
#include "KFMScalarMomentDistributor.hh"
#include "KFMScalarMomentInitializer.hh"
#include "KFMScalarMultipoleExpansion.hh"


namespace KEMField
{

/**
*
*@file KFMElementScalarMomentDistributor.hh
*@class KFMElementScalarMomentDistributor
*@brief acts as an interface between the region tree and the Moment calculator for the elements
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Fri Jun  7 17:18:13 EDT 2013 J. Barrett (barrettj@mit.edu) First Version
*
*/

template<typename ObjectTypeList, typename ScalarMomentType, unsigned int SpatialNDIM>
class KFMElementScalarMomentDistributor
{
  public:
    KFMElementScalarMomentDistributor()
    {
        fBatchCalc = nullptr;

        fElementIDList = nullptr;
        fNodeList = nullptr;
        fOriginList = nullptr;

        fElementIDs = nullptr;
        fOrigins = nullptr;
    }

    virtual ~KFMElementScalarMomentDistributor()
    {
        ;
    };

    void SetBatchCalculator(KFMElementMomentBatchCalculator* batchCalc)
    {
        fBatchCalc = batchCalc;
    };

    void SetElementIDList(const std::vector<unsigned int>* elementIDs)
    {
        fElementIDList = elementIDs;
    };

    void SetNodeList(const std::vector<KFMNode<ObjectTypeList>*>* nodeList)
    {
        fNodeList = nodeList;
    };

    void SetOriginList(const std::vector<KFMPoint<SpatialNDIM>>* origins)
    {
        fOriginList = origins;
    };

    void ProcessAndDistributeMoments()
    {
        if (fBatchCalc != nullptr && SpatialNDIM == fBatchCalc->GetDimension()) {
            fDegree = fBatchCalc->GetDegree();
            fNTerms = (fDegree + 1) * (fDegree + 1);
            fTempMoments.resize((fDegree + 1) * (fDegree + 1));
            fTargetCoeff.SetNumberOfTermsInSeries(fNTerms);
            fNElements = fBatchCalc->GetIDBufferSize();
            fStride = fBatchCalc->GetMomentBufferStride();

            fTotalElementsToProcess = fElementIDList->size();
            fRemainingElementsToProcess = fTotalElementsToProcess;
            fCurrentElementIndex = 0;

            fElementIDs = fBatchCalc->GetIDBuffer();
            fOrigins = fBatchCalc->GetOriginBuffer();
            fMoments = fBatchCalc->GetMomentBuffer();

            fMomentInitializer.SetNumberOfTermsInSeries(fNTerms);
            fMomentDistributor.SetNumberOfTermsInSeries(fNTerms);

            do {
                if (fRemainingElementsToProcess < fNElements) {
                    fNumberOfElementsToProcessOnThisPass = fRemainingElementsToProcess;
                }
                else {
                    fNumberOfElementsToProcessOnThisPass = fNElements;
                }

                FillInputBuffers();

                fBatchCalc->ComputeMoments();

                DistributeMomentBuffer();

                fCurrentElementIndex += fNumberOfElementsToProcessOnThisPass;

                fRemainingElementsToProcess = fRemainingElementsToProcess - fNumberOfElementsToProcessOnThisPass;
            } while (fRemainingElementsToProcess > 0);

            //fBatchCalc->ReleaseMemory();
        }
        else {
            //warning
            kfmout
                << "KFMElementScalarMomentDistributor::ProcessAndDistributeMoments: Dimension of batch calculator does not equal "
                << SpatialNDIM << ". Aborting." << std::endl;
        }
    }

  protected:
    void FillInputBuffers()
    {
        fBatchCalc->SetSizeOfValidElements(fNumberOfElementsToProcessOnThisPass);

        for (unsigned int i = 0; i < fNumberOfElementsToProcessOnThisPass; i++) {
            fElementIDs[i] = fElementIDList->at(fCurrentElementIndex + i);

            for (unsigned int j = 0; j < SpatialNDIM; j++) {
                fOrigins[i * SpatialNDIM + j] = (fOriginList->at(fCurrentElementIndex + i))[j];
            }
        }
    }


    void DistributeMomentBuffer()
    {

        std::vector<double>* rmoments = fTargetCoeff.GetRealMoments();
        std::vector<double>* imoments = fTargetCoeff.GetImaginaryMoments();

        for (unsigned int i = 0; i < fNumberOfElementsToProcessOnThisPass; i++) {
            //extract moments from buffer
            //and insert into a scalar expansion class
            int si;  // psi, nsi;
            double real, imag;
            for (int l = 0; l <= fDegree; l++) {
                for (int m = 0; m <= l; m++) {
                    si = KFMScalarMultipoleExpansion::RealBasisIndex(l, m);  // l*(l+1)/2 + m;
                    real = fMoments[i * 2 * fStride + 2 * si];
                    imag = fMoments[i * 2 * fStride + 2 * si + 1];

                    (*rmoments)[si] = real;
                    (*imoments)[si] = imag;
                }
            }

            //find the associated node
            KFMNode<ObjectTypeList>* node = fNodeList->at(i + fCurrentElementIndex);

            if (node != nullptr) {
                if (KFMObjectRetriever<ObjectTypeList, ScalarMomentType>::GetNodeObject(node) == nullptr) {
                    fMomentInitializer.ApplyAction(node);
                }

                fMomentDistributor.SetExpansionToAdd(&fTargetCoeff);
                fMomentDistributor.ApplyAction(node);
            }
        }
    }


    KFMElementMomentBatchCalculator* fBatchCalc;

    //these vectors make up the map: elements <-> node <--> origin
    const std::vector<unsigned int>* fElementIDList;
    const std::vector<KFMNode<ObjectTypeList>*>* fNodeList;
    const std::vector<KFMPoint<SpatialNDIM>>* fOriginList;

    int fDegree;
    unsigned int fNTerms;
    unsigned int fTotalElementsToProcess;
    unsigned int fCurrentElementIndex;
    unsigned int fRemainingElementsToProcess;
    unsigned int fNumberOfElementsToProcessOnThisPass;

    int* fElementIDs;
    double* fOrigins;
    double* fMoments;

    unsigned int fStride;
    unsigned int fNElements;

    std::vector<std::complex<double>> fTempMoments;
    ScalarMomentType fTargetCoeff;

    KFMScalarMomentInitializer<ObjectTypeList, ScalarMomentType> fMomentInitializer;
    KFMScalarMomentDistributor<ObjectTypeList, ScalarMomentType> fMomentDistributor;
};


}  // namespace KEMField

#endif /* KFMElementScalarMomentDistributor_H__ */
