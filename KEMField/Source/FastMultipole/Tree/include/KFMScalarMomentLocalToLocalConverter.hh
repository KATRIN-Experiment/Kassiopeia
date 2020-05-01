#ifndef KFMScalarMomentLocalToLocalConverter_H__
#define KFMScalarMomentLocalToLocalConverter_H__


#include "KFMArrayScalarMultiplier.hh"
#include "KFMArrayWrapper.hh"
#include "KFMCube.hh"
#include "KFMCubicSpaceNodeNeighborFinder.hh"
#include "KFMKernelExpansion.hh"
#include "KFMKernelResponseArray.hh"
#include "KFMNodeActor.hh"
#include "KFMObjectRetriever.hh"
#include "KFMPointwiseArrayAdder.hh"
#include "KFMPointwiseArrayMultiplier.hh"
#include "KFMScalarMomentCollector.hh"
#include "KFMScalarMomentDistributor.hh"
#include "KFMScalarMomentInitializer.hh"
#include "KFMScaleInvariantKernelExpansion.hh"

#include <complex>
#include <cstdlib>
#include <vector>

namespace KEMField
{

/**
*
*@file KFMScalarMomentLocalToLocalConverter.hh
*@class KFMScalarMomentLocalToLocalConverter
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Fri Oct 12 13:24:38 EDT 2012 J. Barrett (barrettj@mit.edu) First Version
*
*/


template<typename ObjectTypeList, typename ScalarMomentType, typename KernelType, unsigned int SpatialNDIM>
class KFMScalarMomentLocalToLocalConverter : public KFMNodeActor<KFMNode<ObjectTypeList>>
{
  public:
    KFMScalarMomentLocalToLocalConverter()
    {

        fNTerms = 0;
        fTotalSpatialSize = 0;
        fDiv = 0;
        fTopLevelDivisions = 0;
        fZeroMaskSize = 0;
        fLength = 1.0;

        fKernelResponse = new KFMKernelResponseArray<KernelType, true, SpatialNDIM>();  //true -> origin is source
        fIsScaleInvariant = fKernelResponse->GetKernel()->IsScaleInvariant();

        fScalarMultCalc = new KFMArrayScalarMultiplier<std::complex<double>, SpatialNDIM>();

        fMultCalc = new KFMPointwiseArrayMultiplier<std::complex<double>, SpatialNDIM>();

        fAddCalc = new KFMPointwiseArrayAdder<std::complex<double>, SpatialNDIM>();

        fCollector = new KFMScalarMomentCollector<ObjectTypeList, ScalarMomentType, SpatialNDIM>();
        fDistributor = new KFMScalarMomentDistributor<ObjectTypeList, ScalarMomentType>();

        fInitialized = false;

        fAllocated = false;

        fCBSI = nullptr;
        fRBSI = nullptr;
    };


    ~KFMScalarMomentLocalToLocalConverter() override
    {
        DeallocateArrays();
        delete fKernelResponse;
        delete fScalarMultCalc;
        delete fMultCalc;
        delete fCollector;
        delete fDistributor;
        delete fAddCalc;

        delete[] fCBSI;
        delete[] fRBSI;
    };

    bool IsScaleInvariant() const
    {
        return fIsScaleInvariant;
    };

    void SetLength(double length)
    {
        fLength = length;
        fInitialized = false;
    };

    ////////////////////////////////////////////////////////////////////////
    void SetNumberOfTermsInSeries(unsigned int n_terms)
    {
        fNTerms = n_terms;

        KFMScalarMultipoleExpansion expan;
        expan.SetNumberOfTermsInSeries(fNTerms);
        fDegree = expan.GetDegree();

        fCollector->SetNumberOfTermsInSeries(fNTerms);
        fDistributor->SetNumberOfTermsInSeries(fNTerms);

        fParentsLocalCoeff.clear();
        fParentsLocalCoeff.resize(fNTerms);

        fKernelResponse->SetNumberOfTermsInSeries(fNTerms);

        fInitialized = false;

        fLowerLimits[0] = 0;
        fLowerLimits[1] = 0;
        fUpperLimits[0] = fNTerms;
        fUpperLimits[1] = fNTerms;
        fDimensionSize[0] = fNTerms;
        fDimensionSize[1] = fNTerms;

        fTempCoeff.clear();
        fTempCoeff.resize(fNTerms);

        fTargetCoeff.SetNumberOfTermsInSeries(fNTerms);

        delete[] fCBSI;
        fCBSI = new unsigned int[fNTerms];
        delete[] fRBSI;
        fRBSI = new unsigned int[fNTerms];

        unsigned int count = 0;
        for (int j = 0; j <= fDegree; j++) {
            for (int k = 0; k <= j; k++) {
                fCBSI[count] = KFMScalarMultipoleExpansion::ComplexBasisIndex(j, k);
                fRBSI[count] = KFMScalarMultipoleExpansion::RealBasisIndex(j, k);
                count++;
            }
        }

        fNecessaryTerms = count;
    };

    virtual void SetTopLevelDivisions(int div)
    {
        fTopLevelDivisions = div;
    }

    ////////////////////////////////////////////////////////////////////////
    void SetDivisions(int div)
    {
        fDiv = std::abs(div);

        for (unsigned int i = 0; i < SpatialNDIM; i++) {
            fLowerLimits[i + 2] = 0;
            fUpperLimits[i + 2] = fDiv;
            fDimensionSize[i + 2] = fDiv;
        }

        fTotalSpatialSize = KFMArrayMath::TotalArraySize<SpatialNDIM>(&(fDimensionSize[2]));

        fKernelResponse->SetLowerSpatialLimits(&(fLowerLimits[2]));
        fKernelResponse->SetUpperSpatialLimits(&(fUpperLimits[2]));

        //set the source origin here...the position of the source
        //origin should be measured with respect to the center of the child node that is
        //indexed by (0,0,0), spacing between child nodes should be equal to 1.0
        //scaling for various tree levels is handled elsewhere

        double source_origin[SpatialNDIM] = {0., 0., 0.};

        for (unsigned int i = 0; i < SpatialNDIM; i++) {
            if (fDiv % 2 == 0) {
                source_origin[i] = 0.5 * fLength;
            }
            else {
                source_origin[i] = 0.0;
            }
        }


        int shift[SpatialNDIM];
        for (unsigned int i = 0; i < SpatialNDIM; i++) {
            shift[i] = -1 * ((int) (std::ceil(1.0 * (((double) fDiv) / 2.0))) - 1);
        }

        fKernelResponse->SetOrigin(source_origin);
        fKernelResponse->SetShift(shift);
        fCollector->SetDivisions(fDiv);

        fInitialized = false;
    }


    ////////////////////////////////////////////////////////////////////////
    virtual void Initialize()
    {
        if (!fInitialized) {
            AllocateArrays();

            //here we need to initialize the L2L calculator
            //and fill the array of L2L coefficients
            fKernelResponse->SetZeroMaskSize(fZeroMaskSize);
            fKernelResponse->SetDistance(fLength);
            fKernelResponse->SetOutput(fL2LCoeff);
            fKernelResponse->Initialize();
            fKernelResponse->ExecuteOperation();
            fCollector->SetOutput(fLocalCoeffOrig);
            fCollector->Initialize();

            fInitialized = true;
        }
    }


    ////////////////////////////////////////////////////////////////////////
    void ApplyAction(KFMNode<ObjectTypeList>* node) override
    {
        if (node != nullptr && node->HasChildren() && node->GetLevel() != 0) {

            ScalarMomentType* mom = KFMObjectRetriever<ObjectTypeList, ScalarMomentType>::GetNodeObject(node);
            if (mom != nullptr) {
                //collect the local coefficients of this node
                mom->GetMoments(&fParentsLocalCoeff);

                //if we have a scale invariant kernel, so upon having computed the kernel reponse array once
                //we only have to re-scale the moments, we don't have to recompute the array at each tree level
                //any recomputation of the kernel reponse array for non-invariant kernels must be managed by an external class
                double child_side_length =
                    KFMObjectRetriever<ObjectTypeList, KFMCube<SpatialNDIM>>::GetNodeObject(node->GetChild(0))
                        ->GetLength();
                if (fIsScaleInvariant) {
                    //apply the needed re-scaling for this tree level
                    std::complex<double> scale = std::complex<double>(child_side_length, 0.0);
                    for (unsigned int si = 0; si < fNTerms; si++) {
                        fPtrLocalCoeffSource[si] =
                            (fParentsLocalCoeff[si]) * (fKernelResponse->GetKernel()->GetSourceScaleFactor(si, scale));
                    }
                }

                //copy source moments into the array
                FillParentCoefficientArray();

                //collect the local coefficients of the children of this node
                fCollector->ApplyAction(node);

                //compute the down conversion of this node's local coefficients to it's children
                //by pointwise multiply and sum

                unsigned int tsi = 0;
                for (int j = 0; j <= fDegree; j++) {
                    for (int k = 0; k <= j; k++) {
                        tsi = KFMScalarMultipoleExpansion::ComplexBasisIndex(j, k);

                        //reset to zero
                        //KFMArrayOperator<std::complex<double>, SpatialNDIM>::ResetArray(fIndexedLocalCoeffContrib[tsi], zero);
                        KFMArrayOperator<std::complex<double>, SpatialNDIM>::ZeroArray(fIndexedLocalCoeffContrib[tsi]);

                        for (unsigned int ssi = 0; ssi < fNTerms; ssi++) {

                            //reset to zero
                            KFMArrayOperator<std::complex<double>, SpatialNDIM>::ZeroArray(fTempOut);

                            fMultCalc->SetFirstInput(fDoubleIndexedL2LCoeff[tsi][ssi]);
                            fMultCalc->SetSecondInput(fIndexedLocalCoeffParentArray[ssi]);
                            fMultCalc->SetOutput(fTempOut);
                            fMultCalc->Initialize();
                            fMultCalc->ExecuteOperation();

                            fAddCalc->SetFirstInput(fIndexedLocalCoeffContrib[tsi]);
                            fAddCalc->SetSecondInput(fTempOut);
                            fAddCalc->SetOutput(fIndexedLocalCoeffContrib[tsi]);
                            fAddCalc->Initialize();
                            fAddCalc->ExecuteOperation();
                        }

                        if (fIsScaleInvariant) {
                            //apply the needed re-scaling for this tree level
                            std::complex<double> scale = std::complex<double>(child_side_length, 0.0);
                            //rescale the local coefficient constributions depending on the tree level
                            fScalarMultCalc->SetScalarMultiplicationFactor(
                                fKernelResponse->GetKernel()->GetTargetScaleFactor(tsi, scale));
                            fScalarMultCalc->SetInput(fIndexedLocalCoeffContrib[tsi]);
                            fScalarMultCalc->SetOutput(fIndexedLocalCoeffContrib[tsi]);
                            fScalarMultCalc->Initialize();
                            fScalarMultCalc->ExecuteOperation();
                        }

                        //add in originally existing local coeff
                        fAddCalc->SetFirstInput(fIndexedLocalCoeffContrib[tsi]);
                        fAddCalc->SetSecondInput(fIndexedLocalCoeffOrig[tsi]);
                        fAddCalc->SetOutput(fIndexedLocalCoeffOrig[tsi]);
                        fAddCalc->Initialize();
                        fAddCalc->ExecuteOperation();
                    }
                }


                DistributeParentsCoefficients(node);
            }
        }
    }

  protected:
    ////////////////////////////////////////////////////////////////////////
    void AllocateArrays()
    {
        //raw arrays to store data
        fPtrL2LCoeff = new std::complex<double>[fNTerms * fNTerms * fTotalSpatialSize];
        fPtrLocalCoeffOrig = new std::complex<double>[fNTerms * fTotalSpatialSize];
        fPtrLocalCoeffContrib = new std::complex<double>[fNTerms * fTotalSpatialSize];
        fPtrLocalCoeffParentArray = new std::complex<double>[fNTerms * fTotalSpatialSize];
        fPtrLocalCoeffSource = new std::complex<double>[fNTerms];
        fPtrTempOut = new std::complex<double>[fTotalSpatialSize];

        //array wrappers to access and manipulate data all together
        fL2LCoeff = new KFMArrayWrapper<std::complex<double>, SpatialNDIM + 2>(fPtrL2LCoeff, fDimensionSize);

        fLocalCoeffOrig =
            new KFMArrayWrapper<std::complex<double>, SpatialNDIM + 1>(fPtrLocalCoeffOrig, &(fDimensionSize[1]));

        fLocalCoeffContrib =
            new KFMArrayWrapper<std::complex<double>, SpatialNDIM + 1>(fPtrLocalCoeffContrib, &(fDimensionSize[1]));

        fLocalCoeffParentArray =
            new KFMArrayWrapper<std::complex<double>, SpatialNDIM + 1>(fPtrLocalCoeffParentArray, &(fDimensionSize[1]));

        fTempOut = new KFMArrayWrapper<std::complex<double>, SpatialNDIM>(fPtrTempOut, &(fDimensionSize[2]));

        fL2LCoeff->SetArrayBases(fLowerLimits);

        std::complex<double>* ptr;

        fIndexedL2LCoeff.resize(fNTerms);
        fDoubleIndexedL2LCoeff.resize(fNTerms);
        fIndexedLocalCoeffOrig.resize(fNTerms);
        fIndexedLocalCoeffContrib.resize(fNTerms);
        fIndexedLocalCoeffParentArray.resize(fNTerms);
        for (unsigned int tsi = 0; tsi < fNTerms; tsi++) {
            ptr = &(fPtrL2LCoeff[(tsi * fNTerms) * fTotalSpatialSize]);
            fIndexedL2LCoeff[tsi] =
                new KFMArrayWrapper<std::complex<double>, SpatialNDIM + 1>(ptr, &(fDimensionSize[1]));

            ptr = &(fPtrLocalCoeffOrig[(tsi * fTotalSpatialSize)]);
            fIndexedLocalCoeffOrig[tsi] =
                new KFMArrayWrapper<std::complex<double>, SpatialNDIM>(ptr, &(fDimensionSize[2]));

            ptr = &(fPtrLocalCoeffContrib[(tsi * fTotalSpatialSize)]);
            fIndexedLocalCoeffContrib[tsi] =
                new KFMArrayWrapper<std::complex<double>, SpatialNDIM>(ptr, &(fDimensionSize[2]));

            ptr = &(fPtrLocalCoeffParentArray[(tsi * fTotalSpatialSize)]);
            fIndexedLocalCoeffParentArray[tsi] =
                new KFMArrayWrapper<std::complex<double>, SpatialNDIM>(ptr, &(fDimensionSize[2]));

            fDoubleIndexedL2LCoeff[tsi].resize(fNTerms);

            for (unsigned int ssi = 0; ssi < fNTerms; ssi++) {
                ptr = &(fPtrL2LCoeff[(ssi + tsi * fNTerms) * fTotalSpatialSize]);
                fDoubleIndexedL2LCoeff[tsi][ssi] =
                    new KFMArrayWrapper<std::complex<double>, SpatialNDIM>(ptr, &(fDimensionSize[2]));
            }
        }

        fAllocated = true;
    }


    ////////////////////////////////////////////////////////////////////////
    void DeallocateArrays()
    {

        if (fAllocated) {
            delete[] fPtrL2LCoeff;
            fPtrL2LCoeff = nullptr;
            delete[] fPtrLocalCoeffOrig;
            fPtrLocalCoeffOrig = nullptr;
            delete[] fPtrLocalCoeffContrib;
            fPtrLocalCoeffContrib = nullptr;
            delete[] fPtrLocalCoeffParentArray;
            fPtrLocalCoeffParentArray = nullptr;
            delete[] fPtrLocalCoeffSource;
            fPtrLocalCoeffSource = nullptr;
            delete[] fPtrTempOut;
            fPtrTempOut = nullptr;

            //array wrappers to access and manipulate data all together
            delete fL2LCoeff;
            fL2LCoeff = nullptr;
            delete fLocalCoeffOrig;
            fLocalCoeffOrig = nullptr;
            delete fLocalCoeffContrib;
            fLocalCoeffContrib = nullptr;
            delete fLocalCoeffParentArray;
            fLocalCoeffParentArray = nullptr;
            delete fTempOut;
            fTempOut = nullptr;

            for (unsigned int tsi = 0; tsi < fNTerms; tsi++) {
                delete fIndexedL2LCoeff[tsi];
                delete fIndexedLocalCoeffOrig[tsi];
                delete fIndexedLocalCoeffContrib[tsi];
                delete fIndexedLocalCoeffParentArray[tsi];
                for (unsigned int ssi = 0; ssi < fNTerms; ssi++) {
                    delete fDoubleIndexedL2LCoeff[tsi][ssi];
                }
            }
        }
    }


    ////////////////////////////////////////////////////////////////////////
    void FillParentCoefficientArray()
    {
        //collect the local coefficients of this node
        unsigned int index_t[SpatialNDIM + 1];
        int index[SpatialNDIM + 1];

        for (unsigned int i = 0; i < fNTerms; i++) {
            index_t[0] = i;
            for (unsigned int j = 0; j < fTotalSpatialSize; j++) {
                KFMArrayMath::RowMajorIndexFromOffset<SpatialNDIM>(j, &(fDimensionSize[2]), &(index_t[1]));

                for (unsigned int k = 0; k < SpatialNDIM + 1; k++) {
                    index[k] = index_t[k];
                };
                (*fLocalCoeffParentArray)[index] = fPtrLocalCoeffSource[i];
            }
        }
    }


    ////////////////////////////////////////////////////////////////////////
    virtual void DistributeParentsCoefficients(KFMNode<ObjectTypeList>* parent)
    {
        KFMNode<ObjectTypeList>* child;
        std::complex<double> temp;
        int cbsi, rbsi;

        for (unsigned int n = 0; n < fTotalSpatialSize; n++) {
            child = parent->GetChild(n);

            if (child != nullptr) {
                //retrieve moments
                ScalarMomentType* set = KFMObjectRetriever<ObjectTypeList, ScalarMomentType>::GetNodeObject(child);
                if (set != nullptr) {
                    //we use raw ptr for speed
                    double* rmoments = &((*(set->GetRealMoments()))[0]);
                    double* imoments = &((*(set->GetImaginaryMoments()))[0]);
                    for (unsigned int i = 0; i < fNecessaryTerms; ++i) {
                        cbsi = fCBSI[i];
                        rbsi = fRBSI[i];
                        temp = fPtrLocalCoeffOrig[cbsi * fTotalSpatialSize + n];
                        rmoments[rbsi] = temp.real();
                        imoments[rbsi] = temp.imag();
                    }
                }
            }
        }
    }

    ////////////////////////////////////////////////////////////////////////
    //internal data, basic properties and current state
    int fDegree;
    unsigned int fNTerms;
    unsigned int fNecessaryTerms;
    unsigned int fTotalSpatialSize;
    int fDiv;
    int fTopLevelDivisions;
    int fZeroMaskSize;  //this is always set to zero!
    double fLength;
    bool fInitialized;
    bool fIsScaleInvariant;
    bool fAllocated;

    //array access storage indices
    unsigned int* fCBSI;
    unsigned int* fRBSI;

    //limits, and size
    int fLowerLimits[SpatialNDIM + 2];
    int fUpperLimits[SpatialNDIM + 2];
    unsigned int fDimensionSize[SpatialNDIM + 2];

    //calculators
    KFMKernelResponseArray<KernelType, true, SpatialNDIM>* fKernelResponse;
    KFMArrayScalarMultiplier<std::complex<double>, SpatialNDIM>* fScalarMultCalc;
    KFMPointwiseArrayAdder<std::complex<double>, SpatialNDIM>* fAddCalc;
    KFMPointwiseArrayMultiplier<std::complex<double>, SpatialNDIM>* fMultCalc;

    //collector and distributor of the local coeff
    KFMScalarMomentCollector<ObjectTypeList, ScalarMomentType, SpatialNDIM>* fCollector;
    KFMScalarMomentDistributor<ObjectTypeList, ScalarMomentType>* fDistributor;

    //array wrappers to access and manipulate data all together
    KFMArrayWrapper<std::complex<double>, SpatialNDIM + 2>* fL2LCoeff;
    std::vector<KFMArrayWrapper<std::complex<double>, SpatialNDIM + 1>*> fIndexedL2LCoeff;
    std::vector<std::vector<KFMArrayWrapper<std::complex<double>, SpatialNDIM>*>> fDoubleIndexedL2LCoeff;
    KFMArrayWrapper<std::complex<double>, SpatialNDIM + 1>* fLocalCoeffOrig;
    std::vector<KFMArrayWrapper<std::complex<double>, SpatialNDIM>*> fIndexedLocalCoeffOrig;
    KFMArrayWrapper<std::complex<double>, SpatialNDIM + 1>* fLocalCoeffContrib;
    std::vector<KFMArrayWrapper<std::complex<double>, SpatialNDIM>*> fIndexedLocalCoeffContrib;
    KFMArrayWrapper<std::complex<double>, SpatialNDIM + 1>* fLocalCoeffParentArray;
    std::vector<KFMArrayWrapper<std::complex<double>, SpatialNDIM>*> fIndexedLocalCoeffParentArray;
    KFMArrayWrapper<std::complex<double>, SpatialNDIM>* fTempOut;

    //raw arrays to store coefficient data
    std::complex<double>* fPtrL2LCoeff;
    std::complex<double>* fPtrLocalCoeffOrig;
    std::complex<double>* fPtrLocalCoeffContrib;
    std::complex<double>* fPtrTempOut;
    std::complex<double>* fPtrLocalCoeffSource;            //parents local coeff in small array
    std::complex<double>* fPtrLocalCoeffParentArray;       //parents local coeff in large array
    std::vector<std::complex<double>> fParentsLocalCoeff;  //just for retrieval

    std::vector<std::complex<double>> fTempCoeff;  //just for distribution
    ScalarMomentType fTargetCoeff;
};


}  // namespace KEMField


#endif /* __KFMScalarMomentLocalToLocalConverter_H__ */
