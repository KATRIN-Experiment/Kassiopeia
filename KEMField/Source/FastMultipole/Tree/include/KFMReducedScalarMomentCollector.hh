#ifndef KFMReducedScalarMomentCollector_H__
#define KFMReducedScalarMomentCollector_H__


#include "KFMArrayFillingOperator.hh"
#include "KFMNode.hh"
#include "KFMNodeActor.hh"
#include "KFMObjectRetriever.hh"

#include <cmath>
#include <complex>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace KEMField
{

/**
*
*@file KFMReducedScalarMomentCollector.hh
*@class KFMReducedScalarMomentCollector
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Mon Oct  1 16:22:57 EDT 2012 J. Barrett (barrettj@mit.edu) First Version
*
*/

//ScalarMomentType should inherit from KFMScalarMomentExpansion

template<typename ObjectTypeList, typename ScalarMomentType, unsigned int SpatialNDIM>
class KFMReducedScalarMomentCollector :
    public KFMNodeActor<KFMNode<ObjectTypeList>>,
    public KFMArrayFillingOperator<std::complex<double>, SpatialNDIM + 1>
{
  public:
    KFMReducedScalarMomentCollector() : KFMArrayFillingOperator<std::complex<double>, SpatialNDIM + 1>()
    {
        fChild = nullptr;
        for (unsigned int i = 0; i < SpatialNDIM + 1; i++) {
            fLowerLimits[i] = 0;
            fUpperLimits[i] = 0;
        }

        fInitialized = false;
    }


    ~KFMReducedScalarMomentCollector() override
    {
        ;
    };

    void Initialize() override
    {
        fInitialized = false;
        if (this->fOutput != nullptr) {
            if (this->IsBoundedDomainSubsetOfArray(this->fOutput, fLowerLimits, fUpperLimits)) {
                fInitialized = true;
            }
        }
    };

    virtual void SetNumberOfTermsInSeries(unsigned int n_terms)
    {
        fNTerms = n_terms;
        fLowerLimits[0] = 0;
        fUpperLimits[0] = fNTerms;
    }

    virtual void SetDivisions(int div)
    {
        fDiv = std::abs(div);
        for (unsigned int i = 1; i < SpatialNDIM + 1; i++) {
            fLowerLimits[i] = 0;
            fUpperLimits[i] = fDiv;
        }
    };

    void ApplyAction(KFMNode<ObjectTypeList>* node) override
    {
        fPrimaryNode = node;
        ExecuteOperation();
    };

    void ExecuteOperation() override
    {
        this->ZeroArray(this->fOutput);

        if (fPrimaryNode != nullptr && fInitialized) {
            if (fPrimaryNode->HasChildren()) {
                if (fPrimaryNode->GetLevel() == 0) {
                    //loop over top level children
                    for (unsigned int n = 0; n < fPrimaryNode->GetNChildren(); n++) {
                        fChild = fPrimaryNode->GetChild(n);
                        FillFromTopLevelChild(fChild);
                    }
                }
                else {
                    //loop over children
                    for (unsigned int n = 0; n < fPrimaryNode->GetNChildren(); n++) {
                        fChild = fPrimaryNode->GetChild(n);
                        FillFromChild(fChild);
                    }
                }
            }
        }
    }

  protected:
    virtual void FillFromChild(KFMNode<ObjectTypeList>* child)
    {
        unsigned int child_storage_index = child->GetIndex();

        int index[SpatialNDIM + 1];
        unsigned int spatial_index[SpatialNDIM];

        if (child != nullptr && KFMObjectRetriever<ObjectTypeList, ScalarMomentType>::GetNodeObject(child) != nullptr) {
            ScalarMomentType* scalar_moments =
                KFMObjectRetriever<ObjectTypeList, ScalarMomentType>::GetNodeObject(child);

            std::vector<double>* real_moments = scalar_moments->GetRealMoments();
            std::vector<double>* imag_moments = scalar_moments->GetImaginaryMoments();

            const unsigned int* spatial_dim_size =
                KFMObjectRetriever<ObjectTypeList, KFMCubicSpaceTreeProperties<SpatialNDIM>>::GetNodeObject(child)
                    ->GetDimensions();

            KFMArrayMath::RowMajorIndexFromOffset<SpatialNDIM>(child_storage_index, spatial_dim_size, spatial_index);
            for (unsigned int i = 0; i < SpatialNDIM; i++) {
                index[i + 1] = spatial_index[i];
            }

            for (unsigned int n = 0; n < fNTerms; n++) {
                index[0] = n;
                (*(this->fOutput))[index] = std::complex<double>((*real_moments)[n], (*imag_moments)[n]);
            }
        }
    }

    virtual void FillFromTopLevelChild(KFMNode<ObjectTypeList>* child)
    {
        unsigned int child_storage_index = child->GetIndex();

        int index[SpatialNDIM + 1];
        unsigned int spatial_index[SpatialNDIM];

        if (child != nullptr && KFMObjectRetriever<ObjectTypeList, ScalarMomentType>::GetNodeObject(child) != nullptr) {
            ScalarMomentType* scalar_moments =
                KFMObjectRetriever<ObjectTypeList, ScalarMomentType>::GetNodeObject(child);

            std::vector<double>* real_moments = scalar_moments->GetRealMoments();
            std::vector<double>* imag_moments = scalar_moments->GetImaginaryMoments();

            const unsigned int* spatial_dim_size =
                KFMObjectRetriever<ObjectTypeList, KFMCubicSpaceTreeProperties<SpatialNDIM>>::GetNodeObject(child)
                    ->GetTopLevelDimensions();

            KFMArrayMath::RowMajorIndexFromOffset<SpatialNDIM>(child_storage_index, spatial_dim_size, spatial_index);
            for (unsigned int i = 0; i < SpatialNDIM; i++) {
                index[i + 1] = spatial_index[i];
            }

            for (unsigned int n = 0; n < fNTerms; n++) {
                index[0] = n;
                (*(this->fOutput))[index] = std::complex<double>((*real_moments)[n], (*imag_moments)[n]);
            }
        }
    }

    bool fInitialized;
    double fLength;  //side length of the region
    unsigned int fNTerms;
    unsigned int fDiv;  //number of divisions along each side of region

    int fLowerLimits[SpatialNDIM + 1];
    int fUpperLimits[SpatialNDIM + 1];
    double fOrigin[SpatialNDIM];

    KFMNode<ObjectTypeList>* fPrimaryNode;
    KFMNode<ObjectTypeList>* fChild;

    unsigned int fIndex[SpatialNDIM];
};


}  // namespace KEMField


#endif /* __KFMReducedScalarMomentCollector_H__ */
