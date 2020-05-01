#ifndef KFMScalarMomentDistributor_H__
#define KFMScalarMomentDistributor_H__

#include "KFMArrayFillingOperator.hh"
#include "KFMNode.hh"
#include "KFMNodeActor.hh"
#include "KFMObjectRetriever.hh"

#include <cmath>
#include <complex>
#include <vector>

namespace KEMField
{

/**
*
*@file KFMScalarMomentDistributor.hh
*@class KFMScalarMomentDistributor
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Wed Oct  3 19:41:01 EDT 2012 J. Barrett (barrettj@mit.edu) First Version
*
*/

template<typename ObjectTypeList, typename ScalarMomentType>
class KFMScalarMomentDistributor : public KFMNodeActor<KFMNode<ObjectTypeList>>
{
  public:
    KFMScalarMomentDistributor()
    {
        fNTerms = 0;
    };
    ~KFMScalarMomentDistributor() override
    {
        ;
    };

    virtual void SetNumberOfTermsInSeries(unsigned int n_terms)
    {
        fNTerms = n_terms;
        fOriginalMoments.clear();
        fOriginalMoments.resize(fNTerms);
        fAddMoments.clear();
        fAddMoments.resize(fNTerms);

        for (unsigned int i = 0; i < fNTerms; i++) {
            fOriginalMoments[i] = std::complex<double>(0, 0);
            fAddMoments[i] = std::complex<double>(0, 0);
        }
    }

    void SetExpansionToAdd(const ScalarMomentType* expansion_to_add)
    {
        fAddExpansion = expansion_to_add;
        fUseExpansionAdd = true;
    };

    void SetExpansionToSet(const ScalarMomentType* expansion_to_add)
    {
        fAddExpansion = expansion_to_add;
        fUseExpansionAdd = false;
    };


    void ApplyAction(KFMNode<ObjectTypeList>* node) override
    {
        if (node != nullptr) {
            if (KFMObjectRetriever<ObjectTypeList, ScalarMomentType>::GetNodeObject(node) != nullptr) {
                if (fUseExpansionAdd) {
                    (*(KFMObjectRetriever<ObjectTypeList, ScalarMomentType>::GetNodeObject(node))) += (*fAddExpansion);
                }
                else {
                    (*(KFMObjectRetriever<ObjectTypeList, ScalarMomentType>::GetNodeObject(node))) = (*fAddExpansion);
                }
            }
        }
    }

  private:
    unsigned int fNTerms;
    std::vector<std::complex<double>> fAddMoments;
    std::vector<std::complex<double>> fOriginalMoments;

    bool fUseExpansionAdd;
    const ScalarMomentType* fAddExpansion;
};


}  // namespace KEMField

#endif /* __KFMScalarMomentDistributor_H__ */
