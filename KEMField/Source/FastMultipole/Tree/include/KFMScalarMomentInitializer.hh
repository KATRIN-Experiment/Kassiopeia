#ifndef __KFMScalarMomentInitializer_H__
#define __KFMScalarMomentInitializer_H__

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
*@file KFMScalarMomentInitializer.hh
*@class KFMScalarMomentInitializer
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Thu Aug 28 21:47:19 EDT 2012 J. Barrett (barrettj@mit.edu) First Version
*
*/

template<typename ObjectTypeList,
         typename ScalarMomentType>  //ScalarMomentType should inherit from KFMScalarMomentExpansion
class KFMScalarMomentInitializer : public KFMNodeActor<KFMNode<ObjectTypeList>>
{
  public:
    KFMScalarMomentInitializer()
    {
        ;
    };
    ~KFMScalarMomentInitializer() override
    {
        ;
    };

    virtual void SetNumberOfTermsInSeries(unsigned int n_terms)
    {
        fNTerms = n_terms;
    }

    void ApplyAction(KFMNode<ObjectTypeList>* node) override
    {
        if (node != nullptr) {
            if (KFMObjectRetriever<ObjectTypeList, ScalarMomentType>::GetNodeObject(node) != nullptr) {
                delete KFMObjectRetriever<ObjectTypeList, ScalarMomentType>::GetNodeObject(node);
                auto* set = new ScalarMomentType();
                set->SetNumberOfTermsInSeries(fNTerms);
                set->Clear();
                KFMObjectRetriever<ObjectTypeList, ScalarMomentType>::SetNodeObject(set, node);
            }
            else {
                auto* set = new ScalarMomentType();
                set->SetNumberOfTermsInSeries(fNTerms);
                set->Clear();
                KFMObjectRetriever<ObjectTypeList, ScalarMomentType>::SetNodeObject(set, node);
            }
        }
    }

  private:
    unsigned int fNTerms;
};

}  // namespace KEMField

#endif /* __KFMScalarMomentInitializer_H__ */
