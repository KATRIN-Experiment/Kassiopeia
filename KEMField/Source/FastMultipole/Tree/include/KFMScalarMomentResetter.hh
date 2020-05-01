#ifndef __KFMScalarMomentResetter_H__
#define __KFMScalarMomentResetter_H__

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
*@file KFMScalarMomentResetter.hh
*@class KFMScalarMomentResetter
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
class KFMScalarMomentResetter : public KFMNodeActor<KFMNode<ObjectTypeList>>
{
  public:
    KFMScalarMomentResetter()
    {
        ;
    };
    ~KFMScalarMomentResetter() override
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
                ScalarMomentType* set = KFMObjectRetriever<ObjectTypeList, ScalarMomentType>::GetNodeObject(node);
                set->SetNumberOfTermsInSeries(fNTerms);
                set->Clear();
            }
        }
    }

  private:
    unsigned int fNTerms;
};

}  // namespace KEMField

#endif /* __KFMScalarMomentResetter_H__ */
