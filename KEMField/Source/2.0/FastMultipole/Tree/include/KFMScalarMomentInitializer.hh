#ifndef __KFMScalarMomentInitializer_H__
#define __KFMScalarMomentInitializer_H__

#include <vector>
#include <complex>
#include <cmath>

#include "KFMNode.hh"
#include "KFMNodeActor.hh"
#include "KFMObjectRetriever.hh"

#include "KFMArrayFillingOperator.hh"

namespace KEMField{

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

template< typename ObjectTypeList, typename ScalarMomentType> //ScalarMomentType should inherit from KFMScalarMomentExpansion
class KFMScalarMomentInitializer: public KFMNodeActor< KFMNode<ObjectTypeList> >
{
    public:
        KFMScalarMomentInitializer(){;};
        virtual ~KFMScalarMomentInitializer(){;};

        virtual void SetNumberOfTermsInSeries(unsigned int n_terms)
        {
            fNTerms = n_terms;
        }

        virtual void ApplyAction(KFMNode<ObjectTypeList>* node)
        {
            if(node != NULL)
            {
                if(KFMObjectRetriever<ObjectTypeList, ScalarMomentType>::GetNodeObject(node) != NULL)
                {
                    delete KFMObjectRetriever<ObjectTypeList, ScalarMomentType>::GetNodeObject(node);
                    ScalarMomentType* set = new ScalarMomentType();
                    set->SetNumberOfTermsInSeries(fNTerms);
                    set->Clear();
                    KFMObjectRetriever<ObjectTypeList, ScalarMomentType>::SetNodeObject(set, node);
                }
                else
                {
                    ScalarMomentType* set = new ScalarMomentType();
                    set->SetNumberOfTermsInSeries(fNTerms);
                    set->Clear();
                    KFMObjectRetriever<ObjectTypeList, ScalarMomentType>::SetNodeObject(set, node);
                }
            }
        }

    private:

        unsigned int fNTerms;
};

}//end of namespace

#endif /* __KFMScalarMomentInitializer_H__ */
