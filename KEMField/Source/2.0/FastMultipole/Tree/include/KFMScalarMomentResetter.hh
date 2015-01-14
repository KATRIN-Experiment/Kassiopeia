#ifndef __KFMScalarMomentResetter_H__
#define __KFMScalarMomentResetter_H__

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

template< typename ObjectTypeList, typename ScalarMomentType> //ScalarMomentType should inherit from KFMScalarMomentExpansion
class KFMScalarMomentResetter: public KFMNodeActor< KFMNode<ObjectTypeList> >
{
    public:
        KFMScalarMomentResetter(){;};
        virtual ~KFMScalarMomentResetter(){;};

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
                    ScalarMomentType* set = KFMObjectRetriever<ObjectTypeList, ScalarMomentType>::GetNodeObject(node);
                    set->SetNumberOfTermsInSeries(fNTerms);
                    set->Clear();
                }
            }
        }

    private:

        unsigned int fNTerms;
};

}//end of namespace

#endif /* __KFMScalarMomentResetter_H__ */
