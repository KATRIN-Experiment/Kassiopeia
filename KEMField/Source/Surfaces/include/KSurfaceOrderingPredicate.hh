#ifndef KSurfaceOrderingPredicate_HH__
#define KSurfaceOrderingPredicate_HH__

#include "../../../Surfaces/include/KSurfaceContainer.hh"
#include "../../../Surfaces/include/KSurfaceVisitors.hh"

namespace KEMField
{

/*
*
*@file KSurfaceOrderingPredicate.hh
*@class KSurfaceOrderingPredicate
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Fri Apr 11 10:23:57 EDT 2014 J. Barrett (barrettj@mit.edu) First Version
*
*/

template<class Visitor, class VisitedList>
class KSurfaceOrderingPredicate : public KSelectiveVisitor<Visitor, VisitedList>
{
  public:
    KSurfaceOrderingPredicate(){};
    virtual ~KSurfaceOrderingPredicate(){};

    virtual void SetSurfaceContainer(const KSurfaceContainer& container)
    {
        fSurfaceContainer = container;
    };
    virtual void Initialize()
    {
        ;
    }

    //the ordering operator
    virtual bool operator()(int i, int j)
    {
        return i < j;
    }  //default is normal ordering

  private:
    const KSurfaceContainer& fSurfaceContainer;
};


}  // namespace KEMField


#endif /* KSurfaceOrderingPredicate_H__ */
