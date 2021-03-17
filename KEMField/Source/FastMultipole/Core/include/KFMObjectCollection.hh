#ifndef KFMObjectCollection_HH__
#define KFMObjectCollection_HH__


#include "KFMObjectHolder.hh"
#include "KTypelist.hh"

namespace KEMField
{

/*
*
*@file KFMObjectCollection.hh
*@class KFMObjectCollection
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Mon Aug 12 04:44:27 EDT 2013 J. Barrett (barrettj@mit.edu) First Version
*
*/

template<typename TypeList> class KFMObjectCollection : public KGenScatterHierarchy<TypeList, KFMObjectHolder>
{
  public:
    KFMObjectCollection() = default;
    ;
    ~KFMObjectCollection() override = default;
    ;

  private:
};


}  // namespace KEMField


#endif /* KFMObjectCollection_H__ */
