#ifndef KGObjectCollection_HH__
#define KGObjectCollection_HH__


#include "KGObjectHolder.hh"
#include "KGTypelist.hh"

namespace KGeoBag
{

/*
*
*@file KGObjectCollection.hh
*@class KGObjectCollection
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Mon Aug 12 04:44:27 EDT 2013 J. Barrett (barrettj@mit.edu) First Version
*
*/

template<typename TypeList> class KGObjectCollection : public KGGenScatterHierarchy<TypeList, KGObjectHolder>
{
  public:
    KGObjectCollection() = default;
    ;
    ~KGObjectCollection() override = default;
    ;

  private:
};


}  // namespace KGeoBag


#endif /* KGObjectCollection_H__ */
