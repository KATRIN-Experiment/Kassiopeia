#ifndef KFMObjectHolder_HH__
#define KFMObjectHolder_HH__

#include <cstddef>

namespace KEMField
{

/**
*
*@file KFMObjectHolder.hh
*@class KFMObjectHolder
*@brief;
*@details
* Container for a pointer to an object. If the holder is not the owner of the object point
* then external management must ensure that the pointer is set to null before the destructor is called
*<b>Revision History:<b>
*Date Name Brief Description
* J. Barrett (barrettj@mit.edu) First Version
*
*/


template<typename T> class KFMObjectHolder
{
  public:
    KFMObjectHolder() : fObject(nullptr)
    {
        ;
    };

    virtual ~KFMObjectHolder()
    {
        delete fObject;
    };

    T* fObject;
};


}  // namespace KEMField


#endif /* KFMObjectHolder_H__ */
