#ifndef KGObjectHolder_HH__
#define KGObjectHolder_HH__

#include <cstddef>

namespace KGeoBag{

/**
*
*@file KGObjectHolder.hh
*@class KGObjectHolder
*@brief;
*@details
* Container for a pointer to an object. If the holder is not the owner of the object point
* then external management must ensure that the pointer is set to null before the destructor is called
*<b>Revision History:<b>
*Date Name Brief Description
* J. Barrett (barrettj@mit.edu) First Version
*
*/


template<typename T>
class KGObjectHolder
{
    public:

        KGObjectHolder():fObject(NULL){;};

        virtual ~KGObjectHolder()
        {
            delete fObject;
        };

        T* fObject;

};



}//end of KGeoBag



#endif /* KGObjectHolder_H__ */
