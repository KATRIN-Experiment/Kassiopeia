#ifndef KFMNodeActor_HH__
#define KFMNodeActor_HH__

namespace KEMField{

/**
*
*@file KFMNodeActor.hh
*@class KFMNodeActor
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Fri Aug  9 13:34:56 EDT 2013 J. Barrett (barrettj@mit.edu) First Version
*
*/

template<typename NodeType >
class KFMNodeActor
{
    public:
        KFMNodeActor(){};
        virtual ~KFMNodeActor(){};

        virtual void ApplyAction( NodeType* node) = 0;

    private:

};




} //end of KEMField

#endif /* KFMNodeActor_H__ */
