#ifndef KSAFixedSizeInputOutputObject_HH__
#define KSAFixedSizeInputOutputObject_HH__

#include "KSAInputOutputObject.hh"

namespace KEMField{

/**
*
*@file KSAFixedSizeInputOutputObject.hh
*@class KSAFixedSizeInputOutputObject
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Thu Jan 31 15:02:11 EST 2013 J. Barrett (barrettj@mit.edu) First Version
*
*/


class KSAFixedSizeInputOutputObject: public KSAInputOutputObject
{
    public:
        KSAFixedSizeInputOutputObject(){;};
        virtual ~KSAFixedSizeInputOutputObject(){;};

        //we probably ought to add a function which declares the size (number of items) in this classes

//        //inherits these functions, which are overloaded in the derived class
//        virtual void Initialize(){;};
//        virtual void DefineOutputNode(KSAOutputNode* node) const = 0;
//        virtual void DefineInputNode(KSAInputNode* node) = 0;

    protected:

};


}


#endif /* KSAFixedSizeInputOutputObject_H__ */
