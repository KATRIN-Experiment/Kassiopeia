#ifndef KSAOutputObject_HH__
#define KSAOutputObject_HH__



namespace KEMField{

/**
*
*@file KSAOutputObject.hh
*@class KSAOutputObject
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Sat Dec 29 19:24:13 EST 2012 J. Barrett (barrettj@mit.edu) First Version
*
*/

class KSAOutputNode;

class KSAOutputObject
{
    public:
        KSAOutputObject(){;};
        virtual ~KSAOutputObject(){;};

        //defines the children to add to the node associated with this object
        virtual void DefineOutputNode(KSAOutputNode* node) const = 0;

    protected:

};



}

#endif /* KSAOutputObject_H__ */
