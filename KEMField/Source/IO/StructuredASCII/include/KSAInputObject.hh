#ifndef KSAInputObject_HH__
#define KSAInputObject_HH__


namespace KEMField
{

/**
*
*@file KSAInputObject.hh
*@class KSAInputObject
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Sat Dec 29 19:24:13 EST 2012 J. Barrett (barrettj@mit.edu) First Version
*
*/

class KSAInputNode;

class KSAInputObject
{
  public:
    KSAInputObject()
    {
        ;
    };
    virtual ~KSAInputObject()
    {
        ;
    };

    virtual void Initialize()
    {
        ;
    };

    //defines the children to add to the node associated with this object
    virtual void DefineInputNode(KSAInputNode* node) = 0;

  protected:
};


}  // namespace KEMField

#endif /* KSAInputObject_H__ */
