#ifndef KGNodeActor_HH__
#define KGNodeActor_HH__

namespace KGeoBag
{

/**
*
*@file KGNodeActor.hh
*@class KGNodeActor
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Fri Aug  9 13:34:56 EDT 2013 J. Barrett (barrettj@mit.edu) First Version
*
*/

template<typename NodeType> class KGNodeActor
{
  public:
    KGNodeActor(){};
    virtual ~KGNodeActor(){};

    virtual void ApplyAction(NodeType* node) = 0;

  private:
};


}  // namespace KGeoBag

#endif /* KGNodeActor_H__ */
