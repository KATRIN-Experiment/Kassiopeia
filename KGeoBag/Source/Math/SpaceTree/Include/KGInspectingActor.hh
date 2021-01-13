#ifndef KGInspectingActor_HH__
#define KGInspectingActor_HH__

namespace KGeoBag
{

/*
*
*@file KGInspectingActor.hh
*@class KGInspectingActor
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Mon Aug 12 11:46:55 EDT 2013 J. Barrett (barrettj@mit.edu) First Version
*
*/

template<typename NodeType> class KGInspectingActor
{
  public:
    KGInspectingActor() = default;
    ;
    virtual ~KGInspectingActor() = default;
    ;

    //needs to answer this question about whether this node statisfies a condition
    virtual bool ConditionIsSatisfied(NodeType* node) = 0;

  private:
};


}  // namespace KGeoBag

#endif /* KGInspectingActor_H__ */
