#ifndef KTYPELISTVISITOR_DEF
#define KTYPELISTVISITOR_DEF

#include "KTypelist.hh"

namespace KEMField
{

  class KVisitorBase
  {
  public:
    virtual ~KVisitorBase() {}

    virtual void Visit(KEmptyType&) {}
  };

  template <class Policy, class Base=KVisitorBase>
  class KVisitorType : public Base
  {
  public:
    using Base::Visit;

    virtual ~KVisitorType() {}

    virtual void Visit(Policy&) = 0;
  };

  template <class Policy, class Base>
  class KNonVisitorType : public Base
  {
  public:
    using Base::Visit;

    KNonVisitorType() {}
      virtual ~KNonVisitorType() {}

    virtual void Visit(Policy&) {}
  };

/**
* @class KSelectiveVisitor
*
* @brief A visitor for a subset of a typelist.
*
* @author T.J. Corona
*/

  template <class Visitor, class VisitedList>
  class KSelectiveVisitor :
    public KGenLinearHierarchy<typename RemoveTypelist<typename Visitor::AcceptedTypes,VisitedList>::Result,KNonVisitorType,Visitor>
  {
  public:
    virtual ~KSelectiveVisitor() {}
  };

  template <class Visitor>
  class KSelectiveVisitor<Visitor,typename Visitor::AcceptedTypes> :
    public Visitor
  {
  public:
    virtual ~KSelectiveVisitor() {}
  };
}

#endif /* KTYPELISTVISITOR_DEF */
