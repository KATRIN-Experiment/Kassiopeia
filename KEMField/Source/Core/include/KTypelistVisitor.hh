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

template<class Policy, class Base = KVisitorBase> class KVisitorType : public Base
{
  public:
    using Base::Visit;

    ~KVisitorType() override {}

    virtual void Visit(Policy&) = 0;
};

template<class Policy, class Base> class KNonVisitorType : public Base
{
  public:
    using Base::Visit;

    KNonVisitorType() {}
    ~KNonVisitorType() override {}

    void Visit(Policy&) override {}
};

/**
* @class KSelectiveVisitor
*
* @brief A visitor for a subset of a typelist.
*
* @author T.J. Corona
*/

template<class Visitor, class VisitedList>
class KSelectiveVisitor :
    public KGenLinearHierarchy<typename RemoveTypelist<typename Visitor::AcceptedTypes, VisitedList>::Result,
                               KNonVisitorType, Visitor>
{
  public:
    ~KSelectiveVisitor() override {}
};

template<class Visitor> class KSelectiveVisitor<Visitor, typename Visitor::AcceptedTypes> : public Visitor
{
  public:
    ~KSelectiveVisitor() override {}
};
}  // namespace KEMField

#endif /* KTYPELISTVISITOR_DEF */
