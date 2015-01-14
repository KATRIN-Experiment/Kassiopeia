#ifndef KELECTROMAGNETVISITOR_DEF
#define KELECTROMAGNETVISITOR_DEF

#include "KElectromagnetTypes.hh"

#include "KTypelistVisitor.hh"

namespace KEMField
{
  class KElectromagnetVisitor : public KGenLinearHierarchy<KElectromagnetTypes,
							   KVisitorType,
							   KVisitorBase>
  {
  public:
    typedef KElectromagnetTypes AcceptedTypes;

    KElectromagnetVisitor() {}
    virtual ~KElectromagnetVisitor() {}
  };
}

#endif /* KELECTROMAGNETVISITOR_DEF */
