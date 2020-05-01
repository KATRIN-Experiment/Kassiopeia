#ifndef KELECTROMAGNETVISITOR_DEF
#define KELECTROMAGNETVISITOR_DEF

#include "KElectromagnetTypes.hh"
#include "KTypelistVisitor.hh"

namespace KEMField
{
class KElectromagnetVisitor : public KGenLinearHierarchy<KElectromagnetTypes, KVisitorType, KVisitorBase>
{
  public:
    typedef KElectromagnetTypes AcceptedTypes;

    KElectromagnetVisitor() {}
    ~KElectromagnetVisitor() override {}
};
}  // namespace KEMField

#endif /* KELECTROMAGNETVISITOR_DEF */
