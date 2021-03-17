#ifndef KELECTROMAGNETVISITOR_DEF
#define KELECTROMAGNETVISITOR_DEF

#include "KElectromagnetTypes.hh"
#include "KTypelistVisitor.hh"

namespace KEMField
{
class KElectromagnetVisitor : public KGenLinearHierarchy<KElectromagnetTypes, KVisitorType, KVisitorBase>
{
  public:
    using AcceptedTypes = KElectromagnetTypes;

    KElectromagnetVisitor() = default;
    ~KElectromagnetVisitor() override = default;
};
}  // namespace KEMField

#endif /* KELECTROMAGNETVISITOR_DEF */
