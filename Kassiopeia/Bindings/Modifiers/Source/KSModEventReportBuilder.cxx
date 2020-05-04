#include "KSModEventReportBuilder.h"

#include "KSRootBuilder.h"

#include <string>

using namespace Kassiopeia;
namespace katrin
{

template<> KSModEventReportBuilder::~KComplexElement() {}

STATICINT sKSModEventReportStructure = KSModEventReportBuilder::Attribute<std::string>("name");

STATICINT sKSModEventReport = KSRootBuilder::ComplexElement<KSModEventReport>("ksmod_event_report");

}  // namespace katrin
