#include "KMessageBuilder.h"

#include "KElementProcessor.hh"
#include "KRoot.h"

using namespace std;

namespace katrin
{

KMessageData::KMessageData() :
    fKey("none"),
    fFormat(cout.flags()),
    fPrecision(cout.precision()),
    fShowShutdownMessage(true),
    fTerminalVerbosity(eNormal),
    fLogVerbosity(eNormal)
{}
KMessageData::~KMessageData() {}

template<> KMessageDataBuilder::~KComplexElement() {}

STATICINT sKMessageDataStructure =
    KMessageDataBuilder::Attribute<string>("key") + KMessageDataBuilder::Attribute<string>("terminal") +
    KMessageDataBuilder::Attribute<string>("log") + KMessageDataBuilder::Attribute<string>("format") +
    KMessageDataBuilder::Attribute<KMessagePrecision>("precision") +
    KMessageDataBuilder::Attribute<bool>("shutdown_message");

template<> KMessageTableBuilder::~KComplexElement() {}

STATICINT sKMessageTableStructure = KMessageTableBuilder::Attribute<string>("terminal") +
                                    KMessageTableBuilder::Attribute<string>("log") +
                                    KMessageTableBuilder::Attribute<string>("format") +
                                    KMessageTableBuilder::Attribute<KMessagePrecision>("precision") +
                                    KMessageTableBuilder::Attribute<bool>("shutdown_message") +
                                    KMessageTableBuilder::ComplexElement<KTextFile>("file") +
                                    KMessageTableBuilder::ComplexElement<KMessageData>("message");

STATICINT sMessageTable = KRootBuilder::ComplexElement<KMessageTable>("messages");
STATICINT sMessageTableCompat = KElementProcessor::ComplexElement<KMessageTable>("messages");

}  // namespace katrin
