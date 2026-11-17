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
    fShowParserContext(true),
    fTerminalVerbosity(eNormal),
    fLogVerbosity(eNormal)
{}
KMessageData::~KMessageData() = default;

template<> KMessageDataBuilder::~KComplexElement() = default;

STATICINT sKMessageDataStructure =
    KMessageDataBuilder::Attribute<std::string>("key") + KMessageDataBuilder::Attribute<std::string>("terminal") +
    KMessageDataBuilder::Attribute<std::string>("log") + KMessageDataBuilder::Attribute<std::string>("format") +
    KMessageDataBuilder::Attribute<KMessagePrecision>("precision") +
    KMessageDataBuilder::Attribute<bool>("shutdown_message") + KMessageDataBuilder::Attribute<bool>("parser_context");

template<> KMessageTableBuilder::~KComplexElement() = default;

STATICINT sKMessageTableStructure =
    KMessageTableBuilder::Attribute<std::string>("terminal") + KMessageTableBuilder::Attribute<std::string>("log") +
    KMessageTableBuilder::Attribute<std::string>("format") +
    KMessageTableBuilder::Attribute<KMessagePrecision>("precision") +
    KMessageTableBuilder::Attribute<bool>("shutdown_message") +
    KMessageTableBuilder::Attribute<bool>("parser_context") +
    //! the log file to which messages are written. Its default directory is the log directory of the
    //! installation and its default name ``KasperLog.txt`` (``default_path`` and ``default_base`` are
    //! ignored here).
    KMessageTableBuilder::ComplexElement<KTextFile>("file") +
    KMessageTableBuilder::ComplexElement<KMessageData>("message");

STATICINT sMessageTable = KRootBuilder::ComplexElement<KMessageTable>("messages");
STATICINT sMessageTableCompat = KElementProcessor::ComplexElement<KMessageTable>("messages");

}  // namespace katrin
