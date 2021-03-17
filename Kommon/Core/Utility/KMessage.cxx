#include "KMessage.h"

#include "KException.h"
#include "KXMLInitializer.hh"

#include <iomanip>
#include <ostream>

#ifdef KASPER_USE_BOOST
#include <boost/stacktrace.hpp>
#else
#include <execinfo.h>
#endif

using namespace std;

namespace katrin
{

KMessage::KMessage(const string& aKey, const string& aDescription, const string& aPrefix, const string& aSuffix) :
    fKey(aKey),
    fSystemDescription(aDescription),
    fSystemPrefix(aPrefix),
    fSystemSuffix(aSuffix),

    fErrorColorPrefix("\33[31;1m"),  // bold red
    fErrorColorSuffix("\33[0m"),
    fErrorDescription("ERROR"),

    fWarningColorPrefix("\33[33;1m"),  // bold yellow
    fWarningColorSuffix("\33[0m"),
    fWarningDescription("WARNING"),

    fNormalColorPrefix("\33[32;1m"),  // bold green
    fNormalColorSuffix("\33[0m"),
    fNormalDescription("NORMAL"),

    fInfoColorPrefix("\33[34;1m"),  // bold blue
    fInfoColorSuffix("\33[0m"),
    fInfoDescription("INFO"),

    fDebugColorPrefix("\33[36;1m"),  // bold cyan
    fDebugColorSuffix("\33[0m"),
    fDebugDescription("DEBUG"),

    fDefaultColorPrefix("\33[37;1m"),  // bold white
    fDefaultColorSuffix("\33[0m"),
    fDefaultDescription("UNKNOWN"),

    fSeverity(eNormal),

    fColorPrefix(&KMessage::fNormalColorPrefix),
    fDescription(&KMessage::fNormalDescription),
    fColorSuffix(&KMessage::fNormalColorSuffix),

    fMessageLine(),
    fMessageLines(),

    fShowShutdownMessage(false),
    fShowParserContext(false),
    fTerminalVerbosity(KMessageTable::GetInstance().GetTerminalVerbosity()),
    fTerminalStream(KMessageTable::GetInstance().GetTerminalStream()),
    fLogVerbosity(KMessageTable::GetInstance().GetLogVerbosity()),
    fLogStream(KMessageTable::GetInstance().GetLogStream())
{
    fMessageLine.setf(KMessageTable::GetInstance().GetFormat(), std::ios::floatfield);
    fMessageLine.precision(KMessageTable::GetInstance().GetPrecision());
    KMessageTable::GetInstance().Add(this);
}
KMessage::~KMessage()
{
    /** In UnitTestKasper it seems, a static deinitialization fiasco happens:
     * KMessageTable is destroyed before the last KMessage instance is destroyed.
     * According to the standard, this should not happen. But the following workaround
     * (double checking the initialization state) seems to work.
     */
    if (KMessageTable::IsInitialized())
        KMessageTable::GetInstance().Remove(this);
}

const string& KMessage::GetKey()
{
    return fKey;
}
void KMessage::SetKey(const string& aKey)
{
    fKey = aKey;
    return;
}

void KMessage::SetSeverity(const KMessageSeverity& aSeverity)
{
    fSeverity = aSeverity;

    switch (fSeverity) {
        case eError:
            fColorPrefix = &KMessage::fErrorColorPrefix;
            fDescription = &KMessage::fErrorDescription;
            fColorSuffix = &KMessage::fErrorColorSuffix;
            break;

        case eWarning:
            fColorPrefix = &KMessage::fWarningColorPrefix;
            fDescription = &KMessage::fWarningDescription;
            fColorSuffix = &KMessage::fWarningColorSuffix;
            break;

        case eNormal:
            fColorPrefix = &KMessage::fNormalColorPrefix;
            fDescription = &KMessage::fNormalDescription;
            fColorSuffix = &KMessage::fNormalColorSuffix;
            break;

        case eInfo:
            fColorPrefix = &KMessage::fInfoColorPrefix;
            fDescription = &KMessage::fInfoDescription;
            fColorSuffix = &KMessage::fInfoColorSuffix;
            break;

        case eDebug:
        default:
            fColorPrefix = &KMessage::fDebugColorPrefix;
            fDescription = &KMessage::fDebugDescription;
            fColorSuffix = &KMessage::fDebugColorSuffix;
            break;
    }

    return;
}

void KMessage::EndLine(const KMessageLineEnd& aLineEnd)
{
    switch (aLineEnd) {
        case eNewline:
        case eNewlineEnd:
            fMessageLines.emplace_back(fMessageLine.str(), NewLine);
            break;
        case eOverline:
        case eOverlineEnd:
            fMessageLines.emplace_back(fMessageLine.str(), OverLine);
            break;
    }
    fMessageLine.clear();
    fMessageLine.str("");

    return;
}

void KMessage::Flush()
{
    if ((fSeverity <= fTerminalVerbosity) && (fTerminalStream != nullptr) && (fTerminalStream->good() == true)) {
        for (auto& It : fMessageLines) {
            (*fTerminalStream) << Prefix() << (It == fMessageLines.front() ? "" : TabIndent) << It.first << Suffix()
                               << It.second;
        }
        (*fTerminalStream).flush();
    }

    if ((fSeverity <= fLogVerbosity) && (fLogStream != nullptr) && (fLogStream->good() == true)) {
        for (auto& It : fMessageLines) {
            (*fLogStream) << Prefix() << (It == fMessageLines.front() ? "" : TabIndent) << It.first << Suffix()
                          << It.second;
        }
        (*fLogStream).flush();
    }

    if (fSeverity == eError) {
        Shutdown();
    }

    fMessageLines.clear();

    return;
}

void KMessage::Stacktrace(std::ostream& aStream)
{
    aStream << Prefix() << "stack trace:" << Suffix() << NewLine;
#ifdef KASPER_USE_BOOST
    for (auto& frame : boost::stacktrace::stacktrace()) {
        boost::stacktrace::detail::to_string_impl impl;
        aStream << Prefix() << TabIndent << impl(frame.address()) << " [" << frame.address() << "]" << Suffix()
                << NewLine;
    }
#else
    const size_t MaxFrameCount = 512;
    void* FrameArray[MaxFrameCount];
    const size_t FrameCount = backtrace(FrameArray, MaxFrameCount);
    char** FrameSymbols = backtrace_symbols(FrameArray, FrameCount);

    for (size_t Index = 0; Index < FrameCount; Index++) {
        aStream << Prefix() << TabIndent << FrameSymbols[Index] << Suffix() << NewLine;
    }

    free(FrameSymbols);
#endif
}

void KMessage::ParserContext(std::ostream& aStream)
{
    const auto* ctx = KXMLInitializer::GetInstance().GetContext();
    if (ctx && ctx->IsValid()) {
        aStream << Prefix() << TabIndent << "while parsing element <" << ctx->GetElement() << "> in file <"
                << ctx->GetName() << "> at line <" << ctx->GetLine() << ">, column <" << ctx->GetColumn() << ">"
                << NewLine;
    }
}

void KMessage::Shutdown()
{

    if ((fSeverity <= fTerminalVerbosity) && (fTerminalStream != nullptr) && (fTerminalStream->good() == true)) {
        if (fShowParserContext) {
            ParserContext(*fTerminalStream);
        }
        if (fShowShutdownMessage) {
            (*fTerminalStream) << Prefix() << "shutting down..." << Suffix() << NewLine;
            Stacktrace(*fTerminalStream);
        }
        (*fTerminalStream).flush();
    }

    if ((fSeverity <= fLogVerbosity) && (fLogStream != nullptr) && (fLogStream->good() == true)) {
        if (fShowParserContext) {
            ParserContext(*fLogStream);
        }
        if (fShowShutdownMessage) {
            (*fLogStream) << Prefix() << "shutting down..." << Suffix() << NewLine;
            Stacktrace(*fLogStream);
        }
        (*fLogStream).flush();
    }

    string tWhat = "runtime error";
    if (!fMessageLines.empty()) {
        tWhat = fMessageLines.front().first;
        fMessageLines.clear();  // important if exception is handled outside
    }

    throw KException() << tWhat;
    //exit( -1 );

    return;
}

void KMessage::SetFormat(const KMessageFormat& aFormat)
{
    fMessageLine.setf(aFormat, std::ios::floatfield);
    return;
}
void KMessage::SetPrecision(const KMessagePrecision& aPrecision)
{
    fMessageLine.precision(aPrecision);
    return;
}
void KMessage::SetShowShutdownMessage(bool aFlag)
{
    fShowShutdownMessage = aFlag;
    return;
}
void KMessage::SetShowParserContext(bool aFlag)
{
    fShowParserContext = aFlag;
    return;
}
void KMessage::SetTerminalVerbosity(const KMessageSeverity& aVerbosity)
{
    fTerminalVerbosity = aVerbosity;
    return;
}
void KMessage::SetTerminalStream(ostream* aTerminalStream)
{
    fTerminalStream = aTerminalStream;
    return;
}
void KMessage::SetLogVerbosity(const KMessageSeverity& aVerbosity)
{
    fLogVerbosity = aVerbosity;
    return;
}
void KMessage::SetLogStream(ostream* aLogStream)
{
    fLogStream = aLogStream;
    return;
}

const KMessageSeverity& KMessage::GetTerminalVerbosity()
{
    return fTerminalVerbosity;
}
std::ostream* KMessage::GetTerminalStream()
{
    return fTerminalStream;
}
const KMessageSeverity& KMessage::GetLogVerbosity()
{
    return fLogVerbosity;
}
std::ostream* KMessage::GetLogStream()
{
    return fLogStream;
}

}  // namespace katrin

namespace katrin
{
KMessageTable::KMessageTable() :
    fMessageMap(),
    fFormat(cerr.flags()),
    fPrecision(cerr.precision()),
    fShowShutdownMessage(false),
    fTerminalVerbosity(eNormal),
    fTerminalStream(&cerr),
    fLogVerbosity(eInfo),
    fLogStream(nullptr)
{}

KMessageTable::~KMessageTable() = default;

//********
//messages
//********

void KMessageTable::Add(KMessage* aMessage)
{
    auto tIter = fMessageMap.find(aMessage->GetKey());
    if (tIter == fMessageMap.end()) {
        fMessageMap.insert(MessageEntry(aMessage->GetKey(), aMessage));
    }
    return;
}
KMessage* KMessageTable::Get(const string& aKey)
{
    auto tIter = fMessageMap.find(aKey);
    if (tIter != fMessageMap.end()) {
        return tIter->second;
    }
    return nullptr;
}
void KMessageTable::Remove(KMessage* aMessage)
{
    Remove(aMessage->GetKey());
    return;
}
void KMessageTable::Remove(const string& aKey)
{
    fMessageMap.erase(aKey);
    return;
}

void KMessageTable::SetFormat(const KMessageFormat& aFormat)
{
    fFormat = aFormat;
    MessageIt tIter;
    for (tIter = fMessageMap.begin(); tIter != fMessageMap.end(); tIter++) {
        tIter->second->SetFormat(fFormat);
    }
    return;
}
const KMessageFormat& KMessageTable::GetFormat() const
{
    return fFormat;
}

void KMessageTable::SetPrecision(const KMessagePrecision& aPrecision)
{
    fPrecision = aPrecision;
    MessageIt tIter;
    for (tIter = fMessageMap.begin(); tIter != fMessageMap.end(); tIter++) {
        tIter->second->SetPrecision(fPrecision);
    }
    return;
}
const KMessagePrecision& KMessageTable::GetPrecision() const
{
    return fPrecision;
}

void KMessageTable::SetShowShutdownMessage(bool aFlag)
{
    fShowShutdownMessage = aFlag;
    MessageIt tIter;
    for (tIter = fMessageMap.begin(); tIter != fMessageMap.end(); tIter++) {
        tIter->second->SetShowShutdownMessage(fShowShutdownMessage);
    }
    return;
}
bool KMessageTable::GetShowShutdownMessage() const
{
    return fShowShutdownMessage;
}

void KMessageTable::SetShowParserContext(bool aFlag)
{
    fShowParserContext = aFlag;
    MessageIt tIter;
    for (tIter = fMessageMap.begin(); tIter != fMessageMap.end(); tIter++) {
        tIter->second->SetShowParserContext(fShowParserContext);
    }
    return;
}
bool KMessageTable::GetShowParserContext() const
{
    return fShowParserContext;
}

void KMessageTable::SetTerminalVerbosity(const KMessageSeverity& aVerbosity)
{
    fTerminalVerbosity = aVerbosity;
    MessageIt tIter;
    for (tIter = fMessageMap.begin(); tIter != fMessageMap.end(); tIter++) {
        tIter->second->SetTerminalVerbosity(fTerminalVerbosity);
    }
    return;
}
const KMessageSeverity& KMessageTable::GetTerminalVerbosity() const
{
    return fTerminalVerbosity;
}

void KMessageTable::SetTerminalStream(ostream* aTerminalStream)
{
    fTerminalStream = aTerminalStream;
    MessageIt tIter;
    for (tIter = fMessageMap.begin(); tIter != fMessageMap.end(); tIter++) {
        tIter->second->SetTerminalStream(fTerminalStream);
    }
    return;
}
ostream* KMessageTable::GetTerminalStream()
{
    return fTerminalStream;
}

void KMessageTable::SetLogVerbosity(const KMessageSeverity& aVerbosity)
{
    fLogVerbosity = aVerbosity;
    MessageIt tIter;
    for (tIter = fMessageMap.begin(); tIter != fMessageMap.end(); tIter++) {
        tIter->second->SetLogVerbosity(fLogVerbosity);
    }
    return;
}
const KMessageSeverity& KMessageTable::GetLogVerbosity() const
{
    return fLogVerbosity;
}

void KMessageTable::SetLogStream(ostream* aLogStream)
{
    fLogStream = aLogStream;
    MessageIt tIter;
    for (tIter = fMessageMap.begin(); tIter != fMessageMap.end(); tIter++) {
        tIter->second->SetLogStream(fLogStream);
    }
    return;
}
ostream* KMessageTable::GetLogStream()
{
    return fLogStream;
}

}  // namespace katrin
