#ifndef KMESSAGE_H_
#define KMESSAGE_H_

#include <cstdlib>
#include <iomanip>
#include <map>
#include <ostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include <set>
#include <functional>

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

namespace katrin
{
enum KMessageLineEnd : int
{
    eNewline = 0,
    eOverline = 1,
    eNewlineEnd = 10,
    eOverlineEnd = 11,
};

enum KMessageSeverity : int
{
    eErrorMessage = 0,
    eWarningMessage = 1,
    eNormalMessage = 2,
    eInfoMessage = 3,
    eDebugMessage = 4,
};

using KMessageFormat = std::ios_base::fmtflags;
using KMessagePrecision = std::streamsize;
using KMessageLine = std::pair<std::string, std::string>;

static const KMessageSeverity eError = KMessageSeverity::eErrorMessage;
static const KMessageSeverity eWarning = KMessageSeverity::eWarningMessage;
static const KMessageSeverity eNormal = KMessageSeverity::eNormalMessage;
static const KMessageSeverity eInfo = KMessageSeverity::eInfoMessage;
static const KMessageSeverity eDebug = KMessageSeverity::eDebugMessage;
static const KMessageLineEnd ret = KMessageLineEnd::eNewline;
static const KMessageLineEnd rret = KMessageLineEnd::eOverline;
static const KMessageLineEnd eom = KMessageLineEnd::eNewlineEnd;
static const KMessageLineEnd reom = KMessageLineEnd::eOverlineEnd;

static const std::string NewLine = "\n";
static const std::string OverLine = "\r";
static const std::string TabIndent = "    ";

class KMessage
{
  public:
    KMessage(const std::string& aKey, const std::string& aDescription, const std::string& fTerminalStreamaPrefix,
             const std::string& aSuffix);
    virtual ~KMessage();

  private:
    KMessage();
    KMessage(const KMessage&);

    //**************
    //identification
    //**************

  public:
    const std::string& GetKey();
    void SetKey(const std::string& aKey);

  protected:
    std::string fKey;

    //*********
    //interface
    //*********

  public:
    KMessage& operator()(const KMessageSeverity& = KMessageSeverity::eNormalMessage);
    KMessage& operator<<(const KMessageSeverity& aSeverity);
    KMessage& operator<<(const KMessageLineEnd& aLineEnd);

    template<class XPrintable> KMessage& operator<<(const XPrintable& aFragment);

  public:
    void SetSeverity(const KMessageSeverity& aSeverity);
    void EndLine(const KMessageLineEnd& aLineEnd);
    void Flush();

  private:
    void Shutdown();
    void Stacktrace(std::ostream& aStream);
    void ParserContext(std::ostream& aStream);
    std::string Prefix() const;
    std::string Suffix() const;

  protected:
    std::string fSystemDescription;
    std::string fSystemPrefix;
    std::string fSystemSuffix;

    std::string fErrorColorPrefix;
    std::string fErrorColorSuffix;
    std::string fErrorDescription;

    std::string fWarningColorPrefix;
    std::string fWarningColorSuffix;
    std::string fWarningDescription;

    std::string fNormalColorPrefix;
    std::string fNormalColorSuffix;
    std::string fNormalDescription;

    std::string fInfoColorPrefix;
    std::string fInfoColorSuffix;
    std::string fInfoDescription;

    std::string fDebugColorPrefix;
    std::string fDebugColorSuffix;
    std::string fDebugDescription;

    std::string fDefaultColorPrefix;
    std::string fDefaultColorSuffix;
    std::string fDefaultDescription;

  private:
    KMessageSeverity fSeverity;

    std::string KMessage::*fColorPrefix;
    std::string KMessage::*fDescription;
    std::string KMessage::*fColorSuffix;

    std::stringstream fMessageLine;
    std::vector<KMessageLine> fMessageLines;

    //********
    //settings
    //********

  public:
    void SetFormat(const KMessageFormat& aFormat);
    void SetPrecision(const KMessagePrecision& aPrecision);
    void SetShowShutdownMessage(bool aFlag = true);
    void SetShowParserContext(bool aFlag = true);
    
    void SetTerminalVerbosity(const KMessageSeverity& aVerbosity);
    void SetTerminalStream(std::ostream* aTerminalStream);
    void SetLogVerbosity(const KMessageSeverity& aVerbosity);
    void SetLogStream(std::ostream* aLogStream);

    const KMessageSeverity& GetTerminalVerbosity();
    std::ostream* GetTerminalStream();
    const KMessageSeverity& GetLogVerbosity();
    std::ostream* GetLogStream();

  private:
    bool fShowShutdownMessage;
    bool fShowParserContext;
    KMessageSeverity fTerminalVerbosity;
    std::ostream* fTerminalStream;
    KMessageSeverity fLogVerbosity;
    std::ostream* fLogStream;
};

inline KMessage& KMessage::operator()(const KMessageSeverity& aSeverity)
{
    SetSeverity(aSeverity);
    return *this;
}
inline KMessage& KMessage::operator<<(const KMessageSeverity& aSeverity)
{
    SetSeverity(aSeverity);
    return *this;
}
inline KMessage& KMessage::operator<<(const KMessageLineEnd& aLineEnd)
{
    EndLine(aLineEnd);
    if (aLineEnd >= eNewlineEnd)
        Flush();
    return *this;
}
template<class XPrintable> KMessage& KMessage::operator<<(const XPrintable& aFragment)
{
    fMessageLine << aFragment;
    return *this;
}

inline std::string KMessage::Prefix() const
{
    std::ostringstream stream;
    stream << this->*fColorPrefix << fSystemPrefix << "[" << fSystemDescription << " " << this->*fDescription
           << " MESSAGE] ";
    return stream.str();
}

inline std::string KMessage::Suffix() const
{
    std::ostringstream stream;
    stream << fSystemSuffix << this->*fColorSuffix;
    return stream.str();
}

}  // namespace katrin

#include "KSingleton.h"

namespace katrin
{

class KMessageTable : public KSingleton<KMessageTable>
{
  public:
    friend class KSingleton<KMessageTable>;

  private:
    KMessageTable();
    ~KMessageTable() override;

  public:
    void Add(KMessage* aMessage);
    void Remove(KMessage* aMessage);
    void Remove(const std::string& aKey);
    KMessage* Get(const std::string& aKey);

    void SetFormat(const KMessageFormat& aFormat);
    const KMessageFormat& GetFormat() const;

    void SetPrecision(const KMessagePrecision& aPrecision);
    const KMessagePrecision& GetPrecision() const;

    void SetShowShutdownMessage(bool aFlag = true);
    bool GetShowShutdownMessage() const;

    void SetShowParserContext(bool aFlag = true);
    bool GetShowParserContext() const;
    
    std::function<void(std::ostream&)> SetParserContextPrinterCallback(std::function<void(std::ostream&)>);
    std::function<void(std::ostream&)> GetParserContextPrinterCallback() const;
    void RemoveParserContextPrinterCallback();
    
    std::function<void(std::ostream&)> SetStacktracePrinterCallback(std::function<void(std::ostream&)>);
    std::function<void(std::ostream&)> GetStacktracePrinterCallback() const;

    void SetTerminalVerbosity(const KMessageSeverity& aVerbosity);
    const KMessageSeverity& GetTerminalVerbosity() const;

    void SetTerminalStream(std::ostream* aTerminalStream);
    std::ostream* GetTerminalStream();

    void SetLogVerbosity(const KMessageSeverity& aVerbosity);
    const KMessageSeverity& GetLogVerbosity() const;

    void SetLogStream(std::ostream* aLogStream);
    std::ostream* GetLogStream();

    void SetVerbosityLevel(int level = 0);
    KMessageSeverity CorrectedLevel(const KMessageSeverity& level) const;
    
    static void DefaultStacktracePrinter(std::ostream& aStream);

  private:
    using MessageMap = std::map<std::string, KMessage*>;
    using MessageEntry = MessageMap::value_type;
    using MessageIt = MessageMap::iterator;
    using MessageCIt = MessageMap::const_iterator;

    MessageMap fMessageMap;

    KMessageFormat fFormat;
    KMessagePrecision fPrecision;
    bool fShowShutdownMessage;
    bool fShowParserContext;
    KMessageSeverity fTerminalVerbosity;
    std::ostream* fTerminalStream;
    KMessageSeverity fLogVerbosity;
    std::ostream* fLogStream;
    int fVerbosityLevel;
    std::function<void(std::ostream&)> fParserContextPrinterCallback;
    std::function<void(std::ostream&)> fStacktracePrinterCallback;
};

}  // namespace katrin

#include "KInitializer.h"

#define KMESSAGE_DECLARE(xNAMESPACE, xNAME)                                                                            \
    namespace xNAMESPACE                                                                                               \
    {                                                                                                                  \
    class __attribute__((__may_alias__)) KMessage_##xNAME : public katrin::KMessage                                    \
    {                                                                                                                  \
      public:                                                                                                          \
        KMessage_##xNAME();                                                                                            \
        virtual ~KMessage_##xNAME();                                                                                   \
    };                                                                                                                 \
                                                                                                                       \
    using katrin::eDebug;                                                                                              \
    using katrin::eInfo;                                                                                               \
    using katrin::eNormal;                                                                                             \
    using katrin::eWarning;                                                                                            \
    using katrin::eError;                                                                                              \
                                                                                                                       \
    using katrin::ret;                                                                                                 \
    using katrin::rret;                                                                                                \
    using katrin::eom;                                                                                                 \
    using katrin::reom;                                                                                                \
                                                                                                                       \
    extern KMessage_##xNAME& xNAME;                                                                                    \
    static katrin::KInitializer<KMessage_##xNAME> xNAME##_initializer;                                                 \
    }

#define KMESSAGE_DEFINE(xNAMESPACE, xNAME, xKEY, xLABEL)                                                               \
    namespace xNAMESPACE                                                                                               \
    {                                                                                                                  \
    KMessage_##xNAME::KMessage_##xNAME() : katrin::KMessage(#xKEY, #xLABEL, "", "") {}                                 \
    KMessage_##xNAME::~KMessage_##xNAME() {}                                                                           \
                                                                                                                       \
    KMessage_##xNAME& xNAME = *((KMessage_##xNAME*) (katrin::KInitializer<KMessage_##xNAME>::fData));                  \
    }

#define KMESSAGE_DEFINE_FULL(xNAMESPACE, xNAME, xKEY, xLABEL, xPREFIX, xSUFFIX)                                        \
    namespace xNAMESPACE                                                                                               \
    {                                                                                                                  \
    KMessage_##xNAME::KMessage_##xNAME() : katrin::KMessage(#xKEY, #xLABEL, #xPREFIX, #xSUFFIX) {}                     \
    KMessage_##xNAME::~KMessage_##xNAME() {}                                                                           \
                                                                                                                       \
    KMessage_##xNAME& xNAME = *((KMessage_##xNAME*) (katrin::KInitializer<KMessage_##xNAME>::fData));                  \
    }

#endif
