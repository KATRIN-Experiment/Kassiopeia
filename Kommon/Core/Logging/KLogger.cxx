/*
 * KLogger.cxx
 *
 *  Created on: 18.11.2011
 *      Author: Marco Kleesiek <marco.kleesiek@kit.edu>
 */

#include "KLogger.h"

#include <chrono>
#include <iomanip>

using namespace std;
using namespace katrin;

static const char* skEndColor =   COLOR_PREFIX COLOR_NORMAL COLOR_SUFFIX;
static const char* skFatalColor = COLOR_PREFIX COLOR_BRIGHT COLOR_SEPARATOR COLOR_FOREGROUND_RED    COLOR_SUFFIX;
static const char* skErrorColor = COLOR_PREFIX COLOR_BRIGHT COLOR_SEPARATOR COLOR_FOREGROUND_RED    COLOR_SUFFIX;
static const char* skWarnColor =  COLOR_PREFIX COLOR_BRIGHT COLOR_SEPARATOR COLOR_FOREGROUND_YELLOW COLOR_SUFFIX;
static const char* skInfoColor =  COLOR_PREFIX COLOR_BRIGHT COLOR_SEPARATOR COLOR_FOREGROUND_GREEN  COLOR_SUFFIX;
static const char* skDebugColor = COLOR_PREFIX COLOR_BRIGHT COLOR_SEPARATOR COLOR_FOREGROUND_CYAN   COLOR_SUFFIX;
static const char* skOtherColor = COLOR_PREFIX COLOR_BRIGHT COLOR_SEPARATOR COLOR_FOREGROUND_WHITE  COLOR_SUFFIX;

namespace {

inline const char* level2Color(KLogger::ELevel level)
{
    switch(level) {
        case KLogger::eFatal:
            return skFatalColor;
            break;
        case KLogger::eError:
            return skErrorColor;
            break;
        case KLogger::eWarn:
            return skWarnColor;
            break;
        case KLogger::eInfo:
            return skInfoColor;
            break;
        case KLogger::eDebug:
            return skDebugColor;
            break;
        case KLogger::eTrace:
            return skDebugColor;
            break;
        default:
            return skOtherColor;
    }
}

}

#if defined(LOG4CXX)

/*
 * Default implementation for systems with the 'log4cxx' library installed.
 */

#include <log4cxx/logstring.h>
#include <log4cxx/basicconfigurator.h>
#include <log4cxx/propertyconfigurator.h>
#include <log4cxx/patternlayout.h>
#include <log4cxx/level.h>
#include <log4cxx/consoleappender.h>
#include <log4cxx/logmanager.h>
#include <log4cxx/logger.h>

using namespace log4cxx;

namespace {

inline LevelPtr level2Ptr(KLogger::ELevel level)
{
    switch(level) {
        case KLogger::eTrace : return Level::getTrace();
        case KLogger::eDebug : return Level::getDebug();
        case KLogger::eInfo  : return Level::getInfo();
        case KLogger::eWarn  : return Level::getWarn();
        case KLogger::eError : return Level::getError();
        case KLogger::eFatal : return Level::getFatal();
        default     : return Level::getOff();
    }
}

inline KLogger::ELevel ptr2Level(LevelPtr ptr)
{
    if (!ptr)
        return KLogger::eUndefined;
    switch(ptr->toInt()) {
        case Level::TRACE_INT : return KLogger::eTrace;
        case Level::DEBUG_INT : return KLogger::eDebug;
        case Level::INFO_INT  : return KLogger::eInfo;
        case Level::WARN_INT  : return KLogger::eWarn;
        case Level::ERROR_INT : return KLogger::eError;
        case Level::FATAL_INT : return KLogger::eFatal;
        default               : return KLogger::eUndefined;
    }
}

}

#ifndef _LOG4CXX_COLORED_PATTERN_LAYOUT_H
#define _LOG4CXX_COLORED_PATTERN_LAYOUT_H

namespace log4cxx {

class LOG4CXX_EXPORT ColoredPatternLayout : public PatternLayout
{
public:
    DECLARE_LOG4CXX_OBJECT(ColoredPatternLayout)
    BEGIN_LOG4CXX_CAST_MAP()
    LOG4CXX_CAST_ENTRY(ColoredPatternLayout)
    LOG4CXX_CAST_ENTRY_CHAIN(Layout)
    END_LOG4CXX_CAST_MAP()

    ColoredPatternLayout() : PatternLayout() {}
    ColoredPatternLayout(const LogString& pattern) : PatternLayout(pattern) {};
    virtual ~ColoredPatternLayout() {}

protected:
    virtual void format(LogString& output, const spi::LoggingEventPtr& event, helpers::Pool& pool) const;
};

LOG4CXX_PTR_DEF(ColoredPatternLayout);

}

#endif /* _LOG4CXX_COLORED_PATTERN_LAYOUT_H */

IMPLEMENT_LOG4CXX_OBJECT(ColoredPatternLayout)

void ColoredPatternLayout::format(LogString& output, const spi::LoggingEventPtr& event, helpers::Pool& pool) const
{
    PatternLayout::format(output, event, pool);
    output = level2Color( ptr2Level(event->getLevel()) ) + output + skEndColor;
    return;
}


struct StaticInitializer {
    StaticInitializer() {

        if (LogManager::getLoggerRepository()->isConfigured())
            return;

        char* envLoggerConfig;
        envLoggerConfig = getenv("LOGGER_CONFIGURATION");
        if (envLoggerConfig != 0) {
            PropertyConfigurator::configure(envLoggerConfig);
//            #ifndef NDEBUG
//                std::cout << "Logger configuration: " << envLoggerConfig << std::endl;
//            #endif
        }
        else {
            #ifdef LOGGER_CONFIGURATION
                PropertyConfigurator::configure( TOSTRING(LOGGER_CONFIGURATION) );
            #else
                LogManager::getLoggerRepository()->setConfigured(true);
                LoggerPtr root = Logger::getRootLogger();
                #ifdef NDEBUG
                    Logger::getRootLogger()->setLevel(Level::getInfo());
                #endif
                const LogString TTCC_CONVERSION_PATTERN(LOG4CXX_STR("%r [%-5p] %16c: %m%n"));
                LayoutPtr layout(new PatternLayout(TTCC_CONVERSION_PATTERN));
                AppenderPtr appender(new ConsoleAppender(layout));
                root->addAppender(appender);
            #endif
        }

    }
} static sLoggerInitializer;


struct KLogger::Private
{
    void log(const LevelPtr& level, const string& message, const Location& loc)
    {
        fLogger->forcedLog(level, message, ::log4cxx::spi::LocationInfo(loc.fFileName, loc.fFunctionName, loc.fLineNumber));
    }

    LoggerPtr fLogger;
};

KLogger::KLogger(const char* name) : fPrivate(new Private())
{
    fPrivate->fLogger = (name == 0) ? Logger::getRootLogger() : Logger::getLogger(name);
}

KLogger::KLogger(const std::string& name) : fPrivate(new Private())
{
    fPrivate->fLogger = Logger::getLogger(name);
}

KLogger::~KLogger()
{
    delete fPrivate;
}

bool KLogger::IsLevelEnabled(ELevel level) const
{
    return fPrivate->fLogger->isEnabledFor( level2Ptr(level) );
}

KLogger::ELevel KLogger::GetLevel() const
{
    return ptr2Level( fPrivate->fLogger->getLevel() );
}

void KLogger::SetLevel(ELevel level)
{
    fPrivate->fLogger->setLevel( level2Ptr(level) );
}

void KLogger::Log(ELevel level, const string& message, const Location& loc)
{
    fPrivate->log(level2Ptr(level), message, loc);
}



#else

/**
 * Fallback solution for systems without log4cxx.
 */

#include <iomanip>
#include <map>
#include <utility>

typedef map<string, KLogger::ELevel> LoggerMap;
static LoggerMap sLoggerMap;
static const bool skColored = true;

namespace {

const char* level2Str(KLogger::ELevel level)
{
    switch(level) {
        case KLogger::eTrace : return "TRACE";
        case KLogger::eDebug : return "DEBUG";
        case KLogger::eInfo  : return "INFO";
        case KLogger::eWarn  : return "WARN";
        case KLogger::eError : return "ERROR";
        case KLogger::eFatal : return "FATAL";
        default     : return "XXX";
    }
}

void printTime(ostream& strm)
{
    using namespace std::chrono;

    const auto now = system_clock::now();
    const time_t cTime = system_clock::to_time_t(now);

    auto duration = now.time_since_epoch();
    duration -= duration_cast<seconds>(duration);

    /*
     * Unfortunately, g++ < 5.0 does not implement std::put_time, so I have to
     * resort to strftime at this point:
     */
    char dateTimeStr[24];
    strftime(dateTimeStr, sizeof(dateTimeStr), "%F %T", localtime(&cTime));
    strm << dateTimeStr;
    strm << "." << setfill('0') << setw(3) << duration_cast<milliseconds>(duration).count();
    strm << setfill(' ');
}

}

struct KLogger::Private
{
    Private(const string& name) : fLoggerIt( sLoggerMap.find(name) )
    {
        if (fLoggerIt == sLoggerMap.end()) {
            #ifdef NDEBUG
                ELevel newLevel = eInfo;
            #else
                ELevel newLevel = eDebug;
            #endif
            pair<LoggerMap::iterator, bool> v = sLoggerMap.insert(make_pair(name, newLevel));
            fLoggerIt = v.first;
        }

    }

    LoggerMap::iterator fLoggerIt;

    void logCout(const char* level, const string& message, const Location& /*loc*/, const char* color = skOtherColor) {
        if (skColored) {
            cout << color;
            printTime(cout);
            cout << " [" << setw(5) << level << "] " << setw(10) << fLoggerIt->first << ": " << message;
            cout << skEndColor;
            cout << endl;
        }
        else {
            printTime(cout);
            cout << " [" << setw(5) << level << "] " << setw(10) << fLoggerIt->first << ": " << message;
            cout << endl;
        }
    }

    void logCerr(const char* level, const string& message, const Location& /*loc*/, const char* color = skOtherColor) {
        if (skColored) {
            cerr << color;
            printTime(cerr);
            cerr << " [" << setw(5) << level << "] " << setw(10) << fLoggerIt->first << ": " << message;
            cerr << skEndColor;
            cerr << endl;
        }
        else {
            printTime(cerr);
            cerr << " [" << setw(5) << level << "] " << setw(10) << fLoggerIt->first << ": " << message;
            cerr << endl;
        }
    }
};

KLogger::KLogger(const char* name) :
    fPrivate( new Private( name == 0 ? "root" : name ) )
{ }

KLogger::KLogger(const std::string& name) :
    fPrivate( new Private( name.empty() ? "root" : name ) )
{ }

KLogger::~KLogger()
{
    delete fPrivate;
}

bool KLogger::IsLevelEnabled(ELevel level) const
{
    return fPrivate->fLoggerIt->second <= level;
}

KLogger::ELevel KLogger::GetLevel() const
{
    return fPrivate->fLoggerIt->second;
}

void KLogger::SetLevel(ELevel level)
{
    fPrivate->fLoggerIt->second = level;
}

void KLogger::Log(ELevel level, const string& message, const Location& loc)
{
    const char* levelStr = level2Str(level);
    const char* color = level2Color(level);

    if (level >= eError)
        fPrivate->logCerr(levelStr, message, loc, color);
    else
        fPrivate->logCout(levelStr, message, loc, color);
}

#endif
