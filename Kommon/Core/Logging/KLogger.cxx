/*
 * KLogger.cxx
 *
 *  Created on: 18.11.2011
 *      Author: Marco Kleesiek <marco.kleesiek@kit.edu>
 */

#include "KLogger.h"

#include <cstdlib>
#include <cstring>

using namespace std;
using namespace katrin;

static const string skEndColor =   COLOR_PREFIX COLOR_NORMAL COLOR_SUFFIX;
static const string skFatalColor = COLOR_PREFIX COLOR_BRIGHT COLOR_SEPARATOR COLOR_FOREGROUND_RED    COLOR_SUFFIX;
static const string skErrorColor = COLOR_PREFIX COLOR_BRIGHT COLOR_SEPARATOR COLOR_FOREGROUND_RED    COLOR_SUFFIX;
static const string skWarnColor =  COLOR_PREFIX COLOR_BRIGHT COLOR_SEPARATOR COLOR_FOREGROUND_YELLOW COLOR_SUFFIX;
static const string skInfoColor =  COLOR_PREFIX COLOR_BRIGHT COLOR_SEPARATOR COLOR_FOREGROUND_GREEN  COLOR_SUFFIX;
static const string skDebugColor = COLOR_PREFIX COLOR_BRIGHT COLOR_SEPARATOR COLOR_FOREGROUND_CYAN   COLOR_SUFFIX;
static const string skOtherColor = COLOR_PREFIX COLOR_BRIGHT COLOR_SEPARATOR COLOR_FOREGROUND_WHITE  COLOR_SUFFIX;


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
    virtual std::string getColor(const LevelPtr& level) const;

};

LOG4CXX_PTR_DEF(ColoredPatternLayout);

}

#endif /* _LOG4CXX_COLORED_PATTERN_LAYOUT_H */

IMPLEMENT_LOG4CXX_OBJECT(ColoredPatternLayout)

void ColoredPatternLayout::format(LogString& output, const spi::LoggingEventPtr& event, helpers::Pool& pool) const
{
    PatternLayout::format(output, event, pool);
    output = getColor(event->getLevel()) + output + skEndColor;
    return;
}

string ColoredPatternLayout::getColor(const LevelPtr& level) const
{
    if (!level)
        return skOtherColor;
    switch(level->toInt())
    {
        case Level::FATAL_INT:
            return skFatalColor;
            break;
        case Level::ERROR_INT:
            return skErrorColor;
            break;
        case Level::WARN_INT:
            return skWarnColor;
            break;
        case Level::INFO_INT:
            return skInfoColor;
            break;
        case Level::DEBUG_INT:
            return skDebugColor;
            break;
        case Level::TRACE_INT:
            return skDebugColor;
            break;
        default:
            return skOtherColor;
    }
}

namespace {

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
                static const LogString TTCC_CONVERSION_PATTERN(LOG4CXX_STR("%r [%-5p] %16c: %m%n"));
                LayoutPtr layout(new PatternLayout(TTCC_CONVERSION_PATTERN));
//                LayoutPtr layout(new ColoredPatternLayout(TTCC_CONVERSION_PATTERN));
                AppenderPtr appender(new ConsoleAppender(layout));
                root->addAppender(appender);
            #endif
        }

    }
} static sLoggerInitializer;

}

struct KLogger::Private
{
    void log(const LevelPtr& level, const string& message, const Location& loc)
    {
        fLogger->forcedLog(level, message, ::log4cxx::spi::LocationInfo(loc.fFileName, loc.fFunctionName, loc.fLineNumber));
    }

    static LevelPtr level2Ptr(ELevel level)
    {
        switch(level) {
            case eTrace : return Level::getTrace();
            case eDebug : return Level::getDebug();
            case eInfo  : return Level::getInfo();
            case eWarn  : return Level::getWarn();
            case eError : return Level::getError();
            case eFatal : return Level::getFatal();
            default     : return Level::getOff();
        }
    }

    static ELevel ptr2Level(LevelPtr ptr)
    {
        if (!ptr)
            return eUndefined;
        switch(ptr->toInt()) {
            case Level::TRACE_INT : return eTrace;
            case Level::DEBUG_INT : return eDebug;
            case Level::INFO_INT  : return eInfo;
            case Level::WARN_INT  : return eWarn;
            case Level::ERROR_INT : return eError;
            case Level::FATAL_INT : return eFatal;
            default               : return eUndefined;
        }
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
    return fPrivate->fLogger->isEnabledFor( Private::level2Ptr(level) );
}

KLogger::ELevel KLogger::GetLevel() const
{
    return Private::ptr2Level( fPrivate->fLogger->getLevel() );
}

void KLogger::SetLevel(ELevel level) const
{
    fPrivate->fLogger->setLevel( Private::level2Ptr(level) );
}

void KLogger::Log(ELevel level, const string& message, const Location& loc)
{
    fPrivate->log(Private::level2Ptr(level), message, loc);
}



#else

/**
 * Fallback solution for systems without log4cxx.
 */

#include <iomanip>

struct KLogger::Private
{
    const char* fLogger;
    bool fColored;

    static const char* level2Str(ELevel level)
    {
        switch(level) {
            case eTrace : return "TRACE";
            case eDebug : return "DEBUG";
            case eInfo  : return "INFO";
            case eWarn  : return "WARN";
            case eError : return "ERROR";
            case eFatal : return "FATAL";
            default     : return "XXX";
        }
    }

    void logCout(const char* level, const string& message, const Location& /*loc*/, const string& color = skOtherColor) {
        if (fColored)
            cout << color << __DATE__ " " __TIME__ " [" << setw(5) << level << "] " << setw(16) << fLogger << ": " << message << skEndColor << endl;
        else
            cout << __DATE__ " " __TIME__ " [" << setw(5) << level << "] " << setw(16) << fLogger << ": " << message << endl;
    }

    void logCerr(const char* level, const string& message, const Location& /*loc*/, const string& color = skOtherColor) {
        if (fColored)
            cerr << color << __DATE__ " " __TIME__ " [" << setw(5) << level << "] " << setw(16) << fLogger << ": " << message << skEndColor << endl;
        else
            cerr << __DATE__ " " __TIME__ " [" << setw(5) << level << "] " << setw(16) << fLogger << ": " << message << endl;
    }
};

KLogger::KLogger(const char* name) : fPrivate(new Private())
{
    if (name == 0)
        fPrivate->fLogger = "root";
    else {
        const char* logName = strrchr(name, '/') ? strrchr(name, '/') + 1 : name;
        fPrivate->fLogger = logName;
    }
    fPrivate->fColored = false;
}

KLogger::KLogger(const std::string& name) : fPrivate(new Private())
{
    if (name.empty())
        fPrivate->fLogger = "root";
    else
        fPrivate->fLogger = name.c_str();
    fPrivate->fColored = false;
}

KLogger::~KLogger()
{
    delete fPrivate;
}

bool KLogger::IsLevelEnabled(ELevel level) const
{
    return GetLevel() >= level;
}

KLogger::ELevel KLogger::GetLevel() const
{
    // TODO: implement levels for fallback logger
#ifdef NDEBUG
    return eInfo;
#else
    return eDebug;
#endif
}

void KLogger::SetLevel(ELevel /*level*/) const
{
    // TODO: implement levels for fallback logger
}

void KLogger::Log(ELevel level, const string& message, const Location& loc)
{
    const char* levelStr = Private::level2Str(level);

    if (level >= eFatal) {
        fPrivate->logCerr(levelStr, message, loc);
    }
    else {
        fPrivate->logCout(levelStr, message, loc);
    }
}

#endif
