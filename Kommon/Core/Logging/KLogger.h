#ifndef KLOGGER_H_
#define KLOGGER_H_

/**
 * @file
 * @brief Contains the katrin::KLogger class and macros.
 * @date Created on: 18.11.2011
 * @author Marco Kleesiek <marco.kleesiek@kit.edu>
 *
 */

// UTILITY MACROS

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)
#define __FILE_LINE__      __FILE__ " (" TOSTRING(__LINE__) ")"
#define __FILENAME_LINE__  (strrchr(__FILE__, '/') ? strrchr(__FILE_LINE__, '/') + 1 : __FILE_LINE__)

#if defined(_MSC_VER)
#if _MSC_VER >= 1300
      #define __FUNC__ __FUNCSIG__
#endif
#else
#if defined(__GNUC__)
      #define __FUNC__ __PRETTY_FUNCTION__
#endif
#endif
#if !defined(__FUNC__)
#define __FUNC__ ""
#endif

#define va_num_args(...) va_num_args_impl(__VA_ARGS__, 5,4,3,2,1)
#define va_num_args_impl(_1,_2,_3,_4,_5,N,...) N

#define macro_dispatcher(func, ...) macro_dispatcher_(func, va_num_args(__VA_ARGS__))
#define macro_dispatcher_(func, nargs) macro_dispatcher__(func, nargs)
#define macro_dispatcher__(func, nargs) func ## nargs

// COLOR DEFINITIONS
#define COLOR_NORMAL "0"
#define COLOR_BRIGHT "1"
#define COLOR_FOREGROUND_RED "31"
#define COLOR_FOREGROUND_GREEN "32"
#define COLOR_FOREGROUND_YELLOW "33"
#define COLOR_FOREGROUND_CYAN "36"
#define COLOR_FOREGROUND_WHITE "37"
#define COLOR_PREFIX "\033["
#define COLOR_SUFFIX "m"
#define COLOR_SEPARATOR ";"

// INCLUDES

#include <string>
#include <iostream>
#include <sstream>

// CLASS DEFINITIONS

/**
 * The standard KATRIN namespace.
 */
namespace katrin {

/**
 * The KATRIN Logger.
 *
 * The usage and syntax is inspired by log4j. KLogger itself uses the log4cxx library if it
 * was available on the system during compiling, otherwise it falls back to std::stream.
 *
 * The logger output can be configured in a file specified with the environment variable
 * @a LOGGER_CONFIGURATION (by default log4cxx.properties in the kasper config directory).
 *
 * In most cases the following macro can be used
 * to instantiate a Logger in your code:
 * <pre>KLOGGER(myLogger, "loggerName");</pre>
 *
 * This is equivalent to:
 * <pre>static katrin::KLogger myLogger("loggerName");</pre>
 *
 * For logging the following macros can be used. The source code location will then automatically
 * included in the output:
 *
 * <pre>
 * KLOG(myLogger, level, "message");
 * KTRACE(myLogger, "message");
 * KDEBUG(myLogger, "message");
 * KINFO(myLogger, "message");
 * KWARN(myLogger, "message");
 * KERROR(myLogger, "message");
 * KFATAL(myLogger, "message");
 *
 * KASSERT(myLogger, assertion, "message");
 *
 * KLOG_ONCE(myLogger, level, "message");
 * KTRACE_ONCE(myLogger, "message");
 * KDEBUG_ONCE(myLogger, "message");
 * KINFO_ONCE(myLogger, "message");
 * KWARN_ONCE(myLogger, "message");
 * KERROR_ONCE(myLogger, "message");
 * KFATAL_ONCE(myLogger, "message");
 * </pre>
 *
 */
class KLogger
{
public:
    enum ELevel {
        eTrace, eDebug, eInfo, eWarn, eError, eFatal, eUndefined
    };

public:
    /**
     * A simple struct used by the Logger macros to pass information about the filename and line number.
     * Not to be used directly by the user!
     */
    struct Location {
        Location(const char* const fileName = "", const char* const functionName = "", int lineNumber = -1) :
            fLineNumber(lineNumber), fFileName(fileName), fFunctionName(functionName)
            { }
        int fLineNumber;
        const char* fFileName;
        const char* fFunctionName;
    };

public:
    /**
     * Standard constructor assigning a name to the logger instance.
     * @param name The logger name.
     */
    KLogger(const char* name = 0);
    /// @overload
    KLogger(const std::string& name);

    virtual ~KLogger();

    /**
     * Check whether a certain log-level is enabled.
     * @param level The log level as string representation.
     * @return
     */
    bool IsLevelEnabled(ELevel level) const;

    /**
     * Get a loggers minimum logging level
     * @return level enum item identifying the log level
     */
    ELevel GetLevel() const;

    /**
     * Set a loggers minimum logging level
     * @param level enum item identifying the log level
     */
    void SetLevel(ELevel level);

    /**
     * Log a message with the specified level.
     * Use the macro KLOG(logger, level, message).
     * @param level The log level.
     * @param message The message.
     * @param loc Source code location (set automatically by the corresponding macro).
     */
    void Log(ELevel level, const std::string& message, const Location& loc = Location());

    /**
     * Log a message at TRACE level.
     * Use the macro KTRACE(logger, message).
     * @param message The message.
     * @param loc Source code location (set automatically by the corresponding macro).
     */
    void LogTrace(const std::string& message, const Location& loc = Location())
    {
        Log(eTrace, message, loc);
    }
    /**
     * Log a message at DEBUG level.
     * Use the macro KDEBUG(logger, message).
     * @param message The message.
     * @param loc Source code location (set automatically by the corresponding macro).
     */
    void LogDebug(const std::string& message, const Location& loc = Location())
    {
        Log(eDebug, message, loc);
    }
    /**
     * Log a message at DEBUG level.
     * Use the macro KDEBUG(logger, message).
     * @param message The message.
     * @param loc Source code location (set automatically by the corresponding macro).
     */
    void LogInfo(const std::string& message, const Location& loc = Location())
    {
        Log(eInfo, message, loc);
    }
    /**
     * Log a message at INFO level.
     * Use the macro KINFO(logger, message).
     * @param message The message.
     * @param loc Source code location (set automatically by the corresponding macro).
     */
    void LogWarn(const std::string& message, const Location& loc = Location())
    {
        Log(eWarn, message, loc);
    }
    /**
     * Log a message at ERROR level.
     * Use the macro KERROR(logger, message).
     * @param message The message.
     * @param loc Source code location (set automatically by the corresponding macro).
     */
    void LogError(const std::string& message, const Location& loc = Location())
    {
        Log(eError, message, loc);
    }
    /**
     * Log a message at FATAL level.
     * Use the macro KFATAL(logger, message).
     * @param message The message.
     * @param loc Source code location (set automatically by the corresponding macro).
     */
    void LogFatal(const std::string& message, const Location& loc = Location())
    {
        Log(eFatal, message, loc);
    }

private:
    struct Private;
    Private* fPrivate;
};

}

// PRIVATE MACROS

#define __KLOG_LOCATION         katrin::KLogger::Location(__FILE__, __FUNC__, __LINE__)

#define __KLOG_DEFINE_2(I,K)    static katrin::KLogger I(K);
#define __KLOG_DEFINE_1(K)      static katrin::KLogger sLocalLoggerInstance(K);

#define __KLOG_LOG_4(I,L,M,O) \
{ \
    if (I.IsLevelEnabled(katrin::KLogger::e##L)) { \
        static bool _sLoggerMarker = false; \
        if (!O || !_sLoggerMarker) { \
            _sLoggerMarker = true; \
            std::ostringstream stream; stream << M; \
            I.Log(katrin::KLogger::e##L, stream.str(), __KLOG_LOCATION); \
        } \
    } \
}

#define __KLOG_LOG_3(I,L,M)     __KLOG_LOG_4(I,L,M,false)
#define __KLOG_LOG_2(L,M)       __KLOG_LOG_4(sLocalLoggerInstance,L,M,false)
#define __KLOG_LOG_1(M)         __KLOG_LOG_4(sLocalLoggerInstance,Debug,M,false)

#define __KLOG_TRACE_2(I,M)     __KLOG_LOG_4(I,Trace,M,false)
#define __KLOG_TRACE_1(M)       __KLOG_LOG_4(sLocalLoggerInstance,Trace,M,false)

#define __KLOG_DEBUG_2(I,M)     __KLOG_LOG_4(I,Debug,M,false)
#define __KLOG_DEBUG_1(M)       __KLOG_LOG_4(sLocalLoggerInstance,Debug,M,false)

#define __KLOG_INFO_2(I,M)      __KLOG_LOG_4(I,Info,M,false)
#define __KLOG_INFO_1(M)        __KLOG_LOG_4(sLocalLoggerInstance,Info,M,false)

#define __KLOG_WARN_2(I,M)      __KLOG_LOG_4(I,Warn,M,false)
#define __KLOG_WARN_1(M)        __KLOG_LOG_4(sLocalLoggerInstance,Warn,M,false)

#define __KLOG_ERROR_2(I,M)     __KLOG_LOG_4(I,Error,M,false)
#define __KLOG_ERROR_1(M)       __KLOG_LOG_4(sLocalLoggerInstance,Error,M,false)

#define __KLOG_FATAL_2(I,M)     __KLOG_LOG_4(I,Fatal,M,false)
#define __KLOG_FATAL_1(M)       __KLOG_LOG_4(sLocalLoggerInstance,Fatal,M,false)

#define __KLOG_ASSERT_3(I,C,M)  if (!(C)) { __KLOG_ERROR_2(I,M) }
#define __KLOG_ASSERT_2(C,M)    __KLOG_ASSERT_3(sLocalLoggerInstance,C,M)


#define __KLOG_LOG_ONCE_3(I,L,M)     __KLOG_LOG_4(I,L,M,true)
#define __KLOG_LOG_ONCE_2(L,M)       __KLOG_LOG_4(sLocalLoggerInstance,L,M,true)
#define __KLOG_LOG_ONCE_1(M)         __KLOG_LOG_4(sLocalLoggerInstance,Debug,M,true)

#define __KLOG_TRACE_ONCE_2(I,M)     __KLOG_LOG_4(I,Trace,M,true)
#define __KLOG_TRACE_ONCE_1(M)       __KLOG_LOG_4(sLocalLoggerInstance,Trace,M,true)

#define __KLOG_DEBUG_ONCE_2(I,M)     __KLOG_LOG_4(I,Debug,M,true)
#define __KLOG_DEBUG_ONCE_1(M)       __KLOG_LOG_4(sLocalLoggerInstance,Debug,M,true)

#define __KLOG_INFO_ONCE_2(I,M)      __KLOG_LOG_4(I,Info,M,true)
#define __KLOG_INFO_ONCE_1(M)        __KLOG_LOG_4(sLocalLoggerInstance,Info,M,true)

#define __KLOG_WARN_ONCE_2(I,M)      __KLOG_LOG_4(I,Warn,M,true)
#define __KLOG_WARN_ONCE_1(M)        __KLOG_LOG_4(sLocalLoggerInstance,Warn,M,true)

#define __KLOG_ERROR_ONCE_2(I,M)     __KLOG_LOG_4(I,Error,M,true)
#define __KLOG_ERROR_ONCE_1(M)       __KLOG_LOG_4(sLocalLoggerInstance,Error,M,true)

#define __KLOG_FATAL_ONCE_2(I,M)     __KLOG_LOG_4(I,Fatal,M,true)
#define __KLOG_FATAL_ONCE_1(M)       __KLOG_LOG_4(sLocalLoggerInstance,Fatal,M,true)


// PUBLIC MACROS

#define KLOGGER(...)      macro_dispatcher(__KLOG_DEFINE_, __VA_ARGS__)(__VA_ARGS__)

#define KLOG(...)         macro_dispatcher(__KLOG_LOG_, __VA_ARGS__)(__VA_ARGS__)
#define KTRACE(...)       macro_dispatcher(__KLOG_TRACE_, __VA_ARGS__)(__VA_ARGS__)
#define KDEBUG(...)       macro_dispatcher(__KLOG_DEBUG_, __VA_ARGS__)(__VA_ARGS__)
#define KINFO(...)        macro_dispatcher(__KLOG_INFO_, __VA_ARGS__)(__VA_ARGS__)
#define KWARN(...)        macro_dispatcher(__KLOG_WARN_, __VA_ARGS__)(__VA_ARGS__)
#define KERROR(...)       macro_dispatcher(__KLOG_ERROR_, __VA_ARGS__)(__VA_ARGS__)
#define KFATAL(...)       macro_dispatcher(__KLOG_FATAL_, __VA_ARGS__)(__VA_ARGS__)
#define KASSERT(...)      macro_dispatcher(__KLOG_ASSERT_, __VA_ARGS__)(__VA_ARGS__)

#define KLOG_ONCE(...)    macro_dispatcher(__KLOG_LOG_ONCE_, __VA_ARGS__)(__VA_ARGS__)
#define KTRACE_ONCE(...)  macro_dispatcher(__KLOG_TRACE_ONCE_, __VA_ARGS__)(__VA_ARGS__)
#define KDEBUG_ONCE(...)  macro_dispatcher(__KLOG_DEBUG_ONCE_, __VA_ARGS__)(__VA_ARGS__)
#define KINFO_ONCE(...)   macro_dispatcher(__KLOG_INFO_ONCE_, __VA_ARGS__)(__VA_ARGS__)
#define KWARN_ONCE(...)   macro_dispatcher(__KLOG_WARN_ONCE_, __VA_ARGS__)(__VA_ARGS__)
#define KERROR_ONCE(...)  macro_dispatcher(__KLOG_ERROR_ONCE_, __VA_ARGS__)(__VA_ARGS__)
#define KFATAL_ONCE(...)  macro_dispatcher(__KLOG_FATAL_ONCE_, __VA_ARGS__)(__VA_ARGS__)

#endif /* KLOGGER_H_ */
