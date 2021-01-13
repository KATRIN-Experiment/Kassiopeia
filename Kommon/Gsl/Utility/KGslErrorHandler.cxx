#include "KGslErrorHandler.h"

#include "KLogger.h"

#include <gsl/gsl_errno.h>

#ifdef KASPER_USE_BOOST
#include <boost/stacktrace.hpp>
#endif

#include <sstream>

KLOGGER("kommon.gsl");

namespace katrin
{

bool KGslErrorHandler::fGslErrorFlag = false;
gsl_error_context_t KGslErrorHandler::fGslErrorContext;

KGslErrorHandler::KGslErrorHandler() : fThrowOnError(true), fErrorCallbacks()
{
    Enable();
}

KGslErrorHandler::~KGslErrorHandler()
{
    Disable();
}

void KGslErrorHandler::Enable()
{
    ClearError();

    KDEBUG("enabling gsl error handler");
    gsl_set_error_handler(KGslErrorHandler::GslErrorHandler);
}

void KGslErrorHandler::Disable()
{
    KDEBUG("disabling gsl error handler");
    gsl_set_error_handler_off();
}

void KGslErrorHandler::Reset()
{
    fThrowOnError = true;
    fErrorCallbacks.clear();
    ClearError();
}

std::string KGslErrorHandler::AsString()
{
    if (!fGslErrorFlag)
        return "";

    std::stringstream ss;
    ss << "GSL error: " << fGslErrorContext.file << ":" << fGslErrorContext.line << ": " << fGslErrorContext.what
       << " (" << fGslErrorContext.error << ")";
    return ss.str();
}

void KGslErrorHandler::GslErrorHandler(const char* reason, const char* file, int line, int gsl_errno) /* static */
{
    // Ignore additional errors once the flag has been set.
    // This also avoids spamming the console before the error can be processed.
    if (fGslErrorFlag)
        return;

    fGslErrorContext.what = reason;
    fGslErrorContext.error = gsl_errno;
    fGslErrorContext.file = file;
    fGslErrorContext.line = line;
    fGslErrorFlag = true;

    auto& handler = KGslErrorHandler::GetInstance();

    // Callbacks can abort the error handler if they return false
    for (auto& func : handler.fErrorCallbacks) {
        if (!(*func)(gsl_errno) /*callback*/)
            return;
    }

    KERROR(KGslErrorHandler::AsString());

#ifdef KASPER_USE_BOOST
    // Show up to 5 stack frames below current function
    const auto& stackFrames = boost::stacktrace::stacktrace();
    for (size_t depth = 0; depth < stackFrames.size(); ++depth) {
        const auto& frame = stackFrames[depth];
        if (depth == 0 || frame.name().empty())
            continue;
        if (depth > 5 || frame.empty())
            break;

        std::cout << "  " << depth << "# " << frame.name();
        if (!frame.source_file().empty())
            std::cout << " [" << frame.source_file() << ":" << frame.source_line() << "]";
        std::cout << std::endl;
    }
#endif

    if (handler.fThrowOnError)
        throw KGslException() << KGslErrorHandler::AsString();
}

}  // namespace katrin
