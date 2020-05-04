#include "KGslErrorHandler.h"

#include "KLogger.h"

#include <gsl/gsl_errno.h>
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

    for (auto& func : handler.fErrorCallbacks)
        if (!(*func)(gsl_errno) /*callback*/)
            return;

    KERROR(KGslErrorHandler::AsString());

    if (handler.fThrowOnError)
        throw KGslException() << KGslErrorHandler::AsString();
}

}  // namespace katrin
