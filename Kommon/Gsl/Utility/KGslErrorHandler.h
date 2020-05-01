#ifndef Kommon_KGslErrorHandler_h_
#define Kommon_KGslErrorHandler_h_

#include <KException.h>
#include <KSingleton.h>
#include <list>
#include <set>
#include <string>

namespace katrin
{

typedef bool gsl_error_callback_t(int gsl_errno);

typedef struct
{
    std::string what;
    int error;
    std::string file;
    int line;
} gsl_error_context_t;

class KGslException : public KExceptionPrototype<KGslException, KException>
{};

class KGslErrorHandler : public KSingleton<KGslErrorHandler>
{
  public:
    KGslErrorHandler();
    ~KGslErrorHandler();

    /**
         * @brief Enable GSL error handling.
         */
    void Enable();

    /**
         * @brief Disable GSL error handling and switch back to default behavior.
         */
    void Disable();

    /**
         * @brief Reset error handler.
         *
         * This clears the error status and removes any callback functions.
         */
    void Reset();

    /**
         * @brief Throw exception after an error was encountered.
         */
    void ThrowOnError(bool enable = true);

    /**
         * @brief Add a callback function to be executed in the error handler.
         * @param func: A callback function with signature `(bool)(int)`.
         * @sa Reset()
         *
         * All callback functions are called in the order they were added.
         * This happends after the error flag and context have been set,
         * and before any error messages are printed to the terminal.
         * If a callback returns `false`, the error handling stops at that point.
         */
    void AddCallback(gsl_error_callback_t* func);

    /**
         * @brief Clear the error status.
         */
    static void ClearError();

    /**
         * @brief Check the error status.
         * @return True if error flag has been set.
         *
         * Use Clear() to clear the error status.
         */
    static bool HasError();

    /**
         * @brief Return the current error.
         * @return The raw GSL error context.
         */
    static gsl_error_context_t GetContext();

    /**
         * @brief Return the current error as a readable message.
         * @return The GSL error context as string.
         */
    static std::string AsString();

  private:
    static void GslErrorHandler(const char* reason, const char* file, int line, int gsl_errno);

    static bool fGslErrorFlag;
    static gsl_error_context_t fGslErrorContext;

    bool fThrowOnError;
    std::set<gsl_error_callback_t*> fErrorCallbacks;
};

inline void KGslErrorHandler::ThrowOnError(bool enable)
{
    fThrowOnError = enable;
}

inline void KGslErrorHandler::AddCallback(gsl_error_callback_t* func)
{
    fErrorCallbacks.insert(func);
}

inline void KGslErrorHandler::ClearError()
{
    fGslErrorFlag = false;
    fGslErrorContext = gsl_error_context_t();
}

inline bool KGslErrorHandler::HasError()
{
    return fGslErrorFlag;
}

inline gsl_error_context_t KGslErrorHandler::GetContext()
{
    return fGslErrorContext;
}

}  // namespace katrin

#endif /* Kommon_KGslErrorHandler_h_ */
