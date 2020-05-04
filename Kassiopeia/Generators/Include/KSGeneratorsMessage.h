#ifndef Kassiopeia_KSGeneratorsMessage_h_
#define Kassiopeia_KSGeneratorsMessage_h_

#include "KMessage.h"

KMESSAGE_DECLARE(Kassiopeia, genmsg)

#ifdef Kassiopeia_ENABLE_DEBUG

#define genmsg_debug(xCONTENT) genmsg(eDebug) << xCONTENT;

#define genmsg_assert(xVARIABLE, xASSERTION)                                                                           \
    if (!(xVARIABLE xASSERTION))                                                                                       \
        genmsg(eError) << "Assertion failed: " << #xVARIABLE << " " << #xASSERTION << " but " << #xVARIABLE << " is "  \
                       << (xVARIABLE) << eom;

#endif

#ifndef genmsg_debug
#define genmsg_debug(xCONTENT)
#define genmsg_assert(xVARIABLE, xASSERTION)
#endif

#endif
