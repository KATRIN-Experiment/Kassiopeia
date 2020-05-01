#ifndef Kassiopeia_KSNavigatorsMessage_h_
#define Kassiopeia_KSNavigatorsMessage_h_

#include "KMessage.h"

KMESSAGE_DECLARE(Kassiopeia, navmsg)

#ifdef Kassiopeia_ENABLE_DEBUG

#define navmsg_debug(xCONTENT) navmsg(eDebug) << xCONTENT;

#define navmsg_assert(xVARIABLE, xASSERTION)                                                                           \
    if (!(xVARIABLE xASSERTION))                                                                                       \
        fieldmsg(eError) << "Assertion failed: " << #xVARIABLE << " " << #xASSERTION << " but " << #xVARIABLE          \
                         << " is " << (xVARIABLE) << eom;

#endif

#ifndef navmsg_debug
#define navmsg_debug(xCONTENT)
#define navmsg_assert(xVARIABLE, xASSERTION)
#endif

#endif
