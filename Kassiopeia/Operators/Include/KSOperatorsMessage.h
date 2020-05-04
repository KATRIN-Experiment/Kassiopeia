#ifndef Kassiopeia_KSOperatorsMessage_h_
#define Kassiopeia_KSOperatorsMessage_h_

#include "KMessage.h"

KMESSAGE_DECLARE(Kassiopeia, oprmsg)

#ifdef Kassiopeia_ENABLE_DEBUG
#define oprmsg_debug(xCONTENT) oprmsg(eDebug) << xCONTENT;

#define oprmsg_assert(xVARIABLE, xASSERTION)                                                                           \
    if (!(xVARIABLE xASSERTION))                                                                                       \
        oprmsg(eError) << "Assertion failed: " << #xVARIABLE << " " << #xASSERTION << " but " << #xVARIABLE << " is "  \
                       << (xVARIABLE) << eom;

#endif

#ifndef oprmsg_debug
#define oprmsg_debug(xCONTENT)
#define oprmsg_assert(xVARIABLE, xASSERTION)
#endif

#endif
