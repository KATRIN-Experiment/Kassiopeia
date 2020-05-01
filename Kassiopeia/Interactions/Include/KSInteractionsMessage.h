#ifndef Kassiopeia_KSInteractionsMessage_h_
#define Kassiopeia_KSInteractionsMessage_h_

#include "KMessage.h"

KMESSAGE_DECLARE(Kassiopeia, intmsg)

#ifdef Kassiopeia_ENABLE_DEBUG

#define intmsg_debug(xCONTENT) intmsg(eDebug) << xCONTENT;

#define intmsg_assert(xVARIABLE, xASSERTION)                                                                           \
    if (!(xVARIABLE xASSERTION))                                                                                       \
        fieldmsg(eError) << "Assertion failed: " << #xVARIABLE << " " << #xASSERTION << " but " << #xVARIABLE          \
                         << " is " << (xVARIABLE) << eom;

#endif

#ifndef intmsg_debug
#define intmsg_debug(xCONTENT)
#define intmsg_assert(xVARIABLE, xASSERTION)
#endif

#endif
