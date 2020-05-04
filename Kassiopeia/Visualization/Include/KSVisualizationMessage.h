#ifndef KSVISUALIZATIONMESSAGE_HH_
#define KSVISUALIZATIONMESSAGE_HH_

#include "KMessage.h"

KMESSAGE_DECLARE(Kassiopeia, vismsg)

#ifdef Kassiopeia_ENABLE_DEBUG

#define vismsg_debug(xCONTENT) vismsg(eDebug) << xCONTENT;

#define vismsg_assert(xVARIABLE, xASSERTION)                                                                           \
    if (!(xVARIABLE xASSERTION))                                                                                       \
        vismsg(eError) << "Assertion failed: " << #xVARIABLE << " " << #xASSERTION << " but " << #xVARIABLE << " is "  \
                       << (xVARIABLE) << eom;

#endif

#ifndef vismsg_debug
#define vismsg_debug(xCONTENT)
#define vismsg_assert(xVARIABLE, xASSERTION)
#endif

#endif
