#ifndef Kassiopeia_KSMathMessage_h_
#define Kassiopeia_KSMathMessage_h_

#include "KMessage.h"

KMESSAGE_DECLARE(Kassiopeia, mathmsg)

#ifdef Kassiopeia_ENABLE_DEBUG

#define mathmsg_debug(xCONTENT) mathmsg(eDebug) << xCONTENT;

#define mathmsg_assert(xVARIABLE, xASSERTION)                                                                          \
    if (!(xVARIABLE xASSERTION))                                                                                       \
        fieldmsg(eError) << "Assertion failed: " << #xVARIABLE << " " << #xASSERTION << " but " << #xVARIABLE          \
                         << " is " << (xVARIABLE) << eom;

#endif

#ifndef mathmsg_debug
#define mathmsg_debug(xCONTENT)
#define mathmsg_assert(xVARIABLE, xASSERTION)
#endif

#endif
