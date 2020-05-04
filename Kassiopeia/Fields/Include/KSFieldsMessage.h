#ifndef Kassiopeia_KSFieldMessage_h_
#define Kassiopeia_KSFieldMessage_h_

#include "KMessage.h"

KMESSAGE_DECLARE(Kassiopeia, fieldmsg)

#ifdef Kassiopeia_ENABLE_DEBUG

#define fieldmsg_debug(xCONTENT) fieldmsg(eDebug) << xCONTENT;

#define fieldmsg_assert(xVARIABLE, xASSERTION)                                                                         \
    if (!(xVARIABLE xASSERTION))                                                                                       \
        fieldmsg(eError) << "Assertion failed: " << #xVARIABLE << " " << #xASSERTION << " but " << #xVARIABLE          \
                         << " is " << (xVARIABLE) << eom;

#endif

#ifndef fieldmsg_debug
#define fieldmsg_debug(xCONTENT)
#define fieldmsg_assert(xVARIABLE, xASSERTION)
#endif

#endif
