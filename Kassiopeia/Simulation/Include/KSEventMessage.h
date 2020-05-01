#ifndef KSEVENTMESSAGE_H_
#define KSEVENTMESSAGE_H_

#include "KMessage.h"

KMESSAGE_DECLARE(Kassiopeia, eventmsg)

#ifdef Kassiopeia_ENABLE_DEBUG

#define eventmsg_debug(xCONTENT) eventmsg(eDebug) << xCONTENT;

#define eventmsg_assert(xVARIABLE, xASSERTION)                                                                         \
    if (!(xVARIABLE xASSERTION))                                                                                       \
        fieldmsg(eError) << "Assertion failed: " << #xVARIABLE << " " << #xASSERTION << " but " << #xVARIABLE          \
                         << " is " << (xVARIABLE) << eom;

#endif

#ifndef eventmsg_debug
#define eventmsg_debug(xCONTENT)
#define eventmsg_assert(xVARIABLE, xASSERTION)
#endif

#endif
