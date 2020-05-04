#ifndef KSSTEPMESSAGE_H_
#define KSSTEPMESSAGE_H_

#include "KMessage.h"

KMESSAGE_DECLARE(Kassiopeia, stepmsg)

#ifdef Kassiopeia_ENABLE_DEBUG

#define stepmsg_debug(xCONTENT) stepmsg(eDebug) << xCONTENT;

#define stepmsg_assert(xVARIABLE, xASSERTION)                                                                          \
    if (!(xVARIABLE xASSERTION))                                                                                       \
        stepmsg(eError) << "Assertion failed: " << #xVARIABLE << " " << #xASSERTION << " but " << #xVARIABLE << " is " \
                        << (xVARIABLE) << eom;

#endif

#ifndef stepmsg_debug
#define stepmsg_debug(xCONTENT)
#define stepmsg_assert(xVARIABLE, xASSERTION)
#endif

#endif
