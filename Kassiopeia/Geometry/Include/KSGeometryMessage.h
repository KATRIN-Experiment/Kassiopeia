#ifndef Kassiopeia_KSGeometriesMessage_h_
#define Kassiopeia_KSGeometriesMessage_h_

#include "KMessage.h"

KMESSAGE_DECLARE(Kassiopeia, geomsg)

#ifdef Kassiopeia_ENABLE_DEBUG

#define geomsg_debug(xCONTENT) geomsg(eDebug) << xCONTENT;

#define geomsg_assert(xVARIABLE, xASSERTION)                                                                           \
    if (!(xVARIABLE xASSERTION))                                                                                       \
        fieldmsg(eError) << "Assertion failed: " << #xVARIABLE << " " << #xASSERTION << " but " << #xVARIABLE          \
                         << " is " << (xVARIABLE) << eom;

#endif

#ifndef geomsg_debug
#define geomsg_debug(xCONTENT)
#define geomsg_assert(xVARIABLE, xASSERTION)
#endif

#endif
