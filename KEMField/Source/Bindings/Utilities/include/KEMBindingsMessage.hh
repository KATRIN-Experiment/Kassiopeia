#ifndef KEMBindingsMessage_hh_
#define KEMBindingsMessage_hh_

#include "KMessage.h"

KMESSAGE_DECLARE(katrin, BINDINGMSG)

#ifdef KEMField_ENABLE_DEBUG

#define BINDINGMSG_DEBUG(xCONTENT) BINDINGMSG(eDebug) << xCONTENT;

#define BINDINGMSG_ASSERT(xVARIABLE, xASSERTION)                                                                       \
    if (!(xVARIABLE xASSERTION))                                                                                       \
        BINDINGMSG(eError) << "Assertion failed: " << #xVARIABLE << " " << #xASSERTION << " but " << #xVARIABLE        \
                           << " is " << (xVARIABLE) << eom;

#endif

#ifndef BINDINGMSG_DEBUG
#define BINDINGMSG_DEBUG(xCONTENT)
#define BINDINGMSG_ASSERT(xVARIABLE, xASSERTION)
#endif

#endif
