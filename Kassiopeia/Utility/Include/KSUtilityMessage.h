#ifndef KSUTILITYMESSAGE_H_
#define KSUTILITYMESSAGE_H_

#include "KMessage.h"

KMESSAGE_DECLARE( Kassiopeia, ksutilmsg )

#ifdef Kassiopeia_ENABLE_DEBUG

#define ksutilmsg_debug( xCONTENT )\
    ksutilmsg( eDebug ) << xCONTENT;

#define ksutilmsg_assert( xVARIABLE, xASSERTION )\
    if (! (xVARIABLE xASSERTION)) fieldmsg( eError ) << "Assertion failed: " << #xVARIABLE << " " << #xASSERTION << " but " << #xVARIABLE << " is " << (xVARIABLE) << eom;

#endif

#ifndef ksutilmsg_debug
#define ksutilmsg_debug( xCONTENT )
#define ksutilmsg_assert( xVARIABLE, xASSERTION )
#endif

#endif
