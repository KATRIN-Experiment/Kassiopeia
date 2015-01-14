#ifndef Kassiopeia_KSTerminatorsMessage_h_
#define Kassiopeia_KSTerminatorsMessage_h_

#include "KMessage.h"

KMESSAGE_DECLARE( Kassiopeia, termmsg )

#ifdef Kassiopeia_ENABLE_DEBUG

#define termmsg_debug( xCONTENT )\
    termmsg( eDebug ) << xCONTENT;

#define termmsg_assert( xVARIABLE, xASSERTION )\
    if (! (xVARIABLE xASSERTION)) fieldmsg( eError ) << "Assertion failed: " << #xVARIABLE << " " << #xASSERTION << " but " << #xVARIABLE << " is " << (xVARIABLE) << eom;

#endif

#ifndef termmsg_debug
#define termmsg_debug( xCONTENT )
#define termmsg_assert( xVARIABLE, xASSERTION )
#endif

#endif
