#ifndef Kassiopeia_KSWritersMessage_h_
#define Kassiopeia_KSWritersMessage_h_

#include "KMessage.h"

KMESSAGE_DECLARE( Kassiopeia, wtrmsg )

#ifdef Kassiopeia_ENABLE_DEBUG

#define wtrmsg_debug( xCONTENT )\
    wtrmsg( eDebug ) << xCONTENT;

#define wtrmsg_assert( xVARIABLE, xASSERTION )\
    if (! (xVARIABLE xASSERTION)) fieldmsg( eError ) << "Assertion failed: " << #xVARIABLE << " " << #xASSERTION << " but " << #xVARIABLE << " is " << (xVARIABLE) << eom;

#endif

#ifndef wtrmsg_debug
#define wtrmsg_debug( xCONTENT )
#define wtrmsg_assert( xVARIABLE, xASSERTION )
#endif

#endif
