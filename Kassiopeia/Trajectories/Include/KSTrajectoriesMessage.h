#ifndef Kassiopeia_KSTrajectoriesMessage_h_
#define Kassiopeia_KSTrajectoriesMessage_h_

#include "KMessage.h"

KMESSAGE_DECLARE( Kassiopeia, trajmsg )

#ifdef Kassiopeia_ENABLE_DEBUG

#define trajmsg_debug( xCONTENT )\
    trajmsg( eDebug ) << xCONTENT;

#define trajmsg_assert( xVARIABLE, xASSERTION )\
    if (! (xVARIABLE xASSERTION)) trajmsg( eError ) << "Assertion failed: " << #xVARIABLE << " " << #xASSERTION << " but " << #xVARIABLE << " is " << (xVARIABLE) << eom;

#endif

#ifndef trajmsg_debug
#define trajmsg_debug( xCONTENT )
#define trajmsg_assert( xVARIABLE, xASSERTION )
#endif

#endif
