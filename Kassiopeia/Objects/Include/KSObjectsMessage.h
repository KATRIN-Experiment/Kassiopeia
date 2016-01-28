#ifndef Kassopieia_KSObjectsMessage_h_
#define Kassopieia_KSObjectsMessage_h_

#include "KMessage.h"

KMESSAGE_DECLARE( Kassiopeia, objctmsg )

#ifdef Kassiopeia_ENABLE_DEBUG

#define objctmsg_debug( xCONTENT )\
        objctmsg( eDebug ) << xCONTENT;

#define objctmsg_assert( xVARIABLE, xASSERTION )\
    if (! (xVARIABLE xASSERTION)) fieldmsg( eError ) << "Assertion failed: " << #xVARIABLE << " " << #xASSERTION << " but " << #xVARIABLE << " is " << (xVARIABLE) << eom;

#endif

#ifndef objctmsg_debug
#define objctmsg_debug( xCONTENT )
#define objctmsg_assert( xVARIABLE, xASSERTION )
#endif

#endif
