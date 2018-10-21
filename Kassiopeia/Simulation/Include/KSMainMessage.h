#ifndef KSMAINMESSAGE_H_
#define KSMAINMESSAGE_H_

#include "KMessage.h"

KMESSAGE_DECLARE( Kassiopeia, mainmsg )

#ifdef Kassiopeia_ENABLE_DEBUG

#define mainmsg_debug( xCONTENT )\
    mainmsg( eDebug ) << xCONTENT;

#define mainmsg_assert( xVARIABLE, xASSERTION )\
    if (! (xVARIABLE xASSERTION)) mainmsg( eError ) << "Assertion failed: " << #xVARIABLE << " " << #xASSERTION << " but " << #xVARIABLE << " is " << (xVARIABLE) << eom;

#endif

#ifndef mainmsg_debug
#define mainmsg_debug( xCONTENT )
#define mainmsg_assert( xVARIABLE, xASSERTION )
#endif

#endif
