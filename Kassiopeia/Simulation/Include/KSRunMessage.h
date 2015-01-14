#ifndef KSRUNMESSAGE_H_
#define KSRUNMESSAGE_H_

#include "KMessage.h"

KMESSAGE_DECLARE( Kassiopeia, runmsg )

#ifdef Kassiopeia_ENABLE_DEBUG

#define runmsg_debug( xCONTENT )\
    runmsg( eDebug ) << xCONTENT;

#define runmsg_assert( xVARIABLE, xASSERTION )\
    if (! (xVARIABLE xASSERTION)) fieldmsg( eError ) << "Assertion failed: " << #xVARIABLE << " " << #xASSERTION << " but " << #xVARIABLE << " is " << (xVARIABLE) << eom;

#endif

#ifndef runmsg_debug
#define runmsg_debug( xCONTENT )
#define runmsg_assert( xVARIABLE, xASSERTION )
#endif

#endif
