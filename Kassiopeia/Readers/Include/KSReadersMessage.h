#ifndef KSREADERMESSAGE_H_
#define KSREADERMESSAGE_H_

#include "KMessage.h"

KMESSAGE_DECLARE( Kassiopeia, readermsg )

#ifdef Kassiopeia_ENABLE_DEBUG

#define readermsg_debug( xCONTENT )\
    readermsg( eDebug ) << xCONTENT;

#define readermsg_assert( xVARIABLE, xASSERTION )\
    if (! (xVARIABLE xASSERTION)) fieldmsg( eError ) << "Assertion failed: " << #xVARIABLE << " " << #xASSERTION << " but " << #xVARIABLE << " is " << (xVARIABLE) << eom;

#endif

#ifndef readermsg_debug
#define readermsg_debug( xCONTENT )
#define readermsg_assert( xVARIABLE, xASSERTION )
#endif

#endif
