#ifndef KSTRACKMESSAGE_H_
#define KSTRACKMESSAGE_H_

#include "KMessage.h"

KMESSAGE_DECLARE( Kassiopeia, trackmsg )

#ifdef Kassiopeia_ENABLE_DEBUG

#define trackmsg_debug( xCONTENT )\
     trackmsg( eDebug ) << xCONTENT;

#define trackmsg_assert( xVARIABLE, xASSERTION )\
    if (! (xVARIABLE xASSERTION)) fieldmsg( eError ) << "Assertion failed: " << #xVARIABLE << " " << #xASSERTION << " but " << #xVARIABLE << " is " << (xVARIABLE) << eom;

#endif

#ifndef trackmsg_debug
#define trackmsg_debug( xCONTENT )
#define trackmsg_assert( xVARIABLE, xASSERTION )
#endif

#endif
