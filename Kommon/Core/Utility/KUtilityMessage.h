#ifndef KUTILITYMESSAGE_H_
#define KUTILITYMESSAGE_H_

#include "KMessage.h"

KMESSAGE_DECLARE( katrin, utilmsg )

#ifdef Kommon_ENABLE_DEBUG

#define utilmsg_debug( xCONTENT )\
    utilmsg( eDebug ) << xCONTENT;

#endif

#ifndef utilmsg_debug
#define utilmsg_debug( xCONTENT )
#endif

#endif
