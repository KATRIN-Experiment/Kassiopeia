#ifndef KGCOREMESSAGE_HH_
#define KGCOREMESSAGE_HH_

#include "KMessage.h"

KMESSAGE_DECLARE( KGeoBag, coremsg )

#ifdef KGeoBag_ENABLE_DEBUG

#define coremsg_debug( xCONTENT )\
    coremsg( eDebug ) << xCONTENT;

#endif

#ifndef coremsg_debug
#define coremsg_debug( xCONTENT )
#endif

#endif
