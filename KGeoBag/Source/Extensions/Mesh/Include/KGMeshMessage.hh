#ifndef KGeoBag_KGMeshMessage_hh_
#define KGeoBag_KGMeshMessage_hh_

#include "KMessage.h"

KMESSAGE_DECLARE( KGeoBag, meshmsg )

#ifdef KGeoBag_ENABLE_DEBUG

#define meshmsg_debug( xCONTENT )\
    meshmsg( eDebug ) << xCONTENT;

#endif

#ifndef meshmsg_debug
#define meshmsg_debug( xCONTENT )
#endif

#endif
