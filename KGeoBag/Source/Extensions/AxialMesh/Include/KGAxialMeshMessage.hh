#ifndef KGeoBag_KGAxialMeshMessage_hh_
#define KGeoBag_KGAxialMeshMessage_hh_

#include "KMessage.h"

KMESSAGE_DECLARE( KGeoBag, axialmeshmsg )

#ifdef KGeoBag_ENABLE_DEBUG

#define axialmeshmsg_debug( xCONTENT )\
    axialmeshmsg( eDebug ) << xCONTENT;

#endif

#ifndef axialmeshmsg_debug
#define axialmeshmsg_debug( xCONTENT )
#endif



#endif
