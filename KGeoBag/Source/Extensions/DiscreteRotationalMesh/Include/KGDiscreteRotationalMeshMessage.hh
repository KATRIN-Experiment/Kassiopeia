#ifndef KGeoBag_KGDiscreteRotationalMeshMessage_hh_
#define KGeoBag_KGDiscreteRotationalMeshMessage_hh_

#include "KMessage.h"

KMESSAGE_DECLARE( KGeoBag, drmeshmsg )

#ifdef KGeoBag_ENABLE_DEBUG

#define drmeshmsg_debug( xCONTENT )\
    drmeshmsg( eDebug ) << xCONTENT;

#endif

#ifndef drmeshmsg_debug
#define drmeshmsg_debug( xCONTENT )
#endif

#endif
