#ifndef Kassiopeia_KSModifiersMessage_h_
#define Kassiopeia_KSModifiersMessage_h_

#include "KMessage.h"

KMESSAGE_DECLARE( Kassiopeia, modmsg )

#ifdef Kassiopeia_ENABLE_DEBUG

#define modmsg_debug( xCONTENT )\
		modmsg( eDebug ) << xCONTENT;

#endif

#ifndef modmsg_debug
#define modmsg_debug( xCONTENT )
#endif

#endif
