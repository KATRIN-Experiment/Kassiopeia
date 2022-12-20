#ifndef Kommon_KInitializationMessage_hh_
#define Kommon_KInitializationMessage_hh_

#include "KMessage.h"

KMESSAGE_DECLARE(katrin, initmsg)

#ifdef Kommon_ENABLE_DEBUG

#define initmsg_debug(xCONTENT) initmsg(eDebug) << xCONTENT;

#endif

#ifndef initmsg_debug
#define initmsg_debug(xCONTENT)
#endif

#endif
