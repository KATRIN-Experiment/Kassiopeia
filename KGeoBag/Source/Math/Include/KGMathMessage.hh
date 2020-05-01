#ifndef KGMATHMESSAGE_HH_
#define KGMATHMESSAGE_HH_

#include "KMessage.h"

KMESSAGE_DECLARE(KGeoBag, mathmsg)

#ifdef KGeoBag_ENABLE_DEBUG

#define mathmsg_debug(xCONTENT) mathmsg(eDebug) << xCONTENT;

#endif

#ifndef mathmsg_debug
#define mathmsg_debug(xCONTENT)
#endif

#endif
