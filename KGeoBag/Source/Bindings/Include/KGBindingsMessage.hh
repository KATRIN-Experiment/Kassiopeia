#ifndef KGBINDINGSMESSAGE_HH_
#define KGBINDINGSMESSAGE_HH_

#include "KMessage.h"

KMESSAGE_DECLARE(KGeoBag, bindmsg)

#ifdef KGeoBag_ENABLE_DEBUG

#define bindmsg_DEBUG(xCONTENT) bindmsg(eDebug) << xCONTENT;

#endif

#ifndef bindmsg_DEBUG
#define bindmsg_DEBUG(xCONTENT)
#endif

#endif
