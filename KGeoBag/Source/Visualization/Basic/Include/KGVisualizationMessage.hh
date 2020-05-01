#ifndef KGVISUALIZATIONMESSAGE_HH_
#define KGVISUALIZATIONMESSAGE_HH_

#include "KMessage.h"

KMESSAGE_DECLARE(KGeoBag, vismsg)

#ifdef KGeoBag_ENABLE_DEBUG

#define vismsg_debug(xCONTENT) vismsg(eDebug) << xCONTENT;

#endif

#ifndef vismsg_debug
#define vismsg_debug(xCONTENT)
#endif

#endif
