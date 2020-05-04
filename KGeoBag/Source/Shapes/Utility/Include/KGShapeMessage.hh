#ifndef KGSHAPEMESSAGE_HH_
#define KGSHAPEMESSAGE_HH_

#include "KMessage.h"

KMESSAGE_DECLARE(KGeoBag, shapemsg)

#ifdef KGeoBag_ENABLE_DEBUG

#define shapemsg_debug(xCONTENT) shapemsg(eDebug) << xCONTENT;

#endif

#ifndef shapemsg_debug
#define shapemsg_debug(xCONTENT)
#endif

#endif
