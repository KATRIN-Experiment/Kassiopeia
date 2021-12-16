#ifndef KFILEMESSAGE_H_
#define KFILEMESSAGE_H_

#include "KMessage.h"

KMESSAGE_DECLARE(katrin, filemsg)

#ifdef Kommon_ENABLE_DEBUG

#define filemsg_debug(xCONTENT) filemsg(eDebug) << xCONTENT;

#endif

#ifndef filemsg_debug
#define filemsg_debug(xCONTENT)
#endif

#endif
