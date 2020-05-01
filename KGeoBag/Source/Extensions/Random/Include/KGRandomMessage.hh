/*
 * KGRandomMessage.hh
 *
 *  Created on: 21.05.2014
 *      Author: user
 */

#ifndef KGRANDOMMESSAGE_HH_
#define KGRANDOMMESSAGE_HH_

#include "KMessage.h"

KMESSAGE_DECLARE(KGeoBag, randommsg)

#ifdef KGeoBag_ENABLE_DEBUG

#define randommsg_debug(xCONTENT) metricsmsg(eDebug) << xCONTENT;

#endif

#ifndef randommsg_debug
#define randommsg_debug(xCONTENT)
#endif

#endif /* KGRANDOMMESSAGE_HH_ */
