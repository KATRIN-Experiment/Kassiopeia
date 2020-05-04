/*
 * KGMetricsMessage.hh
 *
 *  Created on: 21.05.2014
 *      Author: Jan Oertlin
 */

#ifndef KGMETRICSMESSAGE_HH_
#define KGMETRICSMESSAGE_HH_

#include "KMessage.h"

KMESSAGE_DECLARE(KGeoBag, metricsmsg)

#ifdef KGeoBag_ENABLE_DEBUG

#define metricsmsg_debug(xCONTENT) metricsmsg(eDebug) << xCONTENT;

#endif

#ifndef metricsmsg_debug
#define metricsmsg_debug(xCONTENT)
#endif

#endif /* KGMETRICSMESSAGE_HH_ */
