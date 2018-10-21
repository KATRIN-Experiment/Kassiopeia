/*
 * KTimer.hh
 *
 *  Created on: 21 Sep 2015
 *      Author: wolfgang
 */

#ifndef KEMFIELD_SOURCE_2_0_CORE_INCLUDE_KTIMER_HH_
#define KEMFIELD_SOURCE_2_0_CORE_INCLUDE_KTIMER_HH_

#include <time.h>

namespace KEMField {

class KTimer {
public:
	KTimer(std::string timedActionDescription);
	virtual ~KTimer();
	void start();
	void end();
	void display();

private:
	std::string fDescription;

#ifdef KEMFIELD_USE_REALTIME_CLOCK
	timespec TimeDifference(timespec start, timespec end);

	timespec fStart, fEnd;
#endif
	clock_t fCStart, fCEnd;
};

} /* namespace Kassiopeia */

#endif /* KEMFIELD_SOURCE_2_0_CORE_INCLUDE_KTIMER_HH_ */
