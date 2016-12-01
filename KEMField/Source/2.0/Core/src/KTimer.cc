/*
 * KTimer.cc
 *
 *  Created on: 21 Sep 2015
 *      Author: wolfgang
 */
#include "KEMCout.hh"


#include "KTimer.hh"

namespace KEMField {

KTimer::KTimer(std::string timedActionDescription) {
 fDescription = timedActionDescription;
}

KTimer::~KTimer() {}

void KTimer::start() {
#ifdef KEMFIELD_USE_REALTIME_CLOCK
	clock_gettime(CLOCK_REALTIME, &fStart);
#endif
    fCStart = clock();
}

void KTimer::end() {
#ifdef KEMFIELD_USE_REALTIME_CLOCK
	clock_gettime(CLOCK_REALTIME, &fEnd);
#endif
	fCEnd = clock();


}

void KTimer::display() {
	KEMField::cout << fDescription << " took ";
#ifdef KEMFIELD_USE_REALTIME_CLOCK
	timespec temp = TimeDifference(fStart, fEnd);
	KEMField::cout << temp.tv_sec << "." << temp.tv_nsec << " real time seconds or ";
#endif
	double time = ((double)(fCEnd - fCStart))/CLOCKS_PER_SEC; // time in seconds
	cout << time << " process/CPU Time seconds to perform." << KEMField::endl;
}

#ifdef KEMFIELD_USE_REALTIME_CLOCK
timespec KTimer::TimeDifference(timespec start, timespec end)
{

    timespec temp;
    if( (end.tv_nsec-start.tv_nsec) < 0)
    {
        temp.tv_sec = end.tv_sec-start.tv_sec-1;
        temp.tv_nsec = 1000000000+end.tv_nsec-start.tv_nsec;
    }
    else
    {
        temp.tv_sec = end.tv_sec-start.tv_sec;
        temp.tv_nsec = end.tv_nsec-start.tv_nsec;
    }
    return temp;
}
#endif

} /* namespace Kassiopeia */
