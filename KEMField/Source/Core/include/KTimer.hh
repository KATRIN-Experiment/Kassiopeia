/*
 * KTimer.hh
 *
 *  Created on: 21 Sep 2015
 *      Author: wolfgang
 */

#ifndef KTIMER_HH_
#define KTIMER_HH_

#include <ctime>
#include <string>

namespace KEMField
{

class KTimer
{
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

}  // namespace KEMField

#endif /* KTIMER_HH_ */
