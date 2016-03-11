#ifndef __KTimeTerminator_H__
#define __KTimeTerminator_H__

#ifdef KEMFIELD_USE_REALTIME_CLOCK
#include <ctime>
#include <time.h>
#endif

#include <cmath>
#include "KEMCout.hh"

#ifdef KEMFIELD_USE_MPI
    #include "KMPIInterface.hh"
    #ifndef MPI_ROOT_PROCESS_ONLY
        #define MPI_ROOT_PROCESS_ONLY if (KMPIInterface::GetInstance()->GetProcess()==0)
    #endif
#else
    #ifndef MPI_ROOT_PROCESS_ONLY
        #define MPI_ROOT_PROCESS_ONLY
    #endif
#endif

namespace KEMField
{


/**
*
*@file KTimeTerminator.hh
*@class KTimeTerminator
*@brief
*@details
*<b>Revision History:<b>
*Date Name Brief Description
*Wed Jul  2 13:21:38 EDT 2014 J. Barrett (barrettj@mit.edu) First Version
*
*/


template <typename ValueType>
class KTimeTerminator: public KIterativeSolver<ValueType>::Visitor
{
    public:

        KTimeTerminator():
            KIterativeSolver<ValueType>::Visitor(),
            fMaxTimeAllowed(1e10),
            fResult(0){};

        KTimeTerminator(double max_time_sec):
            KIterativeSolver<ValueType>::Visitor(),
            fMaxTimeAllowed(std::fabs(max_time_sec)),
            fResult(0){};

        KTimeTerminator(double max_time_sec, unsigned int interval):
            KIterativeSolver<ValueType>::Visitor(),
            fMaxTimeAllowed(std::fabs(max_time_sec)),
            fResult(0)
        {
            this->fInterval = interval;
        };

        virtual ~KTimeTerminator(){};

        void SetMaximumAllowedTime(double time_sec)
        {
            fMaxTimeAllowed = time_sec;
        };

        virtual void Initialize(KIterativeSolver<ValueType>&)
        {
            #ifdef KEMFIELD_USE_REALTIME_CLOCK
            clock_gettime(CLOCK_REALTIME, &fStart);
            this->fTerminate = false;
            fResult = 0;
            #else
            MPI_ROOT_PROCESS_ONLY
            {
                KEMField::cout<<"KTimeTerminator::Initialize: KEMField must be compiled with use of real time clock enabled."<<KEMField::endl;
            }
            #endif
        };

        virtual void Visit(KIterativeSolver<ValueType>&)
        {
            #ifdef KEMFIELD_USE_REALTIME_CLOCK
            timespec now;
            clock_gettime(CLOCK_REALTIME, &now);
            double time_elapsed = TimeDifferenceSec(fStart, now);

            if(time_elapsed >= fMaxTimeAllowed)
            {
                fResult = 1;
            }

            //only use the result determined by the root (0) process
            #ifdef KEMFIELD_USE_MPI
            MPI_Bcast( &(fResult), 1, MPI_INT, 0, MPI_COMM_WORLD );
            #endif

            if(fResult != 0)
            {
                this->fTerminate = true;
                MPI_ROOT_PROCESS_ONLY
                {
                    KEMField::cout<<"KTimeTerminator::Visit: Will terminate iterative solver progress because elapsed time of: ";
                    KEMField::cout<<time_elapsed<<"(s) exceeds the allowed time of: "<<fMaxTimeAllowed<<"(s)."<<KEMField::endl;
                }
            };
            #endif
        }

        virtual void Finalize(KIterativeSolver<ValueType>&){};

    protected:

        #ifdef KEMFIELD_USE_REALTIME_CLOCK
        double TimeDifferenceSec(timespec start, timespec end)
        {
            double end_sec = end.tv_sec;
            double start_sec = start.tv_sec;
            return std::fabs(end_sec - start_sec);
        }
        #endif

        double fMaxTimeAllowed;
        int fResult;
        timespec fStart;

};


}

#endif /* __KTimeTerminator_H__ */
