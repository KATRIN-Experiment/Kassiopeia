#ifndef KIterativeKrylovCondition_HH__
#define KIterativeKrylovCondition_HH__

#include <climits>

namespace KEMField
{

/*
*
*@file KIterativeKrylovRestartCondition.hh
*@class KIterativeKrylovCondition
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Tue Jun  3 12:45:16 EDT 2014 J. Barrett (barrettj@mit.edu) First Version
*
*/


class KIterativeKrylovRestartCondition
{
  public:
    KIterativeKrylovRestartCondition()
    {
        fIterationsSinceRestart = 0;
        //these default values are quite high, so as to avoid triggering restarts
        //except in extreme circumstances, however they are not reasonable for most problems
        fNIterationsBetweenRestart = UINT_MAX;  //number of iterations before a restart is triggered
    };

    virtual ~KIterativeKrylovRestartCondition(){};

    virtual void SetNumberOfIterationsBetweenRestart(unsigned int n)
    {
        fNIterationsBetweenRestart = n;
    };

    virtual void UpdateProgress(double /* residual */)
    {
        fIterationsSinceRestart++;
    }

    virtual bool PerformRestart()
    {
        if (fIterationsSinceRestart == 0) {
            return false;
        };

        if (fIterationsSinceRestart >= fNIterationsBetweenRestart) {
            fIterationsSinceRestart = 0;
            return true;
        }
        return false;
    }


  protected:
    unsigned int fNIterationsBetweenRestart;
    unsigned int fIterationsSinceRestart;
};


}  // namespace KEMField


#endif /* KIterativeKrylovCondition_H__ */
