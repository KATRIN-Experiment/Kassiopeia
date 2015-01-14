#ifndef KIterativeKrylovCondition_HH__
#define KIterativeKrylovCondition_HH__

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
            fPreviousResidual = 1.0;
            fFirstResidual = 1.0;
            fAjacentDivergenceFactor = 0.0;
            fGlobalDivergenceFactor = 0.0;

            //these default values are quite high, so as to avoid triggering restarts
            //except in extreme circumstances, however they are not reasonable for most problems
            fNIterationsBetweenRestart = 10000; //number of iterations before a restart is triggered
            fMaxAjacentDivergenceFactor = 1e6; //residual must grow by this factor during one iteration to trigger a restart
            fMaxGlobalDivergenceFactor = 1e9; //residual must grow by this factor since last restart to trigger a restart
        };

        virtual ~KIterativeKrylovRestartCondition(){};

        void SetNumberOfIterationsBetweenRestart(unsigned int n){fNIterationsBetweenRestart = n;};
        void SetAjacentIterationDivergenceFactor(double fac){fMaxAjacentDivergenceFactor = fac;};
        void SetGlobalIterationDivergenceFactor(double fac){fMaxGlobalDivergenceFactor = fac;};


        virtual void UpdateProgress(double residual)
        {
            if(fIterationsSinceRestart == 0)
            {
                fFirstResidual = residual;
                fAjacentDivergenceFactor = 0.0;
                fGlobalDivergenceFactor = 0.0;
            }
            else
            {
                fAjacentDivergenceFactor = residual/fPreviousResidual;
                fGlobalDivergenceFactor = residual/fFirstResidual;
            }

            fPreviousResidual = residual;

            fIterationsSinceRestart++;
        }

        virtual bool PerformRestart()
        {
            if(fIterationsSinceRestart == 0){return false;};

            if(fIterationsSinceRestart >= fNIterationsBetweenRestart)
            {
                fIterationsSinceRestart = 0;
                fPreviousResidual = 1.0;
                fFirstResidual = 1.0;
                return true;
            }

            if(fGlobalDivergenceFactor > fMaxGlobalDivergenceFactor)
            {
                fIterationsSinceRestart = 0;
                fPreviousResidual = 1.0;
                fFirstResidual = 1.0;
                return true;
            }

            if(fAjacentDivergenceFactor > fMaxAjacentDivergenceFactor)
            {
                fIterationsSinceRestart = 0;
                fPreviousResidual = 1.0;
                fFirstResidual = 1.0;
                return true;
            }

            return false;
        }


    protected:

        unsigned int fNIterationsBetweenRestart;
        double fMaxAjacentDivergenceFactor;
        double fMaxGlobalDivergenceFactor;

        double fPreviousResidual;
        double fFirstResidual;
        double fAjacentDivergenceFactor;
        double fGlobalDivergenceFactor;
        unsigned int fIterationsSinceRestart;






};


}


#endif /* KIterativeKrylovCondition_H__ */
