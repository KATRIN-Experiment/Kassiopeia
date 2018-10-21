#ifndef __KIterationTerminator_H__
#define __KIterationTerminator_H__

#include <iostream>

namespace KEMField
{


/**
*
*@file KIterationTerminator.hh
*@class KIterationTerminator
*@brief
*@details
*<b>Revision History:<b>
*Date Name Brief Description
*Wed Jul  2 13:21:38 EDT 2014 J. Barrett (barrettj@mit.edu) First Version
*
*/


template <typename ValueType>
class KIterationTerminator: public KIterativeSolver<ValueType>::Visitor
{
    public:

        KIterationTerminator():
            KIterativeSolver<ValueType>::Visitor(),
            fMaxIterations(UINT_MAX){};

        KIterationTerminator(unsigned int max_iter):
            KIterativeSolver<ValueType>::Visitor(),
            fMaxIterations(max_iter){};

        KIterationTerminator(unsigned int max_iter, unsigned int interval):
            KIterativeSolver<ValueType>::Visitor(),
            fMaxIterations(max_iter)
        {
            this->fInterval = interval;
        };

        virtual ~KIterationTerminator(){};

        void SetMaximumIterations(unsigned int max_iter)
        {
            fMaxIterations = max_iter;
        };

        virtual void Initialize(KIterativeSolver<ValueType>&){;};

        virtual void Visit(KIterativeSolver<ValueType>& solver)
        {
            unsigned int iter = solver.GetIteration();
            if(iter >= fMaxIterations)
            {
                this->fTerminate = true;
            }
        }

        virtual void Finalize(KIterativeSolver<ValueType>&){};

    protected:

        unsigned int fMaxIterations;

};


}

#endif /* __KIterationTerminator_H__ */
