#ifndef KRobinHoodPreconditioner_HH__
#define KRobinHoodPreconditioner_HH__

#include "KPreconditioner.hh"
#include "KSimpleVector.hh"

#include "KRobinHood.hh"
#include "KIterationTerminator.hh"

namespace KEMField
{

/*
*
*@file KRobinHoodPreconditioner.hh
*@class KRobinHoodPreconditioner
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Tue Jun  3 10:14:40 EDT 2014 J. Barrett (barrettj@mit.edu) First Version
*
*/

template <typename ValueType, template <typename> class ParallelTrait = KRobinHood_SingleThread>
class KRobinHoodPreconditioner: public KPreconditioner< ValueType >
{
    public:

        KRobinHoodPreconditioner(const KSquareMatrix<ValueType>& A, unsigned int max_iter):
            fDimension(A.Dimension()),
            fA(A),
            fMaxIterations(max_iter)
            {
                fB.resize(fDimension);
                fX.resize(fDimension);
            };

        virtual ~KRobinHoodPreconditioner(){};

        virtual void Multiply(const KVector<ValueType>& x, KVector<ValueType>& y) const
        {
            for(unsigned int i=0; i<fDimension; i++)
            {
                fX[i] = 0;
                fB[i] = x(i);
            }

            KRobinHood<ValueType, ParallelTrait> fRobinHoodEngine;
            fRobinHoodEngine.SetTolerance(1e-30);
            fRobinHoodEngine.SetResidualCheckInterval(10);
            fRobinHoodEngine.AddVisitor( new KIterationTerminator<ValueType>(fMaxIterations,20) );
//            fRobinHoodEngine.AddVisitor(new KIterationDisplay<double>());

            fRobinHoodEngine.Solve(fA,fX,fB);

            double sum = 0;
            for(unsigned int i=0; i<fDimension; i++)
            {
                if(fX(i) != fX(i)){std::cout<<"fX("<<i<<") = "<<fX(i);};
                y[i] = fX(i);
                sum += y[i];
            }

            std::cout<<"sum   = "<<sum<<std::endl;

        }

        virtual bool IsStationary(){return false;};

        virtual unsigned int Dimension() const {return fDimension;} ;

        virtual const ValueType& operator()(unsigned int i, unsigned int j) const
        {
            return fA(i,j);
        }

    protected:

        unsigned int fDimension;

        const KSquareMatrix<ValueType>& fA;

        mutable KSimpleVector<ValueType> fB;
        mutable KSimpleVector<ValueType> fX;

        unsigned int fMaxIterations;


};



}


#endif /* KRobinHoodPreconditioner_H__ */
