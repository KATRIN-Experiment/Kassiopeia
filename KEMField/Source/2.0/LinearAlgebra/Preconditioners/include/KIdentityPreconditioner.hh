#ifndef KIdentityPreconditioner_HH__
#define KIdentityPreconditioner_HH__

#include "KPreconditioner.hh"

namespace KEMField
{

/*
*
*@file KIdentityPreconditioner.hh
*@class KIdentityPreconditioner
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Tue Jun  3 10:10:50 EDT 2014 J. Barrett (barrettj@mit.edu) First Version
*
*/

template < typename ValueType >
class KIdentityPreconditioner: public KPreconditioner< ValueType >
{
    public:
        KIdentityPreconditioner(unsigned int dimension):
            fDimension(dimension),
            fOne(1),
            fZero(0){};

        virtual ~KIdentityPreconditioner(){};

        virtual void Multiply(const KVector<ValueType>& x, KVector<ValueType>& y) const
        {
            //copy x into y
            for(unsigned int i=0; i<fDimension; i++)
            {
                y[i] = x(i);
            }
        }

        virtual void MultiplyTranspose(const KVector<ValueType>& x, KVector<ValueType>& y) const
        {
            //copy x into y
            for(unsigned int i=0; i<fDimension; i++)
            {
                y[i] = x(i);
            }
        }

        virtual bool IsStationary(){return true;};

        virtual unsigned int Dimension() const {return fDimension;} ;

        virtual const ValueType& operator()(unsigned int i, unsigned int j) const
        {
            if(i == j){return fOne;};
            return fZero;
        }

    private:

        unsigned int fDimension;
        ValueType fOne;
        ValueType fZero;


};


}

#endif /* KIdentityPreconditioner_H__ */
