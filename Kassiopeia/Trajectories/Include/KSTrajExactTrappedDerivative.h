#ifndef Kassiopeia_KSTrajExactTrappedDerivative_h_
#define Kassiopeia_KSTrajExactTrappedDerivative_h_

#include "KSMathArray.h"

#include "KThreeVector.hh"
using KGeoBag::KThreeVector;

namespace Kassiopeia
{

    class KSTrajExactTrappedDerivative :
        public KSMathArray< 8 >
    {
        public:
            KSTrajExactTrappedDerivative();
            KSTrajExactTrappedDerivative( const KSTrajExactTrappedDerivative& anOperand );
            virtual ~KSTrajExactTrappedDerivative();

            //**********
            //assignment
            //**********

        public:
            KSTrajExactTrappedDerivative& operator=( const double& anOperand );

            KSTrajExactTrappedDerivative& operator=( const KSMathArray< 8 >& anOperand );

            template< class XLeft, class XOperation, class XRight >
            KSTrajExactTrappedDerivative& operator=( const KSMathExpression< XLeft, XOperation, XRight >& anOperand );

            KSTrajExactTrappedDerivative& operator=( const KSTrajExactTrappedDerivative& anOperand );

            //*********
            //variables
            //*********

        public:
            void AddToTime( const double& aTime );
            void AddToSpeed( const double& aSpeed );
            void AddToVelocity( const KThreeVector& aVelocity );
            void AddToForce( const KThreeVector& aForce );
    };

    inline KSTrajExactTrappedDerivative& KSTrajExactTrappedDerivative::operator=( const double& anOperand )
    {
        this->KSMathArray< 8 >::operator =( anOperand );
        return *this;
    }

    inline KSTrajExactTrappedDerivative& KSTrajExactTrappedDerivative::operator=( const KSMathArray< 8 >& anOperand )
    {
        this->KSMathArray< 8 >::operator =( anOperand );
        return *this;
    }

    template< class XLeft, class XOperation, class XRight >
    inline KSTrajExactTrappedDerivative& KSTrajExactTrappedDerivative::operator=( const KSMathExpression< XLeft, XOperation, XRight >& anOperand )
    {
        this->KSMathArray< 8 >::operator =( anOperand );
        return *this;
    }

    inline KSTrajExactTrappedDerivative& KSTrajExactTrappedDerivative::operator=( const KSTrajExactTrappedDerivative& anOperand )
    {
        this->KSMathArray< 8 >::operator =( anOperand );
        return *this;
    }

}

#endif
