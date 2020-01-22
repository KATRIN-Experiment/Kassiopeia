#ifndef Kassiopeia_KSTrajExactSpinDerivative_h_
#define Kassiopeia_KSTrajExactSpinDerivative_h_

#include "KSMathArray.h"

#include "KThreeVector.hh"
using KGeoBag::KThreeVector;

namespace Kassiopeia
{

    class KSTrajExactSpinDerivative :
        public KSMathArray< 12 >
    {
        public:
            KSTrajExactSpinDerivative();
            KSTrajExactSpinDerivative( const KSTrajExactSpinDerivative& anOperand );
            virtual ~KSTrajExactSpinDerivative();

            //**********
            //assignment
            //**********

        public:
            KSTrajExactSpinDerivative& operator=( const double& anOperand );

            KSTrajExactSpinDerivative& operator=( const KSMathArray< 12 >& anOperand );

            template< class XLeft, class XOperation, class XRight >
            KSTrajExactSpinDerivative& operator=( const KSMathExpression< XLeft, XOperation, XRight >& anOperand );

            KSTrajExactSpinDerivative& operator=( const KSTrajExactSpinDerivative& anOperand );

            //*********
            //variables
            //*********

        public:
            void AddToTime( const double& aTime );
            void AddToSpeed( const double& aSpeed );
            void AddToVelocity( const KThreeVector& aVelocity );
            void AddToForce( const KThreeVector& aForce );
            // Omega0 and Omega are ds/dt's 0 and 1-3 components, respectively
            void AddToOmega0( const double& aOmega0 );
            void AddToOmega( const KThreeVector& aOmega );
    };

    inline KSTrajExactSpinDerivative& KSTrajExactSpinDerivative::operator=( const double& anOperand )
    {
        this->KSMathArray< 12 >::operator =( anOperand );
        return *this;
    }

    inline KSTrajExactSpinDerivative& KSTrajExactSpinDerivative::operator=( const KSMathArray< 12 >& anOperand )
    {
        this->KSMathArray< 12 >::operator =( anOperand );
        return *this;
    }

    template< class XLeft, class XOperation, class XRight >
    inline KSTrajExactSpinDerivative& KSTrajExactSpinDerivative::operator=( const KSMathExpression< XLeft, XOperation, XRight >& anOperand )
    {
        this->KSMathArray< 12 >::operator =( anOperand );
        return *this;
    }

    inline KSTrajExactSpinDerivative& KSTrajExactSpinDerivative::operator=( const KSTrajExactSpinDerivative& anOperand )
    {
        this->KSMathArray< 12 >::operator =( anOperand );
        return *this;
    }

}

#endif
