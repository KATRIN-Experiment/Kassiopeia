#ifndef Kassiopeia_KSTrajAdiabaticDerivative_h_
#define Kassiopeia_KSTrajAdiabaticDerivative_h_

#include "KSMathArray.h"

#include "KThreeVector.hh"
using KGeoBag::KThreeVector;

namespace Kassiopeia
{

    class KSTrajAdiabaticDerivative :
        public KSMathArray< 8 >
    {
        public:
            KSTrajAdiabaticDerivative();
            virtual ~KSTrajAdiabaticDerivative();

            //**********
            //assignment
            //**********

        public:
            KSTrajAdiabaticDerivative& operator=( const double& anOperand );

            KSTrajAdiabaticDerivative& operator=( const KSMathArray< 8 >& anOperand );

            template< class XLeft, class XOperation, class XRight >
            KSTrajAdiabaticDerivative& operator=( const KSMathExpression< XLeft, XOperation, XRight >& anOperand );

            KSTrajAdiabaticDerivative& operator=( const KSTrajAdiabaticDerivative& anOperand );

            //*****************
            //dynamic variables
            //*****************

        public:
            void AddToTime( const double& aTime );
            void AddToSpeed( const double& aSpeed );
            void AddToGuidingCenterVelocity( const KThreeVector& aVelocity );
            void AddToLongitudinalForce( const double& aForce );
            void AddToTransverseForce( const double& aForce );
            void AddToPhaseVelocity( const double& aPhaseVelocity );
    };

    inline KSTrajAdiabaticDerivative& KSTrajAdiabaticDerivative::operator=( const double& anOperand )
    {
        this->KSMathArray< 8 >::operator =( anOperand );
        return *this;
    }

    inline KSTrajAdiabaticDerivative& KSTrajAdiabaticDerivative::operator=( const KSMathArray< 8 >& anOperand )
    {
        this->KSMathArray< 8 >::operator =( anOperand );
        return *this;
    }

    template< class XLeft, class XOperation, class XRight >
    inline KSTrajAdiabaticDerivative& KSTrajAdiabaticDerivative::operator=( const KSMathExpression< XLeft, XOperation, XRight >& anOperand )
    {
        this->KSMathArray< 8 >::operator =( anOperand );
        return *this;
    }

    inline KSTrajAdiabaticDerivative& KSTrajAdiabaticDerivative::operator=( const KSTrajAdiabaticDerivative& anOperand )
    {
        this->KSMathArray< 8 >::operator =( anOperand );
        return *this;
    }

}

#endif
