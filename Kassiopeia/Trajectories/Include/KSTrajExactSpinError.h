#ifndef Kassiopeia_KSTrajExactSpinError_h_
#define Kassiopeia_KSTrajExactSpinError_h_

#include "KSMathArray.h"

#include "KThreeVector.hh"
using KGeoBag::KThreeVector;

namespace Kassiopeia
{

    class KSTrajExactSpinError :
        public KSMathArray< 12 >
    {
        public:
            KSTrajExactSpinError();
            KSTrajExactSpinError( const KSTrajExactSpinError& anOperand );
            ~KSTrajExactSpinError();

            //**********
            //assignment
            //**********

        public:
            KSTrajExactSpinError& operator=( const double& anOperand );

            KSTrajExactSpinError& operator=( const KSMathArray< 12 >& anOperand );

            template< class XLeft, class XOperation, class XRight >
            KSTrajExactSpinError& operator=( const KSMathExpression< XLeft, XOperation, XRight >& anOperand );

            KSTrajExactSpinError& operator=( const KSTrajExactSpinError& anOperand );

            //*********
            //variables
            //*********

        public:
            const double& GetTimeError() const;
            const double& GetLengthError() const;
            const KThreeVector& GetPositionError() const;
            const KThreeVector& GetMomentumError() const;
            const double& GetSpin0Error() const;
            const KThreeVector& GetSpinError() const;

        protected:
            mutable double fTimeError;
            mutable double fLengthError;
            mutable KThreeVector fPositionError;
            mutable KThreeVector fMomentumError;
            mutable double fSpin0Error;
            mutable KThreeVector fSpinError;
    };

    inline KSTrajExactSpinError& KSTrajExactSpinError::operator=( const double& anOperand )
    {
        this->KSMathArray< 12 >::operator =( anOperand );
        return *this;
    }

    inline KSTrajExactSpinError& KSTrajExactSpinError::operator=( const KSMathArray< 12 >& anOperand )
    {
        this->KSMathArray< 12 >::operator =( anOperand );
        return *this;
    }

    template< class XLeft, class XOperation, class XRight >
    inline KSTrajExactSpinError& KSTrajExactSpinError::operator=( const KSMathExpression< XLeft, XOperation, XRight >& anOperand )
    {
        this->KSMathArray< 12 >::operator =( anOperand );
        return *this;
    }

    inline KSTrajExactSpinError& KSTrajExactSpinError::operator=( const KSTrajExactSpinError& anOperand )
    {
        this->KSMathArray< 12 >::operator =( anOperand );
        return *this;
    }

}

#endif
