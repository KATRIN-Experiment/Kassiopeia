#ifndef Kassiopeia_KSTrajExactError_h_
#define Kassiopeia_KSTrajExactError_h_

#include "KSMathArray.h"

#include "KThreeVector.hh"
using KGeoBag::KThreeVector;

namespace Kassiopeia
{

    class KSTrajExactError :
        public KSMathArray< 8 >
    {
        public:
            KSTrajExactError();
            KSTrajExactError( const KSTrajExactError& anOperand );
            ~KSTrajExactError();

            //**********
            //assignment
            //**********

        public:
            KSTrajExactError& operator=( const double& anOperand );

            KSTrajExactError& operator=( const KSMathArray< 8 >& anOperand );

            template< class XLeft, class XOperation, class XRight >
            KSTrajExactError& operator=( const KSMathExpression< XLeft, XOperation, XRight >& anOperand );

            KSTrajExactError& operator=( const KSTrajExactError& anOperand );

            //*********
            //variables
            //*********

        public:
            const double& GetTimeError() const;
            const double& GetLengthError() const;
            const KThreeVector& GetPositionError() const;
            const KThreeVector& GetMomentumError() const;

        protected:
            mutable double fTimeError;
            mutable double fLengthError;
            mutable KThreeVector fPositionError;
            mutable KThreeVector fMomentumError;
    };

    inline KSTrajExactError& KSTrajExactError::operator=( const double& anOperand )
    {
        this->KSMathArray< 8 >::operator =( anOperand );
        return *this;
    }

    inline KSTrajExactError& KSTrajExactError::operator=( const KSMathArray< 8 >& anOperand )
    {
        this->KSMathArray< 8 >::operator =( anOperand );
        return *this;
    }

    template< class XLeft, class XOperation, class XRight >
    inline KSTrajExactError& KSTrajExactError::operator=( const KSMathExpression< XLeft, XOperation, XRight >& anOperand )
    {
        this->KSMathArray< 8 >::operator =( anOperand );
        return *this;
    }

    inline KSTrajExactError& KSTrajExactError::operator=( const KSTrajExactError& anOperand )
    {
        this->KSMathArray< 8 >::operator =( anOperand );
        return *this;
    }

}

#endif
