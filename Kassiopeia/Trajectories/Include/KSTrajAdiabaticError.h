#ifndef Kassiopeia_KSTrajAdiabaticError_h_
#define Kassiopeia_KSTrajAdiabaticError_h_

#include "KSMathArray.h"

#include "KThreeVector.hh"
using KGeoBag::KThreeVector;

namespace Kassiopeia
{

    class KSTrajAdiabaticError :
        public KSMathArray< 8 >
    {
        public:
            KSTrajAdiabaticError();
            virtual ~KSTrajAdiabaticError();

            //**********
            //assignment
            //**********

        public:
            KSTrajAdiabaticError& operator=( const double& anOperand );

            KSTrajAdiabaticError& operator=( const KSMathArray< 8 >& anOperand );

            template< class XLeft, class XOperation, class XRight >
            KSTrajAdiabaticError& operator=( const KSMathExpression< XLeft, XOperation, XRight >& anOperand );

            KSTrajAdiabaticError& operator=( const KSTrajAdiabaticError& anOperand );

            //*****************
            //dynamic variables
            //*****************

        public:
            const double& GetTimeError() const;
            const double& GetLengthError() const;
            const KThreeVector& GetGuidingCenterPositionError() const;
            const double& GetLongitudinalMomentumError() const;
            const double& GetTransverseMomentumError() const;
            const double& GetPhaseError() const;

        protected:
            mutable double fTimeError;
            mutable double fLengthError;
            mutable KThreeVector fGuidingCenterPositionError;
            mutable double fLongitudinalMomentumError;
            mutable double fTransverseMomentumError;
            mutable double fPhaseError;
    };

    inline KSTrajAdiabaticError& KSTrajAdiabaticError::operator=( const double& anOperand )
    {
        this->KSMathArray< 8 >::operator =( anOperand );
        return *this;
    }

    inline KSTrajAdiabaticError& KSTrajAdiabaticError::operator=( const KSMathArray< 8 >& anOperand )
    {
        this->KSMathArray< 8 >::operator =( anOperand );
        return *this;
    }

    template< class XLeft, class XOperation, class XRight >
    inline KSTrajAdiabaticError& KSTrajAdiabaticError::operator=( const KSMathExpression< XLeft, XOperation, XRight >& anOperand )
    {
        this->KSMathArray< 8 >::operator =( anOperand );
        return *this;
    }

    inline KSTrajAdiabaticError& KSTrajAdiabaticError::operator=( const KSTrajAdiabaticError& anOperand )
    {
        this->KSMathArray< 8 >::operator =( anOperand );
        return *this;
    }

}

#endif
