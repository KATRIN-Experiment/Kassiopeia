#ifndef Kassiopeia_KSMathArray_h_
#define Kassiopeia_KSMathArray_h_

#include "KAssert.h"

namespace Kassiopeia
{
    // operations

    class KSMathAdd
    {
        public:
            static double Evaluate( double aLeftValue, double aRightValue )
            {
                return (aLeftValue + aRightValue);
            }
    };

    class KSMathSubtract
    {
        public:
            static double Evaluate( double aLeftValue, double aRightValue )
            {
                return (aLeftValue - aRightValue);
            }
    };

    class KSMathMultiply
    {
        public:
            static double Evaluate( double aLeftValue, double aRightValue )
            {
                return (aLeftValue * aRightValue);
            }
    };

    class KSMathDivide
    {
        public:
            static double Evaluate( double aLeftValue, double aRightValue )
            {
                return (aLeftValue * aRightValue);
            }
    };

    // expressions

    template< class XLeft, class XOperation, class XRight >
    class KSMathExpression
    {
        public:
            static const int sDimension = XLeft::sDimension;

        public:
            KSMathExpression( const XLeft& aLeftOperand, const XRight& aRightOperand ) :
                fLeft( aLeftOperand ),
                fRight( aRightOperand )
            {
                KSTATICASSERT( XLeft::sDimension == XRight::sDimension, dimension_mismatch_in_KMathExpression )
            }
            virtual ~KSMathExpression()
            {
            }

            double operator[]( const int& anIndex ) const
            {
                return XOperation::Evaluate( fLeft[anIndex], fRight[anIndex] );
            }

        private:
            const XLeft& fLeft;
            const XRight& fRight;
    };

    template< class XLeft, class XOperation >
    class KSMathExpression< XLeft, XOperation, double >
    {
        public:
            static const int sDimension = XLeft::sDimension;

        public:
            KSMathExpression( const XLeft& aLeftOperand, const double& aRightNumber ) :
                fLeft( aLeftOperand ),
                fRight( aRightNumber )
            {
            }
            virtual ~KSMathExpression()
            {
            }

            double operator[]( const int& anIndex ) const
            {
                return XOperation::Evaluate( fLeft[anIndex], fRight );
            }

        private:
            const XLeft& fLeft;
            const double& fRight;
    };

    template< class XOperation, class XRight >
    class KSMathExpression< double, XOperation, XRight >
    {
        public:
            static const int sDimension = XRight::sDimension;

        public:
            KSMathExpression( const double& aLeftOperand, const XRight& aRightNumber ) :
                fLeft( aLeftOperand ),
                fRight( aRightNumber )
            {
            }
            virtual ~KSMathExpression()
            {
            }

            double operator[]( const int& anIndex ) const
            {
                return XOperation::Evaluate( fLeft, fRight[anIndex] );
            }

        private:
            const double& fLeft;
            const XRight& fRight;
    };

    // array

    template< int XDimension >
    class KSMathArray
    {
        public:
            static const int sDimension = XDimension;

        public:
            KSMathArray()
            {
                (*this) = (0.);
            }
            virtual ~KSMathArray()
            {
            }

        public:
            KSMathArray& operator=( const double& anOperand )
            {
                for( int Index = 0; Index < sDimension; Index++ )
                {
                    fData[Index] = anOperand;
                }
                return *this;
            }

            KSMathArray& operator=( const KSMathArray& anOperand )
            {
                for( int Index = 0; Index < sDimension; Index++ )
                {
                    fData[Index] = anOperand[Index];
                }
                return *this;
            }

            template< class XLeft, class XOperation, class XRight >
            KSMathArray& operator=( const KSMathExpression< XLeft, XOperation, XRight >& anOperand )
            {
                for( int Index = 0; Index < sDimension; Index++ )
                {
                    fData[Index] = anOperand[Index];
                }
                return *this;
            }

            const double& operator[]( const int& anIndex ) const
            {
                return fData[anIndex];
            }
            double& operator[]( const int& anIndex )
            {
                return fData[anIndex];
            }

        protected:
            double fData[sDimension];
    };

    // operators

    template< int XSize >
    inline KSMathExpression< KSMathArray< XSize >, KSMathAdd, KSMathArray< XSize > > operator+( const KSMathArray< XSize >& aLeftOperand, const KSMathArray< XSize >& aRightOperand )
    {
        return KSMathExpression< KSMathArray< XSize >, KSMathAdd, KSMathArray< XSize > >( aLeftOperand, aRightOperand );
    }
    template< class XLeft, class XOperation, class XRight, int XSize >
    KSMathExpression< KSMathExpression< XLeft, XOperation, XRight > , KSMathAdd, KSMathArray< XSize > > operator+( const KSMathExpression< XLeft, XOperation, XRight >& aLeftOperand, const KSMathArray< XSize >& aRightOperand )
    {
        return KSMathExpression< KSMathExpression< XLeft, XOperation, XRight > , KSMathAdd, KSMathArray< XSize > >( aLeftOperand, aRightOperand );
    }
    template< int XSize, class XLeft, class XOperation, class XRight >
    KSMathExpression< KSMathArray< XSize >, KSMathAdd, KSMathExpression< XLeft, XOperation, XRight > > operator+( const KSMathArray< XSize >& aLeftOperand, const KSMathExpression< XLeft, XOperation, XRight >& aRightOperand )
    {
        return KSMathExpression< KSMathArray< XSize >, KSMathAdd, KSMathExpression< XLeft, XOperation, XRight > >( aLeftOperand, aRightOperand );
    }
    template< class XLeftLeft, class XLeftOperation, class XLeftRight, class XRightLeft, class XRightOperation, class XRightRight >
    KSMathExpression< KSMathExpression< XLeftLeft, XLeftOperation, XLeftRight > , KSMathAdd, KSMathExpression< XRightLeft, XRightOperation, XRightRight > > operator+( const KSMathExpression< XLeftLeft, XLeftOperation, XLeftRight >& aLeftOperand, const KSMathExpression< XRightLeft, XRightOperation, XRightRight >& aRightOperand )
    {
        return KSMathExpression< KSMathExpression< XLeftLeft, XLeftOperation, XLeftRight > , KSMathAdd, KSMathExpression< XRightLeft, XRightOperation, XRightRight > >( aLeftOperand, aRightOperand );
    }

    template< int XSize >
    inline KSMathExpression< KSMathArray< XSize >, KSMathSubtract, KSMathArray< XSize > > operator-( const KSMathArray< XSize >& aLeftOperand, const KSMathArray< XSize >& aRightOperand )
    {
        return KSMathExpression< KSMathArray< XSize >, KSMathSubtract, KSMathArray< XSize > >( aLeftOperand, aRightOperand );
    }
    template< class XLeft, class XOperation, class XRight, int XSize >
    KSMathExpression< KSMathExpression< XLeft, XOperation, XRight > , KSMathSubtract, KSMathArray< XSize > > operator-( const KSMathExpression< XLeft, XOperation, XRight >& aLeftOperand, const KSMathArray< XSize >& aRightOperand )
    {
        return KSMathExpression< KSMathExpression< XLeft, XOperation, XRight > , KSMathSubtract, KSMathArray< XSize > >( aLeftOperand, aRightOperand );
    }
    template< int XSize, class XLeft, class XOperation, class XRight >
    KSMathExpression< KSMathArray< XSize >, KSMathSubtract, KSMathExpression< XLeft, XOperation, XRight > > operator-( const KSMathArray< XSize >& aLeftOperand, const KSMathExpression< XLeft, XOperation, XRight >& aRightOperand )
    {
        return KSMathExpression< KSMathArray< XSize >, KSMathSubtract, KSMathExpression< XLeft, XOperation, XRight > >( aLeftOperand, aRightOperand );
    }
    template< class XLeftLeft, class XLeftOperation, class XLeftRight, class XRightLeft, class XRightOperation, class XRightRight >
    KSMathExpression< KSMathExpression< XLeftLeft, XLeftOperation, XLeftRight > , KSMathSubtract, KSMathExpression< XRightLeft, XRightOperation, XRightRight > > operator-( const KSMathExpression< XLeftLeft, XLeftOperation, XLeftRight >& aLeftOperand, const KSMathExpression< XRightLeft, XRightOperation, XRightRight >& aRightOperand )
    {
        return KSMathExpression< KSMathExpression< XLeftLeft, XLeftOperation, XLeftRight > , KSMathAdd, KSMathExpression< XRightLeft, XRightOperation, XRightRight > >( aLeftOperand, aRightOperand );
    }

    template< int XSize >
    inline KSMathExpression< KSMathArray< XSize >, KSMathMultiply, double > operator*( const KSMathArray< XSize >& aLeftOperand, const double& aRightOperand )
    {
        return KSMathExpression< KSMathArray< XSize >, KSMathMultiply, double >( aLeftOperand, aRightOperand );
    }
    template< int XSize >
    inline KSMathExpression< double, KSMathMultiply, KSMathArray< XSize > > operator*( const double& aLeftOperand, const KSMathArray< XSize >& aRightOperand )
    {
        return KSMathExpression< double, KSMathMultiply, KSMathArray< XSize > >( aLeftOperand, aRightOperand );
    }
    template< class XLeft, class XOperation, class XRight >
    KSMathExpression< KSMathExpression< XLeft, XOperation, XRight > , KSMathMultiply, double > operator*( const KSMathExpression< XLeft, XOperation, XRight >& aLeftOperand, const double& aRightOperand )
    {
        return KSMathExpression< KSMathExpression< XLeft, XOperation, XRight > , KSMathMultiply, double >( aLeftOperand, aRightOperand );
    }
    template< class XLeft, class XOperation, class XRight >
    KSMathExpression< double, KSMathMultiply, KSMathExpression< XLeft, XOperation, XRight > > operator*( const double& aLeftOperand, const KSMathExpression< XLeft, XOperation, XRight >& aRightOperand )
    {
        return KSMathExpression< double, KSMathMultiply, KSMathExpression< XLeft, XOperation, XRight > >( aLeftOperand, aRightOperand );
    }

    template< int XSize >
    inline KSMathExpression< KSMathArray< XSize >, KSMathDivide, double > operator/( const KSMathArray< XSize >& aLeftOperand, const double& aRightOperand )
    {
        return KSMathExpression< KSMathArray< XSize >, KSMathDivide, double >( aLeftOperand, aRightOperand );
    }
    template< int XSize >
    inline KSMathExpression< double, KSMathDivide, KSMathArray< XSize > > operator/( const double& aLeftOperand, const KSMathArray< XSize >& aRightOperand )
    {
        return KSMathExpression< double, KSMathDivide, KSMathArray< XSize > >( aLeftOperand, aRightOperand );
    }
    template< class XLeft, class XOperation, class XRight >
    KSMathExpression< KSMathExpression< XLeft, XOperation, XRight > , KSMathDivide, double > operator/( const KSMathExpression< XLeft, XOperation, XRight >& aLeftOperand, const double& aRightOperand )
    {
        return KSMathExpression< KSMathExpression< XLeft, XOperation, XRight > , KSMathDivide, double >( aLeftOperand, aRightOperand );
    }
    template< class XLeft, class XOperation, class XRight >
    KSMathExpression< double, KSMathDivide, KSMathExpression< XLeft, XOperation, XRight > > operator/( const double& aLeftOperand, const KSMathExpression< XLeft, XOperation, XRight >& aRightOperand )
    {
        return KSMathExpression< double, KSMathDivide, KSMathExpression< XLeft, XOperation, XRight > >( aLeftOperand, aRightOperand );
    }

}

#endif
