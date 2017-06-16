#ifndef Kassiopeia_KSComponentValue_h_
#define Kassiopeia_KSComponentValue_h_

#include "KSObject.h"
#include "KSNumerical.h"

#include "KTwoVector.hh"
using KGeoBag::KTwoVector;

#include "KThreeVector.hh"
using KGeoBag::KThreeVector;

#include "KTwoMatrix.hh"
using KGeoBag::KTwoMatrix;

#include "KThreeMatrix.hh"
using KGeoBag::KThreeMatrix;

namespace Kassiopeia
{
    template< class XValueType >
    class KSComponentValue
    {
        public:
            KSComponentValue( XValueType* aParentPointer ) :
                    fOperand( aParentPointer ),
                    fValue( KSNumerical< XValueType >::Zero() )
            {
            }
            KSComponentValue( const KSComponentValue< XValueType >& aCopy ) :
                    fOperand( aCopy.fOperand ),
                    fValue( aCopy.fValue )
            {
            }
            virtual ~KSComponentValue()
            {
            }

        public:
            XValueType* operator&()
            {
                return &fValue;
            }

        protected:
            XValueType* fOperand;
            XValueType fValue;
    };

    //

    template< class XValueType >
    class KSComponentValueMaximum :
        public KSComponentValue< XValueType >
    {
        public:
            KSComponentValueMaximum( XValueType* aParentPointer ) :
                    KSComponentValue< XValueType >( aParentPointer )
            {
            }
            KSComponentValueMaximum( const KSComponentValueMaximum< XValueType >& aCopy ) :
                    KSComponentValue< XValueType >( aCopy )
            {
            }
            virtual ~KSComponentValueMaximum()
            {
            }

        public:
            void Reset();
            bool Update();
    };

    template< class XValueType >
    inline void KSComponentValueMaximum< XValueType >::Reset()
    {
        this->fValue = KSNumerical< XValueType >::Lowest();
        return;
    }
    // initialize non-scalar types with zero because the magnitude will always be positive
    template<  >
    inline void KSComponentValueMaximum< KTwoVector >::Reset()
    {
        this->fValue = KSNumerical< KTwoVector >::Zero();
        return;
    }
    template<  >
    inline void KSComponentValueMaximum< KThreeVector >::Reset()
    {
        this->fValue = KSNumerical< KThreeVector >::Zero();
        return;
    }
    template<  >
    inline void KSComponentValueMaximum< KTwoMatrix >::Reset()
    {
        this->fValue = KSNumerical< KTwoMatrix >::Zero();
        return;
    }
    template<  >
    inline void KSComponentValueMaximum< KThreeMatrix >::Reset()
    {
        this->fValue = KSNumerical< KThreeMatrix >::Zero();
        return;
    }

    template< class XValueType >
    inline bool KSComponentValueMaximum< XValueType >::Update()
    {
        if( this->fValue < *(this->fOperand) )
        {
            this->fValue = *(this->fOperand);
            return true;
        }
        return false;
    }
    template<  >
    inline bool KSComponentValueMaximum< KTwoVector >::Update()
    {
        if( this->fValue.Magnitude() < this->fOperand->Magnitude() )
        {
            this->fValue = *(this->fOperand);
            return true;

        }
        return false;
    }
    template<  >
    inline bool KSComponentValueMaximum< KThreeVector >::Update()
    {
        if( this->fValue.Magnitude() < this->fOperand->Magnitude() )
        {
            this->fValue = *(this->fOperand);
            return true;
        }
        return false;
    }
    // TODO: how to compare Matrices?
    /*
    inline bool KSComponentValueMaximum< KTwoMatrix >::Update()
    {
        if( this->fValue.Determinant() < this->fOperand->Determinant() )
        {
            this->fValue = *(this->fOperand);
            return true;
        }
        return false;
    }
    template<  >
    inline bool KSComponentValueMaximum< KThreeMatrix >::Update()
    {
        if( this->fValue.Determinant() < this->fOperand->Determinant() )
        {
            this->fValue = *(this->fOperand);
            return true;
        }
        return false;
    }
    */

    //

    template< class XValueType >
    class KSComponentValueMinimum :
        public KSComponentValue< XValueType >
    {
        public:
            KSComponentValueMinimum( XValueType* aParentPointer ) :
                    KSComponentValue< XValueType >( aParentPointer )
            {
            }
            KSComponentValueMinimum( const KSComponentValueMinimum< XValueType >& aCopy ) :
                    KSComponentValue< XValueType >( aCopy )
            {
            }
            virtual ~KSComponentValueMinimum()
            {
            }

        public:
            void Reset();
            bool Update();
    };

    template< class XValueType >
    inline void KSComponentValueMinimum< XValueType >::Reset()
    {
        this->fValue = KSNumerical< XValueType >::Maximum();
        return;
    }

    template< class XValueType >
    inline bool KSComponentValueMinimum< XValueType >::Update()
    {
        if( this->fValue > *(this->fOperand) )
        {
            this->fValue = *(this->fOperand);
            return true;
        }
        return false;
    }
    template<  >
    inline bool KSComponentValueMinimum< KTwoVector >::Update()
    {
        if( this->fValue.Magnitude() > this->fOperand->Magnitude() )
        {
            this->fValue = *(this->fOperand);
            return true;
        }
        return false;
    }
    template<  >
    inline bool KSComponentValueMinimum< KThreeVector >::Update()
    {
        if( this->fValue.Magnitude() > this->fOperand->Magnitude() )
        {
            this->fValue = *(this->fOperand);
            return true;
        }
        return false;
    }
    // TODO: how to compare Matrices?
    /*
    inline bool KSComponentValueMinimum< KTwoMatrix >::Update()
    {
        if( this->fValue.Determinant() < this->fOperand->Determinant() )
        {
            this->fValue = *(this->fOperand);
            return true;
        }
        return false;
    }
    template<  >
    inline bool KSComponentValueMinimum< KThreeMatrix >::Update()
    {
        if( this->fValue.Determinant() < this->fOperand->Determinant() )
        {
            this->fValue = *(this->fOperand);
            return true;
        }
        return false;
    }
    */

    //

    template< class XValueType >
    class KSComponentValueIntegral :
        public KSComponentValue< XValueType >
    {
        public:
            KSComponentValueIntegral( XValueType* aParentPointer ) :
                    KSComponentValue< XValueType >( aParentPointer )
            {
            }
            KSComponentValueIntegral( const KSComponentValueIntegral< XValueType >& aCopy ) :
                    KSComponentValue< XValueType >( aCopy )
            {
            }
            virtual ~KSComponentValueIntegral()
            {
            }

        public:
            void Reset();
            bool Update();
    };

    template< class XValueType >
    inline void KSComponentValueIntegral< XValueType >::Reset()
    {
        this->fValue = KSNumerical< XValueType >::Zero();
        return;
    }

    template< class XValueType >
    inline bool KSComponentValueIntegral< XValueType >::Update()
    {
        this->fValue = (this->fValue) + (*(this->fOperand));
        return true;
    }

    //

    template< class XValueType >
    class KSComponentValueDelta :
        public KSComponentValue< XValueType >
    {
        public:
            KSComponentValueDelta( XValueType* aParentPointer ) :
                    KSComponentValue< XValueType >( aParentPointer ),
                    fLastValue( KSNumerical< XValueType >::Zero() )
            {
            }
            KSComponentValueDelta( const KSComponentValueDelta< XValueType >& aCopy ) :
                    KSComponentValue< XValueType >( aCopy ),
                    fLastValue( aCopy.fLastValue )
            {
            }
            virtual ~KSComponentValueDelta()
            {
            }

        public:
            void Reset();
            bool Update();

        protected:
            XValueType fLastValue;

    };

    template< class XValueType >
    inline void KSComponentValueDelta< XValueType >::Reset()
    {
        this->fValue = KSNumerical< XValueType >::Zero();
        this->fLastValue = KSNumerical< XValueType >::Zero();
        return;
    }

    template< class XValueType >
    inline bool KSComponentValueDelta< XValueType >::Update()
    {
        this->fValue = (*(this->fOperand)) - (this->fLastValue);
        this->fLastValue = (*(this->fOperand));
        return true;
    }

}

#endif
