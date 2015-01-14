#ifndef KSARRAY_H_
#define KSARRAY_H_

namespace Kassiopeia
{
    template< size_t XDimension >
    class KSArray
    {
        public:
            KSArray()
            {
            }
            virtual ~KSArray()
            {
            }

            double operator[]( const size_t& anIndex ) const
            {
                return fData[anIndex];
            }
            double& operator[]( const size_t& anIndex )
            {
                return fData[anIndex];
            }

            KSArray< XDimension >& operator=( const double& anOperand )
            {
                for( size_t Index = 0; Index < XDimension; Index++ )
                {
                    fData[Index] = anOperand;
                }
                return *this;
            }
            template< class XType >
            KSArray< XDimension >& operator=( const XType& anOperand )
            {
                for( size_t Index = 0; Index < XDimension; Index++ )
                {
                    fData[Index] = anOperand[Index];
                }
                return *this;
            }

            template< class XType >
            KSArray< XDimension >& operator+=( const XType& anOperand )
            {
                for( size_t Index = 0; Index < XDimension; Index++ )
                {
                    fData[Index] += anOperand[Index];
                }
                return *this;
            }
            template< class XType >
            KSArray< XDimension >& operator-=( const XType& anOperand )
            {
                for( size_t Index = 0; Index < XDimension; Index++ )
                {
                    fData[Index] -= anOperand[Index];
                }
                return *this;
            }
            KSArray< XDimension >& operator*=( const double& aFactor )
            {
                for( size_t Index = 0; Index < XDimension; Index++ )
                {
                    fData[Index] *= aFactor;
                }
                return *this;
            }
            KSArray< XDimension >& operator/=( const double& aFactor )
            {
                for( size_t Index = 0; Index < XDimension; Index++ )
                {
                    fData[Index] /= aFactor;
                }
                return *this;
            }

        private:
            double fData[XDimension];
    };

}

#endif
