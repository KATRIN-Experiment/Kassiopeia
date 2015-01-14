#ifndef KSMARTPOINTER_DEF
#define KSMARTPOINTER_DEF

#include <cstddef>

namespace katrin
{

    template< typename T >
    class KSmartPointer
    {
        private:
            class Counter
            {
                public:
                    int Increment()
                    {
                        return ++fCount;
                    }
                    int Decrement()
                    {
                        return --fCount;
                    }
                private:
                    int fCount;
            };

        public:
            KSmartPointer() :
                    fPointer( 0 ),
                    fCounter( 0 )
            {
                fCounter = new Counter();
                fCounter->Increment();
            }
            KSmartPointer( T* pValue ) :
                    fPointer( pValue ),
                    fCounter( 0 )
            {
                fCounter = new Counter();
                fCounter->Increment();
            }

            KSmartPointer( const KSmartPointer< T >& sp ) :
                    fPointer( sp.fPointer ),
                    fCounter( sp.fCounter )
            {
                fCounter->Increment();
            }

            ~KSmartPointer()
            {
                if( fCounter->Decrement() == 0 )
                {
                    delete fPointer;
                    delete fCounter;
                }
            }

            T& operator*()
            {
                return *fPointer;
            }

            const T& operator*() const
            {
                return *fPointer;
            }

            T* operator->()
            {
                return fPointer;
            }

            const T* operator->() const
            {
                return fPointer;
            }

            bool Null() const
            {
                return fPointer == NULL;
            }

            KSmartPointer< T >& operator=( const KSmartPointer< T >& sp )
            {
                if( this != &sp )
                {
                    if( fCounter->Decrement() == 0 )
                    {
                        delete fPointer;
                        delete fCounter;
                    }

                    fPointer = sp.fPointer;
                    fCounter = sp.fCounter;
                    fCounter->Increment();
                }
                return *this;
            }

            bool operator==( const KSmartPointer< T >& sp ) const
            {
                return fPointer == sp.fPointer;
            }
            bool operator<( const KSmartPointer< T >& sp ) const
            {
                return fPointer < sp.fPointer;
            }
            bool operator>( const KSmartPointer< T >& sp ) const
            {
                return fPointer > sp.fPointer;
            }

        private:
            T* fPointer;
            Counter* fCounter;
    };
}

#endif /* KSMARTPOINTER_DEF */
