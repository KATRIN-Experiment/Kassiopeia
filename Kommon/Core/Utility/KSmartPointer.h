#ifndef KSMARTPOINTER_DEF
#define KSMARTPOINTER_DEF

#include <cstddef>

namespace katrin
{

class Counter
{
    template < typename T>
    friend class KSmartPointer;

    Counter() :fCount(0){}

    int Increment()
    {
        return ++fCount;
    }
    int Decrement()
    {
        return --fCount;
    }

    int fCount;
};

template< typename T >
class KSmartPointer
{
public:
    template <typename U>
    friend class KSmartPointer; // for copy constructor from derived or non const class


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

    /** The non template copy constructor is necessary to avoid the use
     * of the default copy constructor that would not increment the counter.
     * Sadly, the template copy constructor below comes after the default copy
     * constructor in the overloading hierachy.
     */
    KSmartPointer( const KSmartPointer< T >& sp ) :
        fPointer( sp.fPointer ),
        fCounter( sp.fCounter )
    {
        fCounter->Increment();
    }

    template< typename U>
    KSmartPointer( const KSmartPointer< U >& sp ) :
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
