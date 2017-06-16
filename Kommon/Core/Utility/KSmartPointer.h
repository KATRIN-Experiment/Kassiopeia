#ifndef KSMARTPOINTER_DEF
#define KSMARTPOINTER_DEF

#include <memory>
#include <utility>

namespace katrin
{

template< typename T >
class KSmartPointer
{
public:
    template <typename U>
    friend class KSmartPointer; // for copy constructor from derived or non const class

    KSmartPointer() :
        fSharedPtr()
    { }

    KSmartPointer( T* pValue ) :
        fSharedPtr( pValue )
    { }

    template< typename U>
    KSmartPointer( const KSmartPointer< U >& sp ) :
        fSharedPtr( sp.fSharedPtr )
    { }

    template< typename U>
    KSmartPointer( KSmartPointer< U >&& sp ) :
        fSharedPtr( std::move(sp.fSharedPtr) )
    { }

    T& operator*()
    {
        return *fSharedPtr;
    }

    const T& operator*() const
    {
        return *fSharedPtr;
    }

    T* operator->()
    {
        return fSharedPtr.get();
    }

    const T* operator->() const
    {
        return fSharedPtr.get();
    }

    bool Null() const
    {
        return !fSharedPtr;
    }

    KSmartPointer< T >& operator=( const KSmartPointer< T >& sp )
    {
        fSharedPtr = sp.fSharedPtr;
        return *this;
    }

    bool operator==( const KSmartPointer< T >& sp ) const
    {
        return fSharedPtr == sp.fSharedPtr;
    }

    bool operator<( const KSmartPointer< T >& sp ) const
    {
        return fSharedPtr < sp.fSharedPtr;
    }

    bool operator>( const KSmartPointer< T >& sp ) const
    {
        return fSharedPtr > sp.fSharedPtr;
    }

private:
    std::shared_ptr<T> fSharedPtr;
};
}

#endif /* KSMARTPOINTER_DEF */
