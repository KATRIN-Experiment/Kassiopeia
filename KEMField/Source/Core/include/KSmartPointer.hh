#ifndef KEMSMARTPOINTER_DEF
#define KEMSMARTPOINTER_DEF

#include <cstddef>

namespace KEMField
{

class KReferenceCounter
{
  public:
    void AddRef()
    {
        fCount++;
    }
    int Release()
    {
        return --fCount;
    }

  private:
    int fCount;
};

/**
* @struct KSmartPointer
*
* @brief KEMField's smart pointer implementation.
*
* @author T.J. Corona
*/

template<typename T> class KSmartPointer
{
  public:
    template<typename U> friend class KSmartPointer;  // for copy constructor from derived class

    KSmartPointer() : fpData(nullptr), fRef(nullptr)
    {
        fRef = new KReferenceCounter();
        fRef->AddRef();
    }
    KSmartPointer(T* pValue, bool persistent = false) : fpData(pValue), fRef(nullptr)
    {
        fRef = new KReferenceCounter();
        fRef->AddRef();
        if (persistent)
            fRef->AddRef();
    }

    /** The non template copy constructor is necessary to avoid the use
     * of the default copy constructor that would not increment the counter.
     * Sadly, the template copy constructor, see below, comes after the default copy
     * constructor in the overloading hierachy.
     */
    KSmartPointer(const KSmartPointer<T>& sp) : fpData(sp.fpData), fRef(sp.fRef)
    {
        fRef->AddRef();
    }

    template<typename U> KSmartPointer(const KSmartPointer<U>& sp) : fpData(sp.fpData), fRef(sp.fRef)
    {
        fRef->AddRef();
    }

    ~KSmartPointer()
    {
        if (fRef->Release() == 0) {
            delete fpData;
            delete fRef;
        }
    }

    T& operator*() const
    {
        return *fpData;
    }

    T* operator->() const
    {
        return fpData;
    }

    bool Is() const
    {
        return fpData != nullptr;
    }

    bool Null() const
    {
        return fpData == nullptr;
    }

    KSmartPointer<T>& operator=(const KSmartPointer<T>& sp)
    {
        if (this != &sp) {
            if (fRef->Release() == 0) {
                delete fpData;
                delete fRef;
            }

            fpData = sp.fpData;
            fRef = sp.fRef;
            fRef->AddRef();
        }
        return *this;
    }

  private:
    T* fpData;
    KReferenceCounter* fRef;
};

template<typename T> bool operator!(const KSmartPointer<T>& pointer)
{
    return pointer.Null();
}
}  // namespace KEMField

#endif /* KEMSMARTPOINTER_DEF */
