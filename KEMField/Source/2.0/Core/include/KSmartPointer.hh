#ifndef KEMSMARTPOINTER_DEF
#define KEMSMARTPOINTER_DEF

namespace KEMField
{

  class KReferenceCounter
  {
  public:
    void AddRef() { fCount++; }
    int Release() { return --fCount; }
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

  template <typename T> class KSmartPointer
  {
  public:
    KSmartPointer() : fpData(0), fRef(0) 
    {
      fRef = new KReferenceCounter();
      fRef->AddRef();
    }
    KSmartPointer(T* pValue,bool persistent=false) : fpData(pValue), fRef(0)
    {
      fRef = new KReferenceCounter();
      fRef->AddRef();
      if (persistent)
	fRef->AddRef();
    }

    KSmartPointer(const KSmartPointer<T>& sp) : fpData(sp.fpData), fRef(sp.fRef)
    {
      fRef->AddRef();
    }

    ~KSmartPointer()
    {
      if(fRef->Release() == 0)
      {
	delete fpData;
	delete fRef;
      }
    }

    T& operator* ()
    {
      return *fpData;
    }

    T* operator-> ()
    {
      return fpData;
    }
    
    bool Null() const { return fpData == NULL; }

    KSmartPointer<T>& operator= (const KSmartPointer<T>& sp)
    {
      if (this != &sp)
      {
	if(fRef->Release() == 0)
	{
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
}

#endif /* KEMSMARTPOINTER_DEF */
