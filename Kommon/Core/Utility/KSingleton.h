#ifndef KSINGLETON_H_
#define KSINGLETON_H_

namespace katrin
{

template<class XType>
class KSingleton
{
public:
    static XType* GetInstance();
    static void CreateInstance();
    static void DeleteInstance();

private:
    static XType* fInstance;

protected:
    KSingleton() { }
    virtual ~KSingleton() { }
};


template<class XType>
XType* KSingleton<XType>::fInstance(0);

template<class XType>
XType* KSingleton<XType>::GetInstance()
{
    if (fInstance == 0) {
        CreateInstance();
    }
    return fInstance;
}

template<class XType>
void KSingleton<XType>::CreateInstance()
{
    if (!fInstance) {
        fInstance = new XType();
    }
}

template<class XType>
void KSingleton<XType>::DeleteInstance()
{
    if (fInstance != 0) {
        delete fInstance;
        fInstance = 0;
    }
}

template<class XType>
class KSingletonAsReference
{
public:
    static XType& GetInstance();
    static void CreateInstance();
    static void DeleteInstance();

private:
    static XType* fInstance;

protected:
    KSingletonAsReference() { }
    virtual ~KSingletonAsReference() { }
};

template<class XType>
XType* KSingletonAsReference<XType>::fInstance(0);

template<class XType>
XType& KSingletonAsReference<XType>::GetInstance()
{
    if (fInstance == 0) {
        CreateInstance();
    }
    return *fInstance;
}

template<class XType>
void KSingletonAsReference<XType>::CreateInstance()
{
    if (!fInstance) {
        fInstance = new XType();
    }
}

template<class XType>
void KSingletonAsReference<XType>::DeleteInstance()
{
    if (fInstance != 0) {
        delete fInstance;
        fInstance = 0;
    }
}


}

#endif
