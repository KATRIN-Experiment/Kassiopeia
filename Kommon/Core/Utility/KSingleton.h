#ifndef KSINGLETON_H_
#define KSINGLETON_H_

//#include <boost/thread/mutex.hpp>

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
//    static boost::mutex fMutex;

protected:
    KSingleton() { }
    virtual ~KSingleton() { }
};


template<class XType>
XType* KSingleton<XType>::fInstance(0);

//template<class XType>
//boost::mutex KSingleton<XType>::fMutex;

template<class XType>
XType* KSingleton<XType>::GetInstance()
{
//    XType* tmp = fInstance.load(boost::memory_order_consume);
    if (fInstance == 0) {
        CreateInstance();
    }
    return fInstance;
}

template<class XType>
void KSingleton<XType>::CreateInstance()
{
//    boost::mutex::scoped_lock guard(fMutex);
//    XType* tmp = fInstance.load(boost::memory_order_consume);
    if (!fInstance) {
        fInstance = new XType();
//        fInstance.store(tmp, boost::memory_order_release);
    }
}

template<class XType>
void KSingleton<XType>::DeleteInstance()
{
//    boost::mutex::scoped_lock guard(fMutex);
//    XType * tmp = fInstance.load(boost::memory_order_consume);
    if (fInstance != 0) {
        delete fInstance;
        fInstance = 0;
//        fInstance.store(0, boost::memory_order_release);
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
//    static boost::mutex fMutex;

protected:
    KSingletonAsReference() { }
    virtual ~KSingletonAsReference() { }
};

template<class XType>
XType* KSingletonAsReference<XType>::fInstance(0);

//template<class XType>
//boost::mutex KSingleton<XType>::fMutex;

template<class XType>
XType& KSingletonAsReference<XType>::GetInstance()
{
//    XType* tmp = fInstance.load(boost::memory_order_consume);
    if (fInstance == 0) {
        CreateInstance();
    }
    return *fInstance;
}

template<class XType>
void KSingletonAsReference<XType>::CreateInstance()
{
//    boost::mutex::scoped_lock guard(fMutex);
//    XType* tmp = fInstance.load(boost::memory_order_consume);
    if (!fInstance) {
        fInstance = new XType();
//        fInstance.store(tmp, boost::memory_order_release);
    }
}

template<class XType>
void KSingletonAsReference<XType>::DeleteInstance()
{
//    boost::mutex::scoped_lock guard(fMutex);
//    XType * tmp = fInstance.load(boost::memory_order_consume);
    if (fInstance != 0) {
        delete fInstance;
        fInstance = 0;
//        fInstance.store(0, boost::memory_order_release);
    }
}




}

#endif
