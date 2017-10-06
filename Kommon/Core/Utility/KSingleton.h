#ifndef KSINGLETON_H_
#define KSINGLETON_H_

namespace katrin
{

template<class XType>
class KSingleton
{
public:
    static XType& GetInstance();
    static bool IsInitialized();

    KSingleton(KSingleton const&) = delete;             // Copy construct
    KSingleton(KSingleton&&) = delete;                  // Move construct
    KSingleton& operator=(KSingleton const&) = delete;  // Copy assign
    KSingleton& operator=(KSingleton &&) = delete;      // Move assign

protected:
    KSingleton();
    virtual ~KSingleton();

private:
    static bool sInitialized;
};

template<class XType>
XType& KSingleton<XType>::GetInstance()
{
    static XType tInstance;
    return tInstance;
}

template<class XType>
KSingleton<XType>::KSingleton()
{
    sInitialized = true;
}

template<class XType>
KSingleton<XType>::~KSingleton()
{
    sInitialized = false;
}

template<class XType>
bool KSingleton<XType>::IsInitialized()
{
    return sInitialized;
}

template<class XType>
bool KSingleton<XType>::sInitialized = false;

}

#endif
