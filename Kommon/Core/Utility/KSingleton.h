#ifndef KSINGLETON_H_
#define KSINGLETON_H_

namespace katrin
{

template<class XType>
class KSingleton
{
public:
    static XType& GetInstance();

    KSingleton(KSingleton const&) = delete;             // Copy construct
    KSingleton(KSingleton&&) = delete;                  // Move construct
    KSingleton& operator=(KSingleton const&) = delete;  // Copy assign
    KSingleton& operator=(KSingleton &&) = delete;      // Move assign

protected:
    KSingleton() { }
    virtual ~KSingleton() { }
};

template<class XType>
XType& KSingleton<XType>::GetInstance()
{
    static XType tInstance;
    return tInstance;
}

}

#endif
