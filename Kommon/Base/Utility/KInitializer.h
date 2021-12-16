#ifndef KINITIALIZER_H_
#define KINITIALIZER_H_

#include <cstddef>
#include <iostream>

namespace katrin
{

template<class XType> class KInitializer
{
  public:
    KInitializer();
    ~KInitializer();

  private:
    static XType* fInstance;
    static int fCount;

  public:
    static char fData[sizeof(XType)];
};

template<class XType> char KInitializer<XType>::fData[sizeof(XType)] = {};

template<class XType> int KInitializer<XType>::fCount = 0;

template<class XType> XType* KInitializer<XType>::fInstance = nullptr;

template<class XType> KInitializer<XType>::KInitializer()
{
    if (0 == fCount++) {
        fInstance = new (&fData) XType();
    }
}
template<class XType> KInitializer<XType>::~KInitializer()
{
    if (0 == --fCount) {
        fInstance->~XType();
    }
}

}  // namespace katrin

#endif
