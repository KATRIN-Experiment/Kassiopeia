#ifndef KTYPECHAIN_H_
#define KTYPECHAIN_H_

#include "KTypeList.h"

namespace katrin
{

template<template<class> class XContainer, class XTypeList> class KSingleChain;


template<template<class> class XContainer, class XType, class XNextType>
class KSingleChain<XContainer, KTypeList<XType, XNextType>> :
    public XContainer<XType>,
    public KSingleChain<XContainer, XNextType>
{
  public:
    KSingleChain() : XContainer<XType>(), KSingleChain<XContainer, XNextType>() {}
    virtual ~KSingleChain() {}
};

template<template<class> class XContainer> class KSingleChain<XContainer, KTypeNull>
{
  public:
    KSingleChain() {}
    virtual ~KSingleChain() {}
};


template<template<class, class> class XContainer, class XParameter, class XTypeList> class KSingleChainSingleParameter;

template<template<class, class> class XContainer, class XParameter, class XType, class XNextType>
class KSingleChainSingleParameter<XContainer, XParameter, KTypeList<XType, XNextType>> :
    public XContainer<XParameter, XType>,
    public KSingleChainSingleParameter<XContainer, XParameter, XNextType>
{
  public:
    KSingleChainSingleParameter() :
        XContainer<XParameter, XType>(),
        KSingleChainSingleParameter<XContainer, XParameter, XNextType>()
    {}
    virtual ~KSingleChainSingleParameter() {}
};

template<template<class, class> class XContainer, class XParameter, class XType>
class KSingleChainSingleParameter<XContainer, XParameter, KTypeList<XType, KTypeNull>> :
    public XContainer<XParameter, XType>
{
  public:
    KSingleChainSingleParameter() : XContainer<XParameter, XType>() {}
    virtual ~KSingleChainSingleParameter() {}
};


template<template<class, class> class XContainer, class XParameter, class XTypeList> class KSingleChainDoubleParameter;

template<template<class, class> class XContainer, class XParameter, class XType, class XNextType>
class KSingleChainDoubleParameter<XContainer, XParameter, KTypeList<XType, XNextType>> :
    public XContainer<XParameter, XType>,
    public KSingleChainDoubleParameter<XContainer, XParameter, XNextType>
{
  public:
    KSingleChainDoubleParameter() :
        XContainer<XParameter, XType>(),
        KSingleChainDoubleParameter<XContainer, XParameter, XNextType>()
    {}
    virtual ~KSingleChainDoubleParameter() {}
};

template<template<class, class> class XContainer, class XParameter, class XType>
class KSingleChainDoubleParameter<XContainer, XParameter, KTypeList<XType, KTypeNull>> :
    public XContainer<XParameter, XType>
{
  public:
    KSingleChainDoubleParameter() : XContainer<XParameter, XType>() {}
    virtual ~KSingleChainDoubleParameter() {}
};

}  // namespace katrin

#endif
