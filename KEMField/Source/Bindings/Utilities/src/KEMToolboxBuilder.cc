/*
 * KEMToolboxBuilder.cc
 *
 *  Created on: 13.05.2015
 *      Author: gosda
 */
#include "KEMToolboxBuilder.hh"

#include "KElementProcessor.hh"
#include "KRoot.h"

#ifdef KEMFIELD_USE_OPENCL
#include "KOpenCLInterface.hh"
#endif

namespace katrin
{

KEMRoot::KEMRoot()
{
#ifdef KEMFIELD_USE_OPENCL
    KEMField::KOpenCLInterface::GetInstance()->Initialize();
#endif
}

KEMRoot::~KEMRoot()
{
}

template<> KEMToolboxBuilder::~KComplexElement() = default;

STATICINT sKEMRoot = KRootBuilder::ComplexElement<KEMRoot>("kemfield");

STATICINT sKEMRootCompat = KElementProcessor::ComplexElement<KEMRoot>("kemfield");


}  // namespace katrin
