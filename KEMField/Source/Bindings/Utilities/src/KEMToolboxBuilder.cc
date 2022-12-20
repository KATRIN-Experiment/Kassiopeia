/*
 * KEMToolboxBuilder.cc
 *
 *  Created on: 13.05.2015
 *      Author: gosda
 */
#include "KEMToolboxBuilder.hh"

#include "KElementProcessor.hh"
#include "KRoot.h"

#ifdef KEMFIELD_USE_MPI
#include "KMPIInterface.hh"
#endif
#ifdef KEMFIELD_USE_OPENCL
#include "KOpenCLInterface.hh"
#endif

namespace katrin
{

KEMRoot::KEMRoot()
{
#ifdef KEMFIELD_USE_MPI
    // TODO: get cmdline options from KXMLInitializer
    KEMField::KMPIInterface::GetInstance()->Initialize(nullptr, nullptr, true);
#endif
#ifdef KEMFIELD_USE_OPENCL
    KEMField::KOpenCLInterface::GetInstance()->InitializeOpenCL();
#endif
}

KEMRoot::~KEMRoot()
{
#ifdef KEMFIELD_USE_MPI
    KEMField::KMPIInterface::GetInstance()->Finalize();
#endif
}

template<> KEMToolboxBuilder::~KComplexElement() = default;

STATICINT sKEMRoot = KRootBuilder::ComplexElement<KEMRoot>("kemfield");

STATICINT sKEMRootCompat = KElementProcessor::ComplexElement<KEMRoot>("kemfield");


}  // namespace katrin
