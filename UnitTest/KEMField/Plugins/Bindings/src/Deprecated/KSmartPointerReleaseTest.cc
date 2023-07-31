/*
 * KSmartPointerReleaseTest.cc
 *
 *  Created on: 3 Aug 2015
 *      Author: wolfgang
 */

#include "KSmartPointerReleaseTest.hh"

std::string KSmartPointerReleaseTest::GetExceptionText() const
{
    return "Release to KSmartPointer failed due to empty KContainer or incompatible type.";
}

void KSmartPointerReleaseTest::SetUp()
{
    KEMFieldTest::SetUp();
    fContainer.Set(new katrin::A());
    fPtr = katrin::ReleaseToSmartPtr<katrin::A>(&fContainer);
}
