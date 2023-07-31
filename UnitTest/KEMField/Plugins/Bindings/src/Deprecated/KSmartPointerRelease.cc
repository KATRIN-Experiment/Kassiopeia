/*
 * KSmartPointerRelease.cc
 *
 *  Created on: 3 Aug 2015
 *      Author: wolfgang
 */

#include "KSmartPointerRelease.hh"

#include "KSmartPointerReleaseTest.hh"
#include "SimpleClassHierachy.h"

using namespace katrin;
using namespace KEMField;

TEST_F(KSmartPointerReleaseTest, Constructor)
{
    ASSERT_FALSE(fPtr.Null());
}

TEST_F(KSmartPointerReleaseTest, ExceptionOnDoubleRelease)
{
    try {
        KSmartPointer<A> ptr2 = ReleaseToSmartPtr<A>(&fContainer);
    }
    catch (const KException& exception) {
        ASSERT_EQ(exception.what(), GetExceptionText());
    }
}

TEST_F(KSmartPointerReleaseTest, WrongTypeRelease)
{
    try {
        fContainer.Set<A>(new A());
        KSmartPointer<B> wrongTypePointer = ReleaseToSmartPtr<B>(&fContainer);
    }
    catch (const KException& exception) {
        ASSERT_EQ(exception.what(), GetExceptionText());
    }
}
