/*
 * KComplexElement_ObjectRetrieval.cxx
 *
 *  Created on: 27 Jun 2015
 *      Author: wolfgang
 */

#include "KComplexElement.hh"

#include "SimpleClassHierachy.h"
#include "UnitTest.h"

using namespace katrin;

TEST(KComplexElement, KComplexElement_AsReference)
{
    KComplexElement<B> element;
    element.Begin();
    A* ptr = nullptr;
    element.ReleaseTo<A>(ptr);
    EXPECT_EQ(ptr->Number(), 2);
}

TEST(KComplexElement, KComplexElement_AsPointer)
{

    KComplexElement<B> element(nullptr);
    element.Begin();
    auto* ptr = element.AsPointer<A>();
    EXPECT_EQ(ptr->Number(), 2);
    // expect to seg fault due to double deletion on uncommenting the next line
    //	delete ptr;
}
