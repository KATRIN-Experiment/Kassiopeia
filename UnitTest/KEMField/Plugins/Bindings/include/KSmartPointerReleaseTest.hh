/*
 * KSmartPointerReleaseTest.hh
 *
 *  Created on: 3 Aug 2015
 *      Author: wolfgang
 */

#ifndef UNITTEST_KEMFIELD_INCLUDE_KSMARTPOINTERRELEASETEST_HH_
#define UNITTEST_KEMFIELD_INCLUDE_KSMARTPOINTERRELEASETEST_HH_

#include "KEMFieldTest.hh"
#include "KSmartPointerRelease.hh"
#include "SimpleClassHierachy.h"

class KSmartPointerReleaseTest : public KEMFieldTest
{
  protected:
    katrin::KContainer fContainer;
    KEMField::KSmartPointer<katrin::A> fPtr;

    std::string GetExceptionText() const;

    virtual void SetUp();
};


#endif /* UNITTEST_KEMFIELD_INCLUDE_KSMARTPOINTERRELEASETEST_HH_ */
