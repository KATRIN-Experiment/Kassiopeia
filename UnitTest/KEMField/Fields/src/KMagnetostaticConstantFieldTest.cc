/*
 * KMagnetostaticConstantFieldTest.cc
 *
 *  Created on: 22 Oct 2020
 *      Author: jbehrens
 */

#include "KMagnetostaticConstantField.hh"

#include "KEMFieldTest.hh"

using namespace KEMField;

TEST_CASE_FIXTURE(KEMFieldTest, "KEMFieldTest - KMagnetostaticConstantField_Field")
{
    KMagnetostaticConstantField field;
    field.SetField(KDirection(0, 0, 10));
    ASSERT_DOUBLE_EQ(10, field.MagneticField(KPosition(0, 0, 1.e-3)).Z());
    ASSERT_DOUBLE_EQ(10, field.MagneticField(KPosition(0, 0, -1.e-3)).Z());
}

TEST_CASE_FIXTURE(KEMFieldTest, "KEMFieldTest - KMagnetostaticConstantField_Initialize")
{
    KMagnetostaticConstantField field;
    field.Initialize();
}
