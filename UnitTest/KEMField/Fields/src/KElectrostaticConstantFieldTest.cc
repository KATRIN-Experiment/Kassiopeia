/*
 * KElectrostaticConstantFieldTest.cc
 *
 *  Created on: 4 Aug 2015
 *      Author: wolfgang
 */

#include "KElectrostaticConstantField.hh"

#include "KEMFieldTest.hh"

using namespace KEMField;

TEST_F(KEMFieldTest, KElectrostaticConstantField_Potential)
{
    KElectrostaticConstantField field;
    field.SetField(KDirection(2, 0, -1));
    ASSERT_DOUBLE_EQ(-0.001, field.Potential(KPosition(0, 0, 1.e-3)));
    ASSERT_DOUBLE_EQ(0.001, field.Potential(KPosition(0, 0, -1.e-3)));
    ASSERT_DOUBLE_EQ(-2.001, field.Potential(KPosition(-1, 0, 1.e-3)));
    ASSERT_DOUBLE_EQ(-0.001, field.Potential(KPosition(0, -1, 1.e-3)));
    field.SetField(KDirection(1, 2, -3));
    ASSERT_DOUBLE_EQ(6, field.Potential(KPosition(1, 1, -1)));
}

TEST_F(KEMFieldTest, KELectrostaticConstantField_Initialize)
{
    KElectrostaticConstantField field;
    field.Initialize();
}
