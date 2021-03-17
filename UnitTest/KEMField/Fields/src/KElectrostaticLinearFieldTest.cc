/*
 * KElectrostaticLinearFieldTest.cc
 *
 *  Created on: 22 Oct 2020
 *      Author: jbehrens
 */

#include "KElectrostaticLinearField.hh"

#include "KEMFieldTest.hh"

using namespace KEMField;

TEST_F(KEMFieldTest, KElectrostaticLinearField_Potential)
{
    KElectrostaticLinearField field;
    field.SetZ1(0);
    field.SetZ2(1);
    field.SetPotential1(100);
    field.SetPotential2(0);
    ASSERT_DOUBLE_EQ(50, field.Potential(KPosition(0, 0, 0.5)));
    ASSERT_DOUBLE_EQ(0,  field.Potential(KPosition(0, 0, -0.5)));
    ASSERT_DOUBLE_EQ(100, field.ElectricField(KPosition(0, 0, 0.5)).Magnitude());
    ASSERT_DOUBLE_EQ(0,  field.ElectricField(KPosition(0, 0, -0.5)).Magnitude());
    ASSERT_DOUBLE_EQ(100, field.ElectricField(KPosition(0.5, 0, 0)).Magnitude());
    ASSERT_DOUBLE_EQ(100,  field.ElectricField(KPosition(-0.5, 0, 0)).Magnitude());
    ASSERT_DOUBLE_EQ(100, field.ElectricField(KPosition(0, 0.5, 0)).Magnitude());
    ASSERT_DOUBLE_EQ(100,  field.ElectricField(KPosition(0, -0.5, 0)).Magnitude());
}

TEST_F(KEMFieldTest, KElectrostaticLinearField_Initialize)
{
    KElectrostaticLinearField field;
    field.Initialize();
}
