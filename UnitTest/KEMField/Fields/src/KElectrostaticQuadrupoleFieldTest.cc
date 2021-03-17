/*
 * KElectrostaticConstantFieldTest.cc
 *
 *  Created on: 22 Oct 2020
 *      Author: jbehrens
 */

#include "KElectricQuadrupoleField.hh"

#include "KEMFieldTest.hh"

using namespace KEMField;

TEST_F(KEMFieldTest, KElectricQuadrupoleField_Potential)
{
    KElectricQuadrupoleField field;
    field.SetLocation(KPosition(0, 0, 0));
    field.SetStrength(-100.);
    field.SetLength(1);
    field.SetRadius(1);
    ASSERT_DOUBLE_EQ(-50/3., field.Potential(KPosition(0, 0, 0.5)));
    ASSERT_DOUBLE_EQ(-50/3., field.Potential(KPosition(0, 0, -0.5)));
    ASSERT_DOUBLE_EQ(200/3., field.ElectricField(KPosition(0, 0, 0.5)).Magnitude());
    ASSERT_DOUBLE_EQ(200/3., field.ElectricField(KPosition(0, 0, -0.5)).Magnitude());
    ASSERT_DOUBLE_EQ(100/3., field.ElectricField(KPosition(0.5, 0, 0)).Magnitude());
    ASSERT_DOUBLE_EQ(100/3., field.ElectricField(KPosition(-0.5, 0, 0)).Magnitude());
    ASSERT_DOUBLE_EQ(100/3., field.ElectricField(KPosition(0, 0.5, 0)).Magnitude());
    ASSERT_DOUBLE_EQ(100/3., field.ElectricField(KPosition(0, -0.5, 0)).Magnitude());
}

TEST_F(KEMFieldTest, KElectricQuadrupoleField_Initialize)
{
    KElectricQuadrupoleField field;
    field.Initialize();
}
