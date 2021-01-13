/*
 * KMagnetostaticDipoleFieldTest.cc
 *
 *  Created on: 22 Oct 2020
 *      Author: jbehrens
 */

#include "KMagneticDipoleField.hh"

#include "KEMFieldTest.hh"

using namespace KEMField;

TEST_F(KEMFieldTest, KMagneticDipoleField_Field)
{
    const double tol = 1e-14;

    KMagneticDipoleField field;
    field.SetLocation(KPosition(1.0, 0, 0));
    field.SetMoment(KDirection(0, 0, 10));
    ASSERT_NEAR(9.0509668e-07, field.MagneticField(KPosition(0, 0, 0.5)).Magnitude(), tol);
    ASSERT_NEAR(9.0509668e-07, field.MagneticField(KPosition(0, 0, -0.5)).Magnitude(), tol);
    ASSERT_NEAR(8.0e-06, field.MagneticField(KPosition(0.5, 0, 0)).Magnitude(), tol);
    ASSERT_NEAR(2.9629629e-07, field.MagneticField(KPosition(-0.5, 0, 0)).Magnitude(), tol);
    ASSERT_NEAR(7.1554175e-07, field.MagneticField(KPosition(0, 0.5, 0)).Magnitude(), tol);
    ASSERT_NEAR(7.1554175e-07, field.MagneticField(KPosition(0, -0.5, 0)).Magnitude(), tol);
    ASSERT_TRUE(std::isnan(field.MagneticField(KPosition(1.0, 0, 0)).Magnitude()));
}

TEST_F(KEMFieldTest, KMagneticDipoleField_Initialize)
{
    KMagneticDipoleField field;
    field.Initialize();
}
