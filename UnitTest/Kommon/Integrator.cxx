/**
 * Unit testing for numerical integrator classes.
 * @author M. Kleesiek
 *
 * See doctest documentation for more info:
 *   https://github.com/doctest/doctest
 */

#include "KConst.h"
#include "KMathIntegrator.h"

#include <math.h>

#include "UnitTest.h"

using namespace katrin;
using namespace std;

TEST(KommonMath, Integrator)
{
    const double analytical = 0.5 * (erf(3.0 / sqrt(2)) - erf(-3.0 / sqrt(2)));  // range [-3; 3]
    const double analyticalOE = 0.5 * (1. - erf(-3.0 / sqrt(2)));  // range [-3; inf)
    double numerical, precision;

    KMathIntegrator<double> integrator;
    auto integrand = [](double x) {
        return exp(-0.5 * x * x) / sqrt(2 * KConst::Pi());
    };

    // Simpson (default)
    integrator.SetMethod(KEMathIntegrationMethod::Simpson);

    precision = 1E-5;
    integrator.SetPrecision(precision);
    numerical = integrator.Integrate(integrand, -3.0, 3.0);

    EXPECT_EQ(65U, integrator.NumberOfSteps());
    ASSERT_NEAR(analytical, numerical, precision);

    numerical = integrator.Integrate(integrand, -3.0, 9999.);

    EXPECT_EQ(65537U, integrator.NumberOfSteps());
    ASSERT_NEAR(analyticalOE, numerical, precision);

    // Romberg
    integrator.SetMethod(KEMathIntegrationMethod::Romberg);

    precision = 1E-5;
    integrator.SetPrecision(precision);
    numerical = integrator.Integrate(integrand, -3.0, 3.0);

    EXPECT_EQ(33U, integrator.NumberOfSteps());
    ASSERT_NEAR(analytical, numerical, precision);

    precision = 1E-9;
    integrator.SetPrecision(precision);
    numerical = integrator.Integrate(integrand, -3.0, 3.0);

    EXPECT_EQ(65U, integrator.NumberOfSteps());
    ASSERT_NEAR(analytical, numerical, precision);

    precision = 1E-6;
    integrator.SetPrecision(precision);
    numerical = integrator.Integrate(integrand, -3.0, 9999.);

    EXPECT_EQ(65537U, integrator.NumberOfSteps());
    ASSERT_NEAR(analyticalOE, numerical, precision);

    // Trapezoidal
    integrator.SetMethod(KEMathIntegrationMethod::Trapezoidal);

    precision = 1E-9;
    integrator.SetPrecision(precision);
    numerical = integrator.Integrate(integrand, -3.0, 3.0);

    EXPECT_EQ(16385U, integrator.NumberOfSteps());
    ASSERT_NEAR(analytical, numerical, precision);

    precision = 1E-4;
    integrator.SetPrecision(precision);
    numerical = integrator.Integrate(integrand, -3.0, 9999.);

    EXPECT_EQ(65537U, integrator.NumberOfSteps());
    ASSERT_NEAR(analyticalOE, numerical, precision);

    integrator.SetMinSteps(32768);
    EXPECT_EQ(32769U, integrator.GetMinSteps());

    numerical = integrator.Integrate(integrand, -3.0, 3.0);

    EXPECT_EQ(32769U, integrator.NumberOfSteps());
    ASSERT_NEAR(analytical, numerical, precision);

#ifdef KASPER_USE_GSL  // need GSL support in Kommon
    //QAGIU (always used, if upper boundary=INFINITY)
    integrator.SetMethod(KEMathIntegrationMethod::QAGS);

    precision = 1E-6;
    integrator.SetPrecision(precision);
    numerical = integrator.Integrate(integrand, -3.0, INFINITY); // uses QAGIU

    EXPECT_EQ(5U, integrator.NumberOfSteps());
    ASSERT_NEAR(analyticalOE, numerical, precision);

    precision = 1E-9;
    integrator.SetPrecision(precision);
    numerical = integrator.Integrate(integrand, -3.0, 3.0); // uses QAGS

    EXPECT_EQ(2U, integrator.NumberOfSteps());
    ASSERT_NEAR(analytical, numerical, precision);
#endif
}
