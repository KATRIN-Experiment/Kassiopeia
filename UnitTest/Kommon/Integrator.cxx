/**
 * Unit testing for numerical integrator classes.
 * @author M. Kleesiek
 *
 * See the official GoogleTest pages for more info:
 *   https://code.google.com/p/googletest/wiki/Primer
 *   https://code.google.com/p/googletest/wiki/AdvancedGuide
 */

#include "KConst.h"
#include "KMathIntegrator.h"

#include <gtest/gtest.h>

using namespace katrin;
using namespace std;

TEST(KommonMath, Integrator)
{
    const double analytical = 0.5 * (erf(3.0 / sqrt(2)) - erf(-3.0 / sqrt(2)));
    double numerical, precision;

    KMathIntegrator<double> integrator(1E-6, KEMathIntegrationMethod::Romberg);
    auto integrand = [](double x) {
        return exp(-0.5 * x * x) / sqrt(2 * KConst::Pi());
    };

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

    integrator.SetMethod(KEMathIntegrationMethod::Trapezoidal);
    numerical = integrator.Integrate(integrand, -3.0, 3.0);

    EXPECT_EQ(16385U, integrator.NumberOfSteps());
    ASSERT_NEAR(analytical, numerical, precision);

    integrator.SetMinSteps(2048);
    EXPECT_EQ(2049U, integrator.GetMinSteps());
}


#ifdef TBB

#include "KMathIntegratorThreaded.h"

#include <tbb/task_scheduler_init.h>

TEST(KommonMath, ThreadedIntegrator)
{
    int nThreads = tbb::task_scheduler_init::default_num_threads();
    GTEST_LOG_(INFO) << "Default number of TBB threads: " << nThreads;

    //    tbb::task_scheduler_init init( 1 );

    const double analytical = 0.5 * (erf(3.0 / sqrt(2)) - erf(-3.0 / sqrt(2)));
    double numerical, precision;

    precision = 1E-12;
    KMathIntegratorThreaded<double> integrator(precision, KEMathIntegrationMethod::Trapezoidal);
    integrator.SetMaxSteps(1 << 24);
    auto integrand = [](double x) {
        return exp(-0.5 * x * x) / sqrt(2 * KConst::Pi());
    };

    numerical = integrator.Integrate(integrand, -3.0, 3.0);

    EXPECT_EQ(524289U, integrator.NumberOfSteps());
    ASSERT_NEAR(analytical, numerical, precision);
}

#endif  // TBB
