/**
 * @file Math.cxx
 *
 * Unit testing for Kommon's mathematical utilities.
 *
 * @date 23.11.2015
 * @author Marco Kleesiek <marco.kleesiek@kit.edu>
 */

#include "KMathRegulaFalsi.h"

#include <gtest/gtest.h>

using namespace katrin;
using namespace std;

TEST(KommonMath, RegulaFalsi)
{
    const double epsilon = 1E-6;

    KMathRegulaFalsi<double> rf(epsilon, 100, KEMathRegulaFalsiMethod::AndersonBjoerck);
    auto f = [](double x) {
        return (x - 2.0) * (x - 2.0) * (x - 2.0) + 0.3;
    };

    ASSERT_NEAR(rf.FindIntercept(f, -5000.0, +5000.0), 1.3305668797818582, epsilon);
    ASSERT_LT(rf.GetNEvaluations(), 60);
}
