#include "KFMBall.hh"
#include "KFMBoundaryCalculator.hh"
#include "KFMMath.hh"
#include "KFMPoint.hh"
#include "KFMPointCloud.hh"

#include <cmath>
#include <iomanip>
#include <iostream>

//#include <gsl/gsl_rng.h>

#include <cstdlib>

using namespace KEMField;

int main(int /*argc*/, char** /*argv*/)
{

    //	const gsl_rng_type * T;
    //	gsl_rng * r;
    //	gsl_rng_env_setup();

    //	T = gsl_rng_default;
    //	r = gsl_rng_alloc (T);

    const unsigned int NGroups = 100;
    const unsigned int NPointsInGroup = 8;

    double p[NPointsInGroup][3];
    double cent[3];
    KFMPoint<3> centroid;

    KFMBoundaryCalculator<3> calc;

    std::vector<KFMPoint<3>> test_points;

    for (unsigned int n = 0; n < NGroups; n++) {

        cent[0] = 0;
        cent[1] = 0;
        cent[2] = 0;

        test_points.clear();

        //generate three points to make the triangle and compute centroid
        for (auto& j : p) {
            for (unsigned int i = 0; i < 3; i++) {
                j[i] = ((double) rand() / (double) RAND_MAX);
                j[i] = ((double) rand() / (double) RAND_MAX);
                cent[i] += j[i] / ((double) NPointsInGroup);
            }
            test_points.emplace_back(j);
        }

        //compute the approximate bounding sphere
        double radius = 0;
        centroid = KFMPoint<3>(cent);
        for (unsigned int i = 0; i < NPointsInGroup; i++) {
            if ((test_points[i] - centroid).Magnitude() > radius) {
                radius = (test_points[i] - centroid).Magnitude();
            };
        }

        //compute the exact bounding sphere
        calc.Reset();

        for (unsigned int i = 0; i < NPointsInGroup; i++) {
            calc.AddPoint(test_points[i]);
        }

        KFMBall<3> min_ball = calc.GetMinimalBoundingBall();
        KFMPoint<3> center = min_ball.GetCenter();

        if (min_ball.GetRadius() > radius) {
            std::cout << "---------------------------------------------------------------" << std::endl;
            std::cout << " mini_ball R = " << min_ball.GetRadius() << " and rough R = " << radius << std::endl;
            std::cout << " difference (should be negative) = " << min_ball.GetRadius() - radius << std::endl;
            std::cout << "center = " << center[0] << ", " << center[1] << ", " << center[2] << std::endl;
            std::cout << "centroid = " << centroid[0] << ", " << centroid[1] << ", " << centroid[2] << std::endl;
            std::cout << "---------------------------------------------------------------" << std::endl;
            for (unsigned int i = 0; i < NPointsInGroup; i++) {
                std::cout << "p" << i << " = " << test_points[i][0] << ", " << test_points[i][1] << ", "
                          << test_points[i][2] << std::endl;
            }

            std::cout << "---------------------------------------------------------------" << std::endl;
        }
    }


    return 0;
}
