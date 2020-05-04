#ifndef KFMInsertionCondition_HH__
#define KFMInsertionCondition_HH__

#include "KFMBall.hh"
#include "KFMCube.hh"

#include <cmath>
#include <cstddef>
#include <iostream>

namespace KEMField
{


/*
*
*@file KFMInsertionCondition.hh
*@class KFMInsertionCondition
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Mon Aug 26 10:27:26 CEST 2013 J. Barrett (barrettj@mit.edu) First Version
*
*/

template<unsigned int NDIM> class KFMInsertionCondition
{
  public:
    KFMInsertionCondition()
    {
        //the ratio 4/3 is an abitrary choice which seems to work fairly well in practice
        fEta = 4.0 / 3.0;
    };
    virtual ~KFMInsertionCondition(){};

    //user can explicitly set the ratio
    //if the ratio goes to infinity then if the center of a bounding ball under test is
    //inside of a cube, then it will be inserted regardless of its radius
    //if the ratio is 1, then a bounding ball will only be inserted if it fits
    //within the bounding ball of the cube
    //a ratio of zero one will cause the bounding ball under test to never be inserted
    //in the cube
    void SetInsertionRatio(double eta)
    {
        fEta = eta;
    };
    double GetInsertionRatio() const
    {
        return fEta;
    };

    virtual bool CanInsertBallInCube(const KFMBall<NDIM>* ball, const KFMCube<NDIM>* cube) const
    {
        if (ball != nullptr && cube != nullptr) {
            //compute the bounding sphere of this cube
            double length_over_two = (cube->GetLength()) / 2.0;
            double radius_squared = NDIM * length_over_two * length_over_two;
            KFMPoint<NDIM> cube_bball_center = cube->GetCenter();

            KFMBall<NDIM> cube_bball(cube_bball_center, fEta * std::sqrt(radius_squared));

            KFMPoint<NDIM> center = ball->GetCenter();

            //first we require that the center of the bounding ball be inside the cube
            if (cube->PointIsInside(center)) {
                //next we require that the bounding ball entirely fit inside the bounding ball of the cube
                //the actual size of the cube's bounding ball is influenced by the parameter fEta
                if (cube_bball.BallIsInside(ball)) {
                    return true;
                }
                else {
                    return false;
                }
            }
            else {
                return false;
            }
        }
        else {
            return false;
        }
    }

  private:
    double fEta;
};


}  // namespace KEMField

#endif /* KFMInsertionCondition_H__ */
