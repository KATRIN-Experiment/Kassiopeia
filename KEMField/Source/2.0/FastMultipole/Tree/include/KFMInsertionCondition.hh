#ifndef KFMInsertionCondition_HH__
#define KFMInsertionCondition_HH__

#include <cstddef>
#include <cmath>

#include "KFMBall.hh"
#include "KFMCube.hh"

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

template<unsigned int NDIM>
class KFMInsertionCondition
{
    public:
        KFMInsertionCondition(){};
        virtual ~KFMInsertionCondition(){};

        virtual bool CanInsertBallInCube(const KFMBall<NDIM>* ball, const KFMCube<NDIM>* cube) const
        {
            if(ball != NULL && cube !=NULL)
            {
                //compute the bounding sphere of this cube
                double length_over_two = (cube->GetLength())/2.0;
                double radius_squared = NDIM*length_over_two*length_over_two;
                KFMPoint<NDIM> cube_bball_center = cube->GetCenter();

                // the ratio 4/3 is an abitrary choice which seems to work fairly well in practice
                KFMBall<NDIM> cube_bball(cube_bball_center, (4.0/3.0)*std::sqrt(radius_squared) );

                KFMPoint<NDIM> center = ball->GetCenter();

                if(cube->PointIsInside(center)) //first we require that the center of the bounding ball be inside the cube
                {
                    if(cube_bball.BallIsInside(ball)) //next we require that the bounding ball itself fit inside the bounding ball of the cube
                    {
                        return true;
                    }
                    else
                    {
//                        std::cout<<"cube bball radius = "<<(4.0/3.0)*std::sqrt(radius_squared)<<std::endl;
//                        std::cout<<"element bball radius = "<<ball->GetRadius()<<std::endl;

//                        KFMPoint<NDIM> center = ball->GetCenter();

//                        double dist = (center - cube_bball_center).Magnitude();

//                        std::cout<<"distance between centers = "<<dist<<std::endl;

                        return false;
                    }
                }
                else
                {
                    return false;
                }

            }
            else
            {
                return false;
            }
        }

    private:
};



}

#endif /* KFMInsertionCondition_H__ */
