#ifndef KFMBoundaryCalculator_HH__
#define KFMBoundaryCalculator_HH__

#include "KFMMath.hh"

#include "KFMPoint.hh"
#include "KFMBall.hh"
#include "KFMBox.hh"
#include "KFMCube.hh"

#include "KFMBallSupportSet.hh"
#include "KFMBoxSupportSet.hh"
#include "KFMPointCloud.hh"
#include "KFMBallCloud.hh"

#include "KFMNumericalConstants.hh"

namespace KEMField
{

/*
*
*@file KFMBoundaryCalculator.hh
*@class KFMBoundaryCalculator
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Sun Aug 18 19:10:36 CEST 2013 J. Barrett (barrettj@mit.edu) First Version
*
*/

template<unsigned int NDIM>
class KFMBoundaryCalculator
{
    public:
        KFMBoundaryCalculator(){;}
        virtual ~KFMBoundaryCalculator(){;};

        void AddPoint(const KFMPoint<NDIM>* p)
        {
            fBallSupportSet.AddPoint(*p);
            fBoxSupportSet.AddPoint(*p);
            fIsEmpty = false;
        }

        void AddPoint(const KFMPoint<NDIM>& p)
        {
            fBallSupportSet.AddPoint(p);
            fBoxSupportSet.AddPoint(p);
            fIsEmpty = false;
        }

        void AddBall(const KFMBall<NDIM>* sph)
        {
            //this is not exact...just an approximation

            KFMPoint<NDIM> sph_cen = sph->GetCenter();
            double sph_r = sph->GetRadius();
            KFMPoint<NDIM> point_to_add;

            if(!fIsEmpty)
            {
                //compute the point on the Ball's surface which is farthest
                //from the current minimum bounding balls center
                //add that point to the set and the point opposite it to the set
                KFMBall<NDIM> min_ball =fBallSupportSet.GetMinimalBoundingBall();
                fCenter = min_ball.GetCenter();
                fRadius = min_ball.GetRadius();

                KFMPoint<NDIM> del = sph_cen - fCenter;
                double len = del.Magnitude();

                if(len > KFM_EPSILON)
                {
                    del *= (1.0)/len; //normalize

                    //add the two points which are at extrema of the ball to be added
                    //relative to the center of the current bounding ball

                    point_to_add = sph_cen + del*sph_r;
                    fBallSupportSet.AddPoint(point_to_add);

                    point_to_add = sph_cen - del*sph_r;
                    fBallSupportSet.AddPoint(point_to_add);
                }
                else
                {
                    //the center to center distance of the two balls is close to zero
                    if(sph_r > fRadius)
                    {
                        //only need to do something if the new ball has a bigger radius
                        //default is to add two extreme points in the first dimesion
                        //is this sufficient??...needs testing!
                        point_to_add = fCenter;

                        point_to_add[0] += sph_r;
                        fBallSupportSet.AddPoint(point_to_add);

                        point_to_add = fCenter;
                        point_to_add[0] -= sph_r;
                        fBallSupportSet.AddPoint(point_to_add);
                    }
                }
            }
            else
            {
                //add two points on the Ball which are opposite each other
                //for convenience we choose the first dimension, but this is arbitrary
                point_to_add = sph_cen;
                point_to_add[0] += sph_r;
                fBallSupportSet.AddPoint(point_to_add);

                point_to_add = sph_cen;
                point_to_add[0] -= sph_r;
                fBallSupportSet.AddPoint(point_to_add);
            }


            //for the bounding box we add the points on the Ball's surface
            //which are farthest in each dimension
            for(unsigned int i=0; i<NDIM; i++)
            {
                point_to_add = sph_cen;
                point_to_add[i] += sph_r;
                fBoxSupportSet.AddPoint(point_to_add);

                point_to_add = sph_cen;
                point_to_add[i] -= sph_r;
                fBoxSupportSet.AddPoint(point_to_add);
            }

            fIsEmpty = false;
        }


        void AddCube(const KFMCube<NDIM>* cube)
        {
            for(unsigned int i=0; i < KFMArrayMath::PowerOfTwo<NDIM>::value; i++)
            {
                fBallSupportSet.AddPoint(cube->GetCorner(i));
                fBoxSupportSet.AddPoint(cube->GetCorner(i));
            }
            fIsEmpty = false;
        }

        void AddBox(const KFMBox<NDIM>* box)
        {
            for(unsigned int i=0; i< KFMArrayMath::PowerOfTwo<NDIM>::value; i++)
            {
                fBallSupportSet.AddPoint(box->GetCorner(i));
                fBoxSupportSet.AddPoint(box->GetCorner(i));
            }
            fIsEmpty = false;
        }

        void AddPointCloud(const KFMPointCloud<NDIM>* cloud)
        {
            if(cloud->GetNPoints() != 0)
            {
                for(unsigned int i=0; i<cloud->GetNPoints(); i++)
                {
                    fBallSupportSet.AddPoint(cloud->GetPoint(i));
                    fBoxSupportSet.AddPoint(cloud->GetPoint(i));
                }
                fIsEmpty = false;
            }
        }


        void Reset()
        {
            fBallSupportSet.Clear();
            fBoxSupportSet.Clear();
            fIsEmpty = true;
        }


        KFMBall<NDIM> GetMinimalBoundingBall() const {return fBallSupportSet.GetMinimalBoundingBall();};

        KFMBox<NDIM> GetMinimalBoundingBox() const {return fBoxSupportSet.GetMinimalBoundingBox();};

        KFMCube<NDIM> GetMinimalBoundingCube() const
        {
            KFMBall<NDIM> ball = fBallSupportSet.GetMinimalBoundingBall();
            return KFMCube<NDIM>( ball.GetCenter(), 2.0*(ball.GetRadius()) );
        }


    private:

        bool fIsEmpty;
        KFMBallSupportSet<NDIM>fBallSupportSet;
        KFMBoxSupportSet<NDIM> fBoxSupportSet;
        KFMPoint<NDIM> fCenter;
        double fRadius;

};

}


#endif /* KFMBoundaryCalculator_H__ */
