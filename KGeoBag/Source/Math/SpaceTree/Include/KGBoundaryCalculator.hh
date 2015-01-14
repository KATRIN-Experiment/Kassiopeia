#ifndef KGBoundaryCalculator_HH__
#define KGBoundaryCalculator_HH__

#include "KGPoint.hh"
#include "KGBall.hh"
#include "KGAxisAlignedBox.hh"
#include "KGCube.hh"

#include "KGBallSupportSet.hh"
#include "KGAxisAlignedBoxSupportSet.hh"
#include "KGPointCloud.hh"
#include "KGBallCloud.hh"

#include "KGNumericalConstants.hh"

namespace KGeoBag
{

/*
*
*@file KGBoundaryCalculator.hh
*@class KGBoundaryCalculator
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Sun Aug 18 19:10:36 CEST 2013 J. Barrett (barrettj@mit.edu) First Version
*
*/

template<size_t NDIM>
class KGBoundaryCalculator
{
    public:
        KGBoundaryCalculator(){;}
        virtual ~KGBoundaryCalculator(){;};

        void AddPoint(const KGPoint<NDIM>* p)
        {
            fBallSupportSet.AddPoint(*p);
            fBoxSupportSet.AddPoint(*p);
            fIsEmpty = false;
        }

        void AddPoint(const KGPoint<NDIM>& p)
        {
            fBallSupportSet.AddPoint(p);
            fBoxSupportSet.AddPoint(p);
            fIsEmpty = false;
        }

        void AddBall(const KGBall<NDIM>* sph)
        {
            //this is not exact...just an approximation

            KGPoint<NDIM> sph_cen = sph->GetCenter();
            double sph_r = sph->GetRadius();
            KGPoint<NDIM> point_to_add;

            if(!fIsEmpty)
            {
                //compute the point on the Ball's surface which is farthest
                //from the current minimum bounding balls center
                //add that point to the set and the point opposite it to the set
                KGBall<NDIM> min_ball =fBallSupportSet.GetMinimalBoundingBall();
                fCenter = min_ball.GetCenter();
                fRadius = min_ball.GetRadius();

                KGPoint<NDIM> del = sph_cen - fCenter;
                double len = del.Magnitude();

                if(len > KG_EPSILON)
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
            for(size_t i=0; i<NDIM; i++)
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


        void AddCube(const KGCube<NDIM>* cube)
        {
            for(size_t i=0; i < KGArrayMath::PowerOfTwo<NDIM>::value; i++)
            {
                fBallSupportSet.AddPoint(cube->GetCorner(i));
                fBoxSupportSet.AddPoint(cube->GetCorner(i));
            }
            fIsEmpty = false;
        }

        void AddBox(const KGAxisAlignedBox<NDIM>* box)
        {
            for(size_t i=0; i< KGArrayMath::PowerOfTwo<NDIM>::value; i++)
            {
                fBallSupportSet.AddPoint(box->GetCorner(i));
                fBoxSupportSet.AddPoint(box->GetCorner(i));
            }
            fIsEmpty = false;
        }

        void AddPointCloud(const KGPointCloud<NDIM>* cloud)
        {
            if(cloud->GetNPoints() != 0)
            {
                for(size_t i=0; i<cloud->GetNPoints(); i++)
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


        KGBall<NDIM> GetMinimalBoundingBall() const {return fBallSupportSet.GetMinimalBoundingBall();};

        KGAxisAlignedBox<NDIM> GetMinimalBoundingBox() const {return fBoxSupportSet.GetMinimalBoundingBox();};

        KGCube<NDIM> GetMinimalBoundingCube() const
        {
            KGBall<NDIM> ball = fBallSupportSet.GetMinimalBoundingBall();
            return KGCube<NDIM>( ball.GetCenter(), 2.0*(ball.GetRadius()) );
        }


    private:

        bool fIsEmpty;
        KGBallSupportSet<NDIM>fBallSupportSet;
        KGAxisAlignedBoxSupportSet<NDIM> fBoxSupportSet;
        KGPoint<NDIM> fCenter;
        double fRadius;

};

}


#endif /* KGBoundaryCalculator_H__ */
