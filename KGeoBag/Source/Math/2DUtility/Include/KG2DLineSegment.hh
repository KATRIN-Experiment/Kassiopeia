#ifndef __KG2DLineSegment_H__
#define __KG2DLineSegment_H__

#include "KTwoVector.hh"

#include "KG2DShape.hh"

#define SMALLNUMBER 1e-9

namespace KGeoBag
{

/**
*
*@file KG2DLineSegment.hh
*@class KG2DLineSegment
*@brief utility class for 2d line segments
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Fri Jul 27 16:58:02 EDT 2012 J. Barrett (barrettj@mit.edu) First Version
*
*/

class KG2DLineSegment: public KG2DShape
{
    public:
        KG2DLineSegment();
        KG2DLineSegment(double point1[2], double point2[2]);
        KG2DLineSegment(const KTwoVector& point1, const KTwoVector& point2);

        virtual ~KG2DLineSegment(){;};

        //setters
        void SetPoints(const KTwoVector& point1, const KTwoVector& point2);
        void SetFirstPoint(const KTwoVector& point1);
        void SetSecondPoint(const KTwoVector& point2);
        void Initialize();

        //getters
        double GetLength() const {return fLength;};
        KTwoVector GetFirstPoint() const {return fP1;};
        KTwoVector GetSecondPoint() const {return fP2;};
        KTwoVector GetUnitVector() const {return fUnit;};

        //geometry utilities
        virtual void NearestDistance( const KTwoVector& aPoint, double& aDistance ) const;
        virtual KTwoVector Point( const KTwoVector& aPoint ) const;
        virtual KTwoVector Normal( const KTwoVector& /*aPoint*/ ) const;
        virtual void NearestIntersection( const KTwoVector& aStart, const KTwoVector& anEnd, bool& aResult, KTwoVector& anIntersection ) const;

        void NearestIntersection(const KG2DLineSegment& aSegment, bool& aResult, KTwoVector& anIntersection ) const;

        //static helper function
        static int FindIntersection1D(double a, double b, double c, double d, double result[2]);

        static void Swap(double& val1, double& val2)
        {
            double temp;
            temp = val1;
            val1 = val2;
            val2 = temp;
        }

        static void Swap(int& val1, int& val2)
        {
            int temp;
            temp = val1;
            val1 = val2;
            val2 = temp;
        }

        //static utility functions for navigation
        static double NearestDistance(const KTwoVector& p0, const KTwoVector& p1, //(p0,p1) describe 2d line sgment
                                      const KTwoVector& aPoint); // point to be tested against

        static KTwoVector NearestPoint(const KTwoVector& p0, const KTwoVector& p1, //(p0,p1) describe 2d line sgment
                                       const KTwoVector& aPoint ); // point to be tested against

        static KTwoVector NearestNormal(const KTwoVector& p0, const KTwoVector& p1, //(p0,p1) describe 2d line sgment
                                        const KTwoVector& /*aPoint*/ ); // point to be tested against

        static bool NearestIntersection(const KTwoVector& p0, const KTwoVector& p1, //(p0,p1) describe 2d line sgment
                                        const KTwoVector& aStart, const KTwoVector& anEnd, //line segment to be tested against
                                        KTwoVector& anIntersection ); //if return value is true, intersection point will be returned here


    protected:

        KTwoVector fP1;
        KTwoVector fP2;
        KTwoVector fDiff;
        KTwoVector fUnit;
        double fLength;
        double fLength2;

};

}//end of kgeobag namespace

#endif /* __KG2DLineSegment_H__ */
