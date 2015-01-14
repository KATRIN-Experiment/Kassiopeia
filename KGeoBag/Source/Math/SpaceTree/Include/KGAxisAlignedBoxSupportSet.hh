#ifndef KGAxisAlignedBoxSupportSet_HH__
#define KGAxisAlignedBoxSupportSet_HH__


#include "KGPoint.hh"
#include "KGAxisAlignedBox.hh"

namespace KGeoBag
{

/*
*
*@file KGAxisAlignedBoxSupportSet.hh
*@class KGAxisAlignedBoxSupportSet
*@brief computes axis aligned bounding box of a set of points
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Sun Aug 18 15:50:46 CEST 2013 J. Barrett (barrettj@mit.edu) First Version
*
*/

template<size_t NDIM>
class KGAxisAlignedBoxSupportSet
{
    public:
        KGAxisAlignedBoxSupportSet(){;};
        virtual ~KGAxisAlignedBoxSupportSet(){;};

        size_t GetNSupportPoints()
        {
            if(fAllPoints.size() <= 1)
            {
                return fAllPoints.size();
            }
            else
            {
                return 2;
            }
        };

        size_t GetNPoints(){return fAllPoints.size();};

        bool AddPoint(const KGPoint<NDIM>& point)
        {
            if(fAllPoints.size() == 0)
            {
                for(size_t i=0; i<NDIM; i++)
                {
                    fLength[i] = 0;
                }

                fCurrentMinimalBoundingBox.SetCenter(point);
                fCurrentMinimalBoundingBox.SetLength(fLength); //no extent

                for(size_t i=0; i<NDIM; i++)
                {
                    fLowerLimits[i] = point[i];
                    fUpperLimits[i] = point[i];
                }

                fAllPoints.push_back(point);
                return true;
            }


            //check if this point is inside our minimum bounding box
            if(fCurrentMinimalBoundingBox.PointIsInside(point))
            {
                //it is inside the current bounding box, so we
                //only need to update the list of all points
                fAllPoints.push_back(point);
                return true;
            }


            bool update;
            for(size_t i=0; i<NDIM; i++)
            {
                update = false;

                if(point[i] < fLowerLimits[i]){fLowerLimits[i] = point[i]; update = true;};
                if(point[i] > fUpperLimits[i]){fUpperLimits[i] = point[i]; update = true;};

                if(update)
                {
                    fCenter[i] = (fLowerLimits[i] + fUpperLimits[i])/2.0;
                    fLength[i] = (fUpperLimits[i] - fLowerLimits[i]);
                }
            }

            fCurrentMinimalBoundingBox.SetCenter(fCenter);
            fCurrentMinimalBoundingBox.SetLength(fLength);

            return true;
        }

        void GetAllPoints( std::vector< KGPoint<NDIM> >* points) const
        {
            *points = fAllPoints;
        }

        void GetSupportPoints( std::vector< KGPoint<NDIM> >* points) const
        {
            if(fAllPoints.size() <= 1)
            {
                *points = fAllPoints;
            }
            else
            {
                points->clear();
                points->push_back(fLowerLimits);
                points->push_back(fUpperLimits);
            }
        }

        void Clear()
        {
            fAllPoints.clear();
        }


        KGAxisAlignedBox<NDIM> GetMinimalBoundingBox() const
        {
            return fCurrentMinimalBoundingBox;
        }

    private:

        //the two support points
        KGPoint<NDIM> fLowerLimits;
        KGPoint<NDIM> fUpperLimits;

        KGAxisAlignedBox<NDIM> fCurrentMinimalBoundingBox;

        std::vector< KGPoint<NDIM> > fAllPoints;

        //scratch space
        double fCenter[NDIM];
        double fLength[NDIM];

};


}//end of KGeoBag

#endif /* KGAxisAlignedBoxSupportSet_H__ */
