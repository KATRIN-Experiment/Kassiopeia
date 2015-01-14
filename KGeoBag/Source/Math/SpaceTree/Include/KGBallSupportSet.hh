#ifndef KGBallSupportSet_HH__
#define KGBallSupportSet_HH__


#include <vector>
#include <bitset>
#include <utility>

#include "KGNumericalConstants.hh"
#include "KGLinearSystemSolver.hh"

#include "KGPoint.hh"
#include "KGBall.hh"

namespace KGeoBag
{


/*
*
*@file KGBallSupportSet.hh
*@class KGBallSupportSet
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Thu Aug 29 12:19:27 CEST 2013 J. Barrett (barrettj@mit.edu) First Version
*
*/

template<size_t NDIM>
class KGBallSupportSet
{
    public:
        KGBallSupportSet()
        {
            for(size_t i=2; i<NDIM+1; i++)
            {
                fSolvers.push_back( new KGLinearSystemSolver(i) );
            }
            fQ.resize(NDIM);
        };

        virtual ~KGBallSupportSet()
        {
            for(size_t i=0; i<fSolvers.size(); i++)
            {
                delete fSolvers[i];
            }
        };


        size_t GetNSupportPoints(){return fSupportPoints.size();};

        size_t GetNPoints(){return fAllPoints.size();};

        bool AddPoint(const KGPoint<NDIM>& point)
        {
            //returns true when the point has been successfully added to the set
            //and updates the minimum bounding ball of the set
            //if there is an error in computing the new minimum bounding ball
            //the this function returns false and leaves the current point set
            //and bounding ball unaltered

            //first check if the set is empty, if it is, then the minimum bounding ball
            //is trivial
            if(fAllPoints.size() == 0)
            {
                fCurrentMinimalBoundingBall.SetCenter(point);
                fCurrentMinimalBoundingBall.SetRadius(0.);
                fAllPoints.push_back(point);
                fSupportPoints.push_back(point);
                return true;
            }

            if(fAllPoints.size() == 1)
            {
                //only have two points so the minimum bounding ball has its
                //center at the average of the two points
                //and radius equal to half the distance
                fCenter = (fAllPoints[0] + point)/2.0;
                fRadius = ( (fAllPoints[0] - point).Magnitude() )/2.0;
                fCurrentMinimalBoundingBall.SetCenter(fCenter);
                fCurrentMinimalBoundingBall.SetRadius(fRadius);

                fAllPoints.push_back(point);
                fSupportPoints.push_back(point);
                return true;
            }

            //check if this point is inside our minimum bounding ball
            if(fCurrentMinimalBoundingBall.PointIsInside(point))
            {
                //it is inside the current bounding ball, so we
                //only need to update the list of all points, not support set
                fAllPoints.push_back(point);
                return true;
            }
            else
            {
                //point is not inside...but if it is outside by just a little bit
                //we will ignore it because this leads to ill-conditioned matrices
                //when solving for the new sphere

                fRadius = fCurrentMinimalBoundingBall.GetRadius();
                fCenter = fCurrentMinimalBoundingBall.GetCenter();
                double dist = (point - fCenter).Magnitude();

                if(dist < fRadius + KG_EPSILON*fRadius)
                {
                    fAllPoints.push_back(point);
                    return true;
                }
            }

            //set is not empty, and the new point isn't inside the current bounding
            //ball, so we need to compute the new one
            if(fSupportPoints.size() <= NDIM + 1 )
            {
                fValidSetFlags.clear();
                fCandidateBalls.clear();

                size_t SupportSetSize = fSupportPoints.size();
                //compute the number of permutations we must try
                size_t NPermutations = 1;
                for(size_t i=0; i<SupportSetSize; i++)
                {
                    NPermutations *= 2;
                }

                //if the support set size is equal to NDIM+1, then we cannot consider set which considers
                //all of the current points + the new one...to avoid this set, we just reduce the number
                //of permutations by 1

                if(SupportSetSize == NDIM + 1)
                {
                    NPermutations -= 1;
                }

                //for each sub-set in the current set of support points
                //we need to create a new set by joining it with the point we
                //are just adding now, for each of these new sets we must compute
                //the new bounding ball, and check if the points which are not in
                //the new set are inside of it, if this is true it is a 'valid' set
                //then for all valid sets we have found, we must find the one
                //that has the minimum radius
                for(size_t p=1; p<NPermutations; p++) //ignore p=0...because this is an empty set!
                {
                    //convert the count number into a set of bools which we can use
                    //to tell us which points should included int the calculation
                    fTwiddle = std::bitset< sizeof(size_t)*CHAR_BIT >(p);

                    bool success = ComputeBoundingBall(&point, &fSupportPoints, fTwiddle, fScratchBall);

                    if(success)
                    {
                        //now check the validity of this ball

                        //this is the set of points which must be contained in the new ball
                        //in order for it to be valid
                        fTwiddleComplement = ~fTwiddle;

                        if( CheckValidity(&fSupportPoints, fTwiddleComplement, fScratchBall) )
                        {
                            fCandidateBalls.push_back(fScratchBall);
                            fValidSetFlags.push_back(fTwiddle);
                        }
                    }

                }


                //now search the list of candidate balls for the smallest one
                size_t smallest;
                double radius;
                bool found_one = false;
                for(size_t i = 0; i<fCandidateBalls.size(); i++)
                {
                    if(!found_one)
                    {
                        radius = fCandidateBalls[i].GetRadius();
                        found_one = true;
                        smallest = i;
                    }
                    else
                    {
                        if(fCandidateBalls[i].GetRadius()  < radius)
                        {
                            radius = fCandidateBalls[i].GetRadius();
                            smallest = i;
                        }
                    }
                }


                if(found_one)
                {
                    //update bounding ball
                    fCurrentMinimalBoundingBall = fCandidateBalls[smallest];

                    //update the support set
                    fScratchSet.clear();

                    for(size_t i=0; i<fSupportPoints.size(); i++)
                    {
                        if(fValidSetFlags[smallest][i])
                        {
                            fScratchSet.push_back(fSupportPoints[i]);
                        }
                    }
                    fScratchSet.push_back(point);
                    fSupportPoints = fScratchSet;
                    return true;
                }
            }

            //error!
            //if we reach here then we haven't found a solution
            return false;
        }

        void GetAllPoints( std::vector< KGPoint<NDIM> >* points) const
        {
            *points = fAllPoints;
        }

        void GetSupportPoints( std::vector< KGPoint<NDIM> >* points) const
        {
            *points = fSupportPoints;
        }

        void Clear()
        {
            fAllPoints.clear();
            fSupportPoints.clear();
            fScratchSet.clear();
            fCandidateBalls.clear();
            fCurrentMinimalBoundingBall = KGBall<NDIM>();
        }

        KGBall<NDIM> GetMinimalBoundingBall() const {return fCurrentMinimalBoundingBall;};


        bool ComputeBoundingBall(const KGPoint<NDIM>* new_point,
                                   const std::vector< KGPoint<NDIM> >* points,
                                   std::bitset< sizeof(size_t)*CHAR_BIT > mask,
                                   KGBall<NDIM>& ball)
        {
            //just for a new vector of points for the moment
            //(can probably improve this later to avoid copying)
            fScratchSet2.clear();
            for(size_t i=0; i < points->size(); i++)
            {
                if(mask[i])
                {
                    fScratchSet2.push_back(points->at(i));
                }
            }
            fScratchSet2.push_back(*new_point);

            bool success = ComputeBoundingBall(&fScratchSet2, ball);

            return success;
        }


        //checks if the points in the vector specified by the bitset are inside the ball, if they are all inside
        //then this function returns true
        bool CheckValidity(const std::vector< KGPoint<NDIM> >* points,
                             std::bitset< sizeof(size_t)*CHAR_BIT > mask,
                             const KGBall<NDIM>& ball)
        {
            for(size_t i=0; i < points->size(); i++)
            {
                if(mask[i])
                {
                    if( !(ball.PointIsInside( points->at(i) ) ) )
                    {
                        return false;
                    }
                }
            }
            return true;
        }


        bool ComputeBoundingBall( std::vector< KGPoint<NDIM> >* points, KGBall<NDIM>& ball)
        {

            if(points->size() == 0)
            {
                return false;
            }

            if(points->size() == 1)
            {
                ball.SetCenter( points->at(0) );
                ball.SetRadius(0.);
                return true;
            }

            if(points->size() == 2)
            {
                fCenter = ( points->at(0) + points->at(1) )/2.0;
                fRadius = ( ( points->at(0) - points->at(1) ).Magnitude() )/2.0;
                ball.SetCenter(fCenter);
                ball.SetRadius(fRadius);
                return true;
            }

            //from here one the size the size of points is less then or equal to NDIM+1 but greater than 1
            size_t N = points->size();
            size_t NDegreesFreedom = N-1;
            size_t SolverIndex = NDegreesFreedom - 2;
            if(N <= NDIM + 1)
            {
                //pick the first point in the set as the origin
                fOrigin = points->at(0);

                for(size_t i=0; i<NDegreesFreedom; i++)
                {
                    fQ[i] = points->at(i+1) - fOrigin;
                }

                //now we construct the matrix A and the result b
                fSolvers[SolverIndex]->Reset();
                for(size_t i=0; i<NDegreesFreedom; i++)
                {
                    fSolvers[SolverIndex]->SetBVectorElement(i, (fQ[i]*fQ[i]) );
                    for(size_t j=0; j<NDegreesFreedom; j++)
                    {
                        fSolvers[SolverIndex]->SetMatrixElement(i, j, 2.0*(fQ[i]*fQ[j]) );
                    }
                }

                //then solve
                fSolvers[SolverIndex]->Solve();

                //retrieve the solution
                fSolvers[SolverIndex]->GetXVector(fXVector);

                //reset the relative center
                for(size_t i=0; i<NDIM; i++)
                {
                    fRelativeCenter[i] = 0;
                }

                //compute the relative center
                for(size_t i=0; i<NDegreesFreedom; i++)
                {
                    fRelativeCenter += fXVector[i]*fQ[i];
                }

                //compute the radius
                fRadius = std::sqrt( fRelativeCenter*fRelativeCenter );

                //compute the absolute center
                fCenter = fRelativeCenter + fOrigin;

                ball.SetCenter(fCenter);
                ball.SetRadius(fRadius);

                return true;
            }
            else
            {
                //error!!!
                return false;
            }
        }



    private:

        //needed for linear solver
        std::vector< KGLinearSystemSolver* > fSolvers;
        std::vector< KGPoint<NDIM> > fQ;
        KGPoint<NDIM> fRelativeCenter;
        KGPoint<NDIM> fOrigin;
        double fXVector[NDIM];

        //results
        KGPoint<NDIM> fCenter;
        double fRadius;
        KGBall<NDIM> fCurrentMinimalBoundingBall;

        //data sets
        std::vector< KGPoint<NDIM> > fAllPoints;
        std::vector< KGPoint<NDIM> > fSupportPoints;

        //scratch space
        std::vector< KGPoint<NDIM> > fScratchSet;
        std::vector< KGPoint<NDIM> > fScratchSet2;
        KGBall<NDIM> fScratchBall;
        std::vector< KGBall<NDIM> > fCandidateBalls;

        std::bitset< sizeof(size_t)*CHAR_BIT > fTwiddle;
        std::bitset< sizeof(size_t)*CHAR_BIT > fTwiddleComplement;

        std::vector< std::bitset< sizeof(size_t)*CHAR_BIT > > fValidSetFlags;

};


}


#endif /* KGBallSupportSet_H__ */
