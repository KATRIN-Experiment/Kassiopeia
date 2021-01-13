#ifndef KGBallCloud_HH__
#define KGBallCloud_HH__

#include "KGBall.hh"

#include <vector>

namespace KGeoBag
{

/*
*
*@file KGBallCloud.hh
*@class KGBallCloud
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Sat Aug 24 14:27:45 CEST 2013 J. Barrett (barrettj@mit.edu) First Version
*
*/


template<size_t NDIM> class KGBallCloud
{
  public:
    KGBallCloud() = default;
    ;
    virtual ~KGBallCloud() = default;
    ;

    size_t GetNBalls() const
    {
        return fBalls.size();
    }

    void AddBall(const KGBall<NDIM>& Ball)
    {
        fBalls.push_back(Ball);
    }

    void Clear()
    {
        fBalls.clear();
    }

    KGBall<NDIM> GetBall(size_t i) const
    {
        return fBalls[i];
    };  //no check performed

    void GetBalls(std::vector<KGBall<NDIM>>* Balls) const
    {
        *Balls = fBalls;
    }

    std::vector<KGBall<NDIM>>* GetBalls()
    {
        return &fBalls;
    }


  private:
    size_t fID;
    std::vector<KGBall<NDIM>> fBalls;
};


}  // namespace KGeoBag


#endif /* KGBallCloud_H__ */
