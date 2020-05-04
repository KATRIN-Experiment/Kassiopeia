#ifndef KFMTrianglePolarBasisCalculator_HH__
#define KFMTrianglePolarBasisCalculator_HH__


#include "KFMPointCloud.hh"

#include <cmath>
#include <vector>

namespace KEMField
{

/**
*
*@file KFMTrianglePolarBasisCalculator.hh
*@class KFMTrianglePolarBasisCalculator
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Tue Nov 20 13:28:26 EST 2012 J. Barrett (barrettj@mit.edu) First Version
*
*/

struct KFMTrianglePolarBasis
{
    double h;     //height of triangle along x-axis
    double area;  //area of the triangle
    double phi1;  //lower angle from x-axis
    double phi2;  //upper angle from x-axis

    //components of the x-axis unit vector
    double e0x;
    double e0y;
    double e0z;

    //componets of the y-axis unit vector
    double e1x;
    double e1y;
    double e1z;

    //components of the z-axis unit vector
    double e2x;
    double e2y;
    double e2z;
};


class KFMTrianglePolarBasisCalculator
{
  public:
    KFMTrianglePolarBasisCalculator();
    virtual ~KFMTrianglePolarBasisCalculator();

    void Convert(const KFMPointCloud<3>* vertices, KFMTrianglePolarBasis& basis);

    void SetPointCloud(const KFMPointCloud<3>* vertices);

    void ConstructBasis();

    double GetH() const
    {
        return fH;
    };
    double GetPhi1() const
    {
        return fPhi1;
    };
    double GetPhi2() const
    {
        return fPhi2;
    };

  protected:
    double fP[3][3];
    double fQ[3];
    double fN0_1[3];
    double fN0_2[3];
    double fNPerp[3];
    double fX[3];
    double fY[3];
    double fZ[3];

    double fH;
    double fArea;
    double fPhi1;
    double fPhi2;


    static inline void SetEqual(const double* vec1, double* vec2)
    {
        vec2[0] = vec1[0];
        vec2[1] = vec1[1];
        vec2[2] = vec1[2];
    }


    static inline double Magnitude(const double* vec)
    {
        return std::sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]);
    }

    static inline void Normalize(double* vec)
    {
        double norm = std::sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]);
        if (norm != 0) {
            vec[0] /= norm;
            vec[1] /= norm;
            vec[2] /= norm;
        }
    }

    static inline void Cross(const double* vec1, const double* vec2, double* out)
    {
        out[0] = vec1[1] * vec2[2] - vec1[2] * vec2[1];
        out[1] = vec1[2] * vec2[0] - vec1[0] * vec2[2];
        out[2] = vec1[0] * vec2[1] - vec1[1] * vec2[0];
    }

    static inline double Dot(const double* vec1, const double* vec2)
    {
        return vec1[0] * vec2[0] + vec1[1] * vec2[1] + vec1[2] * vec2[2];
    }

    static inline void Add(const double* vec1, const double* vec2, double* out)
    {
        out[0] = vec1[0] + vec2[0];
        out[1] = vec1[1] + vec2[1];
        out[2] = vec1[2] + vec2[2];
    }

    static inline void Subtract(const double* vec1, const double* vec2, double* out)
    {
        out[0] = vec1[0] - vec2[0];
        out[1] = vec1[1] - vec2[1];
        out[2] = vec1[2] - vec2[2];
    }

    static inline void ScalarMultiply(const double& fac, double* vec)
    {
        vec[0] = fac * vec[0];
        vec[1] = fac * vec[1];
        vec[2] = fac * vec[2];
    }
};

}  // namespace KEMField


#endif /* __KFMTrianglePolarBasisCalculator_H__ */
