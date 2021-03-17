#ifndef KFMCube_HH__
#define KFMCube_HH__


#include "KFMArrayMath.hh"
#include "KFMPoint.hh"
#include "KSAStructuredASCIIHeaders.hh"

#include <bitset>
#include <climits>
#include <cmath>
#include <limits>
#include <sstream>


namespace KEMField
{


/*
*
*@file KFMCube.hh
*@class KFMCube
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Tue Aug 13 13:41:27 CEST 2013 J. Barrett (barrettj@mit.edu) First Version
*
*/


template<unsigned int NDIM> class KFMCube : public KSAFixedSizeInputOutputObject
{
  public:
    KFMCube()
    {
        for (unsigned int i = 0; i < NDIM + 1; i++) {
            fData[i] = 0.0;
        }
    };

    KFMCube(const double* center, const double& length)
    {
        SetParameters(center, length);
    };

    ~KFMCube() override = default;
    ;

    unsigned int GetDimension() const
    {
        return NDIM;
    };

    inline KFMCube(const KFMCube& copyObject) : KSAFixedSizeInputOutputObject()
    {
        for (unsigned int i = 0; i < NDIM + 1; i++) {
            fData[i] = copyObject.fData[i];
        }
    }

    //geometric property assignment
    void SetParameters(const double* center, const double& length)
    {
        SetCenter(center);
        SetLength(length);
    }
    void SetLength(const double& len)
    {
        fData[NDIM] = len;
    };
    void SetCenter(const double* center)
    {
        for (unsigned int i = 0; i < NDIM; i++) {
            fData[i] = center[i];
        }
    }
    void SetCenter(const KFMPoint<NDIM>& center)
    {
        for (unsigned int i = 0; i < NDIM; i++) {
            fData[i] = center[i];
        }
    }

    //geometric property retrieval
    KFMPoint<NDIM> GetCenter() const
    {
        return KFMPoint<NDIM>(fData);
    };
    void GetCenter(double* center) const
    {
        for (unsigned int i = 0; i < NDIM; i++) {
            center[i] = fData[i];
        }
    }
    double GetLength() const
    {
        return fData[NDIM];
    };
    KFMPoint<NDIM> GetCorner(unsigned int i) const
    {
        KFMPoint<NDIM> corner;
        double length_over_two = fData[NDIM] / 2.0;

        //lower corner
        if (i == 0) {
            for (unsigned int j = 0; j < NDIM; j++) {
                corner[j] = fData[j] - length_over_two;
            }
            return corner;
        }

        //upper corner
        if (i == KFMArrayMath::PowerOfTwo<NDIM>::value - 1) {
            for (unsigned int j = 0; j < NDIM; j++) {
                corner[j] = fData[j] + length_over_two;
            }
            return corner;
        }

        //convert the count number into a set of bools which we can use
        //to tell us which direction the corner is in for each dimension
        std::bitset<sizeof(unsigned int)* CHAR_BIT> twiddle_index = std::bitset<sizeof(unsigned int) * CHAR_BIT>(i);

        for (unsigned int j = 0; j < NDIM; j++) {
            if (twiddle_index[j]) {
                corner[j] = fData[j] + length_over_two;
            }
            else {
                corner[j] = fData[j] - length_over_two;
            }
        }
        return corner;
    }

    //navigation
    bool PointIsInside(const double* p) const
    {
        double length_over_two = fData[NDIM] / 2.0;
        double distance;
        for (unsigned int i = 0; i < NDIM; i++) {
            distance = p[i] - fData[i];  //distance from center in  i-th dimension
            if (distance < -1.0 * length_over_two) {
                return false;
            }
            if (distance > length_over_two) {
                return false;
            }
        }
        return true;
    }

    bool CubeIsInside(const KFMCube<NDIM>* cube) const
    {
        double distance;
        double cube_len_over_two = cube->GetLength() / 2.0;
        for (size_t i = 0; i < NDIM; i++) {
            distance = std::fabs((*cube)[i] - fData[i]);  //distance from center in  i-th dimension
            if (distance + cube_len_over_two > fData[NDIM] / 2.0) {
                return false;
            }
        }
        return true;
    }


    inline KFMCube& operator=(const KFMCube& rhs)
    {
        if (&rhs != this) {
            for (unsigned int i = 0; i < NDIM + 1; i++) {
                fData[i] = rhs.fData[i];
            }
        }
        return *this;
    }

    //access elements
    double& operator[](unsigned int i);
    const double& operator[](unsigned int i) const;


    //IO
    void DefineOutputNode(KSAOutputNode* node) const override;
    void DefineInputNode(KSAInputNode* node) override;


    virtual std::string ClassName() const
    {
        std::stringstream ss;
        ss << "KFMCube<";
        ss << NDIM;
        ss << ">";
        return ss.str();
    };

  protected:
    double fData[NDIM + 1];  //center position + length, last element is length
};

template<unsigned int NDIM> inline double& KFMCube<NDIM>::operator[](unsigned int i)
{
    return fData[i];
}

template<unsigned int NDIM> inline const double& KFMCube<NDIM>::operator[](unsigned int i) const
{
    return fData[i];
}

template<unsigned int NDIM> void KFMCube<NDIM>::DefineOutputNode(KSAOutputNode* node) const
{
    if (node != nullptr) {
        node->AddChild(
            new KSAAssociatedValuePODOutputNode<KFMCube<NDIM>, double, &KFMCube<NDIM>::GetLength>(std::string("L"),
                                                                                                  this));
        node->AddChild(
            new KSAAssociatedPassedPointerPODArrayOutputNode<KFMCube<NDIM>, double, &KFMCube<NDIM>::GetCenter>(
                std::string("C"),
                NDIM,
                this));
    }
}


template<unsigned int NDIM> void KFMCube<NDIM>::DefineInputNode(KSAInputNode* node)
{
    if (node != nullptr) {
        node->AddChild(
            new KSAAssociatedReferencePODInputNode<KFMCube<NDIM>, double, &KFMCube<NDIM>::SetLength>(std::string("L"),
                                                                                                     this));
        node->AddChild(new KSAAssociatedPointerPODArrayInputNode<KFMCube<NDIM>, double, &KFMCube<NDIM>::SetCenter>(
            std::string("C"),
            NDIM,
            this));
    }
}


template<typename Stream> Stream& operator>>(Stream& s, KFMCube<3>& aData)
{
    s.PreStreamInAction(aData);

    for (unsigned int i = 0; i < 4; i++) {
        s >> aData[i];
    }

    s.PostStreamInAction(aData);
    return s;
}

template<typename Stream> Stream& operator<<(Stream& s, const KFMCube<3>& aData)
{
    s.PreStreamOutAction(aData);

    for (unsigned int i = 0; i < 4; i++) {
        s << aData[i];
    }

    s.PostStreamOutAction(aData);

    return s;
}


//this should cover all useful cases
DefineKSAClassName(KFMCube<1>) DefineKSAClassName(KFMCube<2>) DefineKSAClassName(KFMCube<3>)
    DefineKSAClassName(KFMCube<4>)

}  // namespace KEMField


#endif /* KFMCube_H__ */
