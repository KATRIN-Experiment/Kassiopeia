#ifndef KFMScalarMultipoleExpansion_H
#define KFMScalarMultipoleExpansion_H


#include "KFMMessaging.hh"
#include "KFMScalarMomentExpansion.hh"
#include "KSAStructuredASCIIHeaders.hh"

#include <cmath>
#include <complex>
#include <vector>

#define MULTIPOLE_INDEX_TABLE_SIZE 441

namespace KEMField
{

/**
*
*@file KFMScalarMultipoleExpansion.hh
*@class KFMScalarMultipoleExpansion
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Fri Aug 24 09:56:34 EDT 2012 J. Barrett (barrettj@mit.edu) First Version
*
*/


class KFMScalarMultipoleExpansion : public KFMScalarMomentExpansion, public KSAFixedSizeInputOutputObject
{
  public:
    KFMScalarMultipoleExpansion();
    ~KFMScalarMultipoleExpansion() override;

    void Clear() override;

    void SetNumberOfTermsInSeries(unsigned int n_terms) override;
    unsigned int GetNumberOfTermsInSeries() const override;

    static unsigned int GetNumberOfTermsFromDegree(unsigned int degree);
    static unsigned int GetDegreeFromNumberOfTerms(unsigned int n_terms);

    virtual void SetDegree(const int& l_max);
    virtual int GetDegree() const;

    void SetMoments(const std::vector<std::complex<double>>* mom) override;
    void GetMoments(std::vector<std::complex<double>>* mom) const override;

    KFMScalarMultipoleExpansion(const KFMScalarMultipoleExpansion& copyObject);
    KFMScalarMultipoleExpansion& operator=(const KFMScalarMultipoleExpansion& rhs);

    void MultiplyByScalar(double scale);


    std::vector<double>* GetRealMoments()
    {
        return &fMomentsReal;
    };
    std::vector<double>* GetImaginaryMoments()
    {
        return &fMomentsImag;
    };
    const std::vector<double>* GetRealMoments() const
    {
        return &fMomentsReal;
    };
    const std::vector<double>* GetImaginaryMoments() const
    {
        return &fMomentsImag;
    };
    void GetRealMoments(std::vector<double>* real_mom) const
    {
        *real_mom = fMomentsReal;
    };
    void GetImaginaryMoments(std::vector<double>* imag_mom) const
    {
        *imag_mom = fMomentsImag;
    };
    void SetRealMoments(const std::vector<double>* real_moments);
    void SetImaginaryMoments(const std::vector<double>* imag_moments);

    //IO
    virtual std::string ClassName() const
    {
        return std::string("KFMScalarMultipoleExpansion");
    };
    void DefineOutputNode(KSAOutputNode* node) const override;
    void DefineInputNode(KSAInputNode* node) override;

    static inline int TriangleNumber(int n)
    {
        return (n * (n + 1)) / 2;
    };  //triangle number
    static inline int ComplexBasisIndex(int l, int m)
    {
        return l * (l + 1) + m;
    };  //complex basis multipole index
    static inline int RealBasisIndex(int l, int m)
    {
        return (l * (l + 1)) / 2 + m;
    };  //real basis multipole index
    static int ComplexDegreeReverseLookUp(int storage_index);
    static int ComplexOrderReverseLookUp(int storage_index);


    void PrintMoments() const
    {
        int degree = GetDegree();
        int si;
        for (int l = 0; l <= degree; l++) {
            for (int m = 0; m <= l; m++) {
                si = RealBasisIndex(l, m);
                kfmout << "M(" << l << ", " << m << ") = (" << fMomentsReal[si] << ", " << fMomentsImag[si] << ")"
                       << kfmendl;
            }
        }
    }

  protected:
    static const int fDegreeTable[MULTIPOLE_INDEX_TABLE_SIZE];
    static const int fOrderTable[MULTIPOLE_INDEX_TABLE_SIZE];

    std::vector<double> fMomentsReal;
    std::vector<double> fMomentsImag;
};

inline KFMScalarMultipoleExpansion::KFMScalarMultipoleExpansion(const KFMScalarMultipoleExpansion& copyObject) :
    KFMScalarMomentExpansion(),
    KSAFixedSizeInputOutputObject()
{
    SetDegree(copyObject.GetDegree());

    for (unsigned int i = 0; i < (copyObject.fMomentsReal).size(); i++) {
        fMomentsReal[i] = copyObject.fMomentsReal[i];
    }

    for (unsigned int i = 0; i < (copyObject.fMomentsImag).size(); i++) {
        fMomentsImag[i] = copyObject.fMomentsImag[i];
    }
}

inline KFMScalarMultipoleExpansion& KFMScalarMultipoleExpansion::operator=(const KFMScalarMultipoleExpansion& rhs)
{
    if (this != &rhs) {
        if (GetNumberOfTermsInSeries() != rhs.GetNumberOfTermsInSeries()) {
            SetDegree(rhs.GetDegree());
        }

        for (unsigned int i = 0; i < (rhs.fMomentsReal).size(); i++) {
            fMomentsReal[i] = rhs.fMomentsReal[i];
        }

        for (unsigned int i = 0; i < (rhs.fMomentsImag).size(); i++) {
            fMomentsImag[i] = rhs.fMomentsImag[i];
        }
    }
    return *this;
}


inline KFMScalarMultipoleExpansion operator+(const KFMScalarMultipoleExpansion& left,
                                             const KFMScalarMultipoleExpansion& right)
{
    KFMScalarMultipoleExpansion val(left);

    std::vector<double>* real = val.GetRealMoments();
    std::vector<double>* imag = val.GetImaginaryMoments();
    const std::vector<double>* right_real = right.GetRealMoments();
    const std::vector<double>* right_imag = right.GetImaginaryMoments();

    unsigned int s = real->size();
    for (unsigned int i = 0; i < s; i++) {
        (*real)[i] += (*right_real)[i];
        (*imag)[i] += (*right_imag)[i];
    }

    return val;
}

inline KFMScalarMultipoleExpansion operator-(const KFMScalarMultipoleExpansion& left,
                                             const KFMScalarMultipoleExpansion& right)
{
    KFMScalarMultipoleExpansion val(left);

    std::vector<double>* real = val.GetRealMoments();
    std::vector<double>* imag = val.GetImaginaryMoments();
    const std::vector<double>* right_real = right.GetRealMoments();
    const std::vector<double>* right_imag = right.GetImaginaryMoments();

    unsigned int s = real->size();
    for (unsigned int i = 0; i < s; i++) {
        (*real)[i] -= (*right_real)[i];
        (*imag)[i] -= (*right_imag)[i];
    }

    return val;
}


inline KFMScalarMultipoleExpansion& operator+=(KFMScalarMultipoleExpansion& left,
                                               const KFMScalarMultipoleExpansion& right)
{
    std::vector<double>* real = left.GetRealMoments();
    std::vector<double>* imag = left.GetImaginaryMoments();
    const std::vector<double>* right_real = right.GetRealMoments();
    const std::vector<double>* right_imag = right.GetImaginaryMoments();

    unsigned int s = real->size();
    for (unsigned int i = 0; i < s; ++i) {
        (*real)[i] += (*right_real)[i];
        (*imag)[i] += (*right_imag)[i];
    }
    return left;
}

inline KFMScalarMultipoleExpansion& operator*=(KFMScalarMultipoleExpansion& left, double right)
{
    std::vector<double>* real = left.GetRealMoments();
    std::vector<double>* imag = left.GetImaginaryMoments();

    unsigned int s = real->size();
    for (unsigned int i = 0; i < s; ++i) {
        (*real)[i] += right * (*real)[i];
        (*imag)[i] += right * (*imag)[i];
    }
    return left;
}


inline KFMScalarMultipoleExpansion& operator-=(KFMScalarMultipoleExpansion& left,
                                               const KFMScalarMultipoleExpansion& right)
{
    std::vector<double>* real = left.GetRealMoments();
    std::vector<double>* imag = left.GetImaginaryMoments();
    const std::vector<double>* right_real = right.GetRealMoments();
    const std::vector<double>* right_imag = right.GetImaginaryMoments();

    unsigned int s = real->size();
    for (unsigned int i = 0; i < s; ++i) {
        (*real)[i] -= (*right_real)[i];
        (*imag)[i] -= (*right_imag)[i];
    }
    return left;
}


template<typename Stream> Stream& operator>>(Stream& s, KFMScalarMultipoleExpansion& aData)
{
    s.PreStreamInAction(aData);

    unsigned int size;
    s >> size;

    std::vector<double>* r_mom = aData.GetRealMoments();
    std::vector<double>* i_mom = aData.GetImaginaryMoments();

    r_mom->resize(size);
    i_mom->resize(size);

    for (unsigned int i = 0; i < size; i++) {
        s >> (*r_mom)[i];
        s >> (*i_mom)[i];
    }

    s.PostStreamInAction(aData);
    return s;
}

template<typename Stream> Stream& operator<<(Stream& s, const KFMScalarMultipoleExpansion& aData)
{
    s.PreStreamOutAction(aData);

    const std::vector<double>* r_mom = aData.GetRealMoments();
    const std::vector<double>* i_mom = aData.GetImaginaryMoments();

    unsigned int size = r_mom->size();
    s << size;

    for (unsigned int i = 0; i < size; i++) {
        s << (*r_mom)[i];
        s << (*i_mom)[i];
    }

    s.PostStreamOutAction(aData);

    return s;
}


DefineKSAClassName(KFMScalarMultipoleExpansion)


}  // namespace KEMField

#endif /* KFMScalarMultipoleExpansion_H */
