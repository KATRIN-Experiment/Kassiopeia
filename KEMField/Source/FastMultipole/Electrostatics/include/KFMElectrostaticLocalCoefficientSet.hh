#ifndef KFMElectrostaticLocalCoefficientSet_HH__
#define KFMElectrostaticLocalCoefficientSet_HH__

#include "KFMScalarMultipoleExpansion.hh"

namespace KEMField
{

/*
*
*@file KFMElectrostaticLocalCoefficientSet.hh
*@class KFMElectrostaticLocalCoefficientSet
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Wed Sep  4 10:06:47 CEST 2013 J. Barrett (barrettj@mit.edu) First Version
*
*/

class KFMElectrostaticLocalCoefficientSet : public KFMScalarMultipoleExpansion
{
  public:
    KFMElectrostaticLocalCoefficientSet();
    ~KFMElectrostaticLocalCoefficientSet() override;
    KFMElectrostaticLocalCoefficientSet(const KFMElectrostaticLocalCoefficientSet& copyObject) :
        KFMScalarMultipoleExpansion(copyObject)
    {
        ;
    };
    KFMElectrostaticLocalCoefficientSet& operator=(const KFMElectrostaticLocalCoefficientSet& copyObject) = default;
    ;

    std::string ClassName() const override;

    void DefineOutputNode(KSAOutputNode* node) const override;
    void DefineInputNode(KSAInputNode* node) override;

  private:
};

template<typename Stream> Stream& operator>>(Stream& s, KFMElectrostaticLocalCoefficientSet& aData)
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

template<typename Stream> Stream& operator<<(Stream& s, const KFMElectrostaticLocalCoefficientSet& aData)
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


DefineKSAClassName(KFMElectrostaticLocalCoefficientSet)

}  // namespace KEMField


#endif /* KFMElectrostaticLocalCoefficientSet_H__ */
