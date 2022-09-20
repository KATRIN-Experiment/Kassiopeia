#ifndef Kassiopeia_KSTermZHRadius_h_
#define Kassiopeia_KSTermZHRadius_h_

#include "KSTerminator.h"

namespace KEMField {
class KZonalHarmonicMagnetostaticFieldSolver;
class KElectricZHFieldSolver;
}

namespace Kassiopeia
{

class KSMagneticField;
class KSElectricField;

class KSTermZHRadius : public KSComponentTemplate<KSTermZHRadius, KSTerminator>
{
  public:
    KSTermZHRadius();
    KSTermZHRadius(const KSTermZHRadius& aCopy);
    KSTermZHRadius* Clone() const override;
    ~KSTermZHRadius() override;

    void AddMagneticField(KSMagneticField* field);
    const std::vector<KSMagneticField*> GetMagneticFields() const;

    void AddElectricField(KSElectricField* field);
    const std::vector<KSElectricField*> GetElectricFields() const;

    void SetCheckCentralExpansion(bool aFlag = true);
    bool GetCheckCentralExpansion() const;

    void SetCheckRemoteExpansion(bool aFlag = true);
    bool GetCheckRemoteExpansion() const;

public:
    void CalculateTermination(const KSParticle& anInitialParticle, bool& aFlag) override;
    void ExecuteTermination(const KSParticle& anInitialParticle, KSParticle& aFinalParticle,
                            KSParticleQueue& aParticleQueue) const override;

    void InitializeComponent() override;

  private:
    std::vector<KSMagneticField*> fMagneticFields;
    std::vector<KSElectricField*> fElectricFields;

    bool fCheckCentralExpansion;
    bool fCheckRemoteExpansion;

    std::vector<KEMField::KZonalHarmonicMagnetostaticFieldSolver*> fMagneticSolvers;
    std::vector<KEMField::KElectricZHFieldSolver*> fElectricSolvers;
};

}  // namespace Kassiopeia

#endif
