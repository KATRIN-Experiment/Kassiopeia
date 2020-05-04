#ifndef Kassiopeia_KSGenGeneratorSimulation_h_
#define Kassiopeia_KSGenGeneratorSimulation_h_

#include "KField.h"
#include "KSComponentTemplate.h"
#include "KSGenerator.h"
#include "KSParticle.h"
#include "KSReadFileROOT.h"
#include "TFormula.h"

#include <vector>
using std::vector;

#include <string>
using std::string;

namespace Kassiopeia
{

class KSGenGeneratorSimulation : public KSComponentTemplate<KSGenGeneratorSimulation, KSGenerator>
{
  public:
    KSGenGeneratorSimulation();
    KSGenGeneratorSimulation(const KSGenGeneratorSimulation& aCopy);
    KSGenGeneratorSimulation* Clone() const override;
    ~KSGenGeneratorSimulation() override;

  public:
    void ExecuteGeneration(KSParticleQueue& aPrimaries) override;

  public:
    // note that some of these member variables cannot be set via XML bindings yet
    ;
    K_SET_GET(std::string, Base);
    ;
    K_SET_GET(std::string, Path);
    ;
    K_SET_GET(std::string, PositionX);
    ;
    K_SET_GET(std::string, PositionY);
    ;
    K_SET_GET(std::string, PositionZ);
    ;
    K_SET_GET(std::string, DirectionX);
    ;
    K_SET_GET(std::string, DirectionY);
    ;
    K_SET_GET(std::string, DirectionZ);
    ;
    K_SET_GET(std::string, Energy);
    ;
    K_SET_GET(std::string, Time);
    ;
    K_SET_GET(std::string, Terminator);
    ;
    K_SET_GET(std::string, Generator);
    ;
    K_SET_GET(std::string, TrackGroupName);
    ;
    K_SET_GET(std::string, TerminatorName);
    ;
    K_SET_GET(std::string, GeneratorName);
    ;
    K_SET_GET(std::string, PositionName);
    ;
    K_SET_GET(std::string, MomentumName);
    ;
    K_SET_GET(std::string, KineticEnergyName);
    ;
    K_SET_GET(std::string, TimeName);
    ;
    K_SET_GET(std::string, PIDName);
    ;
    K_SET_GET(KThreeVector, DefaultPosition);
    ;
    K_SET_GET(KThreeVector, DefaultDirection);
    ;
    K_SET_GET(double, DefaultEnergy);
    ;
    K_SET_GET(double, DefaultTime);
    ;
    K_SET_GET(int, DefaultPID);

  protected:
    void InitializeComponent() override;
    void DeinitializeComponent() override;

    void GenerateParticlesFromFile(KSParticleQueue& aParticleQueue);

  private:
    KRootFile* fRootFile;

    TFormula* fFormulaPositionX;
    TFormula* fFormulaPositionY;
    TFormula* fFormulaPositionZ;
    TFormula* fFormulaDirectionX;
    TFormula* fFormulaDirectionY;
    TFormula* fFormulaDirectionZ;
    TFormula* fFormulaEnergy;
    TFormula* fFormulaTime;
};

}  // namespace Kassiopeia

#endif
