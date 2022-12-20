#ifndef Kassiopeia_KSGenGeneratorTextFile_h_
#define Kassiopeia_KSGenGeneratorTextFile_h_

#include "KField.h"
#include "KSComponentTemplate.h"
#include "KSGenerator.h"
#include "KSParticle.h"
#include "KTextFile.h"

#include <string>
#include <vector>

namespace Kassiopeia
{

class KSGenGeneratorTextFile : public KSComponentTemplate<KSGenGeneratorTextFile, KSGenerator>
{
  public:
    KSGenGeneratorTextFile();
    KSGenGeneratorTextFile(const KSGenGeneratorTextFile& aCopy);
    KSGenGeneratorTextFile* Clone() const override;
    ~KSGenGeneratorTextFile() override;

  public:
    void ExecuteGeneration(KSParticleQueue& aPrimaries) override;

  public:
    // note that some of these member variables cannot be set via XML bindings yet
    ;
    K_SET_GET(std::string, Base);
    ;
    K_SET_GET(std::string, Path);
    ;

  protected:
    void InitializeComponent() override;
    void DeinitializeComponent() override;

    void GenerateParticlesFromFile(KSParticleQueue& aParticleQueue);

  private:
    katrin::KTextFile* fTextFile;
};

}  // namespace Kassiopeia

#endif
