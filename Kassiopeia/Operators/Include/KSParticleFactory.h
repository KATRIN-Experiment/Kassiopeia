#ifndef Kassiopeia_KSParticleFactory_h_
#define Kassiopeia_KSParticleFactory_h_

#include "KSParticle.h"
#include "KSingleton.h"

#include <map>

namespace Kassiopeia
{

class KSParticleFactory : public katrin::KSingleton<KSParticleFactory>
{
  public:
    friend class katrin::KSingleton<KSParticleFactory>;

  private:
    KSParticleFactory();
    ~KSParticleFactory() override;

  public:
    KSParticle* Create(const long long& aPID);
    KSParticleQueue Create(const long long& aPID, const size_t& aCount);

    KSParticle* StringCreate(const std::string& aStringID);
    KSParticleQueue StringCreate(const std::string& aStringID, const size_t& aCount);

    int Define(const long long& aPID, const std::string& aStringID, const std::vector<std::string> aAltStringIDs,
               const double& aMass, const double& aCharge, const double& aSpinMagnitude,
               const double& aGyromagneticRatio);

    void SetSpace(KSSpace* aSpace);
    KSSpace* GetSpace();

    void SetMagneticField(KSMagneticField* aMagneticField);
    KSMagneticField* GetMagneticField();

    void SetElectricField(KSElectricField* anElectricField);
    KSElectricField* GetElectricField();

  private:
    using ParticleMap = std::map<long long, KSParticle*>;
    using ParticleIt = ParticleMap::iterator;
    using ParticleCIt = ParticleMap::const_iterator;
    using ParticleEntry = ParticleMap::value_type;

    using ParticleStringMap = std::map<std::string, KSParticle*>;
    using ParticleStringIt = ParticleStringMap::iterator;
    using ParticleStringCIt = ParticleStringMap::const_iterator;
    using ParticleStringEntry = ParticleStringMap::value_type;

    ParticleMap fParticles;
    ParticleStringMap fParticleString;
    KSSpace* fSpace;
    KSMagneticField* fMagneticField;
    KSElectricField* fElectricField;

    long fParticleIndex;
};

}  // namespace Kassiopeia

#endif
