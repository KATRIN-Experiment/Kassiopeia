#ifndef Kassiopeia_KSParticleFactory_h_
#define Kassiopeia_KSParticleFactory_h_

#include "KSingleton.h"
using katrin::KSingleton;

#include "KSParticle.h"

#include <map>
using std::map;

namespace Kassiopeia
{

    class KSParticleFactory :
        public KSingleton< KSParticleFactory >
    {
        public:
            friend class KSingleton< KSParticleFactory >;

        private:
            KSParticleFactory();
            virtual ~KSParticleFactory();

        public:
            KSParticle* Create( const long long& aPID );
	    KSParticle* StringCreate( const std::string& aStringID );
	    int Define( const long long& aPID, const std::string& aStringID, const std::vector<std::string> aAltStringIDs, const double& aMass, const double& aCharge, const double& aSpinMagnitude, const double& aGyromagneticRatio );

            void SetSpace( KSSpace* aSpace );
            KSSpace* GetSpace();

            void SetMagneticField( KSMagneticField* aMagneticField );
            KSMagneticField* GetMagneticField();

            void SetElectricField( KSElectricField* anElectricField );
            KSElectricField* GetElectricField();

        private:
            typedef map< long long, KSParticle* > ParticleMap;
            typedef ParticleMap::iterator ParticleIt;
            typedef ParticleMap::const_iterator ParticleCIt;
            typedef ParticleMap::value_type ParticleEntry;

	    typedef map< std::string, KSParticle* > ParticleStringMap;
            typedef ParticleStringMap::iterator ParticleStringIt;
            typedef ParticleStringMap::const_iterator ParticleStringCIt;
            typedef ParticleStringMap::value_type ParticleStringEntry;

            ParticleMap fParticles;
	    ParticleStringMap fParticleString;
            KSSpace* fSpace;
            KSMagneticField* fMagneticField;
            KSElectricField* fElectricField;
    };

}

#endif
