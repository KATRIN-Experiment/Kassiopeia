#ifndef Kassiopeia_KESSRelaxation_h_
#define Kassiopeia_KESSRelaxation_h_

#include "KField.h"
#include "KSParticle.h"

namespace Kassiopeia
{
	class KESSRelaxation
	{

		public:

			KESSRelaxation();

			~KESSRelaxation();

            void RelaxAtom( unsigned int vacantShell,
                            const KSParticle& aFinalParticle,
                            KSParticleQueue& aQueue );

            K_SET_GET( double, SiliconBandGap )

		private:

            void CreateAugerElectron( const double& augerEnergy_eV,
                                      const KSParticle& aFinalParticle,
                                      KSParticleQueue& aQueue );

            void RelaxKShell( const KSParticle& aFinalParticle,
                              KSParticleQueue& aQueue );
            void RelaxL1Shell( const KSParticle& aFinalParticle,
                               KSParticleQueue& aQueue);
            void RelaxL23Shell( const KSParticle& aFinalParticle,
                                KSParticleQueue& aQueue);

            void CreateMMVacancies(unsigned int fromThisShell,
                                   const KSParticle& aFinalParticle,
                                   KSParticleQueue& aQueue );

            double GetBindingEnergy(unsigned int ionizedShell);
	};

}//end namespace kassiopeia

#endif /* Kassiopeia_KESSRelaxation_h_ */
