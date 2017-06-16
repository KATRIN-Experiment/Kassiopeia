#ifndef Kassiopeia_KESSElasticElsepa_h_
#define Kassiopeia_KESSElasticElsepa_h_

#include "KESSScatteringCalculator.h"
#include <map>

namespace Kassiopeia
{
    class KESSElasticElsepa :
        public KSComponentTemplate< KESSElasticElsepa, KSIntCalculator >,
        public KESSScatteringCalculator
    {
        public:
            KESSElasticElsepa();
            KESSElasticElsepa( const KESSElasticElsepa& aCopy );
            virtual KESSElasticElsepa* Clone() const;
            virtual ~KESSElasticElsepa();

            //***********
            //calculator
            //***********

            virtual void ExecuteInteraction( const KSParticle& anInitialParticle,
                                             KSParticle& aFinalParticle,
                                             KSParticleQueue& aQueue);

            virtual void CalculateCrossSection( const KSParticle& aParticle,
                                                double& aCrossSection );


        private:
            std::map< double, double > fElScMFPMap;

            //!<map contains a dictionary and a std::vector of two std::vectors containing the values
            std::map< double, std::vector< std::vector< double > > > fElScMap;

            double GetScatteringPolarAngle( const double& aKineticEnergy );
    };

}
#endif //Kassiopeia_KESSElasticElsepa_h_
