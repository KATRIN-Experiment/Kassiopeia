#ifndef Kassiopeia_KSIntCalculatorArgon_h_
#define Kassiopeia_KSIntCalculatorArgon_h_

#include "KSIntCalculator.h"
#include "KMathBilinearInterpolator.h"
#include "KSIntCalculatorHydrogen.h"
#include <vector>
#include <map>

/*
 * KSIntCalculatorArgon.h
 *
 *  Created on: 04.12.2013
 *      Author: oertlin
 */

namespace Kassiopeia
{

    /////////////////////////////////
    /////		Data Reader 	/////
    /////////////////////////////////
    class KSIntCalculatorArgonTotalCrossSectionReader
    {
        private:
            std::map< double, double > *fData;
            std::vector< double > *fParameters;
            unsigned int fNumOfParameters;
            std::istream *fStream;

        public:
            KSIntCalculatorArgonTotalCrossSectionReader( std::istream *aStream, unsigned int numOfParameters );
            ~KSIntCalculatorArgonTotalCrossSectionReader();

            /**
             * \brief Read the data file. In case of an error it returns false. Otherwise true.
             * First, this method reads the parameters, e.g. for extrapolation. The number of parameters
             * is defined by numOfParameters. Then it reads data points (energy and cross-section) as many as available.
             */
            bool Read();

            /**
             * \brief Returns the data map. First parameter of this map is the energy, the second
             * the cross section.
             */
            std::map< double, double >* GetData();

            /**
             * \brief Returns a std::vector with read in parameters. The length of this std::vector is
             * defined by numOfParameters.
             */
            std::vector< double >* GetParameters();
    };

    class KSIntCalculatorArgonDifferentialCrossSectionReader
    {
        private:
            std::map< double*, double > *fData;
            std::vector< double > *fParameters;
            unsigned int fNumOfParameters;
            std::istream *fStream;

        public:
            KSIntCalculatorArgonDifferentialCrossSectionReader( std::istream *aStream, unsigned int numOfParameters );
            ~KSIntCalculatorArgonDifferentialCrossSectionReader();

            /**
             * \brief Read the data file. In case of an error it returns false. Otherwise true.
             * First, this method reads the parameters, e.g. for extrapolation. The number of parameters
             * is defined by numOfParameters. Then it reads data points (energy and cross-section) as many as available.
             */
            bool Read();

            /**
             *\brief Reads files in lxcat format
             *
             *
             */
            bool Readlx();

            /**
             * \brief Returns the data map. First parameter of this map is the energy, the second
             * the cross section.
             */
            std::map< double*, double >* GetData();

            /**
             * \brief Returns a std::vector with read in parameters. The length of this std::vector is
             * defined by numOfParameters.
             */
            std::vector< double >* GetParameters();
    };

    /////////////////////////////////
    /////		Mother			/////
    /////////////////////////////////

    class KSIntCalculatorArgon :
        public KSComponentTemplate< KSIntCalculatorArgon, KSIntCalculator >
    {
        protected:
            std::string fDataFileTotalCrossSection;
            std::string fDataFileDifferentialCrossSection;

        protected:
            /**
             * \brief Calculates the supporting points for interpolation/extrapolation/... for the total cross section.
             * The result is stored in fSupportingPointsTotalCrossSection and fParametersTotalCrossSection. Currently it reads the
             * data file and stores the data and the parameters in they corresponding fields.
             *
             * \parameter numOfParameters The number of parameters which have to read in before data
             */
            virtual void InitializeTotalCrossSection( unsigned int numOfParameters );

            /**
             * \brief Calculates the supporting points for interpolation/extrapolation/... for the differential cross section.
             * The result is stored in fSupportingPointsDifferentialCrossSection and fParametersDifferentialCrossSection.
             * Currently it reads the data file and stores the data and the parameters in they corresponding fields.
             *
             * \parameter numOfParameters The number of parameters which have to read in before data
             */
            virtual void InitializeDifferentialCrossSection( unsigned int numOfParameters );

            /**
             * \brief Calculates the cross-section for the interpolation region. The parameter "point" is
             * the data point which is the first element in the map which goes after "anEnergy".
             * The current implementation executes a linear interpolation.
             *
             * \return Cross-section
             */
            virtual double GetInterpolationForTotalCrossSection( const double &anEnergy, std::map< double, double >::iterator &point ) const;

            /**
             * \brief Calculates the cross-section for extrapolation at high energies. The parameter "point" is
             * the data point which is the first element in the map which goes after "anEnergy".
             * The current implementation executes a power law extrapolation with parameters fParameters[0]
             * and fParameters[1].
             *
             * \return Cross-section
             */
            virtual double GetUpperExtrapolationForTotalCrossSection( const double &anEnergy, std::map< double, double >::iterator &point ) const;

            /**
             * \brief Calculates the cross-section for extrapolation at low energies. The parameter "point" is
             * the data point which is the first element in the map which goes after "anEnergy".
             * The current implementation returns zero.
             *
             * \return Cross-section
             */
            virtual double GetLowerExtrapolationForTotalCrossSection( const double &anEnergy, std::map< double, double >::iterator &point ) const;

            /**
             * \brief Calculates the cross-section for the given energy. It splits up
             * the energy region into three parts: 1. Energy lower than the lowest data point. Then
             * it calls the virtual method GetLowerExtrapolation. 2. Region of data points. So
             * we can do an interpolation. So it calls GetInterpolation. 3. Extrapolation for higher
             * energies. It calls GetUpperExtrapolation.
             *
             * \return Cross-section
             */
            double GetTotalCrossSectionAt( const double &anEnergy ) const;
        public:
            /**
             * \brief Calculates the cross-section for the given energy and angle.
             *
             * \param anEnergy Energy in eV
             * \param anAngle Angle in degree
             * \return Cross-section
             */
            virtual double GetDifferentialCrossSectionAt( const double &anEnergy, const double &anAngle ) const;
        protected:
            /**
             * \brief Calculates theta. The current implementation uses GetDifferentialCrossSectionAt().
             *
             * \return theta in radians
             */
            virtual double GetTheta( const double &anEnergy ) const;

            /**
             * \brief Calculates the energy loss.
             *
             * \return Energy loss
             */
            virtual double GetEnergyLoss( const double &anEnergy, const double &theta ) const = 0;

        public:
            KSIntCalculatorArgon();
            virtual ~KSIntCalculatorArgon();
            std::map< double, double >* DEBUG_GetSupportingPoints();
            std::map< double*, double >* DEBUG_GetSupportingPointsDiffX();

        public:
            virtual void CalculateCrossSection( const KSParticle& aParticle, double& aCrossSection );
            virtual void CalculateCrossSection( const double anEnergy, double& aCrossSection );

        protected:
            std::map< double, double > *fSupportingPointsTotalCrossSection;
            std::vector< double > *fParametersTotalCrossSection;

            /**
             * \brief Two dimensional interpolator for calculations of differential cross-section.
             */
            katrin::KMathBilinearInterpolator< double > *fDifferentialCrossSectionInterpolator;

            //std::map<double*, double> *DEBUG_fSupportingPointsDifferentialCrossSection;
    };

    /////////////////////////////////////
    /////		Elastic Child		/////
    /////////////////////////////////////
    class KSIntCalculatorArgonElastic :
        public KSComponentTemplate< KSIntCalculatorArgonElastic, KSIntCalculatorArgon >
    {
        public:
            KSIntCalculatorArgonElastic();
            KSIntCalculatorArgonElastic( const KSIntCalculatorArgonElastic& aCopy );
            KSIntCalculatorArgonElastic* Clone() const;
            virtual ~KSIntCalculatorArgonElastic();

        public:
            void ExecuteInteraction( const KSParticle& anInitialParticle, KSParticle& aFinalParticle, KSParticleQueue& aSecondaries );

        protected:
            virtual void InitializeComponent();
            virtual double GetEnergyLoss( const double &anEnergy, const double &theta ) const;
            virtual double GetDifferentialCrossSectionAt( const double &anEnergy, const double &anAngle ) const;

    };

    /////////////////////////////////////
    /////		Excited Child		/////
    /////////////////////////////////////
    class KSIntCalculatorArgonExcitation :
        public KSComponentTemplate< KSIntCalculatorArgonExcitation, KSIntCalculatorArgon >
    {
        public:
            KSIntCalculatorArgonExcitation();
            KSIntCalculatorArgonExcitation( const KSIntCalculatorArgonExcitation& aCopy );
            KSIntCalculatorArgonExcitation* Clone() const;
            virtual ~KSIntCalculatorArgonExcitation();

        public:
            void ExecuteInteraction( const KSParticle& anInitialParticle, KSParticle& aFinalParticle, KSParticleQueue& aSecondaries );
            void SetExcitationState( unsigned int aState );

        protected:
            virtual void InitializeComponent();
            virtual double GetEnergyLoss( const double &anEnergy, const double &theta ) const;
            virtual double GetDifferentialCrossSectionAt( const double &anEnergy, const double &anAngle ) const;
            virtual void InitializeDifferentialCrossSection( unsigned int numOfParameters );

        protected:
            unsigned int fExcitationState;
    };

    /////////////////////////////////////
    /////		Ionized Child		/////
    /////////////////////////////////////
    class KSIntCalculatorArgonSingleIonisation :
        public KSComponentTemplate< KSIntCalculatorArgonSingleIonisation, KSIntCalculatorArgon >
    {
        protected:
            double fIonizationEnergy;

        public:
            KSIntCalculatorArgonSingleIonisation();
            KSIntCalculatorArgonSingleIonisation( const KSIntCalculatorArgonSingleIonisation& aCopy );
            KSIntCalculatorArgonSingleIonisation* Clone() const;
            virtual ~KSIntCalculatorArgonSingleIonisation();

        public:
            void ExecuteInteraction( const KSParticle& anInitialParticle, KSParticle& aFinalParticle, KSParticleQueue& aSecondaries );

        protected:
            virtual void InitializeComponent();
            virtual double GetEnergyLoss( const double &anEnergy, const double &anAngle ) const;
            double GetTheta( const double &anEnergy, const double &anEloss ) const;
            double GetDifferentialCrossSectionAt( const double &anEnergy, const double &anAngle, const double &anEloss ) const;
            KSIntCalculatorHydrogenIonisation* DiffCrossCalculator;
    };

    /////////////////////////////////////////////
    /////		Double Ionized Child		/////
    /////////////////////////////////////////////
    class KSIntCalculatorArgonDoubleIonisation :
        public KSComponentTemplate< KSIntCalculatorArgonDoubleIonisation, KSIntCalculatorArgon >
    {
        protected:
            std::vector< double >* fIonizationEnergy;

        public:
            KSIntCalculatorArgonDoubleIonisation();
            KSIntCalculatorArgonDoubleIonisation( const KSIntCalculatorArgonDoubleIonisation& aCopy );
            KSIntCalculatorArgonDoubleIonisation* Clone() const;
            virtual ~KSIntCalculatorArgonDoubleIonisation();

        public:
            void ExecuteInteraction( const KSParticle& anInitialParticle, KSParticle& aFinalParticle, KSParticleQueue& aSecondaries );

        protected:
            virtual void InitializeComponent();
            virtual double GetEnergyLoss( const double &anEnergy, const double & ) const;
            double GetTheta( const double &anEnergy, const double &anEloss ) const;
            double GetDifferentialCrossSectionAt( const double &anEnergy, const double &anAngle, const double &anEloss ) const;
            KSIntCalculatorHydrogenIonisation* DiffCrossCalculator;
    };
} /* namespace Kassiopeia */

#endif

