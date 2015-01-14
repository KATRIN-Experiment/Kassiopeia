#ifndef Kassiopeia_KSFieldElectromagnet_h_
#define Kassiopeia_KSFieldElectromagnet_h_

#include "KSMagneticField.h"
#include "KSFieldsMessage.h"

#include "KGElectromagnet.hh"
#include "KGElectromagnetConverter.hh"

#include "KMD5HashGenerator.hh"
#include "KEMFileInterface.hh"
using KEMField::KMD5HashGenerator;
using KEMField::Type2Type;
using KEMField::KEMFileInterface;

#include "KElectromagnetIntegratingFieldSolver.hh"
using KEMField::KElectromagnetIntegrator;
using KEMField::KIntegratingFieldSolver;

#include "KElectromagnetZonalHarmonicFieldSolver.hh"
#include "KZonalHarmonicContainer.hh"
#include "KZonalHarmonicParameters.hh"
using KEMField::KZonalHarmonicFieldSolver;
using KEMField::KZonalHarmonicContainer;
using KEMField::KZonalHarmonicParameters;

using namespace KGeoBag;

namespace Kassiopeia
{

    class KSFieldElectromagnet :
        public KSComponentTemplate< KSFieldElectromagnet, KSMagneticField >
    {
        public:
            KSFieldElectromagnet();
            KSFieldElectromagnet( const KSFieldElectromagnet& aCopy );
            KSFieldElectromagnet* Clone() const;
            virtual ~KSFieldElectromagnet();

        public:
            void CalculatePotential( const KThreeVector& aSamplePoint, const double& aSampleTime, KThreeVector& aPotential );
            void CalculateField( const KThreeVector& aSamplePoint, const double& aSampleTime, KThreeVector& aField );
            void CalculateGradient( const KThreeVector& aSamplePoint, const double& aSampleTime, KThreeMatrix& aGradient );

        public:
            void SetDirectory( const string& aDirectory );
            void SetFile( const string& aFile );
            void SetSystem( KGSpace* aSpace );
            void AddSurface( KGSurface* aSurface );
            void AddSpace( KGSpace* aSpace );

        private:
            string fDirectory;
            string fFile;
            KGSpace* fSystem;
            vector< KGSurface* > fSurfaces;
            vector< KGSpace* > fSpaces;

        public:
            class FieldSolver
            {
                public:
                    FieldSolver();
                    virtual ~FieldSolver();

                    virtual KEMThreeVector VectorPotential( const KPosition& P ) const = 0;
                    virtual KEMThreeVector MagneticField( const KPosition& P ) const = 0;
                    virtual KGradient MagneticFieldGradient( const KPosition& P ) const = 0;

                    virtual void Initialize( KElectromagnetContainer& container ) = 0;
                    virtual void Deinitialize() = 0;

                protected:
                    bool fInitialized;

            };

            class IntegratingFieldSolver :
                public FieldSolver
            {
                public:
                    IntegratingFieldSolver();
                    virtual ~IntegratingFieldSolver();

                    void Initialize( KElectromagnetContainer& container );
                    void Deinitialize();

                    KEMThreeVector VectorPotential( const KPosition& P ) const;
                    KEMThreeVector MagneticField( const KPosition& P ) const;
                    KGradient MagneticFieldGradient( const KPosition& P ) const;

                private:
                    KElectromagnetIntegrator fIntegrator;
                    KIntegratingFieldSolver< KElectromagnetIntegrator >* fIntegratingFieldSolver;
            };

            class ZonalHarmonicFieldSolver :
                public FieldSolver
            {
                public:
                    ZonalHarmonicFieldSolver();
                    virtual ~ZonalHarmonicFieldSolver();

                    void Initialize( KElectromagnetContainer& container );
                    void Deinitialize();

                    KEMThreeVector VectorPotential( const KPosition& P ) const;
                    KEMThreeVector MagneticField( const KPosition& P ) const;
                    KGradient MagneticFieldGradient( const KPosition& P ) const;

                    KZonalHarmonicParameters* GetParameters()
                    {
                        return fParameters;
                    }

                private:
                    KElectromagnetIntegrator fIntegrator;
                    KZonalHarmonicContainer< KMagnetostaticBasis >* fZHContainer;
                    KZonalHarmonicFieldSolver< KMagnetostaticBasis >* fZonalHarmonicFieldSolver;
                    KZonalHarmonicParameters* fParameters;
            };

        public:
            KGElectromagnetConverter* GetConverter()
            {
                return fConverter;
            }
            KElectromagnetContainer* GetContainer()
            {
                return fContainer;
            }

            void SetFieldSolver( FieldSolver* solver )
            {
                if( fFieldSolver != NULL )
                {
                    fieldmsg( eError ) << "tried to assign more than one electromagnet field solver" << eom;
                    return;
                }
                fFieldSolver = solver;
                return;
            }
            FieldSolver* GetFieldSolver()
            {
                return fFieldSolver;
            }

        private:
            void InitializeComponent();
            void DeinitializeComponent();

            KGElectromagnetConverter* fConverter;
            KElectromagnetContainer* fContainer;
            FieldSolver* fFieldSolver;
    };

}

#endif
