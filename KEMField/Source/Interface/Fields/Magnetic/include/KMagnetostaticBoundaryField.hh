/*
 * KMagnetostaticBoundaryField.hh
 *
 *  Created on: 07.05.2025
 *      Author: pslocum
 */

#ifndef KMAGNETOSTATICBOUNDARYFIELD_HH_
#define KMAGNETOSTATICBOUNDARYFIELD_HH_

#include "KMagneticChargeDensitySolver.hh"
#include "KEMFileInterface.hh"
#include "KMagneticFFTMFieldSolver.hh"
#include "KMagnetostaticField.hh"
#include "KSurfaceContainer.hh"

namespace KEMField
{

class KMagnetostaticBoundaryField : public KMagnetostaticField
{
  public:
    class Visitor;

    KMagnetostaticBoundaryField();
    ~KMagnetostaticBoundaryField() override;
    void SetChargeDensitySolver(const std::shared_ptr<KMagneticChargeDensitySolver>& solver);
    std::shared_ptr<KMagneticChargeDensitySolver> GetChargeDensitySolver();
    void SetFieldSolver(const std::shared_ptr<KMagneticFFTMFieldSolver>& solver);
    std::shared_ptr<KMagneticFFTMFieldSolver> GetFieldSolver();
    void SetContainer(const std::shared_ptr<KSurfaceContainer>& container);
    std::shared_ptr<KSurfaceContainer> GetContainer() const;

    void AddVisitor(const std::shared_ptr<Visitor>& visitor);
    std::vector<std::shared_ptr<Visitor>> GetVisitors();

    void SetDirectory(const std::string& aDirectory);
    void SetFile(const std::string& aFile);

    void SetHashMaskedBits(const unsigned int& aMaskedBits);
    void SetHashThreshold(const double& aThreshold);

    class Visitor
    {
      public:
        Visitor();
        virtual ~Visitor() = default;
        void Preprocessing(bool choice);
        void Postprocessing(bool choice);
        void InBetweenProcessing(bool choice);
        bool Preprocessing() const;
        bool InBetweenProcessing() const;
        bool Postprocessing() const;
        virtual void PreVisit(KMagnetostaticBoundaryField&) {}
        virtual void InBetweenVisit(KMagnetostaticBoundaryField&) {}
        virtual void PostVisit(KMagnetostaticBoundaryField&) {}

      private:
        bool fPreprocessing;
        bool fInBetweenProcessing;
        bool fPostprocessing;
    };

  protected:
    void CheckSolverExistance();

    void InitializeCore() override;
    void DeinitializeCore() override;

    KFieldVector MagneticFieldCore(const KPosition& P) const override;
    KFieldVector MagneticPotentialCore(const KPosition& P) const override;
    KGradient MagneticGradientCore(const KPosition& P) const override;


  private:
    void VisitorPreprocessing();
    void VisitorInBetweenProcessing();
    void VisitorPostprocessing();

    std::shared_ptr<KMagneticChargeDensitySolver> fChargeDensitySolver;
    std::shared_ptr<KMagneticFFTMFieldSolver> fFieldSolver;
    std::shared_ptr<KSurfaceContainer> fContainer;
    std::vector<std::shared_ptr<Visitor>> fVisitors;

    std::string fFile;
    std::string fDirectory;

    unsigned int fHashMaskedBits;
    double fHashThreshold;
};

}  // namespace KEMField

#endif /* KMAGNETOSTATICBOUNDARYFIELD_HH_ */
