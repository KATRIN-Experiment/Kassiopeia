/*
 * KElectrostaticBoundaryField.hh
 *
 *  Created on: 01.06.2015
 *      Author: gosda
 */

#ifndef KELECTROSTATICBOUNDARYFIELD_HH_
#define KELECTROSTATICBOUNDARYFIELD_HH_

#include "KChargeDensitySolver.hh"
#include "KEMFileInterface.hh"
#include "KElectricFieldSolver.hh"
#include "KElectrostaticField.hh"
#include "KSurfaceContainer.hh"

namespace KEMField
{

class KElectrostaticBoundaryField : public KElectrostaticField
{
  public:
    class Visitor;

    KElectrostaticBoundaryField();
    ~KElectrostaticBoundaryField() override;
    void SetChargeDensitySolver(const std::shared_ptr<KChargeDensitySolver>& solver);
    std::shared_ptr<KChargeDensitySolver> GetChargeDensitySolver();
    void SetFieldSolver(const std::shared_ptr<KElectricFieldSolver>& solver);
    std::shared_ptr<KElectricFieldSolver> GetFieldSolver();
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
        virtual void PreVisit(KElectrostaticBoundaryField&) {}
        virtual void InBetweenVisit(KElectrostaticBoundaryField&) {}
        virtual void PostVisit(KElectrostaticBoundaryField&) {}

      private:
        bool fPreprocessing;
        bool fInBetweenProcessing;
        bool fPostprocessing;
    };

  protected:
    void CheckSolverExistance();

    void InitializeCore() override;
    void DeinitializeCore() override;

    double PotentialCore(const KPosition& P) const override;
    KFieldVector ElectricFieldCore(const KPosition& P) const override;
    std::pair<KFieldVector, double> ElectricFieldAndPotentialCore(const KPosition& P) const override;

  private:
    void VisitorPreprocessing();
    void VisitorInBetweenProcessing();
    void VisitorPostprocessing();

    std::shared_ptr<KChargeDensitySolver> fChargeDensitySolver;
    std::shared_ptr<KElectricFieldSolver> fFieldSolver;
    std::shared_ptr<KSurfaceContainer> fContainer;
    std::vector<std::shared_ptr<Visitor>> fVisitors;

    std::string fFile;
    std::string fDirectory;

    unsigned int fHashMaskedBits;
    double fHashThreshold;
};

}  // namespace KEMField

#endif /* KELECTROSTATICBOUNDARYFIELD_HH_ */
