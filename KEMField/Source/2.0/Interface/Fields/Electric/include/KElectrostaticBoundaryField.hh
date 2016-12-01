/*
 * KElectrostaticBoundaryField.hh
 *
 *  Created on: 01.06.2015
 *      Author: gosda
 */

#ifndef KEMFIELD_SOURCE_2_0_FIELDS_ELECTRIC_INCLUDE_KELECTROSTATICBOUNDARYFIELD_HH_
#define KEMFIELD_SOURCE_2_0_FIELDS_ELECTRIC_INCLUDE_KELECTROSTATICBOUNDARYFIELD_HH_

#include "KElectrostaticField.hh"
#include "KChargeDensitySolver.hh"
#include "KElectricFieldSolver.hh"

#include "KEMFileInterface.hh"

#include "KSurfaceContainer.hh"

namespace KEMField {

class KElectrostaticBoundaryField : public KElectrostaticField {
public:
	class Visitor;

	KElectrostaticBoundaryField();
	virtual ~KElectrostaticBoundaryField();
	void SetChargeDensitySolver(KSmartPointer<KChargeDensitySolver> solver);
	KSmartPointer<KChargeDensitySolver> GetChargeDensitySolver();
	void SetFieldSolver(KSmartPointer<KElectricFieldSolver> solver);
	KSmartPointer<KElectricFieldSolver> GetFieldSolver();
	void SetContainer(KSmartPointer<KSurfaceContainer> container);
	KSmartPointer<KSurfaceContainer> GetContainer() const;

	void AddVisitor(KSmartPointer<Visitor> visitor);
	std::vector<KSmartPointer<Visitor> > GetVisitors();

    void SetDirectory( const std::string& aDirectory );
    void SetFile( const std::string& aFile );

    void SetHashMaskedBits( const unsigned int& aMaskedBits );
    void SetHashThreshold( const double& aThreshold );

	class Visitor
	{
	public:
		Visitor();
		virtual ~Visitor() {}
		void Preprocessing( bool choice );
		void Postprocessing( bool choice );
		void InBetweenProcessing(bool choice);
		bool Preprocessing() const;
		bool InBetweenProcessing() const;
		bool Postprocessing() const;
		virtual void PreVisit( KElectrostaticBoundaryField& ) {}
		virtual void InBetweenVisit( KElectrostaticBoundaryField& ) {}
		virtual void PostVisit( KElectrostaticBoundaryField& ) {}

	private:
		bool fPreprocessing;
		bool fInBetweenProcessing;
		bool fPostprocessing;
	};

protected:
	void CheckSolverExistance();

	virtual void InitializeCore();

	virtual double PotentialCore(const KPosition& P) const;
	virtual KEMThreeVector ElectricFieldCore(const KPosition& P) const;
    virtual std::pair<KEMThreeVector,double> ElectricFieldAndPotentialCore(const KPosition& P) const;

private:
	void VisitorPreprocessing();
	void VisitorInBetweenProcessing();
	void VisitorPostprocessing();

	KSmartPointer<KChargeDensitySolver> fChargeDensitySolver;
	KSmartPointer<KElectricFieldSolver> fFieldSolver;
	KSmartPointer<KSurfaceContainer> fContainer;
    std::vector< KSmartPointer<Visitor> > fVisitors;

    std::string fFile;
    std::string fDirectory;

    unsigned int fHashMaskedBits;
    double fHashThreshold;
};

} //KEMField

#endif /* KEMFIELD_SOURCE_2_0_FIELDS_ELECTRIC_INCLUDE_KELECTROSTATICBOUNDARYFIELD_HH_ */
