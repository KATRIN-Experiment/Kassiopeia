/*
 * KElectrostaticBoundaryField.cc
 *
 *  Created on: 01.06.2015
 *      Author: gosda
 */
#include "KElectrostaticBoundaryField.hh"
#include "KEMSimpleException.hh"

using namespace std;

namespace KEMField {

KElectrostaticBoundaryField::KElectrostaticBoundaryField() :
		fChargeDensitySolver( NULL ),
		fFieldSolver( NULL ),
		fContainer(),
		fFile(),
		fDirectory( KEMFileInterface::GetInstance()->ActiveDirectory() ),
		fHashMaskedBits( 20 ),
		fHashThreshold( 1.e-14 )
{
	fFile = KEMFileInterface::GetInstance()->GetActiveFileName();
	fFile = fFile.substr( fFile.find_last_of( "/" ) + 1, std::string::npos );
}

KElectrostaticBoundaryField::~KElectrostaticBoundaryField()
{
}

void KElectrostaticBoundaryField::SetChargeDensitySolver(
		KSmartPointer<KChargeDensitySolver> solver) {
	fChargeDensitySolver = solver;
}

KSmartPointer<KChargeDensitySolver> KElectrostaticBoundaryField::GetChargeDensitySolver()
{
	return fChargeDensitySolver;
}

void KElectrostaticBoundaryField::SetFieldSolver(
		KSmartPointer<KElectricFieldSolver> solver) {
	fFieldSolver = solver;
}

KSmartPointer<KElectricFieldSolver> KElectrostaticBoundaryField::GetFieldSolver()
{
	return fFieldSolver;
}

void KElectrostaticBoundaryField::SetContainer(KSmartPointer<KSurfaceContainer> container) {
	fContainer = container;
}

KSmartPointer<KSurfaceContainer> KElectrostaticBoundaryField::GetContainer() const
{
	return fContainer;
}

double KElectrostaticBoundaryField::PotentialCore(
		const KPosition& P) const {
	return fFieldSolver->Potential(P);
}

KThreeVector KElectrostaticBoundaryField::ElectricFieldCore(
		const KPosition& P) const {
	return fFieldSolver->ElectricField(P);
}

std::pair<KThreeVector,double>
KElectrostaticBoundaryField::ElectricFieldAndPotentialCore(const KPosition& P) const
{
    return fFieldSolver->ElectricFieldAndPotential(P);
}


void KElectrostaticBoundaryField::InitializeCore() {

	CheckSolverExistance();
    fChargeDensitySolver->SetHashProperties( fHashMaskedBits, fHashThreshold );

    KEMFileInterface::GetInstance()->ActiveDirectory( fDirectory );
    KEMFileInterface::GetInstance()->ActiveFile( KEMFileInterface::GetInstance()->ActiveDirectory() + "/" + fFile );

    VisitorPreprocessing();

    fChargeDensitySolver->Initialize(*fContainer);

    VisitorInBetweenProcessing();

	fFieldSolver->Initialize(*fContainer);

	VisitorPostprocessing();
}

void KElectrostaticBoundaryField::AddVisitor( KSmartPointer<Visitor> visitor )
{
	fVisitors.push_back(visitor);
}

vector<KSmartPointer<KElectrostaticBoundaryField::Visitor> >
		KElectrostaticBoundaryField::GetVisitors()
{
	return fVisitors;
}

void KElectrostaticBoundaryField::SetDirectory( const string& aDirectory )
{
	fDirectory = aDirectory;
	return;
}
void KElectrostaticBoundaryField::SetFile( const string& aFile )
{
	fFile = aFile;
	return;
}

void KElectrostaticBoundaryField::SetHashMaskedBits( const unsigned int& aMaskedBits )
{
	fHashMaskedBits = aMaskedBits;
	return;
}
void KElectrostaticBoundaryField::SetHashThreshold( const double& aThreshold )
{
	fHashThreshold = aThreshold;
	return;
}

void KElectrostaticBoundaryField::VisitorPreprocessing() {
	typedef vector<KSmartPointer<Visitor> >::iterator Viter;
    for (Viter it = fVisitors.begin();it != fVisitors.end();++it)
      if ((*it)->Preprocessing())
        (*it)->PreVisit(*this);
}

void KElectrostaticBoundaryField::VisitorInBetweenProcessing() {
    typedef vector<KSmartPointer<Visitor> >::iterator Viter;
    for (Viter it = fVisitors.begin();it != fVisitors.end();++it)
        if ((*it)->InBetweenProcessing())
            (*it)->InBetweenVisit(*this);
}

void KElectrostaticBoundaryField::VisitorPostprocessing() {
    typedef vector<KSmartPointer<Visitor> >::iterator Viter;
	for (Viter it = fVisitors.begin();it != fVisitors.end();++it)
      if ((*it)->Postprocessing())
        (*it)->PostVisit(*this);
}

void KElectrostaticBoundaryField::CheckSolverExistance()
{
	if( !fChargeDensitySolver )
		throw KEMSimpleException("Initializing aborted: no charge density solver!");

	if( !fFieldSolver )
		throw KEMSimpleException("Initializing aborted: no field solver!");

}
//**********
//visitor
//**********

KElectrostaticBoundaryField::Visitor::Visitor() :
  fPreprocessing( false ),
  fInBetweenProcessing( false ),
  fPostprocessing( false )
{
}

void KElectrostaticBoundaryField::Visitor::Preprocessing(bool choice)
{
	fPreprocessing = choice;
}

void KElectrostaticBoundaryField::Visitor::InBetweenProcessing(bool choice)
{
    fInBetweenProcessing = choice;
}

void KElectrostaticBoundaryField::Visitor::Postprocessing(bool choice)
{
	fPostprocessing = choice;
}

bool KElectrostaticBoundaryField::Visitor::Preprocessing() const
{
	return fPreprocessing;
}

bool KElectrostaticBoundaryField::Visitor::InBetweenProcessing() const
{
    return fInBetweenProcessing;
}

bool KElectrostaticBoundaryField::Visitor::Postprocessing() const
{
	return fPostprocessing;
}

} //KEMField
