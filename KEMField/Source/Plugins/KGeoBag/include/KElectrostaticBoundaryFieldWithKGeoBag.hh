/*
 * KElectrostaticBoundaryFieldWithKGeoBag.hh
 *
 *  Created on: 15 Jun 2015
 *      Author: wolfgang
 */

#ifndef KEMFIELD_SOURCE_2_0_PLUGINS_BINDINGS_FIELDS_ELECTRIC_INCLUDE_KELECTROSTATICBOUNDARYFIELDWITHKGEOBAG_HH_
#define KEMFIELD_SOURCE_2_0_PLUGINS_BINDINGS_FIELDS_ELECTRIC_INCLUDE_KELECTROSTATICBOUNDARYFIELDWITHKGEOBAG_HH_

#include "KElectrostaticBoundaryField.hh"
#include <string>
#include <vector>
#include "KGCore.hh"

#include "KGBEM.hh"
#include "KGBEMConverter.hh"

namespace KEMField {

class KElectrostaticBoundaryFieldWithKGeoBag :
		public KElectrostaticBoundaryField
{
public:
	KElectrostaticBoundaryFieldWithKGeoBag();
	virtual ~KElectrostaticBoundaryFieldWithKGeoBag();
	enum Symmetry { NoSymmetry , AxialSymmetry , DiscreteAxialSymmetry };

	void SetMinimumElementArea( const double& aArea);
	void SetMaximumElementAspectRatio(const double& aAspect);

	void SetSystem( KGeoBag::KGSpace* aSpace );
	void AddSurface( KGeoBag::KGSurface* aSurface );
	void AddSpace( KGeoBag::KGSpace* aSpace );
	void SetSymmetry( const Symmetry& aSymmetry );
	KSmartPointer<KGeoBag::KGBEMConverter> GetConverter();



private:
	virtual double PotentialCore(const KPosition& P) const;
	virtual KEMThreeVector ElectricFieldCore(const KPosition& P) const;
	virtual void InitializeCore();

	void ConfigureSurfaceContainer();

	double fMinimumElementArea;
	double fMaximumElementAspectRatio;
	KGeoBag::KGSpace* fSystem;
	std::vector< KGeoBag::KGSurface* > fSurfaces;
	std::vector< KGeoBag::KGSpace* > fSpaces;
	Symmetry fSymmetry;

private:
	KSmartPointer<KGeoBag::KGBEMConverter> fConverter;
};

} //KEMField



#endif /* KEMFIELD_SOURCE_2_0_PLUGINS_BINDINGS_FIELDS_ELECTRIC_INCLUDE_KELECTROSTATICBOUNDARYFIELDWITHKGEOBAG_HH_ */
