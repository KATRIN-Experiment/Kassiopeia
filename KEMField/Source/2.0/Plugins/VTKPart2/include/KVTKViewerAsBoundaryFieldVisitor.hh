/*
 * KVTKViewerAsBoundaryFieldVisitor.hh
 *
 *  Created on: 29 Jul 2015
 *      Author: wolfgang
 */

#ifndef KEMFIELD_SOURCE_2_0_PLUGINS_VTK_INCLUDE_KVTKVIEWERASBOUNDARYFIELDVISITOR_HH_
#define KEMFIELD_SOURCE_2_0_PLUGINS_VTK_INCLUDE_KVTKVIEWERASBOUNDARYFIELDVISITOR_HH_

#include "KElectrostaticBoundaryField.hh"

namespace KEMField {

class KVTKViewerAsBoundaryFieldVisitor : public KElectrostaticBoundaryField::Visitor
{
public:
	KVTKViewerAsBoundaryFieldVisitor();
	virtual ~KVTKViewerAsBoundaryFieldVisitor();


	void ViewGeometry( bool choice )
	{
		fViewGeometry = choice;
	}

	void SaveGeometry( bool choice )
	{
		fSaveGeometry = choice;
	}

	void SetFile( string file )
	{
		fFile = file;
	}

	bool ViewGeometry() const
	{
		return fViewGeometry;
	}

	bool SaveGeometry() const
	{
		return fSaveGeometry;
	}

	void PreVisit( KElectrostaticBoundaryField& );
	void PostVisit( KElectrostaticBoundaryField& );

private:

	bool fViewGeometry;
	bool fSaveGeometry;
	std::string fFile;
};

} /* namespace KEMField */

#endif /* KEMFIELD_SOURCE_2_0_PLUGINS_VTK_INCLUDE_KVTKVIEWERASBOUNDARYFIELDVISITOR_HH_ */
