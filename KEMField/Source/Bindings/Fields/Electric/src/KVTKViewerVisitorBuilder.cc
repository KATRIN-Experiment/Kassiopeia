/*
 * KVTKViewerVisitorBuilder.cc
 *
 *  Created on: 30 Jul 2015
 *      Author: wolfgang
 */

#include "KVTKViewerVisitorBuilder.hh"

#include "KElectrostaticBoundaryFieldBuilder.hh"

using namespace KEMField;
using namespace std;

namespace katrin {

template< >
KVTKViewerVisitorBuilder::~KComplexElement()
{
}

STATICINT sKVTKViewerVisitorStructure =
		KVTKViewerVisitorBuilder::Attribute< string >( "file" ) +
		KVTKViewerVisitorBuilder::Attribute< bool >( "view" ) +
		KVTKViewerVisitorBuilder::Attribute< bool >( "save" ) +
		KVTKViewerVisitorBuilder::Attribute< bool >( "preprocessing" ) +
		KVTKViewerVisitorBuilder::Attribute< bool >( "postprocessing" );

STATICINT sKElectrostaticBoundaryField =
		KElectrostaticBoundaryFieldBuilder::ComplexElement< KVTKViewerAsBoundaryFieldVisitor >
		( "viewer" );
} /* namespace katrin */
