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

namespace katrin
{

template<> KVTKViewerVisitorBuilder::~KComplexElement() = default;

STATICINT sKVTKViewerVisitorStructure =
    //! specify the name and location of the image data file in .vti format
    KVTKViewerVisitorBuilder::Attribute<std::string>("file") +
    KVTKViewerVisitorBuilder::Attribute<std::string>("path") +
    //! boolean option if true the image will be displayed
    //! in the preprocessing or postprocessing step of the field computation.
    //! Defaults to true.
    KVTKViewerVisitorBuilder::Attribute<bool>("view") +
    //! boolean option to save the generated geometry as a .vti file. Specify
    //! the name with the file parameter.
    //! Defaults to false.
    KVTKViewerVisitorBuilder::Attribute<bool>("save") +
    //! boolean option to start the viewer before the charge density solver
    //! initialization. Defaults to false.
    KVTKViewerVisitorBuilder::Attribute<bool>("preprocessing") +
    //! boolean option to start the viewer after the field solver
    //! initialization. There is no difference in the display between this
    //! option and the preprocessing variant. If both these parameters are
    //! true, the display pops up twice or the file is saved two times.
    //! The second time overwriting the data from the first save operation.
    //! Defaults to false.
    KVTKViewerVisitorBuilder::Attribute<bool>("postprocessing");

STATICINT sKElectrostaticBoundaryField =
    KElectrostaticBoundaryFieldBuilder::ComplexElement<KVTKViewerAsBoundaryFieldVisitor>("viewer");
} /* namespace katrin */
