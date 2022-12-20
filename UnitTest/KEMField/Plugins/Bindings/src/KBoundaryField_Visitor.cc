/*
 * KBoundaryField_Visitor.cc
 *
 *  Created on: 30 Jul 2015
 *      Author: wolfgang
 */

#include "KVTKViewerAsBoundaryFieldVisitor.hh"
#include "KContainer.hh"
#include "KEMFieldTest.hh"
#include "KElectrostaticBoundaryField.hh"
#include "KElectrostaticBoundaryFieldBuilder.hh"
#include "KVTKViewerVisitorBuilder.hh"

using namespace KEMField;
using namespace katrin;

TEST_F(KEMFieldTest, KBoundaryField_Visitor_VTKViewer_Inheritance)
{
    KContainer container;
    container.Set(new KVTKViewerAsBoundaryFieldVisitor);
    ASSERT_TRUE(container.Is<KElectrostaticBoundaryField::Visitor>());
}

TEST_F(KEMFieldTest, KBoundaryField_Visitor_VTKViewerBuilder_Inheritance)
{
    KVTKViewerVisitorBuilder builder;
    builder.Begin();
    ASSERT_TRUE(builder.Is<KElectrostaticBoundaryField::Visitor>());
}

TEST_F(KEMFieldTest, KBoundaryField_Visitor_Process_VTKViewer)
{
    KVTKViewerVisitorBuilder* viewerBuilder = new KVTKViewerVisitorBuilder;
    viewerBuilder->Begin();
    KElectrostaticBoundaryFieldBuilder fieldBuilder;
    fieldBuilder.Begin();

    KElectrostaticBoundaryField::Visitor* soonDeletedContainerPointer =
        viewerBuilder->AsPointer<KElectrostaticBoundaryField::Visitor>();

    fieldBuilder.AddElement(viewerBuilder);

    KElectrostaticBoundaryField* field = fieldBuilder.AsPointer<KElectrostaticBoundaryField>();

    typedef std::vector<KEMField::KSmartPointer<KElectrostaticBoundaryField::Visitor>> VisitorList;

    VisitorList list = field->GetVisitors();
    KElectrostaticBoundaryField::Visitor* fieldPointer = &(*(list.at(0)));
    ASSERT_EQ(fieldPointer, soonDeletedContainerPointer);
}
