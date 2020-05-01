/*
 * KEMToolboxFixture.cc
 *
 *  Created on: 6 Aug 2015
 *      Author: wolfgang
 */

#include "KEMToolboxFixture.hh"

using namespace KEMField;

const KDirection KEMToolboxFixture::sFieldStrength = KDirection(1, 1, 1);

KEMField::KElectrostaticConstantField* KEMToolboxFixture::sField = NULL;

void KEMToolboxFixture::SetUp()
{
    KEMToolbox::GetInstance().DeleteAll();
}

void KEMToolboxFixture::ToolboxContainerAdd(std::string name)
{
    sField = new KEMField::KElectrostaticConstantField(sFieldStrength);
    katrin::KContainer* container = new katrin::KContainer();
    container->Set(sField);
    KEMField::KEMToolbox::GetInstance().AddContainer(*container, name);
    delete container;
}
