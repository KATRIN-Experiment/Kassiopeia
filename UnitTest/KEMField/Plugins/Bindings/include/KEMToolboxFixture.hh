/*
 * KEMToolboxTest.hh
 *
 *  Created on: 6 Aug 2015
 *      Author: wolfgang
 */

#ifndef UNITTEST_KEMFIELD_PLUGINS_BINDINGS_INCLUDE_KEMTOOLBOXFIXTURE_HH_
#define UNITTEST_KEMFIELD_PLUGINS_BINDINGS_INCLUDE_KEMTOOLBOXFIXTURE_HH_


#include "KEMFieldTest.hh"
#include "KEMToolbox.hh"
#include "KElectrostaticConstantField.hh"

class KEMToolboxFixture : public KEMFieldTest
{
  protected:
    virtual void SetUp();

    template<class Field> void ToolboxAdd(std::string);
    void ToolboxContainerAdd(std::string);

    static const KEMField::KDirection sFieldStrength;
    static KEMField::KElectrostaticConstantField* sField;
};

template<class Field> void KEMToolboxFixture::ToolboxAdd(std::string name)
{
    sField = new KEMField::KElectrostaticConstantField(sFieldStrength);
    KEMField::KEMToolbox::GetInstance().Add<Field>(name, sField);
}

#endif /* UNITTEST_KEMFIELD_PLUGINS_BINDINGS_INCLUDE_KEMTOOLBOXFIXTURE_HH_ */
