/*
 * KEMToolboxTest.cc
 *
 *  Created on: 4 Aug 2015
 *      Author: wolfgang
 */

#include "KEMToolbox.hh"

#include "KEMToolboxFixture.hh"
#include "KElectricQuadrupoleField.hh"
#include "KElectrostaticConstantField.hh"
#include "KKeyNotFoundException.hh"

using namespace KEMField;

TEST_F(KEMToolboxFixture, KEMToolbox_PickingFromEmptyToolbox)
{
    try {
        KEMToolbox::GetInstance().Get<KElectricField>("field1");
        ASSERT_FALSE("Should have thrown exception.");
    }
    catch (const KKeyNotFoundException& exception) {
        ASSERT_EQ(string(exception.what()),
                  string(KKeyNotFoundException("KEMToolbox", "field1", KKeyNotFoundException::noEntry).what()));
    }
}

TEST_F(KEMToolboxFixture, KEMToolbox_SimplePointer)
{
    ToolboxAdd<KElectrostaticConstantField>("field1");
    KElectrostaticConstantField* ptr = KEMToolbox::GetInstance().Get<KElectrostaticConstantField>("field1");
    ASSERT_EQ(ptr, sField);
}

TEST_F(KEMToolboxFixture, KEMToolbox_DeleteAll)
{
    ToolboxAdd<KElectrostaticConstantField>("field1");
    KEMToolbox::GetInstance().DeleteAll();
    try {
        KEMToolbox::GetInstance().Get<KElectrostaticConstantField>("field1");
        ASSERT_FALSE("Should have thrown an exception.");
    }
    catch (const KKeyNotFoundException& exception) {
        ASSERT_EQ(string(exception.what()),
                  string(KKeyNotFoundException("KEMToolbox", "field1", KKeyNotFoundException::noEntry).what()));
    }
}

TEST_F(KEMToolboxFixture, KEMToolbox_BasePointer)
{
    ToolboxAdd<KElectricField>("field2");
    KElectricField* ptr = KEMToolbox::GetInstance().Get<KElectricField>("field2");
    ASSERT_EQ(ptr, sField);
}

TEST_F(KEMToolboxFixture, KEMToolbox_BaseToNormalPtr)
{
    ToolboxAdd<KElectricField>("field3");
    try {
        KEMToolbox::GetInstance().Get<KElectrostaticConstantField>("field3");
        ASSERT_FALSE("Should have thrown an exception.");
    }
    catch (const KKeyNotFoundException& exception) {
        ASSERT_EQ(string(exception.what()),
                  string(KKeyNotFoundException("KEMToolbox", "field3", KKeyNotFoundException::wrongType).what()));
    }
}


TEST_F(KEMToolboxFixture, KEMToolbox_NormalToBasePtr)
{
    ToolboxAdd<KElectrostaticConstantField>("field4");
    KElectricField* ptr = KEMToolbox::GetInstance().Get<KElectricField>("field4");
    ASSERT_EQ(ptr, sField);
}

TEST_F(KEMToolboxFixture, KEMToolbox_BaseToWrongPtr)
{
    ToolboxAdd<KElectricField>("field5");
    try {
        KEMToolbox::GetInstance().Get<KElectricQuadrupoleField>("field5");
        ASSERT_FALSE("Should have thrown an exception.");
    }
    catch (const KKeyNotFoundException& exception) {
        ASSERT_EQ(string(exception.what()),
                  string(KKeyNotFoundException("KEMToolbox", "field5", KKeyNotFoundException::wrongType).what()));
    }
}

TEST_F(KEMToolboxFixture, KEMToolbox_NoSuchEntry)
{
    ToolboxAdd<KElectricField>("field6");
    try {
        KEMToolbox::GetInstance().Get<KElectricField>("noField");
        ASSERT_FALSE("Should have thrown an exception.");
    }
    catch (const KKeyNotFoundException& exception) {
        ASSERT_EQ(string(exception.what()),
                  string(KKeyNotFoundException("KEMToolbox", "noField", KKeyNotFoundException::noEntry).what()));
    }
}

TEST(KEMToolboxDeathTest, KEMToolbox_KeyAlreadyExists)
{
    KDirection fieldStrength(1, 1, 1);
    KElectrostaticConstantField* field = new KElectrostaticConstantField(fieldStrength);
    KEMToolbox::GetInstance().Add<KElectricField>("field7", field);
    ASSERT_DEATH(KEMToolbox::GetInstance().Add<KElectricField>("field7", field), "");
}

TEST_F(KEMToolboxFixture, KEMToolbox_DoubleAccess)
{
    ToolboxAdd<KElectricField>("field8");
    KEMToolbox::GetInstance().Get<KElectricField>("field8");
    KElectricField* ptr = KEMToolbox::GetInstance().Get<KElectricField>("field8");
    ASSERT_EQ(ptr, sField);
}

TEST_F(KEMToolboxFixture, KEMToolbox_ScopeTest)
{
    {
        KElectrostaticConstantField* field = new KElectrostaticConstantField(sFieldStrength);
        KEMToolbox::GetInstance().Add<KElectricField>("field9", field);
    }
    KElectricField* ptr = KEMToolbox::GetInstance().Get<KElectricField>("field9");
    ASSERT_EQ(sFieldStrength.X(), ptr->ElectricField(KPosition(0, 0, 0), 0).X());
}

TEST_F(KEMToolboxFixture, KEMToolbox_DirectFromContainer)
{
    ToolboxContainerAdd("field10");
    auto ptr = KEMToolbox::GetInstance().Get<KElectrostaticField>("field10");
    ASSERT_EQ(ptr, sField);
}

TEST_F(KEMToolboxFixture, KEMToolbox_DirectFromContainer_KeyAlreadyExists)
{
    ToolboxContainerAdd("field11");
    ASSERT_DEATH(ToolboxContainerAdd("field11"), "");
}

TEST_F(KEMToolboxFixture, KEMToolbox_GetAll_EmptyToolbox)
{
    auto vec = KEMToolbox::GetInstance().GetAll<KElectricQuadrupoleField>();
    ASSERT_EQ(vec.size(), 0);
}

TEST_F(KEMToolboxFixture, KEMToolbox_GetAll_FullToolbox)
{
    ToolboxContainerAdd("field12");
    ToolboxContainerAdd("field13");
    KEMToolbox::GetInstance().Add<KElectricQuadrupoleField>("field14", new KElectricQuadrupoleField);
    auto vec = KEMToolbox::GetInstance().GetAll<KElectricField>();
    ASSERT_EQ(vec.size(), 3);
}

TEST_F(KEMToolboxFixture, KEMToolbox_GetAll_One_Wrong_Kind)
{
    ToolboxContainerAdd("field15");
    ToolboxContainerAdd("field16");
    KEMToolbox::GetInstance().Add<KElectricQuadrupoleField>("field17", new KElectricQuadrupoleField);
    auto vec = KEMToolbox::GetInstance().GetAll<KElectrostaticConstantField>();
    ASSERT_EQ(vec.size(), 2);
}
