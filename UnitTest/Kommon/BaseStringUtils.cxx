/*
 * KBaseStringUtils.cxx
 *
 *  Created on: 26.11.2021
 *      Author: <benedikt.b@wwu.de>
 */

#include "UnitTest.h"

#include "KBaseStringUtils.h"
#include "KException.h"

using namespace katrin;

TEST(KBaseStringUtils, Comparison)
{

    const std::string s1 = "1234";
    const std::string s2 = "1234.5678";
    const std::string s3 = "abcd";
    const std::string s5 = "Abcd";

    EXPECT_TRUE(KBaseStringUtils::Equals(s1, s1));
    EXPECT_FALSE(KBaseStringUtils::Equals(s1, s3));
    EXPECT_FALSE(KBaseStringUtils::Equals(s3, s5));

    EXPECT_TRUE(KBaseStringUtils::IEquals(s1, s1));
    EXPECT_FALSE(KBaseStringUtils::IEquals(s1, s3));
    EXPECT_TRUE(KBaseStringUtils::IEquals(s3, s5));
}

TEST(KBaseStringUtils, Conversion)
{

    const std::string s1 = "1234";
    const std::string s2 = "1234.5678";
    const std::string s3 = "abcd";
    const std::string s4 = "0xfe";
    const std::string s5 = "0x090000001";
    const std::string s6 = "0xabcdefg";

    EXPECT_EQ(KBaseStringUtils::Convert<int>(s1), 1234);
    EXPECT_THROW(KBaseStringUtils::Convert<int>(s2), KException);
    EXPECT_THROW(KBaseStringUtils::Convert<int>(s3), KException);
    EXPECT_EQ(KBaseStringUtils::Convert<int>(s4), 0xfe);
    EXPECT_THROW(KBaseStringUtils::Convert<int>(s5), KException);
    EXPECT_THROW(KBaseStringUtils::Convert<int>(s6), KException);

    EXPECT_NEAR(KBaseStringUtils::Convert<float>(s1), 1234., 1e-4);
    EXPECT_NEAR(KBaseStringUtils::Convert<float>(s2), 1234.5678, 1e-4);
    EXPECT_THROW(KBaseStringUtils::Convert<float>(s3), KException);
    EXPECT_THROW(KBaseStringUtils::Convert<float>(s4), KException);
    EXPECT_THROW(KBaseStringUtils::Convert<float>(s5), KException);
    EXPECT_THROW(KBaseStringUtils::Convert<float>(s6), KException);

    EXPECT_EQ(KBaseStringUtils::Convert<unsigned int>(s4), (unsigned) 0xfe);
    EXPECT_EQ(KBaseStringUtils::Convert<unsigned int>(s5), (unsigned) 0x090000001);

    EXPECT_EQ(KBaseStringUtils::Convert<long int>(s5), 0x090000001);
    EXPECT_THROW(KBaseStringUtils::Convert<long int>(s6), KException);
}


TEST(KBaseStringUtils, Replacing)
{
    EXPECT_EQ("a", KBaseStringUtils::Replace("a", "b", "c"));
    EXPECT_EQ("", KBaseStringUtils::Replace("", "b", "c"));
    EXPECT_EQ("", KBaseStringUtils::Replace("b", "b", ""));
    EXPECT_EQ("ab", KBaseStringUtils::Replace("a", "a", "ab"));
    EXPECT_EQ("bb", KBaseStringUtils::Replace("b", "b", "bb"));
    EXPECT_EQ("bb bbbb", KBaseStringUtils::Replace("b bb", "b", "bb"));
    EXPECT_EQ(" ab abab ", KBaseStringUtils::Replace(" b bb ", "b", "ab"));
    EXPECT_EQ("A quick fox jumps", KBaseStringUtils::Replace("A quick placeholder jumps", "placeholder", "fox"));
}

TEST(KBaseStringUtils, Manipulation)
{
    const std::string s1 = "1234";
    const std::string s2 = "1234  ";
    const std::string s3 = "  1234";
    const std::string s4 = "  1234  ";
    const std::string s5 = ". 1234 .";

    EXPECT_EQ(s1, KBaseStringUtils::Trim(s1));
    EXPECT_EQ(s1, KBaseStringUtils::Trim(s3));
    EXPECT_EQ(s1, KBaseStringUtils::Trim(s2));
    EXPECT_EQ(s1, KBaseStringUtils::Trim(s4));
    EXPECT_NE(s1, KBaseStringUtils::Trim(s5));

    EXPECT_NE(s1, KBaseStringUtils::TrimLeft(s2));
    EXPECT_EQ(s1, KBaseStringUtils::TrimLeft(s3));

    EXPECT_EQ(s1, KBaseStringUtils::TrimRight(s2));
    EXPECT_NE(s1, KBaseStringUtils::TrimRight(s3));
    
    const std::string s6 = "a,b,c";
    const std::string s7 = "a__b__c__d";
    const std::vector<std::string> v6 = {"a", "b", "c"};
    const std::vector<std::string> v7 = {"a", "b", "c", "d"};
    
    const std::string s8 = "a, b ,\n  , c  \r";
    const std::string s9 = "1, 2 ,\n  , 3  \r";
    const std::vector<int> v9 = {1, 2, 3};

    EXPECT_EQ(v6, KBaseStringUtils::SplitTrimAndConvert<std::string>(s6, ","));
    EXPECT_NE(v6, KBaseStringUtils::SplitTrimAndConvert<std::string>(s7, ","));
    EXPECT_NE(v7, KBaseStringUtils::SplitTrimAndConvert<std::string>(s6, "__"));
    EXPECT_EQ(v7, KBaseStringUtils::SplitTrimAndConvert<std::string>(s7, "__"));
    EXPECT_EQ(v7, KBaseStringUtils::SplitTrimAndConvert<std::string>(s7, "_"));
    EXPECT_NE(v7, KBaseStringUtils::SplitTrimAndConvert<std::string>(s7, " "));
    EXPECT_EQ(v6, KBaseStringUtils::SplitTrimAndConvert<std::string>(s8, ","));
    EXPECT_EQ(v9, KBaseStringUtils::SplitTrimAndConvert<int>(s9, ","));

    EXPECT_EQ(s6, KBaseStringUtils::Join(v6, ","));
    EXPECT_NE(s6, KBaseStringUtils::Join(v7, ","));
    EXPECT_NE(s7, KBaseStringUtils::Join(v6, "__"));
    EXPECT_EQ(s7, KBaseStringUtils::Join(v7, "__"));
    EXPECT_NE(s7, KBaseStringUtils::Join(v7, " "));
}
