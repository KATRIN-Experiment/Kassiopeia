/*
 * KStringUtils.cxx
 *
 *  Created on: 27 Jun 2015
 *      Author: wolfgang
 */

#include "UnitTest.h"

#ifdef BOOST

#include "KStringUtils.h"

using namespace katrin;


TEST(KStringUtils, Comparison)
{

    const std::string s1 = "1234";
    const std::string s2 = "1234.5678";
    const std::string s3 = "abcd";
    const std::string s4 = "abcd_____";
    const std::string s5 = "Abcd";
    const std::string s6 = "abcX";

    EXPECT_TRUE(KStringUtils::IsNumeric(s1));
    EXPECT_TRUE(KStringUtils::IsNumeric(s2));
    EXPECT_FALSE(KStringUtils::IsNumeric(s3));

    EXPECT_TRUE(KStringUtils::Contains(s2, s1));
    EXPECT_TRUE(KStringUtils::Contains(s4, s3));
    EXPECT_FALSE(KStringUtils::Contains(s4, s5));
    EXPECT_FALSE(KStringUtils::Contains(s4, s6));

    EXPECT_TRUE(KStringUtils::IContains(s2, s1));
    EXPECT_TRUE(KStringUtils::IContains(s4, s3));
    EXPECT_TRUE(KStringUtils::IContains(s4, s5));
    EXPECT_FALSE(KStringUtils::IContains(s4, s6));

    EXPECT_FALSE(KStringUtils::ContainsOneOf(s1, {".", "_"}));
    EXPECT_TRUE(KStringUtils::ContainsOneOf(s2, {".", "_"}));
    EXPECT_FALSE(KStringUtils::ContainsOneOf(s3, {".", "_"}));
    EXPECT_TRUE(KStringUtils::ContainsOneOf(s4, {".", "_"}));
    EXPECT_TRUE(KStringUtils::ContainsOneOf(s3, {"a", "x"}));
    EXPECT_TRUE(KStringUtils::ContainsOneOf(s4, {"a", "x"}));
    EXPECT_FALSE(KStringUtils::ContainsOneOf(s5, {"a", "X"}));
    EXPECT_FALSE(KStringUtils::ContainsOneOf(s6, {"A", "x"}));

    EXPECT_TRUE(KStringUtils::IContainsOneOf(s5, {"a", "X"}));
    EXPECT_TRUE(KStringUtils::IContainsOneOf(s6, {"A", "x"}));
    EXPECT_FALSE(KStringUtils::IContainsOneOf(s5, {"z", "X"}));
    EXPECT_TRUE(KStringUtils::IContainsOneOf(s6, {"Z", "x"}));

    EXPECT_EQ(5U, KStringUtils::Distance(s3, s4));
    EXPECT_EQ(1U, KStringUtils::Distance(s3, s5));
    EXPECT_EQ(2U, KStringUtils::Distance(s5, s6));

    EXPECT_EQ(5U, KStringUtils::IDistance(s3, s4));
    EXPECT_EQ(0U, KStringUtils::IDistance(s3, s5));
    EXPECT_EQ(1U, KStringUtils::IDistance(s5, s6));

    EXPECT_GE(0.5, KStringUtils::Similarity(s5, s6));
    EXPECT_GE(1.0, KStringUtils::ISimilarity(s5, s6));
}

TEST(KStringUtils, Manipulation)
{
    const std::string s8 = "123456789";

    EXPECT_EQ("123,456,789", KStringUtils::GroupDigits(123456789));
    EXPECT_EQ("123,456,789", KStringUtils::GroupDigits(s8));
    EXPECT_NE("123456789", KStringUtils::GroupDigits(s8));
    EXPECT_EQ("123456789", KStringUtils::GroupDigits(s8, ""));
    EXPECT_EQ("123 456 789", KStringUtils::GroupDigits(s8, " "));
}

TEST(KStringUtils, Generation)
{

    for (unsigned i = 0; i < 100; i++) {
        for (unsigned tLength = 1; tLength < 10; tLength++) {
            auto s = KStringUtils::RandomAlphaNum(tLength);
            EXPECT_EQ(tLength, s.length());
            std::for_each(s.begin(), s.end(), [](char c) { EXPECT_TRUE(std::isalnum(c)); });
        }
    }
}

#endif
