/*
 * SimpleClassHierachy.h
 *
 *  Created on: 3 Aug 2015
 *      Author: wolfgang
 */

#ifndef UNITTEST_MOCKUPS_INCLUDE_SIMPLECLASSHIERACHY_H_
#define UNITTEST_MOCKUPS_INCLUDE_SIMPLECLASSHIERACHY_H_

namespace katrin
{

class A
{
    int fNumber;

  public:
    A() : fNumber(0) {}
    virtual ~A() {}
    virtual int Number() const
    {
        return fNumber;
    }
    void SetNumber(int n)
    {
        fNumber = n;
    }
};

class B : public A
{
    int f2ndNumber;

  public:
    B() : f2ndNumber(2)
    {
        SetNumber(1);
    }
    ~B() override {}
    int Number() const override
    {
        return f2ndNumber;
    }
};

}  // namespace katrin
#endif /* UNITTEST_MOCKUPS_INCLUDE_SIMPLECLASSHIERACHY_H_ */
