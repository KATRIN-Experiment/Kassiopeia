/*
 * KSmartPointer_test.cc
 *
 *  Created on: 23 Jun 2015
 *      Author: wolfgang
 */

#include "KSmartPointer.hh"

#include <iostream>

using namespace KEMField;
using namespace std;

class A
{
  public:
    virtual ~A() {}
    virtual void printName()
    {
        cout << "Class A" << endl;
    }
};

class B : public A
{
  public:
    ~B() override {}
    void printName() override
    {
        cout << "Class B" << endl;
    }
};

int main(int, char**)
{
    KSmartPointer<A> ptr = nullptr;
    ptr = new A;
    ptr->printName();

    ptr = new B;
    ptr->printName();

    return 0;
}
