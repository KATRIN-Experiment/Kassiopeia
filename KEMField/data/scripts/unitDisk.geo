scale = .5;

x = 0;
y = 0;
z = 0;
r = 1;

  p1 = newp; Point(p1) = {x,  y,  z,  scale} ;
  p2 = newp; Point(p2) = {x+r,y,  z,   scale} ;
  p3 = newp; Point(p3) = {x,  y+r,z,   scale} ;
  p4 = newp; Point(p4) = {x-r,y,  z,   scale} ;
  p5 = newp; Point(p5) = {x,  y-r,z,   scale} ;
 
  c1 = newreg; Circle(c1) = {p2,p1,p3};
  c2 = newreg; Circle(c2) = {p3,p1,p4};
  c3 = newreg; Circle(c3) = {p4,p1,p5};
  c4 = newreg; Circle(c4) = {p5,p1,p2}; 
 
  ll1 = newreg; Line Loop(ll1) = {c1,c2,c3,c4}; 

  s1 = newreg; Ruled Surface(newreg) = {ll1};
 
