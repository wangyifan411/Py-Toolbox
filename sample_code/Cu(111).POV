#include "colors.inc"
#include "finish.inc"

global_settings {assumed_gamma 1 max_trace_level 6}
background {color White transmit 1.0}
camera {orthographic
  right -14.83*x up 11.28*y
  direction 1.00*z
  location <0,0,50.00> look_at <0,0,0>}
light_source {<  2.00,   3.00,  40.00> color White
  area_light <0.70, 0, 0>, <0, 0.70, 0>, 3, 3
  adaptive 1 jitter}

#declare simple = finish {phong 0.7}
#declare pale = finish {ambient 0.5 diffuse 0.85 roughness 0.001 specular 0.200 }
#declare intermediate = finish {ambient 0.3 diffuse 0.6 specular 0.1 roughness 0.04}
#declare vmd = finish {ambient 0.0 diffuse 0.65 phong 0.1 phong_size 40.0 specular 0.5 }
#declare jmol = finish {ambient 0.2 diffuse 0.6 specular 1 roughness 0.001 metallic}
#declare ase2 = finish {ambient 0.05 brilliance 3 diffuse 0.6 metallic specular 0.7 roughness 0.04 reflection 0.15}
#declare ase3 = finish {ambient 0.15 brilliance 2 diffuse 0.6 metallic specular 1.0 roughness 0.001 reflection 0.0}
#declare glass = finish {ambient 0.05 diffuse 0.3 specular 1.0 roughness 0.001}
#declare glass2 = finish {ambient 0.01 diffuse 0.3 specular 1.0 reflection 0.25 roughness 0.001}
#declare Rcell = 0.020;
#declare Rbond = 0.100;

#macro atom(LOC, R, COL, TRANS, FIN)
  sphere{LOC, R texture{pigment{color COL transmit TRANS} finish{FIN}}}
#end
#macro constrain(LOC, R, COL, TRANS FIN)
union{torus{R, Rcell rotate 45*z texture{pigment{color COL transmit TRANS} finish{FIN}}}
      torus{R, Rcell rotate -45*z texture{pigment{color COL transmit TRANS} finish{FIN}}}
      translate LOC}
#end

atom(< -5.74,  -2.58,  -2.08>, 1.32, rgb <0.78, 0.50, 0.20>, 0.0, ase3) // #0 
atom(< -3.19,  -2.58,  -2.08>, 1.32, rgb <0.78, 0.50, 0.20>, 0.0, ase3) // #1 
atom(< -0.64,  -2.58,  -2.08>, 1.32, rgb <0.78, 0.50, 0.20>, 0.0, ase3) // #2 
atom(<  1.91,  -2.58,  -2.08>, 1.32, rgb <0.78, 0.50, 0.20>, 0.0, ase3) // #3 
atom(< -4.47,  -0.37,  -2.08>, 1.32, rgb <0.78, 0.50, 0.20>, 0.0, ase3) // #4 
atom(< -1.91,  -0.37,  -2.08>, 1.32, rgb <0.78, 0.50, 0.20>, 0.0, ase3) // #5 
atom(<  0.64,  -0.37,  -2.08>, 1.32, rgb <0.78, 0.50, 0.20>, 0.0, ase3) // #6 
atom(<  3.19,  -0.37,  -2.08>, 1.32, rgb <0.78, 0.50, 0.20>, 0.0, ase3) // #7 
atom(< -3.19,   1.84,  -2.08>, 1.32, rgb <0.78, 0.50, 0.20>, 0.0, ase3) // #8 
atom(< -0.64,   1.84,  -2.08>, 1.32, rgb <0.78, 0.50, 0.20>, 0.0, ase3) // #9 
atom(<  1.91,   1.84,  -2.08>, 1.32, rgb <0.78, 0.50, 0.20>, 0.0, ase3) // #10 
atom(<  4.47,   1.84,  -2.08>, 1.32, rgb <0.78, 0.50, 0.20>, 0.0, ase3) // #11 
atom(< -1.91,   4.05,  -2.08>, 1.32, rgb <0.78, 0.50, 0.20>, 0.0, ase3) // #12 
atom(<  0.64,   4.05,  -2.08>, 1.32, rgb <0.78, 0.50, 0.20>, 0.0, ase3) // #13 
atom(<  3.19,   4.05,  -2.08>, 1.32, rgb <0.78, 0.50, 0.20>, 0.0, ase3) // #14 
atom(<  5.74,   4.05,  -2.08>, 1.32, rgb <0.78, 0.50, 0.20>, 0.0, ase3) // #15 
atom(< -5.74,  -4.05,   0.00>, 1.32, rgb <0.78, 0.50, 0.20>, 0.0, ase3) // #16 
atom(< -3.19,  -4.05,   0.00>, 1.32, rgb <0.78, 0.50, 0.20>, 0.0, ase3) // #17 
atom(< -0.64,  -4.05,   0.00>, 1.32, rgb <0.78, 0.50, 0.20>, 0.0, ase3) // #18 
atom(<  1.91,  -4.05,   0.00>, 1.32, rgb <0.78, 0.50, 0.20>, 0.0, ase3) // #19 
atom(< -4.47,  -1.84,   0.00>, 1.32, rgb <0.78, 0.50, 0.20>, 0.0, ase3) // #20 
atom(< -1.91,  -1.84,   0.00>, 1.32, rgb <0.78, 0.50, 0.20>, 0.0, ase3) // #21 
atom(<  0.64,  -1.84,   0.00>, 1.32, rgb <0.78, 0.50, 0.20>, 0.0, ase3) // #22 
atom(<  3.19,  -1.84,   0.00>, 1.32, rgb <0.78, 0.50, 0.20>, 0.0, ase3) // #23 
atom(< -3.19,   0.37,   0.00>, 1.32, rgb <0.78, 0.50, 0.20>, 0.0, ase3) // #24 
atom(< -0.64,   0.37,   0.00>, 1.32, rgb <0.78, 0.50, 0.20>, 0.0, ase3) // #25 
atom(<  1.91,   0.37,   0.00>, 1.32, rgb <0.78, 0.50, 0.20>, 0.0, ase3) // #26 
atom(<  4.47,   0.37,   0.00>, 1.32, rgb <0.78, 0.50, 0.20>, 0.0, ase3) // #27 
atom(< -1.91,   2.58,   0.00>, 1.32, rgb <0.78, 0.50, 0.20>, 0.0, ase3) // #28 
atom(<  0.64,   2.58,   0.00>, 1.32, rgb <0.78, 0.50, 0.20>, 0.0, ase3) // #29 
atom(<  3.19,   2.58,   0.00>, 1.32, rgb <0.78, 0.50, 0.20>, 0.0, ase3) // #30 
atom(<  5.74,   2.58,   0.00>, 1.32, rgb <0.78, 0.50, 0.20>, 0.0, ase3) // #31 
