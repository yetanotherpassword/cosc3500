#percy: percy3.cpp
#	g++ -g -o percy percy3.cpp -std=c++17  
armo: armo.cpp
	g++ armo.cpp -g -o armo -std=c++11  -larmadillo -lblas -Bstatic -Iarmadillo-10.6.2/include/ -Larmadillo-10.6.2/build

